#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation evaluation for PubMedBERT variants:
- Base
- FT definitions (Lin and/or RBP)
- FT PubMed (Lin and/or RBP)

Evaluates:
- Correlation/MSE on the internal test (STS) using the Phase 2 Parquet.
- GSC+ (mention -> HPO).
- Subontology classification (RF, micro-F1) on precomputed labels.

Output: JSON and CSV with metrics per model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase3.train_sentence_transformer import load_pairs_dataframe, evaluate_pairs
from phase4.evaluate_gsc import (
    load_gsc_mentions,
    collect_candidate_terms as collect_gsc_candidates,
    evaluate_model as eval_gsc_model,
)
from phase4.evaluate_jax import (
    read_jax_gold,
    load_documents,
    collect_candidate_terms as collect_jax_candidates,
    evaluate_model as eval_jax_model,
)
from phase4.run_subontology_classification import (
    encode_definitions,
    prepare_dataset,
    get_classifier_configs,
    evaluate_classifier,
)


def eval_sts(model_path: str, test_path: str, batch_size: int) -> Dict[str, float]:
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    df = load_pairs_dataframe(test_path)
    metrics = evaluate_pairs(model, df, batch_size=batch_size) or {}
    return metrics


def eval_gsc(model_path: str, corpus_dir: str, files: List[str], batch_size: int) -> Dict[str, float]:
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    files = [Path(corpus_dir) / f for f in files]
    allowed_ids = None
    candidates = collect_gsc_candidates(allowed_ids)
    candidate_ids = {hpo_id for hpo_id, _ in candidates}
    mention_texts, gold_ids = load_gsc_mentions(files, candidate_ids)
    return eval_gsc_model(model, list(mention_texts), list(gold_ids), candidates, batch_size=batch_size)


def eval_jax(model_path: str, corpus_dir: str, batch_size: int) -> Dict[str, float]:
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    corpus_path = Path(corpus_dir)
    gold = read_jax_gold(corpus_path / "JAX_gold.json")
    allowed_ids = None
    candidates = collect_jax_candidates(allowed_ids)
    candidate_ids = {hpo_id for hpo_id, _ in candidates}
    documents = load_documents(gold, corpus_path / "txt", candidate_ids)
    return eval_jax_model(model, documents, candidates, batch_size=batch_size)


def eval_subontology(
    model_path: str,
    labels_path: str,
    target_level: int,
    min_class_size: int,
    batch_size: int,
    seed: int = 13,
    n_jobs: int = -1,
    cv_folds: int = 5,
) -> Dict[str, float]:
    labels_df = pd.read_parquet(labels_path)
    target_col = f"label_lvl{target_level}"
    emb_df = encode_definitions(model_path, batch_size=batch_size)
    X, y = prepare_dataset(labels_df, emb_df, target_col, min_class_size)
    if X is None or y is None:
        return {}
    configs = get_classifier_configs(n_jobs=n_jobs, seed=seed)
    metrics, _preds = evaluate_classifier("rf", configs["rf"], X, y, cv_folds=cv_folds, seed=seed, n_jobs=n_jobs)
    return {
        "clf": "rf",
        "f1_macro": metrics.get("f1_macro", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMedBERT ablation (STS test, GSC+, classification).")
    parser.add_argument("--test-path", default="data/phase2/splits_struct_rbp/test.parquet")
    parser.add_argument("--jax-dir", default="data/corpus/JAX")
    parser.add_argument("--gsc-dir", default="data/corpus/GSC")
    parser.add_argument("--gsc-files", nargs="*", default=["GSCplus_dev_gold.tsv", "GSCplus_test_gold.tsv"])
    parser.add_argument("--labels-path", default="data/phase4/hpo_subontology_labels.parquet")
    parser.add_argument("--subontology-level", type=int, default=1)
    parser.add_argument("--subontology-min-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "base=NeuML/pubmedbert-base-embeddings",
            "defs_lin=models/phase3/pubmedbert_definition_lin_freeze_angle_with_plots/best_model",
            "defs_rbp=models/phase3/pubmedbert_definition_rbp_freeze_angle_with_plots/best_model",
            "pubmed_lin=models/phase3/pubmedbert_pubmed_lin_best/best_model",
            "pubmed_rbp=models/phase3/pubmedbert_pubmed_rbp_best/best_model",
#            "pubmed_freeze=models/phase3/pubmedbert_rbp_freeze_angle_with_plots/best_model"
        ],
        help="List of variants NAME=PATH.",
    )
    parser.add_argument("--output-json", default="phase4/ablation_pubmedbert.json")
    parser.add_argument("--output-csv", default="phase4/ablation_pubmedbert.csv")
    args = parser.parse_args()

    model_paths: Dict[str, str] = {}
    for item in args.models:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        model_paths[name.strip()] = path.strip()

    rows = []
    summary = {}
    for name, path in model_paths.items():
        print(f"Evaluating {name} ({path})…")
        sts_metrics = eval_sts(path, args.test_path, args.batch_size)
        jax_metrics = eval_jax(path, args.jax_dir, args.batch_size)
        gsc_metrics = eval_gsc(path, args.gsc_dir, args.gsc_files, args.batch_size)
        clf_metrics = eval_subontology(
            path,
            args.labels_path,
            target_level=args.subontology_level,
            min_class_size=args.subontology_min_class,
            batch_size=args.batch_size,
        )
        summary[name] = {"sts": sts_metrics, "jax": jax_metrics, "gsc": gsc_metrics, "clf": clf_metrics}
        row = {
            "model": name,
            "sts_spearman": sts_metrics.get("spearman"),
            "sts_pearson": sts_metrics.get("pearson"),
            "sts_mse": sts_metrics.get("mse"),
            "jax_top1": jax_metrics.get("top1"),
            "jax_top5": jax_metrics.get("top5"),
            "jax_top10": jax_metrics.get("top10"),
            "jax_mrr": jax_metrics.get("mrr"),
            "jax_gold_sim": jax_metrics.get("gold_similarity"),
            "jax_margin": jax_metrics.get("avg_margin"),
            "gsc_top1": gsc_metrics.get("top1"),
            "gsc_top5": gsc_metrics.get("top5"),
            "gsc_mrr": gsc_metrics.get("mrr"),
            "gsc_sim_gold": gsc_metrics.get("gold_similarity"),
            "gsc_margin": gsc_metrics.get("avg_margin"),
            "clf_f1_micro": clf_metrics.get("f1_micro"),
            "clf_f1_macro": clf_metrics.get("f1_macro"),
            "clf_acc": clf_metrics.get("acc"),
        }
        rows.append(row)

    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
