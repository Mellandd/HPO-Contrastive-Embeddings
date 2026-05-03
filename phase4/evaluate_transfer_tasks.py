#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick transfer evaluation on general/biomedical STS datasets to check for
catastrophic forgetting after fine-tuning.

Datasets supported (pulled via Hugging Face `datasets`):
- stsb (GLUE STS-Benchmark): "glue", config "stsb"
- sick (Sentences Involving Compositional Knowledge): "sick"
- biosses (biomedical STS): "biosses"
- medsts (clinical STS): "medsts"

Outputs a CSV with Pearson/Spearman/MSE per model and dataset.

Usage:
  python phase4/evaluate_transfer_tasks.py \
    --models \
      "base=NeuML/pubmedbert-base-embeddings" \
      "ft_pubmed=models/phase3/pubmedbert_pubmed_rbp_angle_best" \
      "ft_defs=models/phase3/pubmedbert_defs_rbp_angle_best" \
    --datasets stsb sick biosses medsts \
    --output phase4/transfer_eval.csv

Notes:
- Requires `datasets` and `sentence-transformers`. Internet is needed to
  download datasets the first time; they will be cached locally.
- For biosses/medsts, labels are in [0, 5]; we normalize to [0, 1] to match
  the internal STS format.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0,1] if they look like 0-5 scale."""
    if not scores:
        return scores
    max_score = max(scores)
    min_score = min(scores)
    if max_score <= 5.0 and min_score >= 0.0:
        return [s / 5.0 for s in scores]
    if max_score > 1.0 or min_score < 0.0:
        # z-normalize then map to 0-1 via min-max for safety
        arr = np.array(scores, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr.tolist()
    return scores


def load_pairs(dataset_name: str) -> Tuple[List[str], List[str], List[float]]:
    name = dataset_name.lower()

    def load_stsb():
        ds = load_dataset("glue", "stsb")
        split = "validation" if "validation" in ds else "test"
        test = ds[split]
        return test["sentence1"], test["sentence2"], normalize_scores(test["label"])

    def load_sick():
        ds = load_dataset("mteb/sickr-sts")
        split = "test" if "test" in ds else "validation"
        test = ds[split]
        return test["sentence1"], test["sentence2"], normalize_scores(test["score"])

    def load_biosses():
        ds = load_dataset("biosses")
        test = ds["test"] if "test" in ds else ds["train"]
        return test["sentence1"], test["sentence2"], normalize_scores(test["score"])

    loaders: Dict[str, Callable[[], Tuple[List[str], List[str], List[float]]]] = {
        "stsb": load_stsb,
        "sick": load_sick,
        "biosses": load_biosses,
    }
    if name not in loaders:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    s1, s2, scores = loaders[name]()
    return list(s1), list(s2), list(scores)


def eval_model(model_path: str, pairs: Tuple[List[str], List[str], List[float]], batch_size: int) -> Dict[str, float]:
    sentences1, sentences2, labels = pairs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    emb1 = model.encode(sentences1, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(sentences2, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    scores = util.cos_sim(emb1, emb2).diag().cpu().numpy()
    labels_arr = np.array(labels, dtype=np.float32)
    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore

        pearson = float(pearsonr(labels_arr, scores)[0])
        spearman = float(spearmanr(labels_arr, scores)[0])
    except Exception:
        pearson = float(np.corrcoef(labels_arr, scores)[0, 1])
        spearman = float(pd.Series(labels_arr).corr(pd.Series(scores), method="spearman"))
    mse = float(np.mean((labels_arr - scores) ** 2))
    return {"pearson": pearson, "spearman": spearman, "mse": mse}


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer evaluation (STS general/biomedical) for multiple models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='Lista NAME=PATH (ej: base=NeuML/pubmedbert-base-embeddings ft_pubmed=models/phase3/pubmedbert_pubmed_rbp_angle_best)',
    )
    parser.add_argument("--datasets", nargs="+", default=["stsb", "sick", "biosses", "medsts"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="phase4/transfer_eval.csv")
    args = parser.parse_args()

    model_paths: Dict[str, str] = {}
    for item in args.models:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        model_paths[name.strip()] = path.strip()

    rows: List[Dict[str, object]] = []

    cache_pairs: Dict[str, Tuple[List[str], List[str], List[float]]] = {}
    for ds_name in args.datasets:
        print(f"Loading dataset {ds_name}…")
        try:
            cache_pairs[ds_name] = load_pairs(ds_name)
        except Exception as exc:
            print(f"WARNING: could not load {ds_name} ({exc}); skipping.")
            continue

    for model_name, model_path in model_paths.items():
        print(f"Evaluating {model_name} ({model_path})…")
        for ds_name, pairs in cache_pairs.items():
            metrics = eval_model(model_path, pairs, batch_size=args.batch_size)
            row = {"model": model_name, "dataset": ds_name, **metrics}
            rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
