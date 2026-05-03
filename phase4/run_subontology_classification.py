#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised classification experiment on HPO subontologies.

Input:
    - Label dataset generated with build_subontology_dataset.py.
    - One or more embedding files per term (parquet with columns
      `hpo_id` and `embedding`).

Output:
    - Cross-validation metrics (accuracy, balanced accuracy, F1 macro)
      for each embedding x classifier combination.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyhpo
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ClassifierConfig = Tuple[Pipeline, Dict[str, List[object]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised classification of HPO subontologies.")
    parser.add_argument("--labels", default="data/phase4/hpo_subontology_labels.parquet")
    parser.add_argument(
        "--embeddings",
        nargs="+",
        help="Lista name=path (e.g. base=data/emb_base.parquet). Optional if using --models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List name=path of SentenceTransformer models to generate embeddings on-the-fly.",
    )
    parser.add_argument("--level", type=int, default=1, help="Label level to predict (label_lvlX).")
    parser.add_argument("--min-class-size", type=int, default=25, help="Drop classes with fewer than N examples.")
    parser.add_argument("--classifiers", nargs="+", default=["logreg", "svc", "rf"], help="Classifiers to evaluate.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", default="phase4/subontology_classification.json")
    parser.add_argument("--bootstrap-samples", type=int, default=0, help="Number of bootstrap samples for p-value (0 = disable).")
    parser.add_argument(
        "--baseline-embedding",
        default=None,
        help="Name of the reference embedding for p-values (defaults to the first from --embeddings).",
    )
    parser.add_argument("--encode-batch-size", type=int, default=256, help="Batch size for encoding definitions.")
    parser.add_argument("--max-terms", type=int, default=None, help="Limit number of HPO terms to encode (debug).")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_embedding_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    df = pd.read_parquet(path)
    if "embedding" not in df.columns or "hpo_id" not in df.columns:
        raise ValueError(f"The file {path} must contain columns 'hpo_id' and 'embedding'.")
    return df[["hpo_id", "embedding"]]


def collect_definitions(max_terms: int | None = None) -> List[Tuple[str, str]]:
    ontology = pyhpo.Ontology()
    records: List[Tuple[str, str]] = []
    for term in ontology:
        text = ""
        if getattr(term, "definition", None):
            text = str(term.definition).strip()
        if not text:
            text = (term.name or "").strip()
        if not text:
            continue
        records.append((term.id, text.replace("\n", " ").strip()))
    if max_terms and max_terms > 0:
        records = records[:max_terms]
    return records


def encode_definitions(model_path: str, batch_size: int, max_terms: int | None = None) -> pd.DataFrame:
    model = SentenceTransformer(model_path)
    records = collect_definitions(max_terms)
    if not records:
        raise RuntimeError("No definitions available.")
    ids, texts = zip(*records)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return pd.DataFrame({"hpo_id": ids, "embedding": [vec.tolist() for vec in embeddings]})


def prepare_dataset(
    labels_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    target_col: str,
    min_class_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    merged = labels_df.merge(embeddings_df, on="hpo_id", how="inner")
    merged = merged.dropna(subset=[target_col]).reset_index(drop=True)

    counts = merged[target_col].value_counts()
    keep_classes = counts[counts >= min_class_size].index
    filtered = merged[merged[target_col].isin(keep_classes)].reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError("No classes remain after applying min-class-size filter.")

    X = np.vstack(filtered["embedding"].tolist())
    y = filtered[target_col].astype(str).to_numpy()
    logging.info(
        "Dataset for %s: %d samples, %d classes (>= %d examples).",
        target_col,
        len(filtered),
        len(keep_classes),
        min_class_size,
    )
    return X, y


def get_classifier_configs(n_jobs: int, seed: int) -> Dict[str, ClassifierConfig]:
    configs: Dict[str, ClassifierConfig] = {}

    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=seed)),
        ]
    )
    configs["logreg"] = (
        logreg,
        {
            "clf__C": [0.1, 1.0, 10.0],
        },
    )

    svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=False, random_state=seed)),
        ]
    )
    configs["svc"] = (
        svc,
        {
            "clf__C": [1.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
    )

    rf = Pipeline(
        [
            ("clf", RandomForestClassifier(random_state=seed, n_jobs=n_jobs)),
        ]
    )
    configs["rf"] = (
        rf,
        {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 30],
            "clf__max_features": ["sqrt", "log2"],
        },
    )

    return configs


def evaluate_classifier(
    name: str,
    config: ClassifierConfig,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Dict[str, object], np.ndarray]:
    pipeline, param_grid = config
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X, y)
    best_estimator = grid.best_estimator_

    # Additional metrics via cross_val_predict with the best hyperparameters.
    preds = cross_val_predict(best_estimator, X, y, cv=cv, n_jobs=n_jobs)
    metrics = {
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "best_params": grid.best_params_,
    }
    return metrics, preds


def bootstrap_p_value(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    samples: int = 1000,
    seed: int = 13,
) -> float:
    """
    Calculates a one-sided p-value via bootstrap for the F1-macro difference
    between two sets of predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        f1_a = f1_score(y_true[idx], preds_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], preds_b[idx], average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)
    diffs = np.array(diffs)
    p_value = float(np.mean(diffs <= 0))
    return p_value


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    labels_df = pd.read_parquet(args.labels)
    target_col = f"label_lvl{args.level}"
    if target_col not in labels_df.columns:
        raise ValueError(f"Column {target_col} not found in {args.labels}")

    embedding_specs: Dict[str, Path] = {}
    if args.embeddings:
        for spec in args.embeddings:
            if "=" not in spec:
                raise ValueError("Each --embeddings entry must be in the format name=path")
            name, path = spec.split("=", 1)
            embedding_specs[name] = Path(path)
    if args.models:
        for spec in args.models:
            if "=" not in spec:
                raise ValueError("Each --models entry must be in the format name=path")
            name, path = spec.split("=", 1)
            tmp_df = encode_definitions(path, batch_size=args.encode_batch_size, max_terms=args.max_terms)
            tmp_path = Path(f"data/phase4/.tmp_{name}_defs_embeddings.parquet")
            tmp_df.to_parquet(tmp_path, index=False)
            embedding_specs[name] = tmp_path

    if not embedding_specs:
        raise ValueError("You must provide --embeddings or --models.")

    clf_configs = get_classifier_configs(args.n_jobs, args.seed)
    selected_models = [m for m in args.classifiers if m in clf_configs]
    if not selected_models:
        raise ValueError("No valid classifier selected.")

    results: Dict[str, Dict[str, object]] = {}
    preds_store: Dict[str, Dict[str, np.ndarray]] = {}
    ys_store: Dict[str, np.ndarray] = {}

    baseline = args.baseline_embedding or next(iter(embedding_specs))
    for emb_name, emb_path in embedding_specs.items():
        logging.info("Processing embeddings '%s' from %s", emb_name, emb_path)
        emb_df = load_embedding_file(emb_path)
        X, y = prepare_dataset(labels_df, emb_df, target_col, args.min_class_size)

        ys_store[emb_name] = y
        emb_results: Dict[str, object] = {
            "num_samples": int(len(y)),
            "num_classes": int(len(np.unique(y))),
        }
        for model_name in selected_models:
            logging.info("Training classifier %s on '%s'", model_name, emb_name)
            metrics, preds = evaluate_classifier(
                model_name,
                clf_configs[model_name],
                X,
                y,
                cv_folds=args.cv_folds,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            emb_results[model_name] = metrics
            preds_store.setdefault(emb_name, {})[model_name] = preds
        results[emb_name] = emb_results

    if args.bootstrap_samples > 0 and baseline in results:
        for emb_name, emb_results in results.items():
            if emb_name == baseline:
                continue
            if len(ys_store.get(emb_name, [])) != len(ys_store.get(baseline, [])):
                logging.warning("Skipping bootstrap between %s and %s due to different sizes.", emb_name, baseline)
                continue
            y_true = ys_store[emb_name]
            for model_name in selected_models:
                preds_a = preds_store.get(emb_name, {}).get(model_name)
                preds_b = preds_store.get(baseline, {}).get(model_name)
                if preds_a is None or preds_b is None:
                    continue
                p_val = bootstrap_p_value(
                    y_true,
                    preds_a,
                    preds_b,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
                emb_results[model_name][f"p_vs_{baseline}"] = p_val

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_level": args.level,
        "min_class_size": args.min_class_size,
        "cv_folds": args.cv_folds,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Results saved to %s", output_path)

    # Compact presentation
    for emb_name, emb_results in results.items():
        print(f"\n=== Embedding: {emb_name} (n={emb_results['num_samples']}, classes={emb_results['num_classes']}) ===")
        for model_name in selected_models:
            metrics = emb_results[model_name]
            print(
                f"{model_name:>8} | F1-macro={metrics['f1_macro']:.3f} "
                f"Acc={metrics['accuracy']:.3f} BalAcc={metrics['balanced_accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
