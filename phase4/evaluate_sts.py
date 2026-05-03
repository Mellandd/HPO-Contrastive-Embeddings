#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STS evaluation (e.g., BIOSSES) for SentenceTransformer models.

The dataset must be a TSV/CSV with columns: sentence1, sentence2, score.
Scores are assumed to be in [0, 5] range and are normalized to [0, 1] for correlations.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util
from tqdm import trange


def load_dataset(path: str | Path) -> pd.DataFrame:
    path_str = str(path)
    # Supports remote paths like hf://
    if path_str.startswith("hf://"):
        df = pd.read_parquet(path_str)
    else:
        path_obj = Path(path_str)
        if not path_obj.exists():
            raise FileNotFoundError(f"STS dataset not found at {path_obj}")
        if path_obj.suffix.lower() == ".csv":
            df = pd.read_csv(path_obj)
        elif path_obj.suffix.lower() in {".tsv", ".txt"}:
            df = pd.read_csv(path_obj, sep="\t")
        else:
            df = pd.read_parquet(path_obj)
    expected = {"sentence1", "sentence2", "score"}
    if not expected.issubset(df.columns):
        raise ValueError(f"The dataset must contain columns {expected}")
    df = df.dropna(subset=["sentence1", "sentence2", "score"]).reset_index(drop=True)
    df["sentence1"] = df["sentence1"].astype(str).str.strip()
    df["sentence2"] = df["sentence2"].astype(str).str.strip()
    df = df[(df["sentence1"] != "") & (df["sentence2"] != "")]
    # Normalize scores to [0,1] if they appear to be in [0,5]
    if df["score"].max() > 1.0:
        df["score"] = df["score"] / 5.0
    return df


def evaluate_model(model: SentenceTransformer, df: pd.DataFrame, batch_size: int) -> Dict[str, float]:
    sentences = pd.concat([df["sentence1"], df["sentence2"]]).unique().tolist()
    embeddings = {}
    for start in trange(0, len(sentences), batch_size, desc="Encoding", leave=False):
        chunk = sentences[start : start + batch_size]
        embs = model.encode(chunk, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        for sent, emb in zip(chunk, embs):
            embeddings[sent] = emb

    sims = []
    golds = df["score"].to_numpy()
    for s1, s2 in zip(df["sentence1"], df["sentence2"]):
        e1 = embeddings[s1]
        e2 = embeddings[s2]
        sims.append(util.cos_sim(e1, e2).item())
    sims_arr = np.array(sims)

    spearman = spearmanr(sims_arr, golds).statistic
    pearson = pearsonr(sims_arr, golds).statistic
    mse = float(np.mean((sims_arr - golds) ** 2))
    return {"spearman": float(spearman), "pearson": float(pearson), "mse": mse}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ST model on an STS benchmark (e.g., BIOSSES).")
    parser.add_argument("--data", required=True, help="Path to TSV/CSV with sentence1, sentence2, score.")
    parser.add_argument("--model", required=True, help="Path or identifier of the SentenceTransformer model.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="cuda | cpu (optional).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_dataset(Path(args.data))
    model = SentenceTransformer(args.model, device=args.device)
    metrics = evaluate_model(model, df, batch_size=args.batch_size)

    print(
        f"{args.model} | Spearman={metrics['spearman']:.4f} "
        f"Pearson={metrics['pearson']:.4f} MSE={metrics['mse']:.5f} (n={len(df)})"
    )


if __name__ == "__main__":
    main()
