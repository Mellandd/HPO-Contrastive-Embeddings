#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP to compare base vs fine-tuned model on multisystemic syndrome phenotypes.

Strategy:
- Selects HPO phenotypes associated with 3 OMIM syndromes (multisystemic).
- For each phenotype, takes up to N PubMed sentences (Phase 2), computes embeddings and averages
  them (mean) -> one point per phenotype (colored by the syndrome).
- Projects with UMAP and compares two models (base vs fine-tuned).

Syndromes:
- Marfan syndrome (OMIM:154700)
- Lowe syndrome (OMIM:309000)
- Bardet-Biedl syndrome (OMIM:209900)
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
import pyhpo
import torch
import umap
from sentence_transformers import SentenceTransformer

OMIM_SYNDROMES: Dict[str, int] = {
    "Marfan": 154700,
    "Lowe": 309000,
    "Bardet-Biedl": 209900,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hpo_terms_for_omim(omim_id: int) -> List[str]:
    _ = pyhpo.Ontology()
    from pyhpo.annotations import Omim

    try:
        dis = Omim.get(omim_id)
    except Exception:
        return []
    ids = list(getattr(dis, "hpo", []) or [])
    return [f"HP:{int(i):07d}" if isinstance(i, int) else str(i) for i in ids]


def _pair_columns(schema_names: set[str]) -> List[Tuple[str, str]]:
    if "hpo_id" in schema_names and "sentence" in schema_names:
        return [("hpo_id", "sentence")]
    pairs: List[Tuple[str, str]] = []
    if "hpo_id1" in schema_names and "sentence1" in schema_names:
        pairs.append(("hpo_id1", "sentence1"))
    if "hpo_id2" in schema_names and "sentence2" in schema_names:
        pairs.append(("hpo_id2", "sentence2"))
    if pairs:
        return pairs
    raise ValueError("Dataset does not contain columns hpo_id/sentence nor hpo_id1/2, sentence1/2.")


def collect_sentences(
    dataset_path: str,
    sentences_per_hpo: int,
    syndrome_terms: Dict[str, List[str]],
    seed: int,
) -> List[Dict[str, str]]:
    base_path = Path(dataset_path)
    if not base_path.exists():
        alt = Path(f"{dataset_path}_dir")
        if alt.exists():
            base_path = alt
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path} or {alt}")

    partitioning = "hive" if base_path.is_dir() else None
    dataset = ds.dataset(base_path, format="parquet", partitioning=partitioning)
    schema_names = set(dataset.schema.names)
    pairs = _pair_columns(schema_names)
    needed_cols = sorted({col for pair in pairs for col in pair})

    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    # Map HPO -> syndromes that include it (to allow shared phenotypes)
    hpo_to_synds: Dict[str, List[str]] = {}
    for synd, hpo_list in syndrome_terms.items():
        for hid in hpo_list:
            hpo_to_synds.setdefault(hid, []).append(synd)
    allowed_hpos = set(hpo_to_synds.keys())

    buckets: Dict[Tuple[str, str], List[str]] = {}  # (syndrome, hpo) -> sentences

    for batch in dataset.to_batches(columns=needed_cols, batch_size=20_000):
        data = batch.to_pydict()
        for hpo_col, sent_col in pairs:
            hpo_ids = data[hpo_col]
            sentences = data[sent_col]
            for hpo_id, sentence in zip(hpo_ids, sentences):
                if not hpo_id or not sentence:
                    continue
                hpo_id = str(hpo_id)
                if hpo_id not in allowed_hpos:
                    continue
                text = str(sentence).strip()
                if not text:
                    continue
                for synd in hpo_to_synds[hpo_id]:
                    key = (synd, hpo_id)
                    pool = buckets.setdefault(key, [])
                    if len(pool) < sentences_per_hpo:
                        pool.append(text)
                    else:
                        idx = rng.randrange(len(pool) + 1)
                        if idx < len(pool):
                            pool[idx] = text

    for (synd, hpo_id), sents in buckets.items():
        for s in sents:
            rows.append({"syndrome": synd, "hpo_id": hpo_id, "sentence": s})
    return rows


def compute_embeddings(
    model: SentenceTransformer,
    sentences: Iterable[str],
    batch_size: int,
) -> np.ndarray:
    return model.encode(
        list(sentences),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def average_by_key(embeddings: np.ndarray, keys: List[Tuple[str, str]]) -> Tuple[np.ndarray, List[str]]:
    """
    Groups by (syndrome, hpo_id) and returns the mean embedding per phenotype and its color label (syndrome).
    """
    idx_map: Dict[Tuple[str, str], List[int]] = {}
    for i, key in enumerate(keys):
        idx_map.setdefault(key, []).append(i)
    means = []
    labels = []
    for key, idxs in idx_map.items():
        means.append(embeddings[idxs].mean(axis=0))
        labels.append(key[0])  # syndrome label
    return np.vstack(means), labels


def reduce_dimensionality(
    matrices: Dict[str, np.ndarray],
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    projections: Dict[str, np.ndarray] = {}
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="cosine",
    )
    for name, matrix in matrices.items():
        projections[name] = reducer.fit_transform(matrix)
    return projections


def plot_two_models(
    projections: Dict[str, np.ndarray],
    labels: List[str],
    syndromes: List[str],
    model_titles: Tuple[str, str],
    output_path: str,
) -> None:
    palette = plt.get_cmap("tab10", len(syndromes))
    color_map = {name: palette(i) for i, name in enumerate(syndromes)}
    colors = [color_map.get(lab, (0.5, 0.5, 0.5, 0.6)) for lab in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    titles = [model_titles[0], model_titles[1]]
    mats = [projections["base"], projections["ft"]]

    for ax, mat, title in zip(axes, mats, titles):
        ax.scatter(
            mat[:, 0],
            mat[:, 1],
            c=colors,
            s=8,
            alpha=0.6,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[name], label=name)
        for name in syndromes
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(syndromes), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP of multisystemic syndrome phenotypes (base vs fine-tuned model).")
    parser.add_argument("--dataset", default="data/phase2/hpo_sentence_pairs_struct_rbp.parquet")
    parser.add_argument("--sentences-per-hpo", type=int, default=15, help="Max sentences per phenotype (reservoir).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="phase4/plots/umap_multisystemic.png")
    parser.add_argument("--base-model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--ft-model", default="models/phase3/pubmedbert_pubmed_rbp_angle_best")
    parser.add_argument("--base-label", default="Base PubMedBERT")
    parser.add_argument("--ft-label", default="FT PubMedBERT (RBP)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)

    # Recoge los fenotipos de cada síndrome
    syndrome_terms: Dict[str, List[str]] = {}
    for name, omim_id in OMIM_SYNDROMES.items():
        terms = get_hpo_terms_for_omim(omim_id)
        if terms:
            syndrome_terms[name] = terms
    if not syndrome_terms:
        raise RuntimeError("Could not load phenotypes for the defined syndromes.")

    rows = collect_sentences(
        dataset_path=args.dataset,
        sentences_per_hpo=args.sentences_per_hpo,
        syndrome_terms=syndrome_terms,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No sentences collected for the selected phenotypes.")

    keys = [(r["syndrome"], r["hpo_id"]) for r in rows]
    sentences = [r["sentence"] for r in rows]
    label_order = list(OMIM_SYNDROMES.keys())

    base_model = SentenceTransformer(args.base_model, device=args.device)
    ft_model = SentenceTransformer(args.ft_model, device=args.device)

    print(f"Encoding {len(sentences)} sentences with {args.base_label}…")
    base_sent_emb = compute_embeddings(base_model, sentences, batch_size=args.batch_size)
    print(f"Encoding {len(sentences)} sentences with {args.ft_label}…")
    ft_sent_emb = compute_embeddings(ft_model, sentences, batch_size=args.batch_size)

    base_term_emb, term_labels = average_by_key(base_sent_emb, keys)
    ft_term_emb, term_labels_ft = average_by_key(ft_sent_emb, keys)
    assert term_labels == term_labels_ft

    projections = reduce_dimensionality(
        {"base": base_term_emb, "ft": ft_term_emb},
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    plot_two_models(
        projections,
        term_labels,
        label_order,
        model_titles=(args.base_label, args.ft_label),
        output_path=args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
