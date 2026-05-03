#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP of 6 main HPO branches using PubMed sentences per phenotype.

For each phenotype:
- Up to N sentences are sampled (reservoir) from the PubMed corpus.
- They are encoded and averaged via mean pooling => one point per phenotype.

Comparison: base model vs fine-tuned model.

Branches included:
- Nervous System (HP:0000707)
- Skeletal System (HP:0000924)
- Cardiovascular (HP:0001626)
- Eye (Vision) (HP:0000478)
- Skin (Integument) (HP:0001574)
- Neoplasm (HP:0002664)
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

# ID de las categorías a comparar y su nombre legible
TOP_CATEGORIES = [
    ("Nervous System", "HP:0000707"),
    ("Skeletal System", "HP:0000924"),
    ("Cardiovascular", "HP:0001626"),
    ("Eye (Vision)", "HP:0000478"),
    ("Skin (Integument)", "HP:0001574"),
    ("Neoplasm", "HP:0002664"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_descendants(term: pyhpo.term.HPOTerm) -> set[str]:
    """Returns the set of descendant IDs (including the term itself)."""
    descendants: set[str] = {term.id}
    stack: List[pyhpo.term.HPOTerm] = list(getattr(term, "children", []) or [])
    while stack:
        child = stack.pop()
        if child.id in descendants:
            continue
        descendants.add(child.id)
        stack.extend(getattr(child, "children", []) or [])
    return descendants


def build_branch_sets() -> Dict[str, set[str]]:
    ontology = pyhpo.Ontology()
    branch_sets: Dict[str, set[str]] = {}
    for name, term_id in TOP_CATEGORIES:
        try:
            term = ontology.get_hpo_object(term_id)
        except RuntimeError as exc:
            raise ValueError(f"HPO term {term_id} not found for branch {name}") from exc
        branch_sets[name] = collect_descendants(term)
    return branch_sets


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
    branch_sets: Dict[str, set[str]],
    seed: int,
    min_branch_terms: int,
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Reads the sentence corpus and samples up to N sentences per phenotype (reservoir).
    Returns rows per sentence and a branch lookup by hpo_id.
    """
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
    buckets: Dict[str, List[str]] = {}
    branch_lookup: Dict[str, str] = {}
    for name, ids in branch_sets.items():
        for term_id in ids:
            branch_lookup[term_id] = name
    allowed_ids = set(branch_lookup.keys())

    for batch in dataset.to_batches(columns=needed_cols, batch_size=20_000):
        data = batch.to_pydict()
        for hpo_col, sent_col in pairs:
            hpo_ids = data[hpo_col]
            sentences = data[sent_col]
            for hpo_id, sentence in zip(hpo_ids, sentences):
                if not hpo_id or not sentence:
                    continue
                if hpo_id not in allowed_ids:
                    continue
                text = str(sentence).strip()
                if not text:
                    continue
                pool = buckets.setdefault(hpo_id, [])
                if len(pool) < sentences_per_hpo:
                    pool.append(text)
                else:
                    idx = rng.randrange(len(pool) + 1)
                    if idx < len(pool):
                        pool[idx] = text

    rows: List[Dict[str, str]] = []
    for hpo_id, sentences in buckets.items():
        branch = branch_lookup.get(hpo_id)
        if branch is None:
            continue
        for sentence in sentences:
            rows.append({"hpo_id": hpo_id, "sentence": sentence, "branch": branch})

    if not rows:
        raise RuntimeError("No sentences collected for the selected branches.")

    # Filter branches with too few terms (not sentences)
    term_counts: Dict[str, int] = {}
    seen_terms: set[str] = set()
    for row in rows:
        if row["hpo_id"] not in seen_terms:
            term_counts[row["branch"]] = term_counts.get(row["branch"], 0) + 1
            seen_terms.add(row["hpo_id"])
    keep_branches = {b for b, c in term_counts.items() if c >= min_branch_terms}
    rows = [r for r in rows if r["branch"] in keep_branches]

    return rows, branch_lookup


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


def average_by_term(embeddings: np.ndarray, hpo_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Group embeddings by hpo_id and return the mean per term."""
    df_idx = {}
    for idx, hpo_id in enumerate(hpo_ids):
        df_idx.setdefault(hpo_id, []).append(idx)
    means = []
    terms = []
    for hpo_id, idxs in df_idx.items():
        means.append(embeddings[idxs].mean(axis=0))
        terms.append(hpo_id)
    return np.vstack(means), terms


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
    term_branches: List[str],
    labels: Tuple[str, str],
    output_path: str,
) -> None:
    present = list(dict.fromkeys(term_branches))
    branch_names = [name for name, _ in TOP_CATEGORIES if name in present]
    palette = plt.get_cmap("tab10", len(branch_names))
    branch_to_color = {name: palette(idx) for idx, name in enumerate(branch_names)}
    colors = [branch_to_color[b] for b in term_branches]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    titles = [labels[0], labels[1]]
    proj_mats = [projections["base"], projections["ft"]]

    for ax, proj, title in zip(axes, proj_mats, titles):
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
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
        plt.Line2D([0], [0], marker="o", linestyle="", color=branch_to_color[name], label=name)
        for name in branch_names
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(branch_names), 3),
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP of 6 HPO branches (PubMed sentences, mean vector per phenotype).")
    parser.add_argument("--dataset", default="data/phase2/hpo_sentence_pairs_struct_rbp.parquet")
    parser.add_argument("--sentences-per-hpo", type=int, default=20, help="Max sentences per phenotype (reservoir).")
    parser.add_argument("--min-branch-terms", type=int, default=30, help="Discard branches with <N phenotypes present.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="phase4/plots/umap_top6_pubmed_meanpool.png")
    parser.add_argument("--base-model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--ft-model", default="models/phase3/pubmedbert_pubmed_rbp_angle_best")
    parser.add_argument("--base-label", default="Base PubMedBERT")
    parser.add_argument("--ft-label", default="FT PubMedBERT (RBP)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    branch_sets = build_branch_sets()

    rows, branch_lookup = collect_sentences(
        dataset_path=args.dataset,
        sentences_per_hpo=args.sentences_per_hpo,
        branch_sets=branch_sets,
        seed=args.seed,
        min_branch_terms=args.min_branch_terms,
    )
    if not rows:
        raise RuntimeError("No sentences collected after filtering.")

    hpo_ids = [r["hpo_id"] for r in rows]
    branches = [r["branch"] for r in rows]
    sentences = [r["sentence"] for r in rows]

    print(f"Total sentences: {len(sentences)} | Phenotypes: {len(set(hpo_ids))} | Branches: {len(set(branches))}")

    base_model = SentenceTransformer(args.base_model, device=args.device)
    ft_model = SentenceTransformer(args.ft_model, device=args.device)

    print(f"Encoding sentences with {args.base_label}…")
    base_sent_emb = compute_embeddings(base_model, sentences, batch_size=args.batch_size)
    print(f"Encoding sentences with {args.ft_label}…")
    ft_sent_emb = compute_embeddings(ft_model, sentences, batch_size=args.batch_size)

    base_term_emb, term_order = average_by_term(base_sent_emb, hpo_ids)
    ft_term_emb, term_order_ft = average_by_term(ft_sent_emb, hpo_ids)
    assert term_order == term_order_ft

    term_branches = []
    for term_id in term_order:
        b = branch_lookup.get(term_id)
        if b is None:
            b = "Other"
        term_branches.append(b)

    projections = reduce_dimensionality(
        {"base": base_term_emb, "ft": ft_term_emb},
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    plot_two_models(
        projections,
        term_branches,
        labels=(args.base_label, args.ft_label),
        output_path=args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
