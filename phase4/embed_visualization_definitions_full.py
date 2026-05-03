#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP visualization for ALL HPO definitions.

Compares two models:
1) Base model (without fine-tuning).
2) Fine-tuned model (on definitions or final corpus).

No subontology filtering; all available terms are projected
with the same color to highlight how the global embedding
distribution changes between the two models.
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer

import pyhpo


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_hpo_definitions(
    max_terms: int | None,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Retrieves definitions (or label if missing) for all HPO terms.
    """
    ontology = pyhpo.Ontology()
    entries: List[Tuple[str, str]] = []
    for term in ontology:
        definition = getattr(term, "definition", None)
        text = ""
        if definition:
            text = str(definition).strip()
        if not text:
            text = (term.name or "").strip()
        if not text:
            continue
        clean = " ".join(text.split())
        entries.append((term.id, clean))

    if not entries:
        raise RuntimeError("No definitions or labels found in the ontology.")

    if max_terms is not None and 0 < max_terms < len(entries):
        rng = random.Random(seed)
        entries = rng.sample(entries, max_terms)

    ids, texts = zip(*entries)
    return list(ids), list(texts)


def compute_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def reduce_umap(
    embeddings: Dict[str, np.ndarray],
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
    for name, matrix in embeddings.items():
        projections[name] = reducer.fit_transform(matrix)
    return projections


def plot_embeddings(
    projections: Dict[str, np.ndarray],
    output_path: str,
    color: str = "#1f77b4",
) -> None:
    titles = list(projections.keys())
    fig, axes = plt.subplots(1, len(titles), figsize=(12, 5), sharex=False, sharey=False)
    if len(titles) == 1:
        axes = [axes]

    for ax, title in zip(axes, titles):
        coords = projections[title]
        ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.5, color=color)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def load_models(
    base_model: str,
    ft_model: str,
    device: str,
) -> Dict[str, SentenceTransformer]:
    specs = {
        "Base Model": base_model,
        "Fine-tuned Model": ft_model,
    }
    return {label: SentenceTransformer(path, device=device) for label, path in specs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Global UMAP of HPO definitions (base vs fine-tuned).")
    parser.add_argument("--max-terms", type=int, default=4000, help="Maximum number of terms to sample (None = all).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--output", default="phase4/plots/umap_definitions_full.png")
    parser.add_argument("--base-model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--ft-model", default="models/phase3/pubmedbert_defs_rbp_angle_best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    _, texts = collect_hpo_definitions(args.max_terms, seed=args.seed)
    print(f"Total definitions used: {len(texts)}")

    models = load_models(
        args.base_model,
        args.ft_model,
        args.device,
    )
    embeddings: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"Encoding with {name}…")
        embeddings[name] = compute_embeddings(model, texts, batch_size=args.batch_size)

    projections = reduce_umap(
        embeddings,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )
    plot_embeddings(projections, args.output)
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
