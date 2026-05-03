#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation on GSC+ (mention → HPO linking) for multiple models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import sys

import pyhpo

DEFAULT_BRANCHES = (
    "HP:0012638",  # Abnormal nervous system physiology
    "HP:0012639",  # Abnormal nervous system morphology
    "HP:0410008",  # Abnormality of the peripheral nervous system
)

_SYNONYM_ATTRS = ("synonyms", "alt_names", "alternative_names")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_allowed_hpo_ids(branch_ids: Iterable[str]) -> set[str]:
    ontology = pyhpo.Ontology()
    allowed: set[str] = set()
    for root_id in branch_ids:
        try:
            term = ontology.get_hpo_object(root_id)
        except RuntimeError:
            continue
        stack = [term]
        while stack:
            current = stack.pop()
            if current.id in allowed:
                continue
            allowed.add(current.id)
            stack.extend(getattr(current, "children", []) or [])
    return allowed


def load_gsc_mentions(
    paths: Iterable[Path],
    allowed_ids: Optional[set[str]],
    context_window: int = 40,
) -> Tuple[List[str], List[str]]:
    mentions: List[str] = []
    labels: List[str] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]

        if len(lines) < 3:
            continue

        doc_text = lines[1]
        for line in lines[2:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            start, end, mention_text, hpo_id = parts[:4]
            start_i: Optional[int] = None
            end_i: Optional[int] = None
            try:
                start_i = int(start)
                end_i = int(end)
                if 0 <= start_i < end_i <= len(doc_text):
                    extracted = doc_text[start_i:end_i].strip()
                    if extracted:
                        mention_text = extracted
            except ValueError:
                pass
            mention_text = mention_text.strip()
            hpo_id = hpo_id.strip()
            if not mention_text or not hpo_id:
                continue
            if allowed_ids is not None and hpo_id not in allowed_ids:
                continue
            if (
                context_window > 0
                and start_i is not None
                and end_i is not None
                and 0 <= start_i < end_i <= len(doc_text)
            ):
                left = doc_text[max(0, start_i - context_window) : start_i]
                right = doc_text[end_i : min(len(doc_text), end_i + context_window)]
                context_span = f"{left}{doc_text[start_i:end_i]}{right}".strip()
                if context_span:
                    mention_text = f"{mention_text}. Context: {context_span}"
            mentions.append(mention_text)
            labels.append(hpo_id)

    return mentions, labels


def get_hpo_text(term: pyhpo.term.HPOTerm) -> str:
    pieces: List[str] = []
    name = (term.name or "").strip()
    if name:
        pieces.append(name)
    for attr in _SYNONYM_ATTRS:
        syns = getattr(term, attr, None)
        if not syns:
            continue
        try:
            for raw in syns:
                text = str(raw or "").strip()
                if text:
                    pieces.append(text)
        except Exception:
            continue
    definition = getattr(term, "definition", None)
    if definition:
        def_text = str(definition).strip()
        if def_text:
            pieces.append(def_text)

    seen: set[str] = set()
    ordered: List[str] = []
    for piece in pieces:
        clean = " ".join(piece.split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ". ".join(ordered)


def collect_candidate_terms(allowed_ids: Optional[set[str]]) -> List[Tuple[str, str]]:
    ontology = pyhpo.Ontology()
    candidates: List[Tuple[str, str]] = []
    for term in ontology:
        if allowed_ids is not None and term.id not in allowed_ids:
            continue
        if (term.name or "").strip().lower() == "all":
            continue
        text = get_hpo_text(term)
        if not text:
            continue
        candidates.append((term.id, text))
    candidates.sort(key=lambda item: item[0])
    return candidates


def build_hpo_embeddings(
    model: SentenceTransformer,
    candidates: List[Tuple[str, str]],
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    ids = [hpo_id for hpo_id, _ in candidates]
    texts = [text for _, text in candidates]

    if not texts:
        raise RuntimeError("Could not generate HPO embeddings.")

    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    embeddings = util.normalize_embeddings(torch.tensor(embeddings)).numpy()
    id_to_index = {hpo_id: idx for idx, hpo_id in enumerate(ids)}
    return embeddings, id_to_index


def evaluate_model(
    model: SentenceTransformer,
    mention_texts: List[str],
    gold_ids: List[str],
    candidate_terms: List[Tuple[str, str]],
    batch_size: int,
) -> Dict[str, float]:
    if not candidate_terms:
        raise RuntimeError("No candidate HPO terms to evaluate.")
    hpo_embeddings, id_to_index = build_hpo_embeddings(model, candidate_terms, batch_size)
    filtered = [(m, g) for m, g in zip(mention_texts, gold_ids) if g in id_to_index]
    if not filtered:
        raise RuntimeError("No valid mentions after filtering by known HPO IDs.")

    mention_texts, gold_ids = zip(*filtered)
    gold_indices = np.array([id_to_index[g] for g in gold_ids])

    mention_embeddings = model.encode(mention_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    mention_embeddings = util.normalize_embeddings(torch.tensor(mention_embeddings)).numpy()

    scores = mention_embeddings @ hpo_embeddings.T

    sorted_idx = np.argsort(-scores, axis=1)
    ranks = np.array([np.where(sorted_idx[i] == gold_indices[i])[0][0] for i in range(len(gold_indices))])

    top1 = np.mean(ranks == 0)
    top5 = np.mean(ranks < 5)
    mrr = np.mean(1.0 / (ranks + 1))

    gold_similarities = scores[np.arange(len(scores)), gold_indices]

    negative_similarities = []
    overall_mean_scores = []
    improvements: List[float] = []
    for i, gold_idx in enumerate(gold_indices):
        row = scores[i]
        overall_mean_scores.append(float(row.mean()))
        candidates = np.delete(sorted_idx[i], np.where(sorted_idx[i] == gold_idx))
        if candidates.size:
            best_neg = row[candidates[0]]
            negative_similarities.append(float(best_neg))
            improvements.append(float(row[gold_idx] - best_neg))

    metrics = {
        "top1": float(top1),
        "top5": float(top5),
        "mrr": float(mrr),
        "gold_similarity": float(gold_similarities.mean()),
        "negative_similarity": float(np.mean(negative_similarities)) if negative_similarities else 0.0,
        "overall_mean_similarity": float(np.mean(overall_mean_scores)) if overall_mean_scores else 0.0,
        "avg_margin": float(np.mean(improvements)) if improvements else 0.0,
        "num_samples": len(scores),
        "num_candidates": len(candidate_terms),
    }
    return metrics


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name}: top1={metrics['top1']:.3f}, top5={metrics['top5']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"sim_gold={metrics['gold_similarity']:.3f}, sim_neg={metrics['negative_similarity']:.3f}, "
        f"sim_mean={metrics['overall_mean_similarity']:.3f}, margin={metrics['avg_margin']:.3f} "
        f"(n={metrics['num_samples']}, candidates={metrics['num_candidates']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GSC+ with different models.")
    parser.add_argument("--corpus-dir", default="data/corpus/GSC")
    parser.add_argument("--files", nargs="*", default=["GSCplus_dev_gold.tsv", "GSCplus_test_gold.tsv"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--branch-ids",
        nargs="*",
        default=None,
        help="HPO root IDs to filter (omitted = use all phenotypes).",
    )
    parser.add_argument("--base-model", default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument("--pubmed-model", default="models/phase3/biobert_cosent/best_model")
    parser.add_argument("--definitions-model", default="models/phase3/biobert_definition_cosent/best_model")
    parser.add_argument("--model", default=None, help="Single model to evaluate (skips the three defaults).")
    parser.add_argument("--model-label", default="model", help="Label for the single model.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--context-window",
        type=int,
        default=40,
        help="Character window around the mention for context (0 = no context).",
    )
    args = parser.parse_args()

    set_seed(13)

    files = [Path(args.corpus_dir) / name for name in args.files]
    allowed_ids = build_allowed_hpo_ids(args.branch_ids) if args.branch_ids else None
    candidate_terms = collect_candidate_terms(allowed_ids)
    candidate_ids = {hpo_id for hpo_id, _ in candidate_terms}
    mention_texts, gold_ids = load_gsc_mentions(files, candidate_ids, context_window=args.context_window)
    if not candidate_terms:
        raise RuntimeError("No candidate HPO terms found.")
    if not mention_texts:
        raise RuntimeError("No mentions found in GSC+ (after filtering by candidates).")

    if args.model:
        specs = {args.model_label: args.model}
    else:
        specs = {
            "Base BioBERT": args.base_model,
            "Fine-tuned (PubMed)": args.pubmed_model,
            "Fine-tuned (Definitions)": args.definitions_model,
        }

    collected = []
    for label, path in specs.items():
        if args.json:
            print(f"Evaluating {label} ({path})…", file=sys.stderr)
        else:
            print(f"Evaluating {label} ({path})…")
        model = SentenceTransformer(path, device=args.device)
        metrics = evaluate_model(
            model,
            list(mention_texts),
            list(gold_ids),
            candidate_terms,
            batch_size=args.batch_size,
        )
        if args.json:
            collected.append({"model": label, **metrics})
        else:
            print(format_metrics(label, metrics))

    if args.json:
        import json
        print(json.dumps(collected, indent=2))


if __name__ == "__main__":
    main()
