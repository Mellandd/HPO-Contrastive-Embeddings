#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document-level evaluation on JAX (full text → HPO phenotypes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import sys

import pyhpo

DEFAULT_BRANCHES = (
    "HP:0012638",
    "HP:0012639",
    "HP:0410008",
)

_SYNONYM_ATTRS = ("synonyms", "alt_names", "alternative_names")


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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jax_gold(json_path: Path) -> Dict[str, Dict[str, str]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    # Normalize keys to strings
    return {str(pmid): {str(hpo_id): label for hpo_id, label in entries.items()} for pmid, entries in data.items()}


def chunk_text(text: str, max_len: int = 800) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        tokens = paragraph.split()
        if current_len + len(tokens) > max_len and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.extend(tokens)
        current_len += len(tokens)
        if current_len >= max_len:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text.strip()[:max_len]]


def load_documents(
    gold: Dict[str, Dict[str, str]],
    txt_dir: Path,
    allowed_ids: Optional[set[str]],
) -> List[Tuple[str, str, Dict[str, str]]]:
    docs = []
    for pmid, annotations in gold.items():
        filtered = {h: label for h, label in annotations.items() if (allowed_ids is None or h in allowed_ids)}
        if not filtered:
            continue
        txt_path = txt_dir / f"{pmid}.txt"
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append((pmid, text, filtered))
    return docs


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
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    ids = [hpo_id for hpo_id, _ in candidates]
    texts = [text for _, text in candidates]
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress)
    embeddings = util.normalize_embeddings(torch.tensor(embeddings)).numpy()
    id_to_index = {hpo_id: idx for idx, hpo_id in enumerate(ids)}
    return embeddings, id_to_index


def evaluate_model(
    model: SentenceTransformer,
    documents: List[Tuple[str, str, Dict[str, str]]],
    candidate_terms: List[Tuple[str, str]],
    batch_size: int,
    show_progress: bool = True,
) -> Dict[str, float]:
    if not candidate_terms:
        raise RuntimeError("No candidate HPO terms to evaluate.")
    candidate_ids = [hpo_id for hpo_id, _ in candidate_terms]
    hpo_embeddings, id_to_index = build_hpo_embeddings(model, candidate_terms, batch_size, show_progress=show_progress)

    total_gold = 0
    hits_top1 = 0
    hits_top5 = 0
    hits_top10 = 0
    gold_similarities: List[float] = []
    negative_similarities: List[float] = []
    overall_mean_scores: List[float] = []
    margins: List[float] = []
    ranks: List[int] = []

    for pmid, text, annotations in documents:
        gold_ids = [h for h in annotations.keys() if h in id_to_index]
        if not gold_ids:
            continue
        chunk_embs = model.encode(chunk_text(text), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        chunk_embs = util.normalize_embeddings(torch.tensor(chunk_embs)).numpy()
        scores = chunk_embs @ hpo_embeddings.T  # (chunks, candidates)
        doc_scores = scores.max(axis=0)  # best chunk per phenotype
        overall_mean_scores.append(float(doc_scores.mean()))
        order = np.argsort(-doc_scores)
        rank_positions = {idx: pos for pos, idx in enumerate(order)}
        gold_set = set(gold_ids)
        best_neg_score = None
        for idx in order:
            if candidate_ids[idx] not in gold_set:
                best_neg_score = doc_scores[idx]
                break

        total_gold += len(gold_ids)
        for hpo_id in gold_ids:
            idx = id_to_index[hpo_id]
            rank = rank_positions.get(idx)
            if rank is None:
                continue
            ranks.append(int(rank))
            if rank < 1:
                hits_top1 += 1
            if rank < 5:
                hits_top5 += 1
            if rank < 10:
                hits_top10 += 1
            gold_similarities.append(float(doc_scores[idx]))

            if best_neg_score is not None:
                negative_similarities.append(float(best_neg_score))
                margins.append(float(doc_scores[idx] - best_neg_score))

    metrics = {
        "top1": hits_top1 / total_gold if total_gold else 0.0,
        "top5": hits_top5 / total_gold if total_gold else 0.0,
        "top10": hits_top10 / total_gold if total_gold else 0.0,
        "mrr": float(np.mean([1.0 / (r + 1) for r in ranks])) if ranks else 0.0,
        "gold_similarity": float(np.mean(gold_similarities)) if gold_similarities else 0.0,
        "negative_similarity": float(np.mean(negative_similarities)) if negative_similarities else 0.0,
        "overall_mean_similarity": float(np.mean(overall_mean_scores)) if overall_mean_scores else 0.0,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "num_labels": total_gold,
        "num_candidates": len(candidate_terms),
    }
    return metrics


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name}: top1={metrics['top1']:.3f}, top5={metrics['top5']:.3f}, top10={metrics['top10']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"sim_gold={metrics['gold_similarity']:.3f}, sim_neg={metrics['negative_similarity']:.3f}, "
        f"sim_mean={metrics['overall_mean_similarity']:.3f}, margin={metrics['avg_margin']:.3f} "
        f"(labels={metrics['num_labels']}, candidates={metrics['num_candidates']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document→HPO matching on the JAX corpus.")
    parser.add_argument("--corpus-dir", default="data/corpus/JAX")
    parser.add_argument("--json-file", default="JAX_gold.json")
    parser.add_argument("--txt-dir", default="txt")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-model", default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument("--pubmed-model", default="models/phase3/biobert_cosent/best_model")
    parser.add_argument("--definitions-model", default="models/phase3/biobert_definition_cosent/best_model")
    parser.add_argument("--model", default=None, help="Single model to evaluate (skips the three defaults).")
    parser.add_argument("--model-label", default="model", help="Label for the single model.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--branch-ids",
        nargs="*",
        default=None,
        help="HPO root IDs to filter (omitted = use all phenotypes).",
    )
    args = parser.parse_args()

    set_seed(13)

    corpus_path = Path(args.corpus_dir)
    gold = read_jax_gold(corpus_path / args.json_file)
    allowed_ids = build_allowed_hpo_ids(args.branch_ids) if args.branch_ids else None
    candidate_terms = collect_candidate_terms(allowed_ids)
    candidate_ids = {hpo_id for hpo_id, _ in candidate_terms}
    documents = load_documents(gold, corpus_path / args.txt_dir, candidate_ids)
    if args.max_docs is not None:
        documents = documents[: args.max_docs]
    if not candidate_terms:
        raise RuntimeError("No candidate HPO terms found.")
    if not documents:
        raise RuntimeError("No valid documents found in JAX (after filtering by candidates).")

    if args.model:
        specs = {args.model_label: args.model}
    else:
        specs = {
            "Base BioBERT": args.base_model,
            "Fine-tuned (PubMed)": args.pubmed_model,
            "Fine-tuned (Definitions)": args.definitions_model,
        }

    collected = []
    show_progress = not args.json

    for label, path in specs.items():
        if args.json:
            print(f"Evaluating {label} ({path})…", file=sys.stderr)
        else:
            print(f"Evaluating {label} ({path})…")
        model = SentenceTransformer(path, device=args.device)
        metrics = evaluate_model(model, documents, candidate_terms, batch_size=args.batch_size, show_progress=show_progress)
        if args.json:
            collected.append({"model": label, **metrics})
        else:
            print(format_metrics(label, metrics))

    if args.json:
        print(json.dumps(collected, indent=2))


if __name__ == "__main__":
    main()
