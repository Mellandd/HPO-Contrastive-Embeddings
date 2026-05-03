#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic patient retrieval against OMIM diseases using HPO phenotypes.

Workflow:
1) Precompute disease embeddings as mean of their HPO term names/definitions.
2) Build synthetic patient notes by sampling HPO phenotypes from diseases and
   concatenating example sentences (from Phase 1/2 corpus) for those phenotypes.
3) Encode patient notes and rank diseases by cosine similarity.
4) Report top-k metrics (top1/top5, MRR) per difficulty bucket.

Assumptions:
- Sentences dataset contains columns hpo_id + sentence (or hpo_id1/2, sentence1/2).
- pyhpo has OMIM disease annotations available (Disease iterator).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import json
import pyarrow.dataset as ds
import torch
from sentence_transformers import SentenceTransformer, util

import pyhpo
from pyhpo.annotations import Omim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sentence_inventory(dataset_path: str, max_per_hpo: int, seed: int) -> Dict[str, List[str]]:
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

    def _pairs() -> List[Tuple[str, str]]:
        if "hpo_id" in schema_names and "sentence" in schema_names:
            return [("hpo_id", "sentence")]
        pairs: List[Tuple[str, str]] = []
        if "hpo_id1" in schema_names and "sentence1" in schema_names:
            pairs.append(("hpo_id1", "sentence1"))
        if "hpo_id2" in schema_names and "sentence2" in schema_names:
            pairs.append(("hpo_id2", "sentence2"))
        if not pairs:
            raise ValueError("Dataset does not have columns hpo_id/sentence nor hpo_id1/2, sentence1/2.")
        return pairs

    pairs = _pairs()
    needed_cols = sorted({c for pair in pairs for c in pair})
    rng = random.Random(seed)
    buckets: Dict[str, List[str]] = {}

    for batch in dataset.to_batches(columns=needed_cols, batch_size=20_000):
        data = batch.to_pydict()
        for hpo_col, sent_col in pairs:
            hpo_ids = data[hpo_col]
            sentences = data[sent_col]
            for hpo_id, sentence in zip(hpo_ids, sentences):
                if not hpo_id or not sentence:
                    continue
                text = str(sentence).strip()
                if not text:
                    continue
                pool = buckets.setdefault(hpo_id, [])
                if len(pool) < max_per_hpo:
                    pool.append(text)
                else:
                    # reservoir sampling to keep max_per_hpo uniformly
                    idx = rng.randrange(len(pool) + 1)
                    if idx < len(pool):
                        pool[idx] = text
    return {k: v for k, v in buckets.items() if v}


def collect_diseases(min_terms: int) -> List[Omim]:
    # Ensure ontology loads annotations (pyhpo-data)
    _ = pyhpo.Ontology()

    ids = []
    if hasattr(Omim, "_indicies"):
        ids = list(getattr(Omim, "_indicies", {}).keys())
    if not ids and hasattr(Omim, "keys"):
        try:
            ids = list(Omim.keys())
        except Exception:
            ids = []
    if not ids:
        raise RuntimeError(
            "pyhpo.annotations.Omim has no loaded IDs. Install pyhpo-data or configure PYHPO_DATA_PATH."
        )

    diseases: List[Omim] = []
    for omim_id in ids:
        try:
            omim = Omim.get(omim_id)
        except Exception:
            continue
        phenos = list(get_hpo_terms(omim))
        if len(phenos) >= min_terms:
            diseases.append(omim)
    if not diseases:
        raise RuntimeError("No OMIM diseases found after filtering by minimum number of terms.")
    return diseases


def get_hpo_terms(disease: object) -> Iterable[pyhpo.term.HPOTerm]:
    if hasattr(disease, "hpo_set"):
        try:
            return disease.hpo_set()
        except Exception:
            return []
    if hasattr(disease, "hpo_terms"):
        return getattr(disease, "hpo_terms", []) or []
    return []


def disease_embedding(model: SentenceTransformer, disease: object, batch_size: int) -> np.ndarray:
    texts: List[str] = []
    for term in get_hpo_terms(disease):
        pieces = []
        if term.name:
            pieces.append(str(term.name))
        definition = getattr(term, "definition", None)
        if definition:
            pieces.append(str(definition))
        if not pieces:
            continue
        texts.append(". ".join(pieces))
    if not texts:
        return np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)
    emb = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    emb = emb.mean(dim=0)
    return util.normalize_embeddings(emb.unsqueeze(0)).squeeze(0).cpu().numpy()


def build_disease_matrix(model: SentenceTransformer, diseases: List[object], batch_size: int) -> Tuple[np.ndarray, List[str], List[str]]:
    embs: List[np.ndarray] = []
    names: List[str] = []
    ids: List[str] = []
    for dis in diseases:
        emb = disease_embedding(model, dis, batch_size=batch_size)
        embs.append(emb)
        label = getattr(dis, "name", None) or getattr(dis, "title", None) or "OMIM"
        omim_id = getattr(dis, "omim_id", None) or getattr(dis, "omim", None)
        if omim_id:
            names.append(f"{label} ({omim_id})")
            ids.append(str(omim_id))
        else:
            names.append(str(label))
            ids.append(str(label))
    mat = np.stack(embs, axis=0)
    return mat, names, ids


def sample_patient_sentences(
    disease: object,
    inventory: Dict[str, List[str]],
    phenos_per_patient: int,
    rng: random.Random,
) -> Optional[List[str]]:
    available_terms = [term for term in get_hpo_terms(disease) if term.id in inventory]
    if len(available_terms) < phenos_per_patient:
        return None
    chosen = rng.sample(available_terms, phenos_per_patient)
    sentences: List[str] = []
    for term in chosen:
        sentences.append(rng.choice(inventory[term.id]))
    return sentences


def embed_sentences_mean(model: SentenceTransformer, sentences: List[str], batch_size: int) -> np.ndarray:
    emb = model.encode(sentences, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    emb = emb.mean(dim=0)
    emb = util.normalize_embeddings(emb.unsqueeze(0)).squeeze(0).cpu().numpy()
    return emb


def evaluate_patients(
    model: SentenceTransformer,
    diseases: List[pyhpo.disease.Disease],
    disease_matrix: np.ndarray,
    disease_ids: List[str],
    inventory: Dict[str, List[str]],
    patients: int,
    phenos_per_patient: int,
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    hits1 = 0
    hits5 = 0
    mrr = 0.0
    total = 0

    for _ in range(patients):
        dis = rng.choice(diseases)
        sentences = sample_patient_sentences(dis, inventory, phenos_per_patient, rng)
        if not sentences:
            continue
        emb = embed_sentences_mean(model, sentences, batch_size=batch_size)
        sims = emb @ disease_matrix.T  # (n_diseases,)
        order = np.argsort(-sims)
        target_id = str(getattr(dis, "omim_id", None) or getattr(dis, "omim", None) or getattr(dis, "name", None))
        true_indices = [idx for idx, did in enumerate(disease_ids) if did == target_id]
        rank = min(order.tolist().index(idx) for idx in true_indices) if true_indices else order.tolist().index(order[0])
        total += 1
        if rank == 0:
            hits1 += 1
        if rank < 5:
            hits5 += 1
        mrr += 1.0 / (rank + 1)

    if total == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "evaluated": 0}
    return {
        "top1": hits1 / total,
        "top5": hits5 / total,
        "mrr": mrr / total,
        "evaluated": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic patient -> OMIM disease retrieval.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List NAME=PATH (e.g. base=NeuML/pubmedbert-base-embeddings)",
    )
    parser.add_argument("--dataset", default="data/hpo_sentences.parquet_dir", help="Parquet con hpo_id/sentence")
    parser.add_argument("--patients", type=int, default=1000)
    parser.add_argument("--phenos-per-patient", type=int, default=4, help="Difficulty: sentences/phenotypes per patient")
    parser.add_argument("--min-terms-disease", type=int, default=3, help="Min HPO terms required for a disease")
    parser.add_argument("--max-sentences-per-hpo", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use-all-hpo", action="store_true", help="Do not filter by subontologies.")
    parser.add_argument("--output", default="phase4/synthetic_patient_results.json")
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading sentence inventory...")
    inventory = load_sentence_inventory(args.dataset, max_per_hpo=args.max_sentences_per_hpo, seed=args.seed)
    print(f"Sentences available for {len(inventory)} HPO terms.")

    print("Loading OMIM diseases from pyhpo...")
    diseases = collect_diseases(args.min_terms_disease)
    print(f"Candidate diseases: {len(diseases)}")

    model_specs: Dict[str, str] = {}
    for item in args.models:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        model_specs[name.strip()] = path.strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name, path in model_specs.items():
        print(f"=== Evaluating model {name} ({path}) ===")
        model = SentenceTransformer(path, device=device)
        disease_matrix, disease_labels, disease_ids = build_disease_matrix(model, diseases, batch_size=args.batch_size)
        disease_matrix = util.normalize_embeddings(torch.tensor(disease_matrix)).numpy()

        metrics = evaluate_patients(
            model,
            diseases,
            disease_matrix,
            disease_ids,
            inventory,
            patients=args.patients,
            phenos_per_patient=args.phenos_per_patient,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        results[name] = metrics
        print(f"{name}: {metrics}")

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
