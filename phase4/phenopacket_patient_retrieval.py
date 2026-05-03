#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation of real patients (Phenopackets) against OMIM/Orphanet diseases.

Steps:
- Reads phenopackets from data/0.1.25 (or specified folder).
- Extracts phenotypicFeatures excluding those with "excluded": true.
- For each HPO, takes 1 random sentence from the PubMed corpus (Phase 2) and computes
  the embedding of each sentence; averages (mean) -> patient vector.
- Computes cosine similarity against the set of diseases (OMIM + Orphanet)
  embedded as the mean of the names/definitions of their HPO terms.
- Measures top1/top5/MRR over all valid patients.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds
import torch
from sentence_transformers import SentenceTransformer, util

import pyhpo
from pyhpo.annotations import Omim, Orpha


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sentence_inventory(dataset_path: str, max_per_hpo: int, seed: int) -> Dict[str, List[str]]:
    """
    Loads sentences from the Phase 2 corpus and keeps up to max_per_hpo per phenotype (reservoir).
    Supports columns hpo_id/sentence or hpo_id1/2, sentence1/2.
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
                    idx = rng.randrange(len(pool) + 1)
                    if idx < len(pool):
                        pool[idx] = text
    return {k: v for k, v in buckets.items() if v}


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
    emb = util.normalize_embeddings(emb.unsqueeze(0)).squeeze(0).cpu().numpy()
    return emb


def collect_diseases(include_omim: bool, include_orpha: bool) -> List[object]:
    _ = pyhpo.Ontology()
    diseases: List[object] = []
    if include_omim:
        ids = []
        if hasattr(Omim, "_indicies"):
            ids = list(getattr(Omim, "_indicies", {}).keys())
        elif hasattr(Omim, "keys"):
            try:
                ids = list(Omim.keys())
            except Exception:
                ids = []
        for omim_id in ids:
            try:
                diseases.append(Omim.get(omim_id))
            except Exception:
                continue
    if include_orpha and hasattr(Orpha, "keys"):
        try:
            for oid in Orpha.keys():
                try:
                    diseases.append(Orpha.get(oid))
                except Exception:
                    continue
        except Exception:
            pass
    if not diseases:
        raise RuntimeError("Could not load OMIM/Orpha diseases from pyhpo.")
    return diseases


def normalize_disease_id(raw: str) -> str:
    """
    Normalizes disease IDs to prefix+number format:
    - OMIM:123456 -> OMIM:123456
    - omim:123456 -> OMIM:123456
    - 123456 -> OMIM:123456 (assume OMIM if only digits)
    - ORPHA:123 -> ORPHA:123
    """
    if not raw:
        return raw
    txt = raw.strip()
    if not txt:
        return txt
    upper = txt.upper()
    if upper.startswith("OMIM:"):
        return f"OMIM:{upper.split(':',1)[1]}"
    if upper.startswith("ORPHA:"):
        return f"ORPHA:{upper.split(':',1)[1]}"
    if upper.startswith("MIM:"):
        return f"OMIM:{upper.split(':',1)[1]}"
    if upper.isdigit():
        return f"OMIM:{upper}"
    return upper


def resolve_identifier(dis: object) -> Optional[str]:
    """Try to get a canonical ID (OMIM/ORPHA) from a pyhpo disease object."""
    omim_id = getattr(dis, "omim_id", None) or getattr(dis, "omim", None)
    orpha_id = getattr(dis, "orpha_id", None) or getattr(dis, "orpha", None)
    if omim_id is not None:
        return normalize_disease_id(f"OMIM:{omim_id}")
    if orpha_id is not None:
        return normalize_disease_id(f"ORPHA:{orpha_id}")
    dis_id = getattr(dis, "id", None)
    dis_type = getattr(dis, "diseasetype", None)
    if dis_id is not None and dis_type:
        if str(dis_type).lower().startswith("omim"):
            return normalize_disease_id(f"OMIM:{dis_id}")
        if str(dis_type).lower().startswith("orpha"):
            return normalize_disease_id(f"ORPHA:{dis_id}")
    return None


def build_disease_matrix(
    model: SentenceTransformer,
    diseases: List[object],
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    embs: List[np.ndarray] = []
    ids: List[str] = []
    for dis in diseases:
        emb = disease_embedding(model, dis, batch_size=batch_size)
        embs.append(emb)
        ident = resolve_identifier(dis)
        if ident is None:
            name = str(getattr(dis, "name", "unknown"))
            ident = normalize_disease_id(name)
        ids.append(ident)
    mat = np.stack(embs, axis=0)
    ids = [normalize_disease_id(x) for x in ids]
    return mat, ids


def parse_phenopacket(path: Path) -> Optional[Tuple[str, List[str]]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    features = data.get("phenotypicFeatures", []) or []
    hpo_ids: List[str] = []
    for feat in features:
        if feat.get("excluded") is True:
            continue
        t = feat.get("type") or {}
        hid = t.get("id") or t.get("term")
        if hid:
            hpo_ids.append(str(hid))
    disease_id: Optional[str] = None
    for interp in data.get("interpretations", []) or []:
        diag = interp.get("diagnosis") or {}
        dis = diag.get("disease") or {}
        if dis.get("id"):
            disease_id = normalize_disease_id(str(dis["id"]))
            break
    if disease_id is None:
        for dis in data.get("diseases", []) or []:
            if dis.get("id"):
                disease_id = normalize_disease_id(str(dis["id"]))
                break
    if not disease_id or not hpo_ids:
        return None
    return disease_id, hpo_ids


def load_patients(
    phenopacket_dir: str,
    inventory: Dict[str, List[str]],
    min_phenos: int,
    seed: int,
) -> List[Tuple[str, List[str]]]:
    rng = random.Random(seed)
    patients: List[Tuple[str, List[str]]] = []
    for json_path in Path(phenopacket_dir).rglob("*.json"):
        parsed = parse_phenopacket(json_path)
        if not parsed:
            continue
        disease_id, hpo_ids = parsed
        # filtra a los que tengan frases disponibles
        hpo_ids = [hid for hid in hpo_ids if hid in inventory]
        if len(hpo_ids) < min_phenos:
            continue
        rng.shuffle(hpo_ids)
        patients.append((disease_id, hpo_ids))
    return patients


def build_patient_vector(
    model: SentenceTransformer,
    hpo_ids: Sequence[str],
    inventory: Dict[str, List[str]],
    batch_size: int,
    rng: random.Random,
) -> Optional[torch.Tensor]:
    sents: List[str] = []
    for hid in hpo_ids:
        choices = inventory.get(hid)
        if not choices:
            continue
        sents.append(rng.choice(choices))
    if not sents:
        return None
    with torch.no_grad():
        emb = model.encode(sents, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        emb = emb.mean(dim=0)
        emb = util.normalize_embeddings(emb.unsqueeze(0)).squeeze(0)
    return emb


def evaluate(
    model: SentenceTransformer,
    disease_matrix: torch.Tensor,
    disease_ids: List[str],
    patients: List[Tuple[str, List[str]]],
    inventory: Dict[str, List[str]],
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    hits1 = hits5 = 0
    mrr = 0.0
    total = 0
    phenos_count = 0
    for disease_id, hpo_ids in patients:
        patient_vec = build_patient_vector(model, hpo_ids, inventory, batch_size, rng)
        if patient_vec is None:
            continue
        phenos_count += len(hpo_ids)
        sims = torch.matmul(patient_vec, disease_matrix.T)  # (n_diseases,)
        order = torch.argsort(sims, descending=True)
        if disease_id not in disease_ids:
            continue
        target_indices = [i for i, did in enumerate(disease_ids) if did == disease_id]
        if not target_indices:
            continue
        ranks = [torch.nonzero(order == idx, as_tuple=False).item() for idx in target_indices]
        rank = min(ranks)
        total += 1
        if rank == 0:
            hits1 += 1
        if rank < 5:
            hits5 += 1
        mrr += 1.0 / (rank + 1)

    if total == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "evaluated": 0}
    avg_phenos = phenos_count / total if total else 0.0
    return {
        "top1": hits1 / total,
        "top5": hits5 / total,
        "mrr": mrr / total,
        "evaluated": total,
        "avg_phenos": avg_phenos,
    }


def resolve_models(specs: List[str]) -> Dict[str, str]:
    base_dir = Path("models/phase3")
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid model {spec!r}, use name=path")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        candidate = base_dir / path
        if candidate.exists():
            path = str(candidate)
        models[name] = path
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Disease retrieval from real phenopackets (phenotypicFeatures).")
    parser.add_argument("--phenopacket-dir", default="data/0.1.25", help="Root folder with phenopackets.")
    parser.add_argument("--sentences", default="data/phase2/hpo_sentence_pairs_struct_rbp.parquet", help="Phase 2 corpus with sentences per HPO.")
    parser.add_argument("--sentences-per-hpo", type=int, default=20, help="Max sentences in the inventory per HPO (reservoir).")
    parser.add_argument("--min-phenos", type=int, default=3, help="Discard patients with <N phenotypes with sentences.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "base_pubmedbert=NeuML/pubmedbert-base-embeddings",
            "ft_pubmed_pubmedbert=models/phase3/pubmedbert_pubmed_rbp_angle_best",
        ],
        help="List name=path. If the path is relative it is resolved against models/phase3.",
    )
    parser.add_argument("--include-omim", action="store_true", default=True, help="Include OMIM diseases.")
    parser.add_argument("--include-orpha", action="store_true", default=False, help="Include Orphanet diseases.")
    parser.add_argument("--output", default="phase4/phenopacket_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)

    inventory = load_sentence_inventory(args.sentences, max_per_hpo=args.sentences_per_hpo, seed=args.seed)
    print(f"Sentence inventory: {len(inventory)} HPO terms with sentences.")
    patients = load_patients(args.phenopacket_dir, inventory, min_phenos=args.min_phenos, seed=args.seed)
    print(f"Patients loaded (with at least {args.min_phenos} phenotypes and sentences): {len(patients)}")

    model_paths = resolve_models(args.models)
    results: Dict[str, Dict[str, float]] = {}

    for name, path in model_paths.items():
        print(f"Evaluating model {name} ({path})…")
        model = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu")
        diseases = collect_diseases(include_omim=args.include_omim, include_orpha=args.include_orpha)
        dis_matrix_np, dis_ids = build_disease_matrix(model, diseases, batch_size=args.batch_size)
        dis_matrix = torch.tensor(dis_matrix_np, device=model.device, dtype=torch.float32)
        dis_id_set = {normalize_disease_id(x) for x in dis_ids}
        patients_with_label = [(d, h) for d, h in patients if normalize_disease_id(d) in dis_id_set]
        if not patients_with_label:
            print("  ⚠️ No patient has a disease present in the embedded OMIM/Orpha set. Check --include-orpha or the data.")
            print(f"  Example patient disease_ids: {list({p[0] for p in patients})[:5]}")
            print(f"  Example embedded disease_ids: {dis_ids[:5]}")
            results[name] = {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "evaluated": 0}
            continue
        missing = len(patients) - len(patients_with_label)
        if missing:
            print(f"  {missing} patients discarded due to disease out of vocabulary.")
        metrics = evaluate(
            model,
            dis_matrix,
            dis_ids,
            patients_with_label,
            inventory,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        results[name] = metrics
        print(
            f"  top1={metrics['top1']:.3f} top5={metrics['top5']:.3f} "
            f"mrr={metrics['mrr']:.3f} avg_phenos={metrics.get('avg_phenos',0):.2f} "
            f"(n={metrics['evaluated']})"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
