#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a text corpus per HPO term combining definitions and PubMed sentences.

The output is a Parquet (or CSV) with columns:
    - hpo_id
    - texts  (list of strings associated with the term)

All models will be able to consume exactly the same set of texts
to produce consistent embeddings.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.dataset as ds
import pyhpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates a combined corpus (definitions + sentences) by HPO.")
    parser.add_argument("--output", default="data/phase4/hpo_text_corpus.parquet")
    parser.add_argument("--dataset", default="data/hpo_sentences.parquet_dir", help="Partitioned Parquet with PubMed sentences.")
    parser.add_argument("--sentences-per-hpo", type=int, default=20, help="Max sentences to retain per term (reservoir sampling).")
    parser.add_argument("--include-definitions", action="store_true", help="Add definition/label to the corpus.")
    parser.add_argument("--include-sentences", action="store_true", help="Add PubMed sentences to the corpus.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--filter-path", type=Path, default=None, help="Parquet/CSV with column 'hpo_id' for optional filtering.")
    parser.add_argument("--filter-column", default="hpo_id")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if not args.include_definitions and not args.include_sentences:
        parser.error("You must enable --include-definitions and/or --include-sentences.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_filter_ids(path: Optional[Path], column: str) -> Optional[set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=[column])
    else:
        df = pd.read_parquet(path, columns=[column])
    return set(df[column].astype(str).tolist())


def add_definition_texts(target: Dict[str, List[str]], filter_ids: Optional[set[str]]) -> None:
    ontology = pyhpo.Ontology()
    for term in ontology:
        if filter_ids and term.id not in filter_ids:
            continue
        definition = getattr(term, "definition", None)
        text = ""
        if definition:
            text = str(definition).strip()
        if not text:
            text = (term.name or "").strip()
        if not text:
            continue
        clean = " ".join(text.split())
        target[term.id].append(clean)


def reservoir_add(pool: List[str], item: str, limit: int, rng: random.Random) -> None:
    if limit <= 0:
        pool.append(item)
        return
    if len(pool) < limit:
        pool.append(item)
        return
    idx = rng.randrange(len(pool) + 1)
    if idx < len(pool):
        pool[idx] = item


def add_pubmed_sentences(
    target: Dict[str, List[str]],
    dataset_path: str,
    sentences_per_hpo: int,
    filter_ids: Optional[set[str]],
    seed: int,
) -> None:
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    rng = random.Random(seed)
    buffers: Dict[str, List[str]] = defaultdict(list)

    for batch in dataset.to_batches(columns=["hpo_id", "sentence"], batch_size=20_000):
        data = batch.to_pydict()
        ids = data["hpo_id"]
        sentences = data["sentence"]
        for hpo_id, sentence in zip(ids, sentences):
            if not hpo_id or not sentence:
                continue
            if filter_ids and hpo_id not in filter_ids:
                continue
            text = str(sentence).strip()
            if not text:
                continue
            reservoir_add(buffers[hpo_id], text, sentences_per_hpo, rng)

    for hpo_id, items in buffers.items():
        target[hpo_id].extend(items)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)

    filter_ids = load_filter_ids(args.filter_path, args.filter_column)
    corpus: Dict[str, List[str]] = defaultdict(list)

    if args.include_definitions:
        logging.info("Adding definitions...")
        add_definition_texts(corpus, filter_ids)
    if args.include_sentences:
        logging.info("Adding PubMed sentences...")
        add_pubmed_sentences(
            corpus,
            dataset_path=args.dataset,
            sentences_per_hpo=args.sentences_per_hpo,
            filter_ids=filter_ids,
            seed=args.seed,
        )

    rows = [
        {"hpo_id": hpo_id, "texts": texts}
        for hpo_id, texts in corpus.items()
        if texts
    ]
    if not rows:
        raise RuntimeError("The resulting corpus is empty.")

    df = pd.DataFrame(rows)
    logging.info("Corpus ready with %d terms.", len(df))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)
    logging.info("Corpus saved to %s", output_path)

    meta = {
        "include_definitions": args.include_definitions,
        "include_sentences": args.include_sentences,
        "sentences_per_hpo": args.sentences_per_hpo,
        "dataset": args.dataset,
        "num_terms": len(df),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
