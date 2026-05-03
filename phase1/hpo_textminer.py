#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main pipeline script for HPO PubMed Miner.
- Loads HPO terms (with optional synonyms)
- Searches PMC Open Access articles per term
- Downloads JATS XML in batches
- Extracts sentences containing the term (or any of its synonyms)
- Saves incrementally to JSONL
- Consolidates to Parquet (single file or partitioned dataset)

Usage:
    python phase1/hpo_textminer.py --limit-terms 500 --synonyms --max-ids-per-term 1000
    python phase1/hpo_textminer.py --only-consolidate
"""
from __future__ import annotations
import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote
from tqdm import tqdm

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import config, pubmed_fetcher, text_extractor, hpo_loader, saver

def build_regex(names: list[str]) -> re.Pattern | None:
    # Escaped alternatives, whole-word match, case-insensitive
    alts: list[str] = []
    for raw in names:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        alts.append(re.escape(text))
    if not alts:
        return None
    pattern = re.compile(r"\b(" + "|".join(alts) + r")\b", re.IGNORECASE)
    return pattern


def process_term(term_info: dict, max_ids: int | None, max_sentences: int | None) -> int:
    """
    For an HPO term:
    - Searches PMC OA by the term label (search seed).
    - Downloads XML in batches.
    - Extracts relevant sentences with metadata (pmcid, section).
    - Saves ONLY sentences containing the label or its synonyms.
    """
    label = str(term_info.get("label", "")).strip()
    synonyms = term_info.get("names") or term_info.get("synonyms") or []
    all_terms = [label] + [s for s in synonyms if s and s != label]
    pattern = build_regex(all_terms)
    if pattern is None:
        return 0

    if not label:
        return 0

    pmc_ids = pubmed_fetcher.search_pmc_open_access(label, retmax=max_ids)
    if not pmc_ids:
        return 0

    total = 0
    for xml_str in pubmed_fetcher.batched_fetch_xml(pmc_ids, config.BATCH_SIZE):
        if max_sentences is not None and total >= max_sentences:
            break
        # batched_fetch_xml may return a string (batch XML) or None
        if not xml_str:
            continue
        records = text_extractor.extract_relevant_sentences(
            xml_str,
            hpo_label=label,
            hpo_id=term_info.get("hpo_id", ""),
            pattern=pattern,
        )
        if not records:
            continue

        if max_sentences is not None:
            remaining = max_sentences - total
            if remaining <= 0:
                break
            if len(records) > remaining:
                records = records[:remaining]

        saver.write_jsonl(records)
        total += len(records)

    logging.info(f"{label}: {total} relevant sentences")
    return total


def load_processed_hpo_ids() -> set[str]:
    """
    Retrieves HPO IDs that already have a partition in the final Parquet.
    Allows skipping already-consolidated terms.
    """
    if not config.PARQUET_PARTITION_BY_HPO:
        return set()

    base_dir = config.PARQUET_PATH
    if base_dir.endswith(".parquet"):
        base_dir = f"{base_dir}_dir"

    if not os.path.isdir(base_dir):
        return set()

    processed: set[str] = set()
    prefix = "hpo_id="
    for entry in os.scandir(base_dir):
        if entry.is_dir() and entry.name.startswith(prefix):
            encoded = entry.name[len(prefix) :]
            processed.add(unquote(encoded))
    return processed


def main():
    parser = argparse.ArgumentParser(description="HPO PubMed Miner")
    parser.add_argument("--limit-terms", type=int, default=None, help="Limit number of HPO terms to process")
    parser.add_argument("--synonyms", action="store_true", help="Include HPO synonyms in regex")
    parser.add_argument("--no-synonyms", dest="synonyms", action="store_false")
    parser.set_defaults(synonyms=True)
    parser.add_argument("--max-ids-per-term", type=int, default=None, help="Override MAX_PMIDS_PER_TERM")
    parser.add_argument(
        "--max-sentences-per-term",
        type=int,
        default=None,
        help="Stop processing a phenotype after saving N sentences (None = no limit)",
    )
    parser.add_argument("--only-consolidate", action="store_true", help="Skip download; just consolidate JSONL->Parquet")

    args = parser.parse_args()

    max_sentences_per_term = (
        args.max_sentences_per_term
        if args.max_sentences_per_term is not None
        else config.MAX_SENTENCES_PER_TERM
    )

    logging.basicConfig(
        filename=config.LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.only_consolidate:
        out = saver.consolidate_jsonl_to_parquet()
        sample = saver.validate_parquet(out)
        print("Consolidation OK. Sample:")
        print(sample.head())
        return

    terms = hpo_loader.load_hpo_terms(include_synonyms=args.synonyms)

    processed_ids = load_processed_hpo_ids()
    if processed_ids:
        logging.info(f"Detected {len(processed_ids)} already-consolidated terms.")
        before_filter = len(terms)
        terms = [t for t in terms if t.get("hpo_id") not in processed_ids]
        skipped = before_filter - len(terms)
        if skipped:
            print(f"Skipping {skipped} terms already present in Parquet.")

    if args.limit_terms:
        terms = terms[: args.limit_terms]

    if not terms:
        print("No new HPO terms found to process. Skipping download.")
    else:
        print(f"Processing {len(terms)} HPO terms in parallel ({config.MAX_THREADS} threads)...")

        with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as ex:
            futures = [
                ex.submit(
                    process_term,
                    t,
                    args.max_ids_per_term or config.MAX_PMIDS_PER_TERM,
                    max_sentences_per_term,
                )
                for t in terms
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processed phenotypes"):
                try:
                    n = f.result()
                    if n:
                        logging.info(f"Saved {n} sentences")
                except Exception as e:
                    logging.error(f"Error processing term: {e}")

    out = saver.consolidate_jsonl_to_parquet()
    print(f"Final Parquet: {out}")


if __name__ == "__main__":
    main()
