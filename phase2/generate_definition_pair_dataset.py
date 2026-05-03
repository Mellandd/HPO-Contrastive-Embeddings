#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sentence pairs using only HPO term definitions.

This mirrors the PubMed-based pair generation but uses the ontology
definition (single sentence) for each term. The same similarity thresholds
and sampling strategy are applied, enabling controlled experiments on
whether incorporating PubMed sentences improves performance.

Outputs:
    - Full pair dataset (Parquet)
    - Train/val/test splits (Parquet) using the same split logic as the
      PubMed corpus (no overlapping HPO IDs between splits).
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Tuple

import pandas as pd
import pyhpo

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase2.generate_pair_dataset import (
    HPOSentencePairGenerator,
    format_input_prompt,
    sample_structured_pairs_with_preassigned_splits,
    sample_pairs_with_preassigned_splits,
    split_pairs_by_hpo,
)


LOGGER = logging.getLogger(__name__)


def build_definition_inventory() -> Dict[str, List[str]]:
    ontology = pyhpo.Ontology()
    inventory: Dict[str, List[str]] = {}
    skipped = 0
    fallback = 0
    for term in ontology:
        definition = getattr(term, "definition", None)
        text = ""
        if definition:
            text = str(definition).strip()
        if not text:
            text = (term.name or "").strip()
            if text:
                fallback += 1
        if not text:
            skipped += 1
            continue
        cleaned = text.replace("\n", " ").strip()
        phenotype_name = (term.name or "").strip() or term.id
        inventory[term.id] = [format_input_prompt(phenotype_name, cleaned)]
    LOGGER.info(
        "Collected definitions for %d HPO terms (fallback to label for %d, skipped %d).",
        len(inventory),
        fallback,
        skipped,
    )
    if not inventory:
        raise RuntimeError("No HPO definitions available; cannot generate pairs.")
    return inventory


def save_pairs(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d pairs to %s", len(df), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HPO definition-based sentence pairs.")
    parser.add_argument("--output", default="data/phase2/hpo_definition_pairs.parquet")
    parser.add_argument("--split-dir", default="data/phase2/definition_splits")
    parser.add_argument(
        "--sampling-strategy",
        choices=["structured", "bucketed"],
        default="structured",
        help="structured: per-term positive/hard/easy triplets; bucketed: legacy threshold-based sampling.",
    )
    parser.add_argument("--passes", type=int, default=1, help="Passes over all phenotypes (structured).")
    parser.add_argument("--hard-min-sim", type=float, default=0.4)
    parser.add_argument("--hard-max-sim", type=float, default=0.7)
    parser.add_argument("--easy-max-sim", type=float, default=0.2)
    parser.add_argument("--hard-attempts", type=int, default=60)
    parser.add_argument("--easy-attempts", type=int, default=60)
    parser.add_argument("--positive-target", type=int, default=80000)
    parser.add_argument("--intermediate-target", type=int, default=120000)
    parser.add_argument("--negative-target", type=int, default=80000)
    parser.add_argument("--positive-threshold", type=float, default=0.7)
    parser.add_argument("--negative-threshold", type=float, default=0.3)
    parser.add_argument("--method", default="lin")
    parser.add_argument(
        "--rbp-alpha",
        type=float,
        default=0.01,
        help="Alpha cap for the RelativeBestPair similarity function.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    inventory = build_definition_inventory()
    generator = HPOSentencePairGenerator(
        dataset_path="data/hpo_sentences.parquet_dir",
        method=args.method,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        max_sentences_per_hpo=1,
        min_sentences_per_hpo=1,
        show_progress=False,
        rbp_alpha=args.rbp_alpha,
    )

    train_ratio = 1.0 - args.val_ratio - args.test_ratio
    if train_ratio <= 0:
        raise ValueError("Invalid split ratios; train proportion must be positive.")

    split_plan = [("train", train_ratio), ("val", args.val_ratio), ("test", args.test_ratio)]
    total_targets = {
        "positive": args.positive_target,
        "intermediate": args.intermediate_target,
        "negative": args.negative_target,
    }

    if args.sampling_strategy == "structured":
        combined_df, split_frames, achieved = sample_structured_pairs_with_preassigned_splits(
            generator,
            inventory,
            split_plan,
            passes=args.passes,
            hard_min_sim=args.hard_min_sim,
            hard_max_sim=args.hard_max_sim,
            easy_max_sim=args.easy_max_sim,
            hard_attempts=args.hard_attempts,
            easy_attempts=args.easy_attempts,
            seed=args.seed,
        )
        save_pairs(combined_df, args.output)

        totals = {"positive": 0, "hard": 0, "easy": 0}
        split_counts: Dict[str, int] = {}
        os.makedirs(args.split_dir, exist_ok=True)
        for name, frame in split_frames.items():
            path = os.path.join(args.split_dir, f"{name}.parquet")
            frame.to_parquet(path, index=False)
            split_counts[name] = len(frame)
            counts = achieved.get(name, {"positive": 0, "hard": 0, "easy": 0})
            LOGGER.info(
                "Split %s counts - positive: %d, hard: %d, easy: %d",
                name,
                counts.get("positive", 0),
                counts.get("hard", 0),
                counts.get("easy", 0),
            )
            for bucket in totals:
                totals[bucket] += counts.get(bucket, 0)

        LOGGER.info("Final split sizes: %s", split_counts)
        LOGGER.info(
            "Aggregate counts - positive: %d, hard: %d, easy: %d",
            totals["positive"],
            totals["hard"],
            totals["easy"],
        )
        return

    combined_df, split_frames, achieved = sample_pairs_with_preassigned_splits(
        generator,
        inventory,
        total_targets,
        split_plan,
        seed=args.seed,
    )
    save_pairs(combined_df, args.output)

    os.makedirs(args.split_dir, exist_ok=True)
    totals = {"positive": 0, "intermediate": 0, "negative": 0}
    split_counts: Dict[str, int] = {}
    for name, frame in split_frames.items():
        path = os.path.join(args.split_dir, f"{name}.parquet")
        frame.to_parquet(path, index=False)
        split_counts[name] = len(frame)
        counts = achieved.get(name, {"positive": 0, "intermediate": 0, "negative": 0})
        LOGGER.info(
            "Split %s counts - positive: %d, intermediate: %d, negative: %d",
            name,
            counts.get("positive", 0),
            counts.get("intermediate", 0),
            counts.get("negative", 0),
        )
        for bucket in totals:
            totals[bucket] += counts.get(bucket, 0)

    LOGGER.info("Final split sizes: %s", split_counts)
    LOGGER.info(
        "Aggregate counts - positive: %d, intermediate: %d, negative: %d",
        totals["positive"],
        totals["intermediate"],
        totals["negative"],
    )


if __name__ == "__main__":
    main()
