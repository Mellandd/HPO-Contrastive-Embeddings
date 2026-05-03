#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a dataset with HPO subontology labels by level.

Level 1 = direct children of the root node (HP:0000118 by default).
Level 2 = grandchildren of the root node, etc.

The script traverses the full ontology (PyHPO) with a BFS from the root node
and assigns each term the ancestor that corresponds to each level. The
labels include both the name and the identifier of the ancestor.
"""
from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyhpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Builds hierarchical HPO subontology labels.")
    parser.add_argument("--output", default="data/phase4/hpo_subontology_labels.parquet")
    parser.add_argument("--root-id", default="HP:0000118", help="Root node (default HP:0000118, Phenotypic abnormality).")
    parser.add_argument("--max-levels", type=int, default=2, help="Number of descendant levels to label.")
    parser.add_argument(
        "--filter-path",
        type=Path,
        default=None,
        help="Parquet/CSV file with column 'hpo_id' to filter terms (optional).",
    )
    parser.add_argument("--filter-column", default="hpo_id", help="Name of the column containing IDs in --filter-path.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


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


def build_label_dataframe(root_id: str, max_levels: int) -> pd.DataFrame:
    ontology = pyhpo.Ontology()
    try:
        root = ontology.get_hpo_object(root_id)
    except RuntimeError as exc:
        raise ValueError(f"Root node not found: {root_id}") from exc

    records: List[Dict[str, Optional[str]]] = []
    queue = deque([(root, [root])])
    visited: set[str] = set()

    while queue:
        node, path = queue.popleft()
        if node.id in visited:
            continue
        visited.add(node.id)

        row: Dict[str, Optional[str]] = {
            "hpo_id": node.id,
            "hpo_label": node.name or node.id,
        }
        for level in range(1, max_levels + 1):
            if len(path) > level:
                ancestor = path[level]
                row[f"label_lvl{level}_id"] = ancestor.id
                row[f"label_lvl{level}"] = ancestor.name or ancestor.id
            else:
                row[f"label_lvl{level}_id"] = None
                row[f"label_lvl{level}"] = None
        records.append(row)

        children = getattr(node, "children", None) or []
        for child in children:
            queue.append((child, path + [child]))

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    labels_df = build_label_dataframe(args.root_id, args.max_levels)
    logging.info("Generated labels for %d HPO terms.", len(labels_df))

    filter_ids = load_filter_ids(args.filter_path, args.filter_column)
    if filter_ids is not None:
        before = len(labels_df)
        labels_df = labels_df[labels_df["hpo_id"].isin(filter_ids)].reset_index(drop=True)
        logging.info("Optional filtering: %d -> %d terms.", before, len(labels_df))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        labels_df.to_csv(output_path, index=False)
    else:
        labels_df.to_parquet(output_path, index=False)
    logging.info("Labels saved to %s", output_path)


if __name__ == "__main__":
    main()
