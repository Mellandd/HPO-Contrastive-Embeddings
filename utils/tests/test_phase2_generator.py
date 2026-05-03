import pathlib
import shutil
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
TMP_ROOT = pathlib.Path(__file__).resolve().parent / "__tmp__"
TMP_ROOT.mkdir(exist_ok=True)

from phase2.generate_pair_dataset import (
    HPOSentencePairGenerator,
    _allocate_counts,
    assign_hpo_ids_to_splits,
    split_pairs_by_hpo,
)


def _prepare_dir(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_partitioned_dataset(base_path, records):
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(table, root_path=str(base_path), partition_cols=["hpo_id"])


def test_load_sentence_inventory_limits_sentences():
    dataset_dir = _prepare_dir(TMP_ROOT / "load_inventory")
    records = [
        {"hpo_id": "HP:0001250", "sentence": "S seizure 1"},
        {"hpo_id": "HP:0001250", "sentence": "S seizure 2"},
        {"hpo_id": "HP:0001250", "sentence": "S seizure 3"},
        {"hpo_id": "HP:0000707", "sentence": "S developmental 1"},
        {"hpo_id": "HP:0000707", "sentence": "S developmental 2"},
        {"hpo_id": "HP:0000707", "sentence": "S developmental 3"},
    ]
    _write_partitioned_dataset(dataset_dir, records)

    generator = HPOSentencePairGenerator(
        dataset_path=str(dataset_dir),
        max_sentences_per_hpo=2,
        min_sentences_per_hpo=2,
        show_progress=False,
    )
    inventory = generator.load_sentence_inventory()

    assert set(inventory.keys()) == {"HP:0001250", "HP:0000707"}
    assert all(len(sentences) == 2 for sentences in inventory.values())


def test_sample_pairs_produces_requested_buckets():
    dataset_dir = _prepare_dir(TMP_ROOT / "sample_pairs")
    # Four HPO terms with varying similarity
    records = [
        {"hpo_id": "HP:0001250", "sentence": "Seizure case A"},
        {"hpo_id": "HP:0001250", "sentence": "Seizure case B"},
        {"hpo_id": "HP:0000707", "sentence": "Developmental delay example"},
        {"hpo_id": "HP:0000707", "sentence": "Developmental delay follow-up"},
        {"hpo_id": "HP:0000505", "sentence": "Visual impairment observation"},
        {"hpo_id": "HP:0000505", "sentence": "Vision changes reported"},
        {"hpo_id": "HP:0000726", "sentence": "Intellectual disability diagnosis"},
        {"hpo_id": "HP:0000726", "sentence": "Cognitive impairment note"},
    ]
    _write_partitioned_dataset(dataset_dir, records)

    generator = HPOSentencePairGenerator(
        dataset_path=str(dataset_dir),
        positive_threshold=0.5,
        negative_threshold=0.05,
        max_sentences_per_hpo=2,
        min_sentences_per_hpo=2,
        show_progress=False,
        seed=7,
    )
    inventory = generator.load_sentence_inventory()
    df, counts = generator.sample_pairs(
        inventory,
        positive_target=1,
        intermediate_target=1,
        negative_target=1,
    )

    assert counts == {"positive": 1, "intermediate": 1, "negative": 1}
    assert set(df.columns) == {"sentence1", "sentence2", "hpo_id1", "hpo_id2", "gold_similarity"}
    assert len(df) == 3
    assert df["gold_similarity"].between(0, 1).all()


def test_split_pairs_by_hpo_respects_disjoint_terms():
    df = pd.DataFrame(
        [
            {
                "sentence1": "a",
                "sentence2": "b",
                "hpo_id1": "HP:0001250",
                "hpo_id2": "HP:0000707",
                "gold_similarity": 0.6,
            },
            {
                "sentence1": "c",
                "sentence2": "d",
                "hpo_id1": "HP:0000726",
                "hpo_id2": "HP:0007104",
                "gold_similarity": 0.4,
            },
            {
                "sentence1": "e",
                "sentence2": "f",
                "hpo_id1": "HP:0100010",
                "hpo_id2": "HP:0100011",
                "gold_similarity": 0.5,
            },
        ]
    )
    splits, dropped = split_pairs_by_hpo(
        df,
        splits=[("train", 0.34), ("val", 0.33), ("test", 0.33)],
        seed=11,
    )

    assert dropped == 0
    total_rows = sum(len(part) for part in splits.values())
    assert total_rows == len(df)

    seen_terms: set[str] = set()
    all_terms = set(df["hpo_id1"]).union(df["hpo_id2"])
    for part in splits.values():
        part_terms = set(part["hpo_id1"]).union(part["hpo_id2"])
        assert seen_terms.isdisjoint(part_terms)
        seen_terms.update(part_terms)
        # Each split should retain the original column schema
        assert set(part.columns) == set(df.columns)
    assert seen_terms == all_terms


def test_allocate_counts_respects_total_and_ratios():
    split_plan = [("train", 0.7), ("val", 0.2), ("test", 0.1)]
    counts = _allocate_counts(100, split_plan)
    assert sum(counts.values()) == 100
    # Ensure ordering roughly follows ratios
    assert counts["train"] >= counts["val"] >= counts["test"]


def test_assign_hpo_ids_to_splits_matches_counts():
    split_plan = [("train", 0.6), ("val", 0.2), ("test", 0.2)]
    ids = [f"HP:{i:07d}" for i in range(20)]
    assignments = assign_hpo_ids_to_splits(ids, split_plan, seed=42)
    counts = {name: len(values) for name, values in assignments.items()}
    expected = _allocate_counts(len(ids), split_plan)
    assert counts == expected
    # Ensure IDs are unique across splits
    combined = sum((values for values in assignments.values()), [])
    assert sorted(combined) == sorted(ids)
