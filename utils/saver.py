"""
Save utilities: incremental JSONL (thread-safe) and robust Parquet consolidation.
- Concurrent-safe JSONL writes
- Optional size-based rotation
- Streaming (chunked) Parquet consolidation with optional hpo_id partitioning
- Optional hash-based deduplication (when the 'hash' column exists)
"""
from __future__ import annotations
from typing import Iterable, List, Optional
import os
import time
import threading
import logging

import jsonlines
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

_lock = threading.Lock()

def write_jsonl(records):
    """
    Write records to JSONL ensuring all values are str.
    Avoids errors such as "a bytes-like object is required, not 'int'".
    """
    os.makedirs(os.path.dirname(config.JSONL_PATH), exist_ok=True)

    def _to_str_dict(rec):
        # convert everything to str; None -> ""
        return {str(k): ("" if v is None else str(v)) for k, v in rec.items()}

    with _lock:
        with jsonlines.open(config.JSONL_PATH, mode="a") as writer:
            for rec in records:
                writer.write(_to_str_dict(rec))


def rotate_jsonl(max_bytes: int = 2_000_000_000, keep: int = 3) -> Optional[str]:
    """Rotate the JSONL file if it exceeds `max_bytes`. Keeps `keep` rotations.
    Returns the active path (post-rotation) or None if no rotation occurred.
    """
    path = config.JSONL_PATH
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    if size < max_bytes:
        return None
    # Shift backups
    for i in range(keep, 0, -1):
        src = f"{path}.{i}"
        dst = f"{path}.{i+1}"
        if os.path.exists(src):
            try:
                if i == keep:
                    os.remove(src)
                else:
                    os.replace(src, dst)
            except Exception:
                pass
    # Rotate current file
    rotated = f"{path}.1"
    os.replace(path, rotated)
    logging.info(f"JSONL rotated: {rotated}")
    return path


def _ensure_schema(df: pd.DataFrame) -> pa.Schema:
    """Define a stable Arrow schema for Parquet."""
    dtypes = {
        "hpo_id": pa.string(),
        "hpo_label": pa.string(),
        "pmcid": pa.string(),
        "section": pa.string(),
        "sentence": pa.string(),
        "hash": pa.string(),
    }
    fields = []
    for col, typ in dtypes.items():
        if col in df.columns:
            fields.append(pa.field(col, typ))
    return pa.schema(fields)


def _to_table(df: pd.DataFrame, schema: Optional[pa.Schema] = None) -> pa.Table:
    """Convert a pandas DataFrame to an Arrow Table with safe types (all string)."""
    # Ensure columns are present and have string type
    expected_cols = ["hpo_id", "hpo_label", "pmcid", "section", "sentence", "hash"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
        else:
            # Convert to string, except if already one
            if df[col].dtype != "object":
                df[col] = df[col].astype(str)
    if schema is None:
        schema = _ensure_schema(df)
    tbl = pa.Table.from_pandas(df[expected_cols], schema=schema, preserve_index=False)
    return tbl


def consolidate_jsonl_to_parquet(chunksize: int = 250_000) -> str:
    """Convert the JSONL to Parquet incrementally.

    - If config.PARQUET_PARTITION_BY_HPO=True, writes a dataset partitioned by 'hpo_id'.
    - If False, writes to a single Parquet file in *streaming* mode using ParquetWriter.
    - If the 'hash' column exists, performs *intra-chunk* and optionally *global in-memory* deduplication.
    """
    jsonl = config.JSONL_PATH
    if not os.path.exists(jsonl):
        raise FileNotFoundError(f"JSONL does not exist at {jsonl}")

    reader = pd.read_json(jsonl, lines=True, chunksize=chunksize)

    # Global deduplication (by hash) — disable if RAM is limited
    seen: set[str] = set()
    use_global_dedup = True

    out_path = config.PARQUET_PATH

    if config.PARQUET_PARTITION_BY_HPO:
        # Partitioned dataset to directory (supports append)
        base_dir = out_path if not out_path.endswith(".parquet") else out_path + "_dir"
        os.makedirs(base_dir, exist_ok=True)
        schema: Optional[pa.Schema] = None
        total_rows = 0
        for chunk in reader:
            if "hash" in chunk.columns:
                chunk.drop_duplicates(subset=["hash"], inplace=True)
                if use_global_dedup:
                    before = len(chunk)
                    chunk = chunk[~chunk["hash"].isin(seen)]
                    seen.update(chunk["hash"].astype(str).tolist())
                    logging.info(f"DEDUP(part): {before} -> {len(chunk)}")
            if chunk.empty:
                continue
            if schema is None:
                schema = _ensure_schema(chunk)
            tbl = _to_table(chunk, schema)
            pq.write_to_dataset(
                tbl,
                root_path=base_dir,
                partition_cols=["hpo_id"],
                existing_data_behavior="overwrite_or_ignore",
            )
            total_rows += len(chunk)
        logging.info(f"Written partitioned dataset to {base_dir} ({total_rows} rows)")
        return base_dir

    else:
        # Single Parquet file using ParquetWriter
        tmp_path = out_path + ".tmp"
        writer: Optional[pq.ParquetWriter] = None
        total_rows = 0
        schema: Optional[pa.Schema] = None
        try:
            for chunk in reader:
                if "hash" in chunk.columns:
                    chunk.drop_duplicates(subset=["hash"], inplace=True)
                    if use_global_dedup:
                        before = len(chunk)
                        chunk = chunk[~chunk["hash"].isin(seen)]
                        seen.update(chunk["hash"].astype(str).tolist())
                        logging.info(f"DEDUP(part): {before} -> {len(chunk)}")
                if chunk.empty:
                    continue
                if schema is None:
                    schema = _ensure_schema(chunk)
                    table = _to_table(chunk, schema)
                    writer = pq.ParquetWriter(
                        tmp_path, schema=schema, compression=config.PARQUET_COMPRESSION
                    )
                    writer.write_table(table)
                else:
                    table = _to_table(chunk, schema)
                    writer.write_table(table)
                total_rows += len(chunk)
            if writer is None:
                raise RuntimeError("No rows found to consolidate")
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp_path, out_path)
        logging.info(f"Written single Parquet to {out_path} ({total_rows} rows)")
        return out_path


def validate_parquet(path: Optional[str] = None, sample_rows: int = 5) -> pd.DataFrame:
    """Read a few rows from the Parquet to validate it is readable."""
    if path is None:
        path = config.PARQUET_PATH
    if os.path.isdir(path):
        import pyarrow.dataset as ds
        dataset = ds.dataset(path, format="parquet")
        table = dataset.head(sample_rows)
        return table.to_pandas()
    else:
        return pd.read_parquet(path)