"""Central configuration for the HPO Text Miner project.
Adjust API keys, paths and performance parameters here.
"""
from __future__ import annotations
import os

# === NCBI / Entrez ===
ENTREZ_EMAIL: str = os.getenv("ENTREZ_EMAIL", "")  # Set via environment variable
ENTREZ_API_KEY: str | None = os.getenv("ENTREZ_API_KEY")  # Set via environment variable
# Respect NCBI limits: ~3 req/s without API key, ~10 req/s with API key
RATE_LIMIT_RPS: float = 3.0 if ENTREZ_API_KEY is None else 10.0

# === Download / processing ===
MAX_THREADS: int = int(os.getenv("MAX_THREADS", "12"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
MAX_PMIDS_PER_TERM: int | None = None  # None = no limit (set to 1000 for testing)
_raw_max_sentences = os.getenv("MAX_SENTENCES_PER_TERM")
MAX_SENTENCES_PER_TERM: int | None = int(_raw_max_sentences) if _raw_max_sentences else None

# === Paths ===
DATA_DIR: str = os.getenv("DATA_DIR", "data")
JSONL_PATH: str = os.path.join(DATA_DIR, "sentences_temp.jsonl")
PARQUET_PATH: str = os.path.join(DATA_DIR, "hpo_sentences.parquet")
LOG_PATH: str = os.path.join(DATA_DIR, "hpo_textminer.log")
STATE_PATH: str = os.path.join(DATA_DIR, "state.json")  # simple checkpoints

# === Parquet ===
PARQUET_COMPRESSION: str = os.getenv("PARQUET_COMPRESSION", "zstd")
PARQUET_PARTITION_BY_HPO: bool = True  # True to partition by hpo_id

# === Tokenisation ===
SENTENCE_MIN_LEN: int = 20
USE_SPACY: bool = True  # True if spaCy is installed and a model is loadable
# === EntityLinker SciSpaCy ===
USE_ENTITY_LINKER = True
SPACY_MODEL = "en_core_sci_sm"  # or "en_core_sci_md"
ENTITY_LINKER_SCORE_THRESHOLD = 0.85

# === Retries / backoff ===
MAX_RETRIES: int = 5
BACKOFF_BASE_SECONDS: float = 1.0

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
