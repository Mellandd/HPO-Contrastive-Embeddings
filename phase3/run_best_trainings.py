#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the 6 training jobs with the optimal hyperparameters found.

Base models:
 - NeuML/pubmedbert-base-embeddings
 - sentence-transformers/all-MiniLM-L6-v2
 - cambridgeltl/SapBERT-from-PubMedBERT-fulltext

For each one, the following are trained:
 - FT with PubMed phrases (Phase2 structured splits, RBP and Lin versions)
 - FT with definitions (structured splits, RBP and Lin versions)
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

# Allow relative imports to the repository without needing to install as a package
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase3.train_sentence_transformer import TrainingConfig, train_model, save_report
from sentence_transformers import SentenceTransformer, models as st_models


BEST_PARAMS = {
    "use_lora": False,
    "freeze_layers": 6,
    "max_lr": 7.873739534437914e-05,
    "batch_size": 64,
    "max_seq_length": 256,
    "warmup_ratio": 0.06040372109274485,
    "weight_decay": 0.05,
    "loss_type": "angle",
    "epochs": 4,
    "min_lr": 1e-6,
    "max_grad_norm": 1.0,
    "use_amp": True,
    "train_eval_sample": 2000,
    "skip_test": True,
    "skip_eval": False,
    "eval_steps": 0,
    "seed": 13,
    "eval_batch_size": 256,
}

FAMILIES = [
    # {
    #     "name": "pubmedbert",
    #     "model_name": "NeuML/pubmedbert-base-embeddings",
    # },
    # {
    #     "name": "minilm",
    #     "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    # },
    {
        "name": "sapbert",
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    },
]

DATASETS = {
    "pubmed_rbp": {
        "train_path": "data/phase2/splits_struct_rbp/train.parquet",
        "val_path": "data/phase2/splits_struct_rbp/val.parquet",
    },
    "pubmed_lin": {
        "train_path": "data/phase2/splits_struct_lin/train.parquet",
        "val_path": "data/phase2/splits_struct_lin/val.parquet",
    },
    "defs_rbp": {
        "train_path": "data/phase2/definition_splits_struct_rbp/train.parquet",
        "val_path": "data/phase2/definition_splits_struct_rbp/val.parquet",
    },
    "defs_lin": {
        "train_path": "data/phase2/definition_splits_struct_lin/train.parquet",
        "val_path": "data/phase2/definition_splits_struct_lin/val.parquet",
    },
}


def ensure_sentence_transformer(model_name: str, init_dir: Path) -> str:
    """
    Some checkpoints (e.g., SapBERT) are HuggingFace-only and cannot be loaded
    directly with SentenceTransformer(model_name). We convert them to a simple
    Transformer+Pooling SentenceTransformer and return the local path.
    """
    # For SapBERT (HF-only), always build a SentenceTransformer wrapper
    if "sapbert" in model_name.lower():
        if init_dir.exists():
            return str(init_dir)
        init_dir.mkdir(parents=True, exist_ok=True)
        word_emb = st_models.Transformer(model_name, max_seq_length=BEST_PARAMS.get("max_seq_length", 256))
        pooling = st_models.Pooling(word_emb.get_word_embedding_dimension())
        st_model = SentenceTransformer(modules=[word_emb, pooling])
        st_model.save(str(init_dir))
        return str(init_dir)

    # Default path: rely on SentenceTransformer loading directly
    try:
        SentenceTransformer(model_name)
        return model_name
    except Exception:
        if init_dir.exists():
            return str(init_dir)
        init_dir.mkdir(parents=True, exist_ok=True)
        word_emb = st_models.Transformer(model_name, max_seq_length=BEST_PARAMS.get("max_seq_length", 256))
        pooling = st_models.Pooling(word_emb.get_word_embedding_dimension())
        st_model = SentenceTransformer(modules=[word_emb, pooling])
        st_model.save(str(init_dir))
        return str(init_dir)


def run_job(model_name: str, output_dir: str, train_path: str, val_path: str) -> None:
    init_dir = Path(output_dir) / "init_model"
    st_model_path = ensure_sentence_transformer(model_name, init_dir)
    params = dict(
        model_name=st_model_path,
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        **BEST_PARAMS,
    )
    config = TrainingConfig(
        test_path="",
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_target_modules=[],
        **params,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = train_model(config)
    save_report(report, output_dir)


def main() -> None:
    jobs = []
    for fam in FAMILIES:
        for ds_key, paths in DATASETS.items():
            jobs.append(
                (
                    fam["model_name"],
                    os.path.join("models/phase3", f"{fam['name']}_{ds_key}_best"),
                    paths["train_path"],
                    paths["val_path"],
                )
            )

    for model_name, out_dir, train_path, val_path in jobs:
        print(f"=== Training {model_name} -> {out_dir} ===")
        run_job(model_name, out_dir, train_path, val_path)


if __name__ == "__main__":
    main()
