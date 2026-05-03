#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune BioBERT (Sentence Transformers) with CoSENTLoss on the Phase 2 pairs.

Main features:
- Loads train/val/test Parquet splits with columns:
  sentence1, sentence2, hpo_id1, hpo_id2, gold_similarity
- Uses CoSENTLoss to regress cosine similarity towards the gold score.
- Freezes the lower Transformer layers and applies discriminative learning rates.
- Reports Pearson/Spearman correlations on validation and test splits.
- Saves the fine-tuned model and training metrics to disk.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from sentence_transformers.models import Transformer
from tqdm.auto import tqdm

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

try:
    from transformers import get_linear_schedule_with_warmup
except Exception:  # pragma: no cover - fallback for older transformers
    from transformers.optimization import get_linear_schedule_with_warmup

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    PeftLoraConfig = None
    TaskType = None
    get_peft_model = None


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str
    train_path: str
    val_path: Optional[str]
    test_path: Optional[str]
    output_dir: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    freeze_layers: int
    min_lr: float
    max_lr: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    max_seq_length: int
    seed: int
    use_amp: bool
    train_eval_sample: Optional[int]
    loss_type: str
    skip_test: bool
    skip_eval: bool
    eval_steps: int
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pairs_dataframe(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"No dataset found at {path}")
    df = pd.read_parquet(path)
    expected_cols = {"sentence1", "sentence2", "gold_similarity"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset {path} is missing required columns: {missing_cols}")
    df = df.dropna(subset=["sentence1", "sentence2", "gold_similarity"]).copy()
    # Filter empty or blank lines to avoid attention errors
    df["sentence1"] = df["sentence1"].astype(str).str.strip()
    df["sentence2"] = df["sentence2"].astype(str).str.strip()
    df = df[(df["sentence1"] != "") & (df["sentence2"] != "")]
    df["gold_similarity"] = df["gold_similarity"].astype(float)
    return df


def dataframe_to_examples(df: pd.DataFrame, label_transform=None) -> List[InputExample]:
    return [
        InputExample(
            texts=[row.sentence1, row.sentence2],
            label=float(label_transform(row.gold_similarity) if label_transform else row.gold_similarity),
        )
        for row in df.itertuples(index=False)
    ]


def sample_dataframe(
    df: pd.DataFrame,
    sample_size: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def create_dataloader(
    examples: Sequence[InputExample],
    model: SentenceTransformer,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    if not examples:
        raise ValueError("No training examples available to create DataLoader.")
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=model.smart_batching_collate,
        drop_last=False,
    )


def attach_lora_adapters(transformer_module: Transformer, config: TrainingConfig) -> None:
    if PeftLoraConfig is None or get_peft_model is None or TaskType is None:
        raise ImportError("The --use-lora option requires installing `peft` (pip install peft).")
    target_modules = config.lora_target_modules or ["query", "value"]
    peft_config = PeftLoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    base_model = transformer_module.auto_model
    transformer_module.auto_model = get_peft_model(base_model, peft_config)
    LOGGER.info(
        "LoRA adapters injected (r=%d, alpha=%d, dropout=%.2f, targets=%s)",
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
        ",".join(target_modules),
    )


def apply_layer_freezing(transformer_module: Transformer, freeze_layers: int) -> Dict[str, bool]:
    """
    Freeze embeddings and the first `freeze_layers` encoder blocks.
    Returns a dict mapping module names to their trainable status for logging/tests.
    """
    status: Dict[str, bool] = {}

    if hasattr(transformer_module.auto_model, "embeddings"):
        for param in transformer_module.auto_model.embeddings.parameters():
            param.requires_grad = False
        status["embeddings"] = False

    encoder_layers = list(transformer_module.auto_model.encoder.layer)
    for idx, layer in enumerate(encoder_layers):
        trainable = idx >= freeze_layers
        for param in layer.parameters():
            param.requires_grad = trainable
        status[f"encoder.layer.{idx}"] = trainable

    if hasattr(transformer_module.auto_model, "pooler") and transformer_module.auto_model.pooler:
        # Pooler stays trainable (will use highest LR later)
        for param in transformer_module.auto_model.pooler.parameters():
            param.requires_grad = True
        status["pooler"] = True

    return status


def _add_param_group(
    param_groups: List[Dict[str, object]],
    summary: List[Tuple[str, float, int]],
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    name: str,
    seen: set[int],
) -> None:
    params = [p for p in params if p.requires_grad]
    unique_params: List[torch.nn.Parameter] = []
    for p in params:
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            unique_params.append(p)
    if not unique_params:
        return
    param_groups.append({"params": unique_params, "lr": lr, "weight_decay": weight_decay})
    total_params = sum(p.numel() for p in unique_params)
    summary.append((name, lr, total_params))


def create_discriminative_param_groups(
    model: SentenceTransformer,
    freeze_layers: int,
    min_lr: float,
    max_lr: float,
    weight_decay: float,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, float, int]]]:
    """
    Creates parameter groups with increasing learning rates for higher layers.
    Returns both the parameter groups (for the optimizer) and a summary for logging/tests.
    """
    if min_lr <= 0 or max_lr <= 0:
        raise ValueError("Learning rates must be positive.")
    if max_lr < min_lr:
        raise ValueError("max_lr must be >= min_lr.")

    transformer_module: Transformer = model[0]
    encoder_layers = list(transformer_module.auto_model.encoder.layer)
    total_layers = len(encoder_layers)
    trainable_layers = max(0, total_layers - freeze_layers)
    if trainable_layers == 0:
        raise ValueError("No trainable layers remain. Reduce freeze_layers.")

    lr_span = max_lr - min_lr
    groups: List[Dict[str, object]] = []
    summary: List[Tuple[str, float, int]] = []
    seen: set[int] = set()

    for idx, layer in enumerate(encoder_layers):
        if idx < freeze_layers:
            continue
        if trainable_layers == 1:
            layer_lr = max_lr
        else:
            relative_pos = (idx - freeze_layers) / (trainable_layers - 1)
            layer_lr = min_lr + lr_span * relative_pos
        _add_param_group(
            groups,
            summary,
            layer.parameters(),
            lr=layer_lr,
            weight_decay=weight_decay,
            name=f"encoder.layer.{idx}",
            seen=seen,
        )

    # Include pooler (if present) with the highest LR
    if hasattr(transformer_module.auto_model, "pooler") and transformer_module.auto_model.pooler:
        _add_param_group(
            groups,
            summary,
            transformer_module.auto_model.pooler.parameters(),
            lr=max_lr,
            weight_decay=weight_decay,
            name="pooler",
            seen=seen,
        )

    # Remaining SentenceTransformer modules (e.g., Pooling or Dense heads)
    for module_id in range(1, len(model)):
        module = model[module_id]
        _add_param_group(
            groups,
            summary,
            module.parameters(),
            lr=max_lr,
            weight_decay=weight_decay,
            name=f"module.{module_id}",
            seen=seen,
        )

    if not groups:
        raise RuntimeError("No parameter groups were created; check freeze/lr settings.")

    return groups, summary


def evaluate_pairs(
    model: SentenceTransformer,
    df: pd.DataFrame,
    batch_size: int,
) -> Optional[Dict[str, float]]:
    if df is None or df.empty:
        return None

    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()
    labels = df["gold_similarity"].astype(float).to_numpy()

    embeddings1 = model.encode(
        sentences1,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    embeddings2 = model.encode(
        sentences2,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    cosine_scores = util.cos_sim(embeddings1, embeddings2).diag().cpu().numpy()

    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore

        pearson = float(pearsonr(labels, cosine_scores)[0])
        spearman = float(spearmanr(labels, cosine_scores)[0])
    except Exception:
        pearson = float(np.corrcoef(labels, cosine_scores)[0, 1])
        spearman = float(pd.Series(labels).corr(pd.Series(cosine_scores), method="spearman"))

    mse = float(np.mean((labels - cosine_scores) ** 2))
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mse": mse,
    }


def generate_training_plots(history: List[Dict[str, float]], output_dir: str) -> Dict[str, str]:
    if not history:
        return {}
    os.makedirs(output_dir, exist_ok=True)

    def _x_value(entry: Dict[str, float]) -> Optional[float]:
        if entry.get("global_step") is not None:
            try:
                return float(entry["global_step"])
            except Exception:
                return None
        if entry.get("epoch") is not None:
            try:
                return float(entry["epoch"])
            except Exception:
                return None
        return None

    xs_all = [_x_value(entry) for entry in history]
    loss_values = [entry.get("train_loss") for entry in history]

    plot_paths: Dict[str, str] = {}

    if any(value is not None for value in loss_values):
        plt.figure(figsize=(6, 4))
        plt.plot(xs_all, loss_values, marker="o", label="Train loss")
        plt.xlabel("Step" if any(entry.get("global_step") is not None for entry in history) else "Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        loss_path = os.path.join(output_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        plot_paths["loss_curve"] = loss_path

    def _extract_series(key: str) -> Tuple[List[float], List[float]]:
        xs, ys = [], []
        for entry in history:
            value = entry.get(key)
            xval = _x_value(entry)
            if value is not None and xval is not None:
                xs.append(xval)
                ys.append(value)
        return xs, ys

    series = {
        "train_pearson": "Train Pearson",
        "train_spearman": "Train Spearman",
        "val_pearson": "Val Pearson",
        "val_spearman": "Val Spearman",
    }

    any_series = False
    plt.figure(figsize=(6, 4))
    for key, label in series.items():
        xs, ys = _extract_series(key)
        if ys:
            plt.plot(xs, ys, marker="o", label=label)
            any_series = True
    if any_series:
        plt.xlabel("Step" if any(entry.get("global_step") is not None for entry in history) else "Epoch")
        plt.ylabel("Correlation")
        plt.title("Pearson / Spearman Evolution")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        corr_path = os.path.join(output_dir, "correlation_curve.png")
        plt.tight_layout()
        plt.savefig(corr_path)
        plot_paths["correlation_curve"] = corr_path
    plt.close()

    return plot_paths


def train_model(config: TrainingConfig) -> Dict[str, object]:
    set_seed(config.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    LOGGER.info("Loading datasets...")
    train_df = load_pairs_dataframe(config.train_path)
    val_df = load_pairs_dataframe(config.val_path) if config.val_path else None
    test_df = load_pairs_dataframe(config.test_path) if config.test_path else None
    if config.skip_test:
        test_df = None

    if config.skip_eval:
        val_df = None
        test_df = None
        train_eval_df = pd.DataFrame()
        LOGGER.info("Evaluation disabled (train/val/test).")
    else:
        train_eval_df = sample_dataframe(train_df, config.train_eval_sample, config.seed)
        LOGGER.info("Train evaluation sample size: %d", len(train_eval_df))

    LOGGER.info(
        "Train pairs: %d | Val pairs: %d | Test pairs: %d",
        len(train_df),
        0 if val_df is None else len(val_df),
        0 if test_df is None else len(test_df),
    )

    LOGGER.info("Initializing model: %s", config.model_name)
    model = SentenceTransformer(config.model_name)
    # Limit sequence length to avoid GPU asserts with SDPA
    model.max_seq_length = getattr(model, "max_seq_length", 256) or 256
    model.max_seq_length = min(model.max_seq_length, config.max_seq_length)
    transformer_module: Transformer = model[0]

    lr_summary: List[Tuple[str, float, int]] = []
    if config.use_lora:
        attach_lora_adapters(transformer_module, config)
        freeze_status: Dict[str, bool] = {"lora_adapters": True}
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters() if p.requires_grad],
                    "lr": config.max_lr,
                    "weight_decay": config.weight_decay,
                }
            ],
            betas=(0.9, 0.999),
            eps=1e-6,
        )
    else:
        freeze_status = apply_layer_freezing(transformer_module, config.freeze_layers)
        param_groups, lr_summary = create_discriminative_param_groups(
            model,
            freeze_layers=config.freeze_layers,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            weight_decay=config.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-6)

    label_transform = None
    if config.loss_type.lower() == "cosine":
        label_transform = lambda x: 2 * float(x) - 1.0  # map [0,1] -> [-1,1]

    train_examples = dataframe_to_examples(train_df, label_transform=label_transform)
    train_loader = create_dataloader(train_examples, model, batch_size=config.batch_size, shuffle=True)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = config.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.use_lora:
        lr_summary = [
            ("lora_adapters", config.max_lr, sum(p.numel() for p in trainable_params)),
        ]

    if config.loss_type.lower() == "cosent":
        loss_fct = losses.CoSENTLoss(model)
    elif config.loss_type.lower() == "cosine":
        loss_fct = losses.CosineSimilarityLoss(model)
    elif config.loss_type.lower() == "angle":
        loss_fct = losses.AnglELoss(model)
    else:
        raise ValueError(f"Unknown loss_type {config.loss_type}")
    model.to(model.device)

    history: List[Dict[str, float]] = []
    best_val_spearman = float("-inf")
    best_state: Optional[str] = None
    global_step = 0

    def run_evaluation(epoch: int, current_step: int, loss_avg: float) -> None:
        nonlocal best_val_spearman, best_state, history
        metrics: Dict[str, float] = {"epoch": float(epoch), "global_step": current_step, "train_loss": loss_avg}
        if not config.skip_eval:
            model.eval()
            train_eval_metrics = None
            if not train_eval_df.empty:
                with torch.no_grad():
                    train_eval_metrics = evaluate_pairs(model, train_eval_df, config.eval_batch_size)
                if train_eval_metrics:
                    metrics.update({f"train_{k}": v for k, v in train_eval_metrics.items()})
            val_metrics = None
            if val_df is not None:
                with torch.no_grad():
                    try:
                        val_metrics = evaluate_pairs(model, val_df, config.eval_batch_size)
                    except Exception as exc:
                        LOGGER.error("Validation evaluation failed: %s", exc)
                        val_metrics = None
                if val_metrics:
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                    if val_metrics["spearman"] > best_val_spearman:
                        best_val_spearman = val_metrics["spearman"]
                        best_state = os.path.join(config.output_dir, "best_model")
                        os.makedirs(config.output_dir, exist_ok=True)
                        model.save(best_state)
                        LOGGER.info("Saved new best model with Spearman=%.4f", best_val_spearman)
            if train_eval_metrics:
                LOGGER.info(
                    "Eval @step %d | Train sample Spearman %.4f Pearson %.4f MSE %.5f",
                    current_step,
                    train_eval_metrics["spearman"],
                    train_eval_metrics["pearson"],
                    train_eval_metrics["mse"],
                )
            if val_metrics:
                LOGGER.info(
                    "Eval @step %d | Val Spearman %.4f Pearson %.4f MSE %.5f",
                    current_step,
                    val_metrics["spearman"],
                    val_metrics["pearson"],
                    val_metrics["mse"],
                )
            model.train()
        history.append(metrics)

    LOGGER.info("Starting training for %d epochs (%d steps total)...", config.epochs, total_steps)
    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        running_loss = 0.0

        for step, batch in enumerate(progress, start=1):
            features, labels = batch
            labels = labels.to(model.device)
            for feature in features:
                for key in feature:
                    feature[key] = feature[key].to(model.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss_value = loss_fct(features, labels)

            scaler.scale(loss_value).backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss_value.item()
            global_step += 1
            progress.set_postfix({"loss": f"{loss_value.item():.4f}"})

            if not config.skip_eval and config.eval_steps > 0 and (global_step % config.eval_steps == 0):
                avg_loss_so_far = running_loss / float(step)
                run_evaluation(epoch, global_step, avg_loss_so_far)

        avg_loss = running_loss / len(train_loader)
        run_evaluation(epoch, global_step, avg_loss)

    final_model_path = best_state or config.output_dir
    os.makedirs(final_model_path, exist_ok=True)
    if best_state is None:
        model.save(final_model_path)
        LOGGER.info("Saved final model to %s", final_model_path)
    else:
        LOGGER.info("Best model already saved at %s", final_model_path)

    test_metrics = None
    if test_df is not None and not config.skip_test:
        best_model = SentenceTransformer(final_model_path)
        best_model.to(best_model.device)
        best_model.max_seq_length = min(getattr(best_model, "max_seq_length", 256) or 256, config.max_seq_length)
        best_model.eval()
        with torch.no_grad():
            try:
                test_metrics = evaluate_pairs(best_model, test_df, config.eval_batch_size)
            except Exception as exc:
                LOGGER.error("Test evaluation failed: %s", exc)
                test_metrics = None
        if test_metrics:
            LOGGER.info(
                "Test | Spearman %.4f | Pearson %.4f | MSE %.5f",
                test_metrics["spearman"],
                test_metrics["pearson"],
                test_metrics["mse"],
            )

    training_report: Dict[str, object] = {
        "history": history,
        "freeze_status": freeze_status,
        "lr_summary": [{"name": name, "lr": lr, "params": params} for name, lr, params in lr_summary],
        "best_val_spearman": best_val_spearman if val_df is not None else None,
        "final_model_path": final_model_path,
        "test_metrics": test_metrics,
        "train_eval_sample": len(train_eval_df),
    }
    plot_paths = generate_training_plots(history, config.output_dir)
    if plot_paths:
        training_report["plots"] = plot_paths
    return training_report


def save_report(report: Dict[str, object], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_report.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOGGER.info("Training report stored at %s", metrics_path)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune BioBERT with CoSENTLoss on HPO sentence pairs.")
    parser.add_argument("--model-name", default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument("--train-path", default="data/phase2/splits/train.parquet")
    parser.add_argument("--val-path", default="data/phase2/splits/val.parquet")
    parser.add_argument("--test-path", default="data/phase2/splits/test.parquet")
    parser.add_argument("--output-dir", default="models/phase3/biobert_cosent")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--freeze-layers", type=int, default=6)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum token length for the encoder.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision training if CUDA is available.")
    parser.add_argument("--skip-test", action="store_true", help="Skip evaluation on test set (useful for HPO).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation on train/val/test.")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Eval every N optimizer steps (0 = only at the end of each epoch).",
    )
    parser.add_argument(
        "--loss-type",
        choices=["cosent", "cosine", "angle"],
        default="angle",
        help="Loss function: CoSENT (default), CosineSimilarityLoss (requires labels in [-1,1]), or AnglELoss.",
    )
    parser.add_argument(
        "--train-eval-sample",
        type=int,
        default=2000,
        help="Number of training pairs to use when computing train correlations (0 = full dataset).",
    )
    parser.add_argument("--use-lora", action="store_true", help="Train only LoRA adapters.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["query", "value"],
        help="Target modules for LoRA (default: query and value).",
    )
    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        freeze_layers=args.freeze_layers,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        use_amp=args.use_amp,
        loss_type=args.loss_type,
        skip_test=args.skip_test,
        skip_eval=args.skip_eval,
        eval_steps=args.eval_steps,
        train_eval_sample=args.train_eval_sample,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )


def main() -> None:
    config = parse_args()
    report = train_model(config)
    save_report(report, config.output_dir)


if __name__ == "__main__":
    main()
