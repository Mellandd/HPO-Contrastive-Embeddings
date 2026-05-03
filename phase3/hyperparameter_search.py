#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna-based hyperparameter search for SentenceTransformer fine-tuning.

Reads a YAML config describing the base training parameters and the search
space for individual hyperparameters (including LoRA options), then launches
multiple trials sequentially. Each trial writes its artifacts to a dedicated
subdirectory under the configured output path.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml
import sys

# Allow relative imports to the repository without needing to install as a package
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase3.train_sentence_transformer import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for HPO SentenceTransformer models.")
    parser.add_argument("--config", default="phase3/hpo_search_space.yaml")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study-name", default="hpo_study")
    parser.add_argument(
        "--storage",
        default="sqlite:///phase3/optuna.db",
        help="Optuna storage URI (default: sqlite:///phase3/optuna.db to use with optuna-dashboard).",
    )
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--resume", action="store_true", help="Resume existing study if available.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def sample_param(trial: optuna.trial.Trial, name: str, spec: Any):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, dict):
        if "choices" in spec:
            return trial.suggest_categorical(name, spec["choices"])
        if "low" in spec and "high" in spec:
            log = bool(spec.get("log", False))
            return trial.suggest_float(name, spec["low"], spec["high"], log=log)
    raise ValueError(f"Unsupported spec for {name}: {spec}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    base_params: Dict[str, Any] = cfg["base_params"]
    search_space: Dict[str, Any] = cfg.get("search_space", {})
    objective_metric: str = cfg.get("objective_metric", "best_val_spearman")

    base_output_dir = Path(base_params.get("output_dir", "models/phase3/hpo_runs"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        params = copy.deepcopy(base_params)

        # Base model
        if "model_name" in search_space:
            params["model_name"] = sample_param(trial, "model_name", search_space["model_name"])

        # Conmutador LoRA
        use_lora = params.get("use_lora", False)
        if "use_lora" in search_space:
            use_lora = bool(sample_param(trial, "use_lora", search_space["use_lora"]))
            params["use_lora"] = use_lora

        for key, spec in search_space.items():
            if key in {"model_name", "use_lora"}:
                continue
            if key.startswith("lora_") and not use_lora:
                # don't sample LoRA parameters if disabled
                continue
            params[key] = sample_param(trial, key, spec)

        if use_lora:
            params.setdefault("lora_target_modules", base_params.get("lora_target_modules", ["query", "value"]))
        else:
            params["use_lora"] = False
            params["lora_r"] = 0
            params["lora_alpha"] = 0
            params["lora_dropout"] = 0.0
            params["lora_target_modules"] = []
            # Ensure freeze_layers exists for classic mode
            params.setdefault("freeze_layers", base_params.get("freeze_layers", 6))

        # Compatibility with fields added in TrainingConfig
        params.setdefault("skip_eval", base_params.get("skip_eval", False))
        params.setdefault("eval_steps", base_params.get("eval_steps", 0))

        trial_dir = base_output_dir / f"trial_{trial.number:04d}"
        params["output_dir"] = str(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        training_config = TrainingConfig(**params)
        try:
            report = train_model(training_config)
        except Exception as exc:
            trial.set_user_attr("failed_reason", str(exc))
            raise

        metric_value = report.get(objective_metric)
        if metric_value is None:
            raise RuntimeError(f"Target metric '{objective_metric}' not found in report.")
        trial.set_user_attr("final_model_path", report.get("final_model_path"))
        trial.set_user_attr("report_path", os.path.join(params["output_dir"], "training_report.json"))
        with open(Path(params["output_dir"]) / "training_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return float(metric_value)

    if args.storage:
        if args.storage.startswith("sqlite:///"):
            db_path = Path(args.storage.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=args.study_name,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=args.resume,
        )
    else:
        study = optuna.create_study(study_name=args.study_name, direction=args.direction)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    logging.info("Best trial: %s", study.best_trial.number)
    logging.info("Best value (%s): %.5f", objective_metric, study.best_value)
    logging.info("Best params: %s", study.best_trial.params)

    summary_path = base_output_dir / f"{args.study_name}_summary.json"
    payload = {
        "study_name": args.study_name,
        "objective_metric": objective_metric,
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
