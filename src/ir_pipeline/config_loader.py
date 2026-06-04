from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ir_pipeline.logging_utils import log

DATASET_MINI = "dataset_mini"
DATASET_V003 = "dataset_v003"
DATASET_V002_FALLBACK = "dataset_v002"


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_train_defaults(train_cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "max_files": 0,
        "random_seed": 42,
        "train_frac": 0.85,
        "model": "sklearn_rf",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "torch_epochs": 30,
        "torch_batch_size": 32,
        "torch_lr": 1e-3,
        "torch_optimizer": "adamw",
        "torch_weight_decay": 1e-4,
        "torch_momentum": 0.9,
        "torch_loss": "bce_with_logits",
        "torch_scheduler": None,
        "torch_scheduler_step_size": 75,
        "torch_scheduler_gamma": 0.2,
        "live_training_plot": False,
        "train_log_tail": 5,
        "label_schema": "structure_smarts",
        "ir_hidden_size": 34,
        "pos_weight_scale": 1.0,
        "use_measurement_context": True,
        "context_dropout_prob": 0.0,
        "use_weighted_sampler": False,
        "early_stop_metric": "val_f1_weighted",
        "early_stop_patience": 0,
        "early_stop_min_epochs": 1,
        "augment_train": False,
        "prediction_threshold": 0.5,
        "rf_backend": "auto",
    }
    out = {**defaults, **train_cfg}
    return out


def resolve_dataset_version_name(
    processed_root: Path,
    profile: str,
) -> str:
    """
    mini → dataset_mini; full → dataset_v003 (fallback v002); auto → v003 если есть, иначе mini.
    """
    p = str(profile or "auto").strip().lower()
    env = os.environ.get("IR_DATASET_PROFILE", "").strip().lower()
    if env:
        p = env

    if p == "mini":
        return DATASET_MINI
    if p == "full":
        if (processed_root / DATASET_V003 / "spectra.npz").is_file():
            return DATASET_V003
        if (processed_root / DATASET_V002_FALLBACK / "spectra.npz").is_file():
            log(f"dataset_v003 не найден, fallback → {DATASET_V002_FALLBACK}")
            return DATASET_V002_FALLBACK
        return DATASET_V003
    # auto
    if (processed_root / DATASET_V003 / "spectra.npz").is_file():
        return DATASET_V003
    if (processed_root / DATASET_MINI / "spectra.npz").is_file():
        return DATASET_MINI
    return DATASET_MINI


def resolve_paths(paths_cfg: dict[str, Any]) -> dict[str, Path]:
    raw = paths_cfg.get("raw_jcamp_dir", "downloaded_jcamp")
    proc = paths_cfg.get("processed_root", "data/processed")
    bands = paths_cfg.get("bands_config", "configs/bands_reference.yaml")
    profile = paths_cfg.get("dataset_profile", "auto")

    raw_path = Path(os.environ.get("IR_RAW_JCAMP_DIR", raw))
    proc_root = Path(os.environ.get("IR_PROCESSED_ROOT", proc))
    bands_path = Path(bands)

    if not bands_path.is_absolute():
        bands_path = Path.cwd() / bands_path

    dv = paths_cfg.get("dataset_version")
    if dv:
        dataset_version = str(dv)
    else:
        dataset_version = resolve_dataset_version_name(proc_root, str(profile))

    return {
        "raw_jcamp_dir": raw_path,
        "processed_root": proc_root,
        "bands_config": bands_path,
        "dataset_version": dataset_version,
        "dataset_profile": str(profile),
    }


def resolve_dataset_dir(paths_cfg: dict[str, Any], dataset_profile: str | None = None) -> Path:
    """Каталог датасета с учётом profile override."""
    cfg = dict(paths_cfg)
    if dataset_profile:
        cfg["dataset_profile"] = dataset_profile
        cfg.pop("dataset_version", None)
    p = resolve_paths(cfg)
    return p["processed_root"] / str(p["dataset_version"])
