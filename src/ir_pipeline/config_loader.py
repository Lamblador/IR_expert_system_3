from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
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
    }
    out = {**defaults, **train_cfg}
    return out


def resolve_paths(paths_cfg: dict[str, Any]) -> dict[str, Path]:
    raw = paths_cfg.get("raw_jcamp_dir", "downloaded_jcamp")
    proc = paths_cfg.get("processed_root", "data/processed")
    bands = paths_cfg.get("bands_config", "configs/bands_reference.yaml")

    raw_path = Path(os.environ.get("IR_RAW_JCAMP_DIR", raw))
    proc_root = Path(os.environ.get("IR_PROCESSED_ROOT", proc))
    bands_path = Path(bands)

    if not bands_path.is_absolute():
        bands_path = Path.cwd() / bands_path

    return {
        "raw_jcamp_dir": raw_path,
        "processed_root": proc_root,
        "bands_config": bands_path,
        "dataset_version": paths_cfg.get("dataset_version", "dataset_v001"),
    }
