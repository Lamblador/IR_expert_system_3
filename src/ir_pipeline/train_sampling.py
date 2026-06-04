"""Веса классов и WeightedRandomSampler для IrResnet (как в оригинальном ноутбуке 72 cls)."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except ImportError:
    torch = None  # type: ignore
    WeightedRandomSampler = None  # type: ignore


def compute_class_pos_weights(
    Y_train: np.ndarray,
    *,
    mode: str = "neg_pos",
    pos_weight_scale: float = 1.0,
    max_weight: float = 50.0,
) -> Any:
    """Тензор pos_weight для BCEWithLogitsLoss."""
    if torch is None:
        raise RuntimeError("torch required")
    pos = Y_train.sum(axis=0).astype(np.float64)
    n = len(Y_train)
    if mode == "inverse_freq":
        total = pos.sum()
        num_classes = Y_train.shape[1]
        w = np.zeros(Y_train.shape[1], dtype=np.float64)
        for j in range(num_classes):
            if pos[j] > 0:
                w[j] = total / (num_classes * pos[j])
            else:
                w[j] = 0.0
        w = np.nan_to_num(w, nan=0.0)
    else:
        neg = n - pos
        w = neg / np.maximum(pos, 1.0)
        w = np.clip(w, 1.0, max_weight)
    w = w * float(pos_weight_scale)
    return torch.tensor(w, dtype=torch.float32)


def build_sample_weights(Y_train: np.ndarray, class_weights: Any) -> np.ndarray:
    """
    Вес образца: сумма (label * class_weight) по классам; пустая метка → 0.1.
    Как weighted_labels в Koshelev notebook 6.
    """
    cw = class_weights.detach().cpu().numpy() if torch is not None and hasattr(class_weights, "detach") else np.asarray(class_weights)
    weights = []
    for i in range(len(Y_train)):
        row = Y_train[i] * cw
        s = float(row.sum())
        if s <= 0 or np.isnan(s):
            s = 0.1
        weights.append(max(s, 1e-6))
    return np.array(weights, dtype=np.float64)


def build_weighted_sampler(
    sample_weights: np.ndarray,
    num_samples: int | None = None,
) -> Any:
    if WeightedRandomSampler is None:
        raise RuntimeError("torch required")
    w = torch.tensor(sample_weights, dtype=torch.double)
    n = num_samples if num_samples is not None else len(sample_weights)
    return WeightedRandomSampler(w, num_samples=n, replacement=True)
