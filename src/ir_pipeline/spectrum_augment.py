"""Аугментация 3-канального входа IrResnet (поглощение + маска пиков)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpectrumAugmentConfig:
    noise_std: float = 0.02
    scale_min: float = 0.9
    scale_max: float = 1.1
    baseline_slope_max: float = 0.05
    shift_max_points: int = 8
    peak_scale_min: float = 0.7
    p_noise: float = 0.8
    p_scale: float = 0.5
    p_baseline: float = 0.4
    p_shift: float = 0.3
    p_peak_attenuate: float = 0.2


def augment_config_from_train_cfg(train_cfg: dict[str, Any]) -> SpectrumAugmentConfig:
    aug = train_cfg.get("augment") or {}
    if isinstance(aug, dict):
        return SpectrumAugmentConfig(**{k: v for k, v in aug.items() if k in SpectrumAugmentConfig.__dataclass_fields__})
    return SpectrumAugmentConfig()


class SpectrumAugmentor:
    """Применяется к X (3, L): канал 0 — wavenumber (не трогаем), 1 — absorbance, 2 — peaks."""

    def __init__(self, cfg: SpectrumAugmentConfig | None = None):
        self.cfg = cfg or SpectrumAugmentConfig()

    def apply(self, x_3ch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = x_3ch.astype(np.float32, copy=True)
        ab = out[1]
        pk = out[2]
        c = self.cfg

        if rng.random() < c.p_noise:
            ab = ab + rng.normal(0, c.noise_std, size=ab.shape).astype(np.float32)
        if rng.random() < c.p_scale:
            ab = ab * float(rng.uniform(c.scale_min, c.scale_max))
        if rng.random() < c.p_baseline:
            n = len(ab)
            slope = float(rng.uniform(-c.baseline_slope_max, c.baseline_slope_max))
            ab = ab + (slope * np.linspace(-1, 1, n, dtype=np.float32))
        if rng.random() < c.p_shift and c.shift_max_points > 0:
            shift = int(rng.integers(-c.shift_max_points, c.shift_max_points + 1))
            ab = np.roll(ab, shift)
            pk = np.roll(pk, shift)
        if rng.random() < c.p_peak_attenuate:
            pk = pk * float(rng.uniform(c.peak_scale_min, 1.0))

        out[1] = np.clip(ab, 0.0, None)
        out[2] = np.clip(pk, 0.0, 1.0)
        return out


def augment_model_inputs_offline(
    X_input: np.ndarray,
    spectrum_ids: list[str],
    train_ids: set[str],
    *,
    n_per_spectrum: int = 1,
    seed: int = 42,
    train_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Доп. строки только для train_ids."""
    cfg = augment_config_from_train_cfg(train_cfg or {})
    aug = SpectrumAugmentor(cfg)
    rng = np.random.default_rng(seed)
    extra_x: list[np.ndarray] = []
    extra_ids: list[str] = []
    for i, sid in enumerate(spectrum_ids):
        if sid not in train_ids:
            continue
        for k in range(n_per_spectrum):
            sub = rng.spawn(1)[0]
            extra_x.append(aug.apply(X_input[i], sub))
            extra_ids.append(f"{sid}_aug{k}")
    if not extra_x:
        return X_input, spectrum_ids
    X_out = np.vstack([X_input, np.stack(extra_x, axis=0)]).astype(np.float32)
    ids_out = spectrum_ids + extra_ids
    return X_out, ids_out
