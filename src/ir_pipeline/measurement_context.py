"""One-hot контекст условий измерения (техника + фаза образца) для IrResnet4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Канонические значения (порядок фиксируется при первой сборке датасета)
MEASUREMENT_MODE_CATEGORIES = ("atr", "transmission", "gas", "solution", "absorbance", "unknown")
SAMPLE_STATE_CATEGORIES = ("solid", "liquid", "gas", "solution", "film", "unknown")


def normalize_measurement_mode(raw: str) -> str:
    s = str(raw or "unknown").strip().lower()
    if s in MEASUREMENT_MODE_CATEGORIES:
        return s
    return "unknown"


def normalize_sample_state(raw: str) -> str:
    s = str(raw or "unknown").strip().lower()
    if "gas" in s:
        return "gas"
    if "liquid" in s or "solution" in s:
        return "solution" if "solution" in s else "liquid"
    if "solid" in s or "powder" in s or "kbr" in s:
        return "solid"
    if "film" in s:
        return "film"
    if s in SAMPLE_STATE_CATEGORIES:
        return s
    return "unknown"


def context_column_names() -> list[str]:
    cols = [f"mm_{c}" for c in MEASUREMENT_MODE_CATEGORIES]
    cols += [f"st_{c}" for c in SAMPLE_STATE_CATEGORIES]
    return cols


def build_context_matrix(
    meta: pd.DataFrame,
    spectrum_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Матрица (N, C) one-hot: measurement_mode + sample_state.
    Строки в порядке spectrum_ids; отсутствующие id → все нули + mm_unknown/st_unknown.
    """
    cols = context_column_names()
    col_index = {c: i for i, c in enumerate(cols)}
    out = np.zeros((len(spectrum_ids), len(cols)), dtype=np.float32)
    meta_i = meta.set_index("spectrum_id") if "spectrum_id" in meta.columns else meta

    for row_i, sid in enumerate(spectrum_ids):
        if sid not in meta_i.index:
            out[row_i, col_index["mm_unknown"]] = 1.0
            out[row_i, col_index["st_unknown"]] = 1.0
            continue
        r = meta_i.loc[sid]
        mm = normalize_measurement_mode(r.get("measurement_mode", "unknown"))
        st = normalize_sample_state(r.get("sample_state", "unknown"))
        out[row_i, col_index[f"mm_{mm}"]] = 1.0
        out[row_i, col_index[f"st_{st}"]] = 1.0

    return out, cols


def load_context_for_spectrum_ids(
    dataset_dir: Path,
    spectrum_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    meta = pd.read_parquet(dataset_dir / "meta.parquet")
    return build_context_matrix(meta, spectrum_ids)


def context_dict_from_row(meta_row: pd.Series) -> dict[str, float]:
    """Для инференса по одному JCAMP: словарь {mm_atr: 1, ...}."""
    cols = context_column_names()
    vec = {c: 0.0 for c in cols}
    mm = normalize_measurement_mode(meta_row.get("measurement_mode", "unknown"))
    st = normalize_sample_state(meta_row.get("sample_state", "unknown"))
    vec[f"mm_{mm}"] = 1.0
    vec[f"st_{st}"] = 1.0
    return vec


def vector_from_dict(ctx: dict[str, float], columns: list[str]) -> np.ndarray:
    return np.array([float(ctx.get(c, 0.0)) for c in columns], dtype=np.float32)


def unknown_context_vector(columns: list[str] | None = None) -> np.ndarray:
    """Вектор «контекст неизвестен» для аугментации при обучении."""
    cols = columns or context_column_names()
    v = np.zeros(len(cols), dtype=np.float32)
    idx = {c: i for i, c in enumerate(cols)}
    if "mm_unknown" in idx:
        v[idx["mm_unknown"]] = 1.0
    if "st_unknown" in idx:
        v[idx["st_unknown"]] = 1.0
    return v
