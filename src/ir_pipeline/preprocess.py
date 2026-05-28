from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass
class GridSpectrum:
    wavenumbers: np.ndarray  # shape (G,)
    absorbance: np.ndarray  # shape (G,)
    absorbance_normalized: np.ndarray  # shape (G,)
    coverage_mask: np.ndarray  # bool (G,)
    absorbance_raw: np.ndarray  # shape (G,) same grid before baseline-ish normalization chain


def wavenumbers_from_jcamp(x: np.ndarray, xunits: str | None) -> np.ndarray:
    """
    Приводит ось X к см⁻¹.
    MICROMETERS -> см⁻¹: ν = 10000 / λ(µm).
    """
    if xunits is None:
        return np.asarray(x, dtype=float)
    u = xunits.upper().replace(" ", "")
    if "MICROM" in u or "UM" == u:
        lam = np.asarray(x, dtype=float)
        lam = np.clip(lam, 1e-6, None)
        return 10000.0 / lam
    # уже см⁻¹
    return np.asarray(x, dtype=float)


def transmittance_to_absorbance(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if np.nanmax(y) > 1.5:
        y = y / 100.0
    y = np.clip(y, 1e-6, 1.0 - 1e-6)
    return -np.log10(y)


def to_absorbance_like(y: np.ndarray, yunits: str | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if yunits is None:
        return y
    u = yunits.upper()
    if "TRANSMITTANCE" in u or "TRANSMISSION" in u or "PERCENT" in u:
        return transmittance_to_absorbance(y)
    if "ABSORBANCE" in u or "ABSORPTION" in u:
        return y
    if "REFLECTANCE" in u:
        # грубое приближение к pseudo-absorbance
        y2 = np.clip(y, 1e-6, None)
        return -np.log10(y2 / np.nanmax(y2))
    return y


def infer_spectrum_y_scale(y: np.ndarray) -> str:
    """
    Эвристика: пропускание держится около 1 (или 100) с провалами,
    поглощение — около 0 с положительными пиками.
    """
    v = np.asarray(y, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return "unknown"
    med = float(np.median(v))
    p90 = float(np.percentile(v, 90))
    p10 = float(np.percentile(v, 10))

    if med > 0.55 and p90 > 0.8:
        return "transmittance"
    if med < 0.5 and p90 <= 3.0:
        return "absorbance"
    if p90 > 1.2:
        return "absorbance"
    return "unknown"


def ensure_absorbance(
    y: np.ndarray,
    yunits: str | None = None,
    *,
    assumed_scale: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Приводит Y к шкале поглощения (-log10 T).
    Не использует (1 - T): это ломает уже переведённые спектры.
    """
    v = np.asarray(y, dtype=np.float64)
    meta: dict[str, object] = {
        "input_scale": None,
        "converted": False,
        "method": None,
    }

    if yunits:
        meta["input_scale"] = str(yunits)
        meta["method"] = "yunits"
        meta["converted"] = "TRANSMITTANCE" in str(yunits).upper() or "TRANSMISSION" in str(yunits).upper()
        return to_absorbance_like(v, yunits), meta

    if assumed_scale == "absorbance":
        meta["input_scale"] = "absorbance"
        meta["method"] = "assumed_absorbance"
        return v, meta
    if assumed_scale == "transmittance":
        meta["input_scale"] = "transmittance"
        meta["method"] = "transmittance_to_absorbance"
        meta["converted"] = True
        return transmittance_to_absorbance(v), meta

    detected = infer_spectrum_y_scale(v)
    meta["input_scale"] = detected
    if detected == "transmittance":
        meta["method"] = "transmittance_to_absorbance"
        meta["converted"] = True
        return transmittance_to_absorbance(v), meta
    if detected == "absorbance":
        meta["method"] = "heuristic_absorbance"
        return v, meta

    if med := float(np.median(v[np.isfinite(v)])) > 0.55:
        meta["input_scale"] = "transmittance_fallback"
        meta["method"] = "fallback_transmittance_to_absorbance"
        meta["converted"] = True
        return transmittance_to_absorbance(v), meta

    meta["method"] = "passthrough"
    return v, meta


def validate_absorbance_spectrum(y: np.ndarray) -> dict[str, object]:
    """QC: похоже ли на поглощение, а не на пропускание около 1."""
    v = np.asarray(y, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return {"ok": False, "issues": ["too_few_points"], "median": None, "p90": None}

    med = float(np.median(v))
    p90 = float(np.percentile(v, 90))
    p10 = float(np.percentile(v, 10))
    issues: list[str] = []

    if med > 0.7 and p90 > 0.9:
        issues.append("looks_like_transmittance")
    if p90 - p10 < 1e-4:
        issues.append("flat_spectrum")
    if np.nanmin(v) < -0.05:
        issues.append("negative_values")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "median": med,
        "p90": p90,
        "p10": p10,
        "inferred_scale": infer_spectrum_y_scale(v),
    }


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens)."""
    L = len(y)
    y = np.asarray(y, dtype=np.float64)
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), dtype=np.float64)
    w = np.ones(L, dtype=np.float64)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L), dtype=np.float64)
        Z = W + lam * (D @ D.T)
        Z = csr_matrix(Z)
        z = spsolve(Z, (w * y).astype(np.float64))
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z, dtype=float)


def preprocess_to_grid(
    x_cm: np.ndarray,
    y_raw: np.ndarray,
    grid_min: float = 400.0,
    grid_max: float = 4000.0,
    grid_step: float = 2.0,
    sg_window: int = 11,
    sg_poly: int = 3,
) -> GridSpectrum:
    """Интерполяция на равномерную сетку см⁻¹, поглощение-подобная шкала, baseline, нормализация."""
    x_cm = np.asarray(x_cm, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    order = np.argsort(x_cm)
    x_cm = x_cm[order]
    y_raw = y_raw[order]
    # уникальные x
    ux, idx = np.unique(x_cm, return_index=True)
    y_u = y_raw[idx]

    grid = np.arange(grid_min, grid_max + 1e-9, grid_step, dtype=float)

    mask_obs = (ux >= grid_min) & (ux <= grid_max)
    coverage = (grid >= ux[mask_obs].min()) & (grid <= ux[mask_obs].max()) if mask_obs.any() else np.zeros_like(grid, dtype=bool)

    f = interp1d(ux, y_u, kind="linear", bounds_error=False, fill_value=np.nan)
    y_g = f(grid)
    valid = np.isfinite(y_g)
    if valid.sum() < 8:
        return GridSpectrum(grid, y_g, y_g, coverage & valid, y_g)

    y_fill = np.asarray(y_g, dtype=float).copy()
    if valid.any():
        idxs = np.where(valid)[0]
        first, last = idxs[0], idxs[-1]
        med = np.nanmedian(y_fill[first : last + 1])
        y_fill = np.nan_to_num(y_fill, nan=med)
        y_fill[:first] = y_fill[first]
        y_fill[last + 1 :] = y_fill[last]
    else:
        y_fill = np.nan_to_num(y_fill, nan=0.0)

    b_full = baseline_als(y_fill)
    abs_corr = y_fill - b_full

    if sg_window >= 5 and sg_window % 2 == 1 and len(abs_corr) >= sg_window:
        abs_smooth = savgol_filter(abs_corr, sg_window, sg_poly)
    else:
        abs_smooth = abs_corr

    peak = np.nanmax(abs_smooth[coverage]) if coverage.any() else np.nanmax(abs_smooth)
    norm = abs_smooth / peak if peak and np.isfinite(peak) and peak != 0 else abs_smooth

    return GridSpectrum(
        wavenumbers=grid,
        absorbance=np.asarray(abs_smooth, dtype=np.float32),
        absorbance_normalized=np.asarray(norm, dtype=np.float32),
        coverage_mask=np.asarray(coverage, dtype=bool),
        absorbance_raw=np.asarray(y_fill, dtype=np.float32),
    )


def intensity_bucket_from_peak(abs_norm: np.ndarray, grid: np.ndarray, peak_cm: float) -> str:
    """Грубые классы vw/w/m/s/vs по доле от глобального максимума на нормализованном спектре."""
    if not np.isfinite(peak_cm):
        return "u"
    idx = int(np.argmin(np.abs(grid - peak_cm)))
    h = float(abs_norm[idx])
    # abs_norm max ~1 на главном пике; локальные пики часто 0.05-0.8
    if h < 0.03:
        return "vw"
    if h < 0.08:
        return "w"
    if h < 0.18:
        return "m"
    if h < 0.35:
        return "s"
    return "vs"
