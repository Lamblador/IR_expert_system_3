"""Препроцессинг спектра в формате FTIR Telegram-бота: 500–4100 см⁻¹, 3 канала."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from ir_pipeline.preprocess import ensure_absorbance, validate_absorbance_spectrum

try:
    import peakutils

    _HAS_PEAKUTILS = True
except ImportError:
    _HAS_PEAKUTILS = False

BOT_WAVENUMBERS = np.arange(500, 4101, 2, dtype=np.float32)
BOT_GRID_LEN = len(BOT_WAVENUMBERS)


@dataclass
class TelegramSpectrum:
    wavenumbers: np.ndarray  # (L,)
    absorption: np.ndarray  # (L,)
    peaks: np.ndarray  # (L,) mask 0..1
    tensor_3ch: np.ndarray  # (3, L)
    scale_meta: dict[str, object] = field(default_factory=dict)


def convert_to_absorption_bot(
    values: np.ndarray,
    yunits: str | None = None,
    *,
    already_absorbance: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Поглощение для бота: -log10(T), без ошибочного (1 - T)."""
    if already_absorbance:
        return ensure_absorbance(values, assumed_scale="absorbance")
    return ensure_absorbance(values, yunits=yunits)


def interpolate_bot_grid(wavenumber: np.ndarray, absorption: np.ndarray) -> np.ndarray:
    wn = np.asarray(wavenumber, dtype=np.float64)
    ab = np.asarray(absorption, dtype=np.float64)
    order = np.argsort(wn)
    wn, ab = wn[order], ab[order]
    ux, idx = np.unique(wn, return_index=True)
    ab_u = ab[idx]
    f = interp1d(
        ux,
        ab_u,
        kind="linear",
        bounds_error=False,
        fill_value=(float(ab_u[0]), float(ab_u[-1])),
    )
    return f(BOT_WAVENUMBERS).astype(np.float32)


def detect_peaks_bot(absorption: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    ab = np.asarray(absorption, dtype=np.float64)
    peaks = np.zeros_like(ab, dtype=np.float32)
    if _HAS_PEAKUTILS:
        peak_indices = peakutils.indexes(ab, thres=threshold, min_dist=10)
    else:
        from scipy.signal import find_peaks

        prom = max(threshold * np.nanmax(ab), 1e-6)
        peak_indices, _ = find_peaks(ab, prominence=prom, distance=10)
    for idx in peak_indices:
        start = max(0, int(idx) - 10)
        end = min(len(ab), int(idx) + 11)
        for i in range(start, end):
            distance = abs(i - int(idx))
            if distance <= 10:
                peaks[i] = 1.0 - distance / 10.0
    return peaks


def jcamp_xy_to_telegram(
    x_cm: np.ndarray,
    y_raw: np.ndarray,
    *,
    yunits: str | None = None,
    already_absorbance: bool = False,
    peak_threshold: float = 0.1,
) -> TelegramSpectrum:
    """x_cm — см⁻¹, y_raw — сырые или уже поглощение-подобные значения."""
    ab, scale_meta = convert_to_absorption_bot(
        y_raw,
        yunits,
        already_absorbance=already_absorbance,
    )
    interp = interpolate_bot_grid(x_cm, ab)
    qc = validate_absorbance_spectrum(interp)
    scale_meta = {**scale_meta, "absorbance_qc": qc}
    pk = detect_peaks_bot(interp, threshold=peak_threshold)
    wn = BOT_WAVENUMBERS.copy()
    tensor = np.vstack([wn, interp, pk]).astype(np.float32)
    return TelegramSpectrum(
        wavenumbers=wn,
        absorption=interp,
        peaks=pk,
        tensor_3ch=tensor,
        scale_meta=scale_meta,
    )
