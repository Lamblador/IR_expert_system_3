"""3-канальный вход IrResnet4: сетка 400–4000 см⁻¹ (как dataset grid)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from ir_pipeline.measurement_context import build_context_matrix, context_column_names
from ir_pipeline.preprocess import ensure_absorbance, validate_absorbance_spectrum

try:
    import peakutils

    _HAS_PEAKUTILS = True
except ImportError:
    _HAS_PEAKUTILS = False

GRID_MIN_CM1 = 400.0
GRID_MAX_CM1 = 4000.0
GRID_STEP_CM1 = 2.0
RESNET_WAVENUMBERS = np.arange(GRID_MIN_CM1, GRID_MAX_CM1 + 1e-9, GRID_STEP_CM1, dtype=np.float32)
RESNET_GRID_LEN = len(RESNET_WAVENUMBERS)
MODEL_INPUTS_FILENAME = "model_inputs.npz"
MODEL_INPUTS_VERSION = "grid_400_4000_ctx_v1"


@dataclass
class ResnetSpectrum:
    wavenumbers: np.ndarray
    absorption: np.ndarray
    peaks: np.ndarray
    tensor_3ch: np.ndarray
    scale_meta: dict[str, object] = field(default_factory=dict)


def convert_to_absorption_resnet(
    values: np.ndarray,
    yunits: str | None = None,
    *,
    already_absorbance: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    if already_absorbance:
        return ensure_absorbance(values, assumed_scale="absorbance")
    return ensure_absorbance(values, yunits=yunits)


def interpolate_resnet_grid(wavenumber: np.ndarray, absorption: np.ndarray) -> np.ndarray:
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
    return f(RESNET_WAVENUMBERS).astype(np.float32)


def detect_peaks_resnet(absorption: np.ndarray, threshold: float = 0.1) -> np.ndarray:
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


def xy_to_resnet_spectrum(
    x_cm: np.ndarray,
    y_raw: np.ndarray,
    *,
    yunits: str | None = None,
    already_absorbance: bool = False,
    peak_threshold: float = 0.1,
) -> ResnetSpectrum:
    ab, scale_meta = convert_to_absorption_resnet(y_raw, yunits, already_absorbance=already_absorbance)
    interp = interpolate_resnet_grid(x_cm, ab)
    qc = validate_absorbance_spectrum(interp)
    scale_meta = {**scale_meta, "absorbance_qc": qc}
    pk = detect_peaks_resnet(interp, threshold=peak_threshold)
    wn = RESNET_WAVENUMBERS.copy()
    tensor = np.vstack([wn, interp, pk]).astype(np.float32)
    return ResnetSpectrum(
        wavenumbers=wn,
        absorption=interp,
        peaks=pk,
        tensor_3ch=tensor,
        scale_meta=scale_meta,
    )


def absorbance_grid_to_resnet_tensor(
    wavenumbers: np.ndarray,
    absorption: np.ndarray,
    *,
    peak_threshold: float = 0.1,
) -> ResnetSpectrum:
    """Из поглощения на сетке датасета (400–4000); при совпадении сетки — почти без искажений."""
    wn_ds = np.asarray(wavenumbers, dtype=np.float64)
    if len(wn_ds) == RESNET_GRID_LEN and np.allclose(wn_ds, RESNET_WAVENUMBERS, atol=0.5):
        interp = np.asarray(absorption, dtype=np.float32)
    else:
        interp = interpolate_resnet_grid(wavenumbers, absorption)
    pk = detect_peaks_resnet(interp, threshold=peak_threshold)
    wn = RESNET_WAVENUMBERS.copy()
    tensor = np.vstack([wn, interp, pk]).astype(np.float32)
    return ResnetSpectrum(
        wavenumbers=wn,
        absorption=interp,
        peaks=pk,
        tensor_3ch=tensor,
        scale_meta={"source": "spectra_npz"},
    )


def build_model_inputs_from_dataset(
    dataset_dir: Path,
    *,
    peak_threshold: float = 0.1,
    include_context: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray | None, list[str]]:
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    wn = np.asarray(z["wavenumbers"], dtype=np.float64)
    X_abs = np.asarray(z["X_absorbance_corrected"], dtype=np.float64)
    spec_ids = [str(s) for s in z["spectrum_id"].tolist()]
    tensors: list[np.ndarray] = []
    for i in range(len(spec_ids)):
        rs = absorbance_grid_to_resnet_tensor(wn, X_abs[i], peak_threshold=peak_threshold)
        tensors.append(rs.tensor_3ch)
    X_in = np.stack(tensors, axis=0).astype(np.float32)

    X_ctx: np.ndarray | None = None
    ctx_cols: list[str] = []
    if include_context:
        import pandas as pd

        meta = pd.read_parquet(dataset_dir / "meta.parquet")
        X_ctx, ctx_cols = build_context_matrix(meta, spec_ids)

    return X_in, spec_ids, X_ctx, ctx_cols


def _npz_grid_compatible(z: np.lib.npyio.NpzFile) -> bool:
    if z.get("version") != MODEL_INPUTS_VERSION:
        return False
    wn = np.asarray(z.get("wavenumbers", []), dtype=np.float64)
    return len(wn) == RESNET_GRID_LEN and np.allclose(wn, RESNET_WAVENUMBERS, atol=0.5)


def ensure_model_inputs_npz(
    dataset_dir: Path,
    *,
    peak_threshold: float = 0.1,
    force: bool = False,
    include_context: bool = True,
) -> Path:
    path = dataset_dir / MODEL_INPUTS_FILENAME
    if path.exists() and not force:
        with np.load(path, allow_pickle=True) as z:
            if _npz_grid_compatible(z):
                return path
    X, ids, X_ctx, ctx_cols = build_model_inputs_from_dataset(
        dataset_dir,
        peak_threshold=peak_threshold,
        include_context=include_context,
    )
    payload: dict[str, object] = {
        "X_input": X,
        "spectrum_id": np.array(ids, dtype=object),
        "wavenumbers": RESNET_WAVENUMBERS,
        "version": MODEL_INPUTS_VERSION,
        "grid_min": GRID_MIN_CM1,
        "grid_max": GRID_MAX_CM1,
        "grid_step": GRID_STEP_CM1,
    }
    if X_ctx is not None:
        payload["X_context"] = X_ctx
        payload["context_columns"] = np.array(ctx_cols, dtype=object)
    np.savez_compressed(path, **payload)
    return path


def load_model_inputs(
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None, list[str]]:
    path = ensure_model_inputs_npz(dataset_dir)
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X_input"], dtype=np.float32)
    wn = np.asarray(z["wavenumbers"], dtype=np.float64)
    ids = [str(s) for s in z["spectrum_id"].tolist()]
    X_ctx = None
    ctx_cols: list[str] = []
    if "X_context" in z:
        X_ctx = np.asarray(z["X_context"], dtype=np.float32)
        ctx_cols = [str(c) for c in z["context_columns"].tolist()]
    return X, wn, ids, X_ctx, ctx_cols
