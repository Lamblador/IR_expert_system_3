from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from rdkit import Chem

from ir_pipeline.bands import BandDef, structure_matches_band
from ir_pipeline.preprocess import intensity_bucket_from_peak


@dataclass
class BandObservation:
    band_id: str
    region_min_cm1: float
    region_max_cm1: float
    structure_match: bool
    observed_peak_cm1: float | None
    intensity_class: str
    label_confidence: float


def find_dominant_peak_in_region(
    wn: np.ndarray,
    absorb_norm: np.ndarray,
    mask: np.ndarray,
    region_min: float,
    region_max: float,
    prominence_rel: float = 0.015,
) -> tuple[float | None, float]:
    """
    Ищет главный пик в регионе на нормализованном спектре.
    Возвращает (peak_cm1, confidence 0..1).
    """
    sel = (wn >= region_min) & (wn <= region_max) & mask
    if sel.sum() < 5:
        return None, 0.0
    y = absorb_norm[sel]
    x = wn[sel]
    prom = max(prominence_rel, float(np.nanmax(y)) * prominence_rel)
    peaks, props = find_peaks(y, prominence=prom)
    if peaks.size == 0:
        # fallback: просто максимум
        i = int(np.argmax(y))
        return float(x[i]), 0.35
    # самый высокий по prominence * height
    scores = props["prominences"] * y[peaks]
    j = int(np.argmax(scores))
    pk = peaks[j]
    peak_cm = float(x[pk])
    conf = float(min(1.0, props["prominences"][j] / (prom + 1e-9)))
    return peak_cm, conf


def label_spectrum_bands(
    spectrum_id: str,
    wn: np.ndarray,
    absorb_norm: np.ndarray,
    coverage_mask: np.ndarray,
    mol: Chem.Mol | None,
    bands: list[BandDef],
    require_structure: bool,
) -> list[BandObservation]:
    """Разметка полос: при require_structure=True метки только если SMARTS совпал."""
    out: list[BandObservation] = []
    for b in bands:
        sm_match = structure_matches_band(mol, b)
        if require_structure and not sm_match:
            continue
        pk, conf = find_dominant_peak_in_region(
            wn,
            absorb_norm,
            coverage_mask,
            b.range_min_cm1,
            b.range_max_cm1,
        )
        icls = intensity_bucket_from_peak(absorb_norm, wn, pk) if pk is not None else "u"
        out.append(
            BandObservation(
                band_id=b.band_id,
                region_min_cm1=b.range_min_cm1,
                region_max_cm1=b.range_max_cm1,
                structure_match=sm_match,
                observed_peak_cm1=pk,
                intensity_class=icls,
                label_confidence=conf,
            )
        )
    return out


def label_spectrum_spectrum_only(
    wn: np.ndarray,
    absorb_norm: np.ndarray,
    coverage_mask: np.ndarray,
    bands: list[BandDef],
) -> list[BandObservation]:
    """Вариант без структуры: все полосы из справочника как кандидаты."""
    return label_spectrum_bands(
        spectrum_id="",
        wn=wn,
        absorb_norm=absorb_norm,
        coverage_mask=coverage_mask,
        mol=None,
        bands=bands,
        require_structure=False,
    )


def label_spectrum_structure_conditioned(
    wn: np.ndarray,
    absorb_norm: np.ndarray,
    coverage_mask: np.ndarray,
    mol: Chem.Mol | None,
    bands: list[BandDef],
) -> list[BandObservation]:
    """Только полосы с совпавшей подструктурой."""
    out: list[BandObservation] = []
    for b in bands:
        if not structure_matches_band(mol, b):
            continue
        pk, conf = find_dominant_peak_in_region(
            wn,
            absorb_norm,
            coverage_mask,
            b.range_min_cm1,
            b.range_max_cm1,
        )
        icls = intensity_bucket_from_peak(absorb_norm, wn, pk) if pk is not None else "u"
        out.append(
            BandObservation(
                band_id=b.band_id,
                region_min_cm1=b.range_min_cm1,
                region_max_cm1=b.range_max_cm1,
                structure_match=True,
                observed_peak_cm1=pk,
                intensity_class=icls,
                label_confidence=conf,
            )
        )
    return out
