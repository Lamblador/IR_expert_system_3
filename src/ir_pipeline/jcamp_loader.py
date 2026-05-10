from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class QCResult:
    ok: bool
    reason: str | None = None


def _try_import_jcamp():
    try:
        import jcamp as jc

        return jc
    except ImportError as e:  # pragma: no cover
        raise ImportError("Установите пакет jcamp: pip install jcamp") from e


def read_jcamp_dict(path: Path) -> dict[str, Any]:
    """Читает JCAMP-DX через пакет jcamp (readfile)."""
    jc = _try_import_jcamp()
    if hasattr(jc, "readfile"):
        return jc.readfile(str(path))
    if hasattr(jc, "jcamp_readfile"):
        return jc.jcamp_readfile(str(path))
    raise RuntimeError("Не найдена функция readfile в пакете jcamp")


def cas_from_filename(path: Path) -> str | None:
    stem = path.stem if path.suffix.lower() == ".jdx" else path.name
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", stem):
        return stem
    if re.fullmatch(r"\d{3,10}-\d{2}-\d", stem):
        return stem
    return None


def extract_measurement_mode(meta: dict[str, Any]) -> str:
    blob = " ".join(
        str(meta.get(k, "")).upper()
        for k in ("title", "jcamp cx notes", "jcamp-dx notes", "comments", "sample description", "origin")
        if k in meta
    )
    if "ATR" in blob or "ATTENUATED TOTAL REFLECTANCE" in blob:
        return "atr"
    if "TRANSMISSION" in blob or "TRANSMITTANCE" in blob:
        return "transmission"
    if "ABSORBANCE" in blob or "ABSORPTION" in blob:
        return "absorbance"
    return "unknown"


def extract_sample_state(meta: dict[str, Any]) -> str:
    st = str(meta.get("state", meta.get("sample state", ""))).lower()
    if not st:
        return "unknown"
    return st.split(",")[0].strip()


def qc_jcamp_dict(d: dict[str, Any]) -> QCResult:
    npoints = int(d.get("npoints", 0) or 0)
    if npoints == 0:
        return QCResult(False, "npoints=0")
    x = d.get("x")
    y = d.get("y")
    if x is None or y is None:
        return QCResult(False, "missing x/y")
    try:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
    except Exception:
        return QCResult(False, "x/y not numeric")
    if xa.size < 8 or ya.size < 8:
        return QCResult(False, "too few points")
    if xa.size != ya.size:
        return QCResult(False, "x/y length mismatch")
    data_proc = str(d.get("data processing", "")).upper()
    if "NO SPECTRUM" in data_proc:
        return QCResult(False, "no spectrum in record")
    return QCResult(True)


def flatten_if_link(d: dict[str, Any]) -> dict[str, Any]:
    """Compound LINK: берём первый ребёнок с XYDATA."""
    dt = str(d.get("data type", d.get("datatype", ""))).lower()
    if "link" in dt and isinstance(d.get("children"), list) and d["children"]:
        for child in d["children"]:
            if isinstance(child, dict) and "x" in child and "y" in child:
                child = dict(child)
                child.update({k: v for k, v in d.items() if k in ("filename", "cas registry no")})
                return child
    return d
