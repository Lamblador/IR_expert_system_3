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
    """Читает JCAMP-DX через пакет jcamp или PerkinElmer ASCII (*.asc)."""
    if path.suffix.lower() == ".asc":
        return _read_perkinelmer_ascii(path)
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
    if "GAS PHASE" in blob or " GAS SAMPLE" in blob or blob.startswith("GAS ") or " FTIR GAS" in blob:
        return "gas"
    if "SOLUTION" in blob or "SOLVENT" in blob or " IN CHLOROFORM" in blob or " IN CCL4" in blob:
        return "solution"
    if "TRANSMISSION" in blob or "TRANSMITTANCE" in blob:
        return "transmission"
    if "ABSORBANCE" in blob or "ABSORPTION" in blob:
        return "absorbance"
    return "unknown"


def extract_sample_state(meta: dict[str, Any]) -> str:
    st = str(meta.get("state", meta.get("sample state", ""))).lower()
    if not st:
        blob = " ".join(
            str(meta.get(k, "")).lower()
            for k in ("title", "sample description", "comments")
            if k in meta
        )
        if "gas" in blob:
            return "gas"
        if "solution" in blob or "liquid" in blob:
            return "solution"
        if "solid" in blob or "powder" in blob:
            return "solid"
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


def _read_perkinelmer_ascii(path: Path) -> dict[str, Any]:
    """
    Читает спектры PerkinElmer ASCII (*.asc), встречающиеся в локальных ATR-экспортах.
    Ожидает блок #DATA с двумя колонками: wavenumber и %T.
    """
    text = path.read_text(encoding="cp1251", errors="ignore")
    lines = text.splitlines()

    data_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "#DATA":
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError(f"ASC parse error: no #DATA block in {path}")

    xs: list[float] = []
    ys: list[float] = []
    for ln in lines[data_idx:]:
        row = ln.strip()
        if not row or row.startswith("#"):
            continue
        parts = re.split(r"[\t ]+", row)
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0].replace(",", "."))
            y = float(parts[1].replace(",", "."))
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)

    if len(xs) < 8:
        raise ValueError(f"ASC parse error: too few numeric points in {path}")

    stem = path.stem
    title = re.sub(r"^\d+\s*", "", stem).replace("_", " ").strip()
    if not title:
        title = stem

    return {
        "filename": str(path),
        "title": title,
        "origin": "PerkinElmer ASCII",
        "xunits": "cm-1",
        "yunits": "TRANSMITTANCE",
        "npoints": len(xs),
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
    }
