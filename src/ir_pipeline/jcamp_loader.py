from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_RE_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


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

    # SDBS_extraction: ##XYDATA=(X++(Y..Y)), но в строках лежат пары X Y X Y…
    # Пакет jcamp тогда сыпет X-Check и получает len(x)!=len(y).
    if _is_mislabeled_xy_pairs(path):
        return _read_mislabeled_xy_pairs(path)

    jc = _try_import_jcamp()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        if hasattr(jc, "readfile"):
            d = jc.readfile(str(path))
        elif hasattr(jc, "jcamp_readfile"):
            d = jc.jcamp_readfile(str(path))
        else:
            raise RuntimeError("Не найдена функция readfile в пакете jcamp")

    x = d.get("x")
    y = d.get("y")
    if x is not None and y is not None and len(np.asarray(x)) != len(np.asarray(y)):
        # запасной путь, если эвристика не сработала на заголовке
        if _is_mislabeled_xy_pairs(path, force_scan=True):
            return _read_mislabeled_xy_pairs(path)
    return d


def _jcamp_header_and_data_lines(path: Path) -> tuple[dict[str, str], list[str]]:
    text = path.read_text(encoding="latin-1", errors="replace")
    header: dict[str, str] = {}
    data_lines: list[str] = []
    in_xydata = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("$$"):
            continue
        if line.startswith("##"):
            body = line[2:]
            if "=" not in body:
                continue
            lhs, rhs = body.split("=", 1)
            key = lhs.strip().lower()
            val = rhs.strip()
            header[key] = val
            if key in {"xydata", "xypoints", "peak table"}:
                in_xydata = True
                data_lines = []
            elif key == "end":
                in_xydata = False
            continue
        if in_xydata:
            data_lines.append(line)
    return header, data_lines


def _first_data_numbers(data_lines: list[str], limit: int = 40) -> list[float]:
    nums: list[float] = []
    for line in data_lines:
        for m in _RE_NUM.finditer(line):
            nums.append(float(m.group(0)))
            if len(nums) >= limit:
                return nums
    return nums


def _looks_like_xy_pairs(nums: list[float]) -> bool:
    """True, если числа чередуются как wavenumber, intensity, wavenumber, …"""
    if len(nums) < 6 or len(nums) % 2 != 0:
        # для эвристики достаточно чётного префикса
        nums = nums[: len(nums) - (len(nums) % 2)]
    if len(nums) < 6:
        return False
    xs = np.asarray(nums[0::2], dtype=float)
    ys = np.asarray(nums[1::2], dtype=float)
    if xs.size < 3:
        return False
    dx = np.diff(xs)
    # X монотонны с почти постоянным шагом; Y по масштабу не похожи на wavenumber-сетку
    if not (np.all(dx > 0) or np.all(dx < 0)):
        return False
    step = float(np.median(np.abs(dx)))
    if step <= 0:
        return False
    if float(np.max(np.abs(np.abs(dx) - step))) > max(0.05 * step, 0.5):
        return False
    # типичный IR: X ~ сотни–тысячи, Y обычно меньше шага сетки / порядка единиц
    y_scale = float(np.nanmax(np.abs(ys))) if ys.size else 0.0
    if y_scale > max(abs(float(xs[0])), 50.0) and y_scale > 10 * step:
        return False
    return True


def _is_mislabeled_xy_pairs(path: Path, *, force_scan: bool = False) -> bool:
    header, data_lines = _jcamp_header_and_data_lines(path)
    xydata = header.get("xydata", "").replace(" ", "").upper()
    if not force_scan and xydata != "(X++(Y..Y))":
        return False
    if not data_lines:
        return False
    return _looks_like_xy_pairs(_first_data_numbers(data_lines))


def _header_float(header: dict[str, str], key: str, default: float | None = None) -> float | None:
    raw = header.get(key)
    if raw is None:
        return default
    try:
        return float(raw.replace(",", ".", 1))
    except ValueError:
        return default


def _read_mislabeled_xy_pairs(path: Path) -> dict[str, Any]:
    """Парсит JCAMP, где под (X++(Y..Y)) лежат пары X Y (экспорт SDBS_extraction)."""
    header, data_lines = _jcamp_header_and_data_lines(path)
    nums: list[float] = []
    for line in data_lines:
        nums.extend(float(m.group(0)) for m in _RE_NUM.finditer(line))
    if len(nums) < 8 or len(nums) % 2 != 0:
        raise ValueError(f"JCAMP XY-pairs parse error in {path}: odd/short numeric stream")

    xs = np.asarray(nums[0::2], dtype=float)
    ys = np.asarray(nums[1::2], dtype=float)
    xfactor = _header_float(header, "xfactor", 1.0) or 1.0
    yfactor = _header_float(header, "yfactor", 1.0) or 1.0
    xs = xs * xfactor
    ys = ys * yfactor

    out: dict[str, Any] = {
        "filename": str(path),
        "title": header.get("title", path.stem),
        "origin": header.get("origin", ""),
        "owner": header.get("owner", ""),
        "xunits": header.get("xunits", "1/CM"),
        "yunits": header.get("yunits", ""),
        "xfactor": xfactor,
        "yfactor": yfactor,
        "firstx": _header_float(header, "firstx", float(xs[0])),
        "lastx": _header_float(header, "lastx", float(xs[-1])),
        "npoints": int(xs.size),
        "xydata": "(XY..XY)",
        "x": xs,
        "y": ys,
        "cas registry no": header.get("cas registry no", ""),
        "molform": header.get("molform", ""),
        "state": header.get("state", header.get("sample state", "")),
        "sample state": header.get("sample state", header.get("state", "")),
        "data type": header.get("data type", "INFRARED SPECTRUM"),
    }
    for k in ("inchi", "inchikey", "sdbs_spcode", "source"):
        if k in header and header[k]:
            out[k] = header[k]
    return out


def extract_jcamp_structure_ids(d: dict[str, Any]) -> tuple[str | None, str | None]:
    """Достаёт (InChI, InChIKey) из заголовка JCAMP, если есть."""
    inchi = d.get("inchi")
    if inchi is None:
        inchi = d.get("INCHI")
    inchikey = d.get("inchikey")
    if inchikey is None:
        inchikey = d.get("INCHIKEY")
    inchi_s = str(inchi).strip() if inchi not in (None, "") else None
    ik_s = str(inchikey).strip().upper() if inchikey not in (None, "") else None
    if inchi_s and inchi_s.upper().startswith("INCHIKEY="):
        # защита от путаницы ключей
        inchi_s = None
    if ik_s and ik_s.startswith("INCHI="):
        ik_s = None
    return inchi_s, ik_s


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
