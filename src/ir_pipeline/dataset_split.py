"""Train/val/test split для multi-label (structure_smarts), split.json v2."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ir_pipeline.dataset_preview import build_multilabel_matrix
from ir_pipeline.logging_utils import log

DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLIT_VERSION = 2


def load_split_ids(dataset_dir: Path) -> dict[str, set[str]]:
    """train/val/test id sets; v1 без val_ids → val пустой."""
    path = dataset_dir / "split.json"
    if not path.exists():
        return {"train": set(), "val": set(), "test": set()}
    sp = json.loads(path.read_text(encoding="utf-8"))
    train = set(map(str, sp.get("train_ids", [])))
    test = set(map(str, sp.get("test_ids", [])))
    val = set(map(str, sp.get("val_ids", [])))
    if not val and int(sp.get("version", 1)) < SPLIT_VERSION:
        log("split.json v1: val_ids отсутствуют — val пустой, test использовался как val при обучении")
    return {"train": train, "val": val, "test": test}


def _validate_fractions(fractions: dict[str, float]) -> dict[str, float]:
    f = {k: float(fractions[k]) for k in ("train", "val", "test")}
    s = sum(f.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {s}")
    return f


def fractions_from_train_frac(train_frac: float) -> dict[str, float]:
    """train_frac → train/val/test; доля val:test как в DEFAULT_FRACTIONS."""
    train = float(train_frac)
    remainder = max(0.0, 1.0 - train)
    val_test_sum = DEFAULT_FRACTIONS["val"] + DEFAULT_FRACTIONS["test"]
    val = remainder * (DEFAULT_FRACTIONS["val"] / val_test_sum)
    test = remainder * (DEFAULT_FRACTIONS["test"] / val_test_sum)
    return _validate_fractions({"train": train, "val": val, "test": test})


def _multilabel_stratified_two_way(
    Y: np.ndarray,
    ids: list[str],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Один отсек test_size доли через iterstrat или fallback."""
    n = len(ids)
    if n < 2:
        return ids, []

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # pip install iterative-stratification

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        idx = np.arange(n)
        tr_idx, te_idx = next(msss.split(idx, Y))
        train_part = [ids[i] for i in tr_idx]
        test_part = [ids[i] for i in te_idx]
        return train_part, test_part
    except ImportError:
        log("iterstrat не установлен (pip install -e '.[split]'); случайный split по образцам")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(max(1, round(test_size * n)))
        if cut >= n:
            cut = n - 1
        te_idx = perm[:cut]
        tr_idx = perm[cut:]
        return [ids[i] for i in tr_idx], [ids[i] for i in te_idx]


def _split_by_groups(
    spectrum_ids: list[str],
    groups: dict[str, str],
    fractions: dict[str, float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split на уровне group key (inchikey), затем развернуть в spectrum_id."""
    gid_to_sids: dict[str, list[str]] = {}
    for sid in spectrum_ids:
        g = groups.get(sid) or sid
        gid_to_sids.setdefault(g, []).append(sid)

    gids = list(gid_to_sids.keys())
    if len(gids) < 3:
        raise ValueError("Слишком мало уникальных групп для group_by_inchikey split")

    rng = np.random.default_rng(seed)
    gperm = rng.permutation(len(gids))
    n_test_g = max(1, int(round(fractions["test"] * len(gids))))
    n_val_g = max(1, int(round(fractions["val"] * len(gids))))
    n_train_g = len(gids) - n_test_g - n_val_g
    if n_train_g < 1:
        n_train_g = 1
        n_val_g = max(0, len(gids) - n_train_g - n_test_g)
    test_groups = {gids[i] for i in gperm[:n_test_g]}
    val_groups = {gids[i] for i in gperm[n_test_g : n_test_g + n_val_g]}
    train_groups = set(gids) - test_groups - val_groups

    def expand(gs: set[str]) -> list[str]:
        out: list[str] = []
        for g in gs:
            out.extend(gid_to_sids.get(g, []))
        return out

    return expand(train_groups), expand(val_groups), expand(test_groups)


def build_stratified_split(
    spectrum_ids: list[str],
    Y: np.ndarray,
    *,
    fractions: dict[str, float] | None = None,
    seed: int = 42,
    label_schema_for_split: str = "structure_smarts",
    group_by_inchikey: bool = False,
    inchikey_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Multi-label stratified split 70/10/20 (по умолчанию).
    Возвращает payload для split.json v2 + coverage stats.
    """
    fractions = _validate_fractions(fractions or DEFAULT_FRACTIONS)
    ids = list(spectrum_ids)
    if len(ids) != Y.shape[0]:
        raise ValueError("len(spectrum_ids) != Y.shape[0]")

    if group_by_inchikey and inchikey_map:
        train_ids, val_ids, test_ids = _split_by_groups(ids, inchikey_map, fractions, seed)
    else:
        train_val, test_ids = _multilabel_stratified_two_way(Y, ids, fractions["test"], seed)
        id_to_i = {s: i for i, s in enumerate(ids)}
        tv_idx = [id_to_i[s] for s in train_val]
        Y_tv = Y[tv_idx]
        val_frac = fractions["val"] / max(fractions["train"] + fractions["val"], 1e-9)
        train_ids, val_ids = _multilabel_stratified_two_way(Y_tv, train_val, val_frac, seed + 1)

    if not train_ids:
        raise RuntimeError("Пустой train после split")
    if not test_ids:
        test_ids = train_ids[: max(1, len(train_ids) // 10)]
    if not val_ids:
        val_ids = train_ids[: max(1, len(train_ids) // 10)]

    coverage = _band_coverage_report(Y, ids, train_ids, val_ids, test_ids)

    return {
        "version": SPLIT_VERSION,
        "seed": int(seed),
        "fractions": fractions,
        "label_schema_for_split": label_schema_for_split,
        "group_by_inchikey": bool(group_by_inchikey),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "coverage": coverage,
    }


def _band_coverage_report(
    Y: np.ndarray,
    all_ids: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> dict[str, Any]:
    id_to_i = {s: i for i, s in enumerate(all_ids)}
    tr = np.array([id_to_i[s] for s in train_ids if s in id_to_i])
    va = np.array([id_to_i[s] for s in val_ids if s in id_to_i])
    te = np.array([id_to_i[s] for s in test_ids if s in id_to_i])
    n_bands = Y.shape[1]
    missing_val: list[int] = []
    missing_test: list[int] = []
    per_band: list[dict[str, Any]] = []
    for b in range(n_bands):
        g_pos = int(Y[:, b].sum())
        tr_p = int(Y[tr, b].sum()) if len(tr) else 0
        va_p = int(Y[va, b].sum()) if len(va) else 0
        te_p = int(Y[te, b].sum()) if len(te) else 0
        per_band.append(
            {"band_index": b, "global_positives": g_pos, "train": tr_p, "val": va_p, "test": te_p}
        )
        if g_pos >= 3 and va_p < 1:
            missing_val.append(b)
        if g_pos >= 3 and te_p < 1:
            missing_test.append(b)
    return {
        "bands_missing_in_val": missing_val,
        "bands_missing_in_test": missing_test,
        "per_band": per_band,
    }


def write_split_json(dataset_dir: Path, payload: dict[str, Any]) -> Path:
    path = dataset_dir / "split.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = dataset_dir / "split_report.json"
    report = {
        "n_train": len(payload.get("train_ids", [])),
        "n_val": len(payload.get("val_ids", [])),
        "n_test": len(payload.get("test_ids", [])),
        "coverage": payload.get("coverage", {}),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_split_for_dataset(
    dataset_dir: Path,
    bands_yaml: Path,
    *,
    label_schema: str = "structure_smarts",
    fractions: dict[str, float] | None = None,
    seed: int = 42,
    group_by_inchikey: bool = False,
) -> dict[str, Any]:
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    spec_ids = [str(s) for s in z["spectrum_id"].tolist()]
    Y, _ = build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml, label_schema=label_schema)
    inchikey_map: dict[str, str] | None = None
    if group_by_inchikey:
        meta = pd.read_parquet(dataset_dir / "meta.parquet")
        if "inchikey" in meta.columns:
            inchikey_map = {
                str(r["spectrum_id"]): str(r["inchikey"]) if pd.notna(r["inchikey"]) else str(r["spectrum_id"])
                for _, r in meta.iterrows()
            }
    payload = build_stratified_split(
        spec_ids,
        Y,
        fractions=fractions,
        seed=seed,
        label_schema_for_split=label_schema,
        group_by_inchikey=group_by_inchikey,
        inchikey_map=inchikey_map,
    )
    write_split_json(dataset_dir, payload)
    log(
        f"split v2: train={len(payload['train_ids'])}, val={len(payload['val_ids'])}, "
        f"test={len(payload['test_ids'])}"
    )
    return payload


def audit_split(dataset_dir: Path, bands_yaml: Path, label_schema: str = "structure_smarts") -> dict[str, Any]:
    sp = load_split_ids(dataset_dir)
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    spec_ids = [str(s) for s in z["spectrum_id"].tolist()]
    Y, class_names = build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml, label_schema=label_schema)
    id_to_i = {s: i for i, s in enumerate(spec_ids)}
    rows = []
    for b, name in enumerate(class_names):
        g = int(Y[:, b].sum())
        tr = sum(1 for s in sp["train"] if s in id_to_i and Y[id_to_i[s], b] > 0)
        va = sum(1 for s in sp["val"] if s in id_to_i and Y[id_to_i[s], b] > 0)
        te = sum(1 for s in sp["test"] if s in id_to_i and Y[id_to_i[s], b] > 0)
        rows.append({"band_id": name, "global": g, "train": tr, "val": va, "test": te})
    return {
        "n_train": len(sp["train"]),
        "n_val": len(sp["val"]),
        "n_test": len(sp["test"]),
        "bands": rows,
    }


def copy_dataset_version(
    processed_root: Path,
    source_version: str,
    target_version: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Копия датасета (например v002 → v003) без пересборки JCAMP."""
    src = processed_root / source_version
    dst = processed_root / target_version
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} уже существует; используйте --overwrite")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log(f"скопировано {src} → {dst}")
    return dst
