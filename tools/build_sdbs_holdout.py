#!/usr/bin/env python3
"""Сборка hold-out набора SDBS (спектры вне dataset_v003) для внешнего теста."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ir_pipeline.dataset_build import _spectrum_id, iter_jcamp_files
from ir_pipeline.dataset_split import load_split_ids
from ir_pipeline.jcamp_loader import (
    extract_measurement_mode,
    extract_sample_state,
    flatten_if_link,
    qc_jcamp_dict,
    read_jcamp_dict,
)
from ir_pipeline.logging_utils import log
from ir_pipeline.preprocess import preprocess_to_grid, to_absorbance_like, wavenumbers_from_jcamp
from ir_pipeline.resnet_input import (
    RESNET_WAVENUMBERS,
    absorbance_grid_to_resnet_tensor,
)


def _is_sdbs_file(path: Path) -> bool:
    name = path.name.upper()
    return name.startswith("IR-NIDA-") and path.suffix.lower() in {".jdx", ".asc", ""}


def _load_v003_ids(dataset_dir: Path) -> set[str]:
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    ids = set(map(str, z["spectrum_id"].tolist()))
    splits = load_split_ids(dataset_dir)
    ids |= splits["train"] | splits["val"] | splits["test"]
    return ids


def _process_jcamp(path: Path, *, peak_threshold: float) -> tuple[np.ndarray, dict] | None:
    try:
        raw = read_jcamp_dict(path)
        flat = flatten_if_link(raw)
        qc = qc_jcamp_dict(flat)
        if not qc.ok:
            return None
        xunits = flat.get("xunits")
        yunits = flat.get("yunits")
        x_cm = wavenumbers_from_jcamp(np.asarray(flat["x"], dtype=float), xunits)
        y_abs_like = to_absorbance_like(np.asarray(flat["y"], dtype=float), yunits)
        grid = preprocess_to_grid(x_cm, y_abs_like)
        rs = absorbance_grid_to_resnet_tensor(
            grid.wavenumbers,
            grid.absorbance,
            peak_threshold=peak_threshold,
        )
        qc_abs = rs.scale_meta.get("absorbance_qc", {})
        if isinstance(qc_abs, dict) and not qc_abs.get("ok", True):
            return None
        meta = {
            "spectrum_id": f"sdbs-{path.stem}",
            "compound_name": str(flat.get("title", path.stem)),
            "smiles": "",
            "source_file": path.name,
            "measurement_mode": extract_measurement_mode(flat),
            "sample_state": extract_sample_state(flat),
            "qc_ok": True,
        }
        return rs.tensor_3ch, meta
    except Exception as exc:
        log(f"skip {path.name}: {exc}")
        return None


def build_sdbs_holdout(
    raw_jcamp_dir: Path,
    dataset_v003_dir: Path,
    out_dir: Path,
    *,
    peak_threshold: float = 0.1,
    max_files: int = 0,
    sdbs_only: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    v003_ids = _load_v003_ids(dataset_v003_dir)
    log(f"v003 known ids: {len(v003_ids)}")

    tensors: list[np.ndarray] = []
    meta_rows: list[dict] = []
    skipped_in_v003 = 0
    skipped_qc_or_parse = 0

    paths = [p for p in iter_jcamp_files(raw_jcamp_dir, max_files=0) if not sdbs_only or _is_sdbs_file(p)]
    if max_files:
        paths = paths[:max_files]
    log(f"candidate SDBS files: {len(paths)}")

    for path in paths:
        sid_hash = _spectrum_id(path)
        if sid_hash in v003_ids:
            skipped_in_v003 += 1
            continue
        out = _process_jcamp(path, peak_threshold=peak_threshold)
        if out is None:
            skipped_qc_or_parse += 1
            continue
        tensor, meta = out
        tensors.append(tensor)
        meta_rows.append(meta)

    if not tensors:
        raise RuntimeError(
            "Нет спектров для sdbs_eval — проверьте raw_jcamp_dir и dataset_v003. "
            f"candidates={len(paths)}, skipped_in_v003={skipped_in_v003}, "
            f"skipped_qc_or_parse={skipped_qc_or_parse}"
        )

    X_in = np.stack(tensors, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)
    npz_path = out_dir / "sdbs_eval.npz"
    np.savez_compressed(
        npz_path,
        X_input=X_in,
        wavenumbers=RESNET_WAVENUMBERS,
        spectrum_id=meta_df["spectrum_id"].astype(str).values,
        compound_name=meta_df["compound_name"].astype(str).values,
        smiles=meta_df["smiles"].astype(str).values,
        source_file=meta_df["source_file"].astype(str).values,
    )
    parquet_path = out_dir / "sdbs_eval_meta.parquet"
    meta_df.to_parquet(parquet_path, index=False)

    manifest = {
        "n_spectra": int(len(tensors)),
        "n_candidates": len(paths),
        "skipped_in_v003": skipped_in_v003,
        "skipped_qc_or_parse": skipped_qc_or_parse,
        "dataset_v003_dir": str(dataset_v003_dir),
        "raw_jcamp_dir": str(raw_jcamp_dir),
        "sdbs_only": sdbs_only,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"sdbs holdout: {len(tensors)} spectra → {npz_path}")
    return npz_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка external_sdbs/sdbs_eval.npz")
    ap.add_argument("--raw-jcamp", type=Path, default=Path("downloaded_jcamp"))
    ap.add_argument("--dataset-v003", type=Path, default=Path("data/processed/dataset_v003"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/external_sdbs"))
    ap.add_argument("--peak-threshold", type=float, default=0.1)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--all-jcamp", action="store_true", help="не только IR-NIDA-*")
    args = ap.parse_args()
    build_sdbs_holdout(
        args.raw_jcamp,
        args.dataset_v003,
        args.out_dir,
        peak_threshold=args.peak_threshold,
        max_files=args.max_files,
        sdbs_only=not args.all_jcamp,
    )


if __name__ == "__main__":
    main()
