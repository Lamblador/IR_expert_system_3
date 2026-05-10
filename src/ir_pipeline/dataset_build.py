from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_pipeline.bands import load_bands
from ir_pipeline.jcamp_loader import (
    cas_from_filename,
    extract_measurement_mode,
    extract_sample_state,
    flatten_if_link,
    qc_jcamp_dict,
    read_jcamp_dict,
)
from ir_pipeline.labeling import BandObservation, label_spectrum_spectrum_only, label_spectrum_structure_conditioned
from ir_pipeline.preprocess import preprocess_to_grid, to_absorbance_like, wavenumbers_from_jcamp
from ir_pipeline.structure_resolver import (
    load_structure_cache,
    mol_from_resolution,
    resolve_structure_for_record,
    save_structure_cache,
    structure_cache_path,
)


def _spectrum_id(path: Path) -> str:
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return h[:16]


def iter_jcamp_files(raw_dir: Path, max_files: int) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Нет каталога JCAMP: {raw_dir}")
    paths = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() == ".jdx" or p.suffix == "":
            paths.append(p)
        if max_files and len(paths) >= max_files:
            break
    return paths


def build_dataset(
    raw_jcamp_dir: Path,
    processed_root: Path,
    dataset_version: str,
    bands_yaml: Path,
    max_files: int = 0,
    pubchem_sleep_s: float = 0.12,
    split_seed: int = 42,
    train_frac: float = 0.85,
) -> Path:
    """Собирает версионированный датасет: spectra.npz, meta.parquet, labels_*.parquet, manifest.json."""
    out_dir = processed_root / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = load_bands(bands_yaml)
    band_ids = [b.band_id for b in bands]

    cache_path = structure_cache_path(processed_root, dataset_version)
    cache = load_structure_cache(cache_path)

    files = iter_jcamp_files(raw_jcamp_dir, max_files)
    if not files:
        raise RuntimeError(f"Не найдено JCAMP файлов в {raw_jcamp_dir}")

    meta_rows: list[dict[str, Any]] = []
    rows_spec: list[dict[str, Any]] = []
    rows_str: list[dict[str, Any]] = []

    spectra_list: list[np.ndarray] = []
    absorb_corr_list: list[np.ndarray] = []
    absorb_interp_list: list[np.ndarray] = []
    spectrum_ids: list[str] = []
    coverage_list: list[np.ndarray] = []

    qc_failed = 0
    struct_failed = 0
    last_wn: np.ndarray | None = None

    for fp in tqdm(files, desc="JCAMP"):
        sid = _spectrum_id(fp)
        try:
            d = read_jcamp_dict(fp)
        except Exception as e:
            qc_failed += 1
            meta_rows.append(
                {
                    "spectrum_id": sid,
                    "path": str(fp),
                    "cas": cas_from_filename(fp),
                    "qc_ok": False,
                    "qc_reason": f"read_error:{e}",
                }
            )
            continue

        d = flatten_if_link(d)
        qc = qc_jcamp_dict(d)
        xunits = d.get("xunits")
        yunits = d.get("yunits")
        x_cm = wavenumbers_from_jcamp(np.asarray(d["x"], dtype=float), xunits)
        y_abs_like = to_absorbance_like(np.asarray(d["y"], dtype=float), yunits)

        meta_common = {
            "spectrum_id": sid,
            "path": str(fp),
            "cas": cas_from_filename(fp) or str(d.get("cas registry no", "")),
            "title": str(d.get("title", "")),
            "molform": str(d.get("molform", "")),
            "xunits_raw": str(xunits or ""),
            "yunits_raw": str(yunits or ""),
            "npoints": int(d.get("npoints", 0) or 0),
            "measurement_mode": extract_measurement_mode(d),
            "sample_state": extract_sample_state(d),
            "origin": str(d.get("origin", "")),
        }

        if not qc.ok:
            qc_failed += 1
            meta_rows.append({**meta_common, "qc_ok": False, "qc_reason": qc.reason})
            continue

        grid = preprocess_to_grid(x_cm, y_abs_like)
        last_wn = grid.wavenumbers

        cas_key = (meta_common.get("cas") or "").strip()
        title_key = (meta_common.get("title") or "").strip()
        res = resolve_structure_for_record(cas_key or None, title_key or None, cache, pubchem_sleep_s)

        mol = mol_from_resolution(res)
        if res.get("smiles") is None:
            struct_failed += 1

        meta_rows.append(
            {
                **meta_common,
                "qc_ok": True,
                "qc_reason": None,
                "smiles": res.get("smiles"),
                "inchikey": res.get("inchikey"),
                "structure_source": res.get("source"),
                "structure_error": res.get("error"),
            }
        )

        spectra_list.append(grid.absorbance_normalized.astype(np.float32))
        absorb_corr_list.append(grid.absorbance.astype(np.float32))
        absorb_interp_list.append(grid.absorbance_raw.astype(np.float32))
        coverage_list.append(grid.coverage_mask.astype(np.uint8))
        spectrum_ids.append(sid)

        obs_spec = label_spectrum_spectrum_only(grid.wavenumbers, grid.absorbance_normalized, grid.coverage_mask, bands)
        for o in obs_spec:
            rows_spec.append(_obs_row(sid, o, label_schema="spectrum_only"))

        obs_str = label_spectrum_structure_conditioned(
            grid.wavenumbers, grid.absorbance_normalized, grid.coverage_mask, mol, bands
        )
        for o in obs_str:
            rows_str.append(_obs_row(sid, o, label_schema="structure_conditioned"))

    save_structure_cache(cache_path, cache)

    if not spectra_list or last_wn is None:
        raise RuntimeError("Не удалось собрать ни одного валидного спектра — проверьте данные и QC.")

    X = np.stack(spectra_list, axis=0)
    X_abs_corr = np.stack(absorb_corr_list, axis=0)
    X_abs_interp = np.stack(absorb_interp_list, axis=0)
    C = np.stack(coverage_list, axis=0)
    np.savez_compressed(
        out_dir / "spectra.npz",
        spectrum_id=np.array(spectrum_ids, dtype=object),
        X=X,
        X_absorbance_corrected=X_abs_corr,
        X_absorbance_like_interp=X_abs_interp,
        coverage=C,
        wavenumbers=last_wn.astype(np.float32),
    )

    rng = np.random.default_rng(int(split_seed))
    uids = np.array(spectrum_ids, dtype=object)
    perm = rng.permutation(len(uids))
    u_shuf = uids[perm]
    split_idx = int(max(1, round(float(train_frac) * len(u_shuf))))
    train_ids = u_shuf[:split_idx].tolist()
    test_ids = u_shuf[split_idx:].tolist()
    if not test_ids:
        test_ids = train_ids
    split_payload = {
        "seed": int(split_seed),
        "train_frac": float(train_frac),
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    (out_dir / "split.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(out_dir / "meta.parquet", index=False)

    pd.DataFrame(rows_spec).to_parquet(out_dir / "labels_spectrum.parquet", index=False)
    pd.DataFrame(rows_str).to_parquet(out_dir / "labels_structure.parquet", index=False)

    unresolved = meta_df[(meta_df["qc_ok"] == True) & (meta_df["smiles"].isna())]  # noqa: E712
    unresolved.to_parquet(out_dir / "unresolved_structures.parquet", index=False)

    manifest = {
        "dataset_version": dataset_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_jcamp_dir": str(raw_jcamp_dir.resolve()),
        "n_files_seen": len(files),
        "n_spectra_ok": int(len(spectrum_ids)),
        "qc_failed": qc_failed,
        "structure_unresolved_estimate": int(struct_failed),
        "band_ids": band_ids,
        "bands_config": str(bands_yaml.resolve()),
        "grid": {"min": 400.0, "max": 4000.0, "step": 2.0},
        "split_seed": int(split_seed),
        "train_frac": float(train_frac),
        "arrays": {
            "X": "normalized absorbance-like on grid",
            "X_absorbance_corrected": "after ALS baseline + smoothing",
            "X_absorbance_like_interp": "interpolated absorbance/transmittance-derived on grid",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return out_dir


def _obs_row(spectrum_id: str, o: BandObservation, label_schema: str) -> dict[str, Any]:
    return {
        "spectrum_id": spectrum_id,
        "band_id": o.band_id,
        "region_min_cm1": o.region_min_cm1,
        "region_max_cm1": o.region_max_cm1,
        "structure_match": o.structure_match,
        "structure_expected": o.structure_match if label_schema == "structure_conditioned" else None,
        "observed_peak_cm1": o.observed_peak_cm1,
        "intensity_class": o.intensity_class,
        "label_confidence": o.label_confidence,
        "label_schema": label_schema,
    }
