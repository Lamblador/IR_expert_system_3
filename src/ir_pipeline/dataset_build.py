from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_pipeline.bands import load_bands
from ir_pipeline.jcamp_loader import (
    cas_from_filename,
    extract_jcamp_structure_ids,
    extract_measurement_mode,
    extract_sample_state,
    flatten_if_link,
    qc_jcamp_dict,
    read_jcamp_dict,
)
from ir_pipeline.labeling import (
    BandObservation,
    label_spectrum_spectrum_only,
    label_spectrum_structure_conditioned,
    label_spectrum_structure_smarts_only,
)
from ir_pipeline.preprocess import preprocess_to_grid, to_absorbance_like, wavenumbers_from_jcamp
from ir_pipeline.structure_resolver import (
    load_structure_cache,
    mol_from_resolution,
    resolve_structure_for_record,
    save_structure_cache,
    seed_structure_cache_from_lamblador,
    structure_cache_path,
)


def _event(message: str) -> None:
    tqdm.write(f"[ir-pipeline] {datetime.now().strftime('%H:%M:%S')} {message}")


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
        if p.suffix.lower() in {".jdx", ".asc"} or p.suffix == "":
            paths.append(p)
        if max_files and len(paths) >= max_files:
            break
    return paths


def _try_resolve_jcamp_dir_from_zip(raw_dir: Path) -> Path:
    """
    Если raw_dir не существует, пытается найти downloaded_jcamp*.zip рядом и распаковать.
    Возвращает путь к каталогу с JCAMP.
    """
    if raw_dir.exists():
        return raw_dir

    search_roots: list[Path] = []
    if raw_dir.parent:
        search_roots.append(raw_dir.parent)
    search_roots.extend([Path.cwd(), Path.cwd().parent])

    seen: set[Path] = set()
    zip_candidates: list[Path] = []
    for root in search_roots:
        rp = root.resolve()
        if rp in seen or not root.exists():
            continue
        seen.add(rp)
        for z in root.glob("**/downloaded_jcamp*.zip"):
            if z.is_file():
                zip_candidates.append(z)

    if not zip_candidates:
        raise FileNotFoundError(f"Нет каталога JCAMP: {raw_dir}")

    zip_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    _event(f"zip fallback candidates found: {len(zip_candidates)}")
    for z in zip_candidates[:20]:
        st = z.stat()
        mod = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        _event(f"zip candidate: path={z} | mtime={mod} | size_bytes={st.st_size}")
    if len(zip_candidates) > 20:
        _event(f"zip candidates truncated in log: shown=20, total={len(zip_candidates)}")

    chosen_zip = zip_candidates[0]
    _event(f"raw_jcamp_dir not found; using zip fallback: {chosen_zip}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(chosen_zip) as zf:
        zf.extractall(raw_dir)

    nested = raw_dir / "downloaded_jcamp"
    if nested.is_dir():
        _event(f"zip extracted nested folder, using: {nested}")
        return nested

    return raw_dir


def build_dataset(
    raw_jcamp_dir: Path,
    processed_root: Path,
    dataset_version: str,
    bands_yaml: Path,
    max_files: int = 0,
    pubchem_sleep_s: float = 0.12,
    resolve_missing_structures: bool = False,
    split_seed: int = 42,
    train_frac: float = 0.85,
    *,
    include_labels: bool = True,
) -> Path:
    """Собирает версионированный датасет: spectra.npz, meta.parquet, manifest.json (+ labels при include_labels)."""
    out_dir = processed_root / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)
    _event(
        f"build-dataset start: version={dataset_version}, max_files={max_files or 'all'}, "
        f"include_labels={include_labels}"
    )

    if include_labels:
        _event(f"loading bands: {bands_yaml}")
        bands = load_bands(bands_yaml)
        band_ids = [b.band_id for b in bands]
        _event(f"loaded bands: {len(band_ids)}")
    else:
        bands = []
        band_ids = []
        _event("labels disabled: skipping bands / SMARTS labeling")

    cache_path = structure_cache_path(processed_root, dataset_version)
    _event(f"loading structure cache: {cache_path}")
    cache = load_structure_cache(cache_path)
    _event(f"structure cache entries before seed: {len(cache)}")
    try:
        seeded_structures = seed_structure_cache_from_lamblador(cache, processed_root)
        _event(f"fast structure seed ready: added_keys={seeded_structures}, cache_entries={len(cache)}")
    except Exception as e:
        seeded_structures = 0
        _event(f"Lamblador/IRSpectra seed skipped: {e}")

    raw_jcamp_dir_resolved = _try_resolve_jcamp_dir_from_zip(raw_jcamp_dir)
    files = iter_jcamp_files(raw_jcamp_dir_resolved, max_files)
    if not files:
        raise RuntimeError(f"Не найдено JCAMP файлов в {raw_jcamp_dir_resolved}")
    _event(f"JCAMP files selected: {len(files)} from {raw_jcamp_dir_resolved}")
    if resolve_missing_structures:
        _event("slow structure resolution enabled: PubChem may be called for cache misses")
    else:
        _event("fast structure resolution: PubChem disabled for cache misses")

    meta_rows: list[dict[str, Any]] = []
    rows_spec: list[dict[str, Any]] = []
    rows_str: list[dict[str, Any]] = []
    rows_smarts: list[dict[str, Any]] = []

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
        jcamp_inchi, jcamp_inchikey = extract_jcamp_structure_ids(d)

        meta_common = {
            "spectrum_id": sid,
            "path": str(fp),
            "cas": cas_from_filename(fp) or str(d.get("cas registry no", "")),
            "title": str(d.get("title", "")),
            "molform": str(d.get("molform", "")),
            "jcamp_inchi": jcamp_inchi,
            "jcamp_inchikey": jcamp_inchikey,
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
        res = resolve_structure_for_record(
            cas_key or None,
            title_key or None,
            cache,
            pubchem_sleep_s,
            allow_network=resolve_missing_structures,
            inchi=jcamp_inchi,
            inchikey=jcamp_inchikey,
        )

        mol = mol_from_resolution(res) if include_labels else None
        if res.get("smiles") is None:
            struct_failed += 1

        meta_rows.append(
            {
                **meta_common,
                "qc_ok": True,
                "qc_reason": None,
                "smiles": res.get("smiles"),
                "inchi": res.get("inchi") or jcamp_inchi,
                "inchikey": res.get("inchikey") or jcamp_inchikey,
                "structure_source": res.get("source"),
                "structure_error": res.get("error"),
            }
        )

        spectra_list.append(grid.absorbance_normalized.astype(np.float32))
        absorb_corr_list.append(grid.absorbance.astype(np.float32))
        absorb_interp_list.append(grid.absorbance_raw.astype(np.float32))
        coverage_list.append(grid.coverage_mask.astype(np.uint8))
        spectrum_ids.append(sid)

        if include_labels:
            obs_spec = label_spectrum_spectrum_only(
                grid.wavenumbers, grid.absorbance_normalized, grid.coverage_mask, bands
            )
            for o in obs_spec:
                rows_spec.append(_obs_row(sid, o, label_schema="spectrum_only"))

            obs_str = label_spectrum_structure_conditioned(
                grid.wavenumbers, grid.absorbance_normalized, grid.coverage_mask, mol, bands
            )
            for o in obs_str:
                rows_str.append(_obs_row(sid, o, label_schema="structure_conditioned"))

            obs_smarts = label_spectrum_structure_smarts_only(
                grid.wavenumbers, grid.absorbance_normalized, grid.coverage_mask, mol, bands
            )
            for o in obs_smarts:
                rows_smarts.append(_obs_row(sid, o, label_schema="structure_smarts_only"))

    _event("saving structure cache")
    save_structure_cache(cache_path, cache)

    if not spectra_list or last_wn is None:
        raise RuntimeError("Не удалось собрать ни одного валидного спектра — проверьте данные и QC.")

    _event(f"stacking spectra arrays: n={len(spectra_list)}")
    X = np.stack(spectra_list, axis=0)
    X_abs_corr = np.stack(absorb_corr_list, axis=0)
    X_abs_interp = np.stack(absorb_interp_list, axis=0)
    C = np.stack(coverage_list, axis=0)
    _event("writing spectra.npz")
    np.savez_compressed(
        out_dir / "spectra.npz",
        spectrum_id=np.array(spectrum_ids, dtype=object),
        X=X,
        X_absorbance_corrected=X_abs_corr,
        X_absorbance_like_interp=X_abs_interp,
        coverage=C,
        wavenumbers=last_wn.astype(np.float32),
    )

    meta_df = pd.DataFrame(meta_rows)
    _event("writing parquet tables")
    meta_df.to_parquet(out_dir / "meta.parquet", index=False)

    # Справочник веществ (уникальные по inchikey / cas / title)
    compounds = _build_compounds_table(meta_df)
    compounds.to_parquet(out_dir / "compounds.parquet", index=False)
    compounds.to_csv(out_dir / "compounds.csv", index=False, encoding="utf-8")
    _event(f"compounds table: n={len(compounds)}")

    _event("writing split.json")
    if include_labels and str(dataset_version) == "dataset_v003":
        from ir_pipeline.dataset_split import build_split_for_dataset, fractions_from_train_frac

        # labels нужны для стратификации — пишем их до split
        pd.DataFrame(rows_spec).to_parquet(out_dir / "labels_spectrum.parquet", index=False)
        pd.DataFrame(rows_str).to_parquet(out_dir / "labels_structure.parquet", index=False)
        pd.DataFrame(rows_smarts).to_parquet(out_dir / "labels_structure_smarts.parquet", index=False)
        build_split_for_dataset(
            out_dir,
            bands_yaml,
            label_schema="structure_smarts",
            fractions=fractions_from_train_frac(train_frac),
            seed=int(split_seed),
            group_by_inchikey=True,
        )
    else:
        if include_labels:
            pd.DataFrame(rows_spec).to_parquet(out_dir / "labels_spectrum.parquet", index=False)
            pd.DataFrame(rows_str).to_parquet(out_dir / "labels_structure.parquet", index=False)
            pd.DataFrame(rows_smarts).to_parquet(out_dir / "labels_structure_smarts.parquet", index=False)
        else:
            for name in (
                "labels_spectrum.parquet",
                "labels_structure.parquet",
                "labels_structure_smarts.parquet",
            ):
                p = out_dir / name
                if p.exists():
                    p.unlink()
        _write_simple_split(out_dir, spectrum_ids, meta_df, split_seed=split_seed, train_frac=train_frac)

    n_spec_pos = int(pd.DataFrame(rows_spec)["observed_peak_cm1"].notna().sum()) if rows_spec else 0
    n_str_pos = int(pd.DataFrame(rows_str)["observed_peak_cm1"].notna().sum()) if rows_str else 0
    n_smarts_pos = len(rows_smarts)

    unresolved = meta_df[(meta_df["qc_ok"] == True) & (meta_df["smiles"].isna())]  # noqa: E712
    unresolved.to_parquet(out_dir / "unresolved_structures.parquet", index=False)

    manifest = {
        "dataset_version": dataset_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_jcamp_dir": str(raw_jcamp_dir_resolved.resolve()),
        "n_files_seen": len(files),
        "n_spectra_ok": int(len(spectrum_ids)),
        "n_compounds": int(len(compounds)),
        "qc_failed": qc_failed,
        "structure_unresolved_estimate": int(struct_failed),
        "structure_cache_seeded_from_lamblador": int(seeded_structures),
        "resolve_missing_structures": bool(resolve_missing_structures),
        "include_labels": bool(include_labels),
        "band_ids": band_ids,
        "bands_config": str(bands_yaml.resolve()) if include_labels else None,
        "grid": {"min": 400.0, "max": 4000.0, "step": 2.0},
        "split_seed": int(split_seed),
        "train_frac": float(train_frac),
        "arrays": {
            "X": "normalized absorbance-like on grid",
            "X_absorbance_corrected": "after ALS baseline + smoothing",
            "X_absorbance_like_interp": "interpolated absorbance/transmittance-derived on grid",
        },
        "compound_files": {
            "meta.parquet": "per-spectrum metadata + structure ids",
            "compounds.parquet": "unique substances aggregated from meta",
            "compounds.csv": "same as compounds.parquet (CSV)",
        },
        "label_files": (
            {
                "labels_spectrum.parquet": "peaks in band regions (no SMARTS filter)",
                "labels_structure.parquet": "SMARTS match + observed peak in region",
                "labels_structure_smarts.parquet": "SMARTS match only (peak optional in optional_peak_cm1)",
            }
            if include_labels
            else {}
        ),
        "label_positive_rows": (
            {
                "spectrum": n_spec_pos,
                "structure": n_str_pos,
                "structure_smarts": n_smarts_pos,
            }
            if include_labels
            else {}
        ),
    }
    _event("writing manifest.json")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _event(
        f"build-dataset done: ok={len(spectrum_ids)}, qc_failed={qc_failed}, unresolved={struct_failed}, "
        f"compounds={len(compounds)}, labels={'on' if include_labels else 'off'}, out={out_dir}"
    )

    return out_dir


def _build_compounds_table(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Уникальные вещества по inchikey (fallback: cas / title) + число спектров."""
    cols_out = [
        "compound_key",
        "inchikey",
        "inchi",
        "smiles",
        "cas",
        "title",
        "molform",
        "n_spectra",
        "structure_source",
    ]
    ok = meta_df[meta_df["qc_ok"] == True].copy() if "qc_ok" in meta_df.columns else meta_df.copy()  # noqa: E712
    if ok.empty:
        return pd.DataFrame(columns=cols_out)

    for col in ("inchikey", "jcamp_inchikey", "cas", "title", "inchi", "smiles", "molform", "structure_source"):
        if col not in ok.columns:
            ok[col] = None

    def _compound_key(row: pd.Series) -> str:
        for col in ("inchikey", "jcamp_inchikey", "cas", "title"):
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip():
                return f"{col}:{str(v).strip()}"
        return f"spectrum:{row.get('spectrum_id')}"

    ok["compound_key"] = ok.apply(_compound_key, axis=1)

    def _first_nonnull(series: pd.Series):
        for v in series:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            s = str(v).strip()
            if s and s.lower() != "nan":
                return v
        return None

    agg = (
        ok.groupby("compound_key", sort=False)
        .agg(
            inchikey=("inchikey", _first_nonnull),
            inchi=("inchi", _first_nonnull),
            smiles=("smiles", _first_nonnull),
            cas=("cas", _first_nonnull),
            title=("title", _first_nonnull),
            molform=("molform", _first_nonnull),
            n_spectra=("spectrum_id", "count"),
            structure_source=("structure_source", _first_nonnull),
        )
        .reset_index()
    )
    return agg.sort_values(["n_spectra", "compound_key"], ascending=[False, True]).reset_index(drop=True)


def _write_simple_split(
    out_dir: Path,
    spectrum_ids: list[str],
    meta_df: pd.DataFrame,
    *,
    split_seed: int,
    train_frac: float,
) -> None:
    """Простой split по группам inchikey (без стратификации по labels)."""
    rng = np.random.default_rng(int(split_seed))
    ok = meta_df[meta_df["qc_ok"] == True]  # noqa: E712
    sid_to_group: dict[str, str] = {}
    for _, r in ok.iterrows():
        sid = str(r["spectrum_id"])
        ik = r.get("inchikey")
        if ik is not None and not (isinstance(ik, float) and np.isnan(ik)) and str(ik).strip():
            sid_to_group[sid] = f"ik:{str(ik).strip()}"
        else:
            sid_to_group[sid] = f"sid:{sid}"

    groups = sorted(set(sid_to_group.get(s, f"sid:{s}") for s in spectrum_ids))
    perm = rng.permutation(len(groups))
    g_shuf = [groups[i] for i in perm]
    split_idx = int(max(1, round(float(train_frac) * len(g_shuf))))
    train_g = set(g_shuf[:split_idx])
    test_g = set(g_shuf[split_idx:]) or train_g
    train_ids = [s for s in spectrum_ids if sid_to_group.get(s, f"sid:{s}") in train_g]
    test_ids = [s for s in spectrum_ids if sid_to_group.get(s, f"sid:{s}") in test_g]
    if not test_ids:
        test_ids = list(train_ids)
    payload = {
        "version": 1,
        "seed": int(split_seed),
        "train_frac": float(train_frac),
        "group_by_inchikey": True,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    (out_dir / "split.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_missing_structures_for_dataset(
    processed_root: Path,
    dataset_version: str,
    pubchem_sleep_s: float = 0.12,
) -> dict[str, int | str]:
    """Медленный опциональный добор структур для unresolved_structures.parquet."""
    dataset_dir = processed_root / dataset_version
    unresolved_path = dataset_dir / "unresolved_structures.parquet"
    if not unresolved_path.exists():
        raise FileNotFoundError(f"Нет {unresolved_path}; сначала запустите build-dataset")

    _event(f"resolve-missing-structures start: version={dataset_version}")
    cache_path = structure_cache_path(processed_root, dataset_version)
    _event(f"loading structure cache: {cache_path}")
    cache = load_structure_cache(cache_path)
    unresolved = pd.read_parquet(unresolved_path)
    _event(f"unresolved rows loaded: {len(unresolved)}")

    attempted = 0
    resolved = 0
    seen: set[tuple[str, str, str, str]] = set()
    for _, row in tqdm(unresolved.iterrows(), total=len(unresolved), desc="Resolve missing structures"):
        cas_value = row.get("cas")
        title_value = row.get("title")
        inchi_value = row.get("jcamp_inchi", row.get("inchi"))
        inchikey_value = row.get("jcamp_inchikey", row.get("inchikey"))
        cas = "" if pd.isna(cas_value) else str(cas_value).strip()
        title = "" if pd.isna(title_value) else str(title_value).strip()
        inchi = "" if pd.isna(inchi_value) else str(inchi_value).strip()
        inchikey = "" if pd.isna(inchikey_value) else str(inchikey_value).strip()
        key = (cas, title[:220], inchi[:80], inchikey)
        if key in seen:
            continue
        seen.add(key)

        attempted += 1
        res = resolve_structure_for_record(
            cas or None,
            title or None,
            cache,
            pubchem_sleep_s,
            allow_network=True,
            inchi=inchi or None,
            inchikey=inchikey or None,
        )
        if res.get("smiles"):
            resolved += 1

    _event("saving updated structure cache")
    save_structure_cache(cache_path, cache)
    report = {
        "dataset_version": dataset_version,
        "attempted": attempted,
        "resolved": resolved,
        "cache_path": str(cache_path),
    }
    (dataset_dir / "resolve_missing_structures_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _event(f"resolve-missing-structures done: attempted={attempted}, resolved={resolved}")
    return report


def _obs_row(spectrum_id: str, o: BandObservation, label_schema: str) -> dict[str, Any]:
    return {
        "spectrum_id": spectrum_id,
        "band_id": o.band_id,
        "region_min_cm1": o.region_min_cm1,
        "region_max_cm1": o.region_max_cm1,
        "structure_match": o.structure_match,
        "structure_expected": o.structure_match
        if label_schema in {"structure_conditioned", "structure_smarts_only"}
        else None,
        "observed_peak_cm1": o.observed_peak_cm1,
        "optional_peak_cm1": o.optional_peak_cm1,
        "intensity_class": o.intensity_class,
        "label_confidence": o.label_confidence,
        "label_schema": label_schema,
    }
