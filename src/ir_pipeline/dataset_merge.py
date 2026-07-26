"""Слияние уже собранных processed-датасетов без пересборки JCAMP."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ir_pipeline.dataset_build import _build_compounds_table, _write_simple_split
from ir_pipeline.logging_utils import log

_SPECTRA_ARRAY_KEYS = (
    "X",
    "X_absorbance_corrected",
    "X_absorbance_like_interp",
    "coverage",
)


def merge_dataset_versions(
    processed_root: Path,
    source_versions: list[str],
    target_version: str,
    *,
    overwrite: bool = False,
    include_labels: bool = False,
    split_seed: int = 42,
    train_frac: float = 0.85,
    source_tags: list[str] | None = None,
) -> Path:
    """
    Склеивает spectra.npz + meta (и опционально labels) из нескольких версий.

    - Берутся только спектры из spectra.npz (QC-ok).
    - Колонки meta объединяются (union); добавляется source_dataset.
    - По умолчанию labels не копируются (архив спектров + вещества).
    """
    if len(source_versions) < 2:
        raise ValueError("Нужно минимум две исходные версии (--from-version …)")

    if source_tags is not None and len(source_tags) != len(source_versions):
        raise ValueError("source_tags должен совпадать по длине с source_versions")

    tags = source_tags or list(source_versions)
    src_dirs = [processed_root / v for v in source_versions]
    for d, v in zip(src_dirs, source_versions):
        if not (d / "spectra.npz").is_file():
            raise FileNotFoundError(f"Нет spectra.npz в {d} (version={v})")
        if not (d / "meta.parquet").is_file():
            raise FileNotFoundError(f"Нет meta.parquet в {d} (version={v})")

    dst = processed_root / target_version
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} уже существует; используйте --overwrite")
        import shutil

        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    all_ids: list[str] = []
    all_meta: list[pd.DataFrame] = []
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in _SPECTRA_ARRAY_KEYS}
    wavenumbers: np.ndarray | None = None
    sources_report: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    id_collisions = 0

    for src_dir, version, tag in zip(src_dirs, source_versions, tags):
        z = np.load(src_dir / "spectra.npz", allow_pickle=True)
        wn = np.asarray(z["wavenumbers"], dtype=np.float32)
        if wavenumbers is None:
            wavenumbers = wn
        elif wn.shape != wavenumbers.shape or not np.allclose(wn, wavenumbers):
            raise ValueError(
                f"Разные сетки wavenumbers: {version} {wn.shape} vs base {wavenumbers.shape}"
            )

        sid_arr = np.asarray([str(s) for s in z["spectrum_id"].tolist()], dtype=object)
        meta = pd.read_parquet(src_dir / "meta.parquet")
        meta = meta.copy()
        meta["spectrum_id"] = meta["spectrum_id"].astype(str)
        # только строки, реально попавшие в spectra.npz
        meta = meta[meta["spectrum_id"].isin(set(sid_arr.tolist()))].copy()
        # порядок как в npz
        meta = meta.set_index("spectrum_id").reindex(sid_arr.tolist()).reset_index()
        meta["source_dataset"] = tag
        if "qc_ok" not in meta.columns:
            meta["qc_ok"] = True

        keep_mask = []
        kept_ids: list[str] = []
        for sid in sid_arr.tolist():
            if sid in seen_ids:
                id_collisions += 1
                keep_mask.append(False)
                continue
            seen_ids.add(sid)
            keep_mask.append(True)
            kept_ids.append(sid)
        keep_mask_np = np.asarray(keep_mask, dtype=bool)
        if not keep_mask_np.any():
            sources_report.append(
                {
                    "version": version,
                    "tag": tag,
                    "n_in": int(len(sid_arr)),
                    "n_kept": 0,
                    "n_id_skipped": int((~keep_mask_np).sum()),
                }
            )
            continue

        sid_kept = sid_arr[keep_mask_np]
        meta_kept = meta[meta["spectrum_id"].isin(set(sid_kept.tolist()))].copy()
        # сохранить порядок sid_kept
        meta_kept = meta_kept.set_index("spectrum_id").reindex(sid_kept.tolist()).reset_index()

        all_ids.extend(sid_kept.tolist())
        all_meta.append(meta_kept)
        for key in _SPECTRA_ARRAY_KEYS:
            if key not in z:
                raise KeyError(f"В {src_dir}/spectra.npz нет массива {key}")
            arrays[key].append(np.asarray(z[key])[keep_mask_np])

        sources_report.append(
            {
                "version": version,
                "tag": tag,
                "n_in": int(len(sid_arr)),
                "n_kept": int(keep_mask_np.sum()),
                "n_id_skipped": int((~keep_mask_np).sum()),
            }
        )
        log(f"merge source {version} ({tag}): kept={int(keep_mask_np.sum())}/{len(sid_arr)}")

    if not all_ids or wavenumbers is None:
        raise RuntimeError("После merge не осталось спектров")

    meta_df = pd.concat(all_meta, ignore_index=True, sort=False)
    X = np.concatenate(arrays["X"], axis=0)
    X_corr = np.concatenate(arrays["X_absorbance_corrected"], axis=0)
    X_interp = np.concatenate(arrays["X_absorbance_like_interp"], axis=0)
    C = np.concatenate(arrays["coverage"], axis=0)
    sid_out = np.asarray(all_ids, dtype=object)

    if not (len(sid_out) == len(meta_df) == X.shape[0]):
        raise RuntimeError(
            f"Рассинхрон после merge: ids={len(sid_out)}, meta={len(meta_df)}, X={X.shape[0]}"
        )

    log(f"writing merged spectra.npz: n={len(sid_out)}")
    np.savez_compressed(
        dst / "spectra.npz",
        spectrum_id=sid_out,
        X=X.astype(np.float32),
        X_absorbance_corrected=X_corr.astype(np.float32),
        X_absorbance_like_interp=X_interp.astype(np.float32),
        coverage=C.astype(np.uint8),
        wavenumbers=wavenumbers.astype(np.float32),
    )

    meta_df.to_parquet(dst / "meta.parquet", index=False)

    compounds = _build_compounds_table(meta_df)
    compounds.to_parquet(dst / "compounds.parquet", index=False)
    compounds.to_csv(dst / "compounds.csv", index=False, encoding="utf-8")
    log(f"compounds: {len(compounds)}")

    unresolved = meta_df[(meta_df["qc_ok"] == True) & (meta_df["smiles"].isna())]  # noqa: E712
    unresolved.to_parquet(dst / "unresolved_structures.parquet", index=False)

    _write_simple_split(
        dst,
        list(sid_out.tolist()),
        meta_df,
        split_seed=split_seed,
        train_frac=train_frac,
    )

    label_stats: dict[str, int] = {}
    if include_labels:
        label_stats = _merge_label_files(src_dirs, source_versions, tags, set(all_ids), dst)

    manifest = {
        "dataset_version": target_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "merge": {
            "sources": sources_report,
            "id_collisions_skipped": int(id_collisions),
            "include_labels": bool(include_labels),
        },
        "n_spectra_ok": int(len(sid_out)),
        "n_compounds": int(len(compounds)),
        "structure_unresolved_estimate": int(len(unresolved)),
        "include_labels": bool(include_labels),
        "grid": {"min": float(wavenumbers[0]), "max": float(wavenumbers[-1]), "step": 2.0},
        "split_seed": int(split_seed),
        "train_frac": float(train_frac),
        "label_rows": label_stats,
        "compound_files": {
            "meta.parquet": "per-spectrum metadata + source_dataset",
            "compounds.parquet": "unique substances",
            "compounds.csv": "same as compounds.parquet",
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(
        f"merge done: n={len(sid_out)}, compounds={len(compounds)}, "
        f"unresolved={len(unresolved)}, out={dst}"
    )
    return dst


def _merge_label_files(
    src_dirs: list[Path],
    versions: list[str],
    tags: list[str],
    keep_ids: set[str],
    dst: Path,
) -> dict[str, int]:
    names = (
        "labels_spectrum.parquet",
        "labels_structure.parquet",
        "labels_structure_smarts.parquet",
    )
    stats: dict[str, int] = {}
    for name in names:
        frames: list[pd.DataFrame] = []
        for src_dir, version, tag in zip(src_dirs, versions, tags):
            path = src_dir / name
            if not path.exists():
                log(f"skip missing labels {version}/{name}")
                continue
            df = pd.read_parquet(path)
            if "spectrum_id" not in df.columns:
                continue
            df = df.copy()
            df["spectrum_id"] = df["spectrum_id"].astype(str)
            df = df[df["spectrum_id"].isin(keep_ids)]
            df["source_dataset"] = tag
            frames.append(df)
        if not frames:
            continue
        out = pd.concat(frames, ignore_index=True, sort=False)
        out.to_parquet(dst / name, index=False)
        stats[name] = int(len(out))
        log(f"merged {name}: rows={len(out)}")
    return stats
