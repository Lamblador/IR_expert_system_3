"""Аудит дублей в датасете (отчёт без автоматического удаления)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ir_pipeline.dataset_preview import build_multilabel_matrix, labels_parquet_path


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def audit_dataset_duplicates(
    dataset_dir: Path,
    bands_yaml: Path,
    out_dir: Path,
    *,
    near_dup_threshold: float = 0.99,
    max_near_pairs: int = 50,
) -> dict[str, Any]:
    """Формирует отчёты о дублях в runs/... (без изменения датасета)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(dataset_dir)

    meta = pd.read_parquet(dataset_dir / "meta.parquet")
    ok = meta[meta["qc_ok"] == True].copy()  # noqa: E712

    report: dict[str, Any] = {"dataset_dir": str(dataset_dir.resolve())}

    dup_sid = ok[ok.duplicated(subset=["spectrum_id"], keep=False)]
    report["duplicate_spectrum_id_rows"] = int(len(dup_sid))
    if len(dup_sid):
        dup_sid[["spectrum_id", "path", "cas", "inchikey"]].to_csv(
            out_dir / "duplicate_spectrum_id.csv", index=False
        )

    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    npz_ids = [str(s) for s in z["spectrum_id"].tolist()]
    report["spectra_npz_count"] = len(npz_ids)
    report["spectra_npz_unique_ids"] = len(set(npz_ids))
    report["spectra_npz_duplicate_ids"] = len(npz_ids) - len(set(npz_ids))

    inchi_groups: list[dict[str, Any]] = []
    if "inchikey" in ok.columns:
        g = ok[ok["inchikey"].notna() & (ok["inchikey"].astype(str).str.len() > 0)]
        for ik, sub in g.groupby("inchikey"):
            if len(sub) < 2:
                continue
            modes = sub["measurement_mode"].dropna().unique().tolist() if "measurement_mode" in sub else []
            states = sub["sample_state"].dropna().unique().tolist() if "sample_state" in sub else []
            inchi_groups.append(
                {
                    "inchikey": str(ik),
                    "n_spectra": int(len(sub)),
                    "spectrum_ids": sub["spectrum_id"].astype(str).tolist()[:20],
                    "measurement_modes": modes[:10],
                    "sample_states": states[:10],
                }
            )
        inchi_groups.sort(key=lambda x: -x["n_spectra"])
        pd.DataFrame(inchi_groups).to_csv(out_dir / "duplicates_inchikey.csv", index=False)
        report["inchikey_groups_ge2"] = len(inchi_groups)
        report["max_spectra_per_inchikey"] = int(inchi_groups[0]["n_spectra"]) if inchi_groups else 0

    cas_groups: list[dict[str, Any]] = []
    if "cas" in ok.columns:
        g = ok[ok["cas"].notna() & (ok["cas"].astype(str).str.len() > 0)]
        for cas, sub in g.groupby("cas"):
            if len(sub) < 2:
                continue
            cas_groups.append(
                {
                    "cas": str(cas),
                    "n_spectra": int(len(sub)),
                    "n_unique_inchikey": int(sub["inchikey"].nunique()) if "inchikey" in sub else 0,
                }
            )
        cas_groups.sort(key=lambda x: -x["n_spectra"])
        pd.DataFrame(cas_groups).to_csv(out_dir / "duplicates_cas.csv", index=False)
        report["cas_groups_ge2"] = len(cas_groups)

    X_abs = np.asarray(z["X_absorbance_corrected"], dtype=np.float64)
    sid_to_i = {s: i for i, s in enumerate(npz_ids)}
    near_pairs: list[dict[str, Any]] = []

    for grp in inchi_groups[:200]:
        ids = [s for s in grp["spectrum_ids"] if s in sid_to_i]
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                si, sj = sid_to_i[ids[i]], sid_to_i[ids[j]]
                sim = _cosine_sim(X_abs[si], X_abs[sj])
                if sim >= near_dup_threshold:
                    near_pairs.append(
                        {
                            "spectrum_id_a": ids[i],
                            "spectrum_id_b": ids[j],
                            "inchikey": grp["inchikey"],
                            "cosine_similarity": round(sim, 6),
                        }
                    )
    near_pairs.sort(key=lambda x: -x["cosine_similarity"])
    near_pairs = near_pairs[:max_near_pairs]
    pd.DataFrame(near_pairs).to_csv(out_dir / "near_duplicate_pairs.csv", index=False)
    report["near_duplicate_pairs_logged"] = len(near_pairs)

    label_overlap: dict[str, Any] = {}
    if labels_parquet_path(dataset_dir, "structure_smarts").exists() and labels_parquet_path(
        dataset_dir, "structure"
    ).exists():
        Y_sm, _ = build_multilabel_matrix(
            dataset_dir, npz_ids, bands_yaml, label_schema="structure_smarts"
        )
        Y_st, _ = build_multilabel_matrix(dataset_dir, npz_ids, bands_yaml, label_schema="structure")
        n_sm = int(Y_sm.sum())
        n_st = int(Y_st.sum())
        both = int(((Y_sm > 0) & (Y_st > 0)).sum())
        sm_only = int(((Y_sm > 0) & (Y_st == 0)).sum())
        label_overlap = {
            "positives_structure_smarts": n_sm,
            "positives_structure_peak": n_st,
            "positives_both": both,
            "positives_smarts_only_no_peak": sm_only,
            "fraction_smarts_lost_when_requiring_peak": round(1.0 - both / max(n_sm, 1), 4),
        }
        report["label_overlap"] = label_overlap

    (out_dir / "duplicates_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Dataset duplicate audit",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- QC-ok spectra: {len(ok)}",
        f"- Duplicate spectrum_id rows in meta: {report.get('duplicate_spectrum_id_rows', 0)}",
        f"- InChIKey groups (≥2 spectra): {report.get('inchikey_groups_ge2', 0)}",
        f"- Near-duplicate pairs (cos≥{near_dup_threshold}): {report.get('near_duplicate_pairs_logged', 0)}",
    ]
    if label_overlap:
        lines.extend(
            [
                "",
                "## Label overlap (structure_smarts vs structure+peak)",
                f"- SMARTS-only positives: {label_overlap['positives_structure_smarts']}",
                f"- SMARTS+peak positives: {label_overlap['positives_structure_peak']}",
                f"- SMARTS-only without peak label: {label_overlap['positives_smarts_only_no_peak']}",
                f"- Fraction lost when requiring peak: {label_overlap['fraction_smarts_lost_when_requiring_peak']}",
            ]
        )
    (out_dir / "duplicates_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
