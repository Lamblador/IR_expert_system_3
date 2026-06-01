"""Превью датасета и вспомогательные функции для multi-label матриц."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ir_pipeline.bands import load_bands
from ir_pipeline.preprocess import validate_absorbance_spectrum

LABEL_SCHEMA_FILES = {
    "spectrum": "labels_spectrum.parquet",
    "structure": "labels_structure.parquet",
    "structure_smarts": "labels_structure_smarts.parquet",
}


def labels_parquet_path(dataset_dir: Path, label_schema: str) -> Path:
    """Путь к parquet-файлу меток для схемы spectrum | structure | structure_smarts."""
    name = LABEL_SCHEMA_FILES.get(label_schema)
    if not name:
        raise ValueError(f"Неизвестная label_schema={label_schema!r}; допустимо: {list(LABEL_SCHEMA_FILES)}")
    return dataset_dir / name


def build_multilabel_matrix(
    dataset_dir: Path,
    spectrum_ids: list[str],
    bands_yaml: Path,
    *,
    label_schema: str = "spectrum",
) -> tuple[np.ndarray, list[str]]:
    """
    Y (N, C) multi-label матрица.
    - spectrum / structure: 1, если observed_peak_cm1 задан;
    - structure_smarts: 1 по факту строки (SMARTS-only, пик не обязателен).
    """
    label_file = labels_parquet_path(dataset_dir, label_schema)
    if not label_file.exists():
        raise FileNotFoundError(f"Нет файла меток: {label_file}")
    labels = pd.read_parquet(label_file)
    bands = load_bands(bands_yaml)
    class_names = [b.band_id for b in bands]
    band_to_idx = {b: i for i, b in enumerate(class_names)}
    Y = np.zeros((len(spectrum_ids), len(class_names)), dtype=np.float32)
    sid_to_i = {s: i for i, s in enumerate(spectrum_ids)}
    if label_schema == "structure_smarts":
        rows = labels
    else:
        rows = labels.dropna(subset=["observed_peak_cm1"])
    for _, r in rows.iterrows():
        sid = str(r["spectrum_id"])
        bid = str(r["band_id"])
        if sid not in sid_to_i or bid not in band_to_idx:
            continue
        Y[sid_to_i[sid], band_to_idx[bid]] = 1.0
    return Y, class_names


def _plot_peak_labels(ax, labels_df: pd.DataFrame, color: str, alpha: float = 0.5) -> int:
    n = 0
    for _, r in labels_df.iterrows():
        cm = r.get("observed_peak_cm1")
        if cm is None or pd.isna(cm):
            continue
        ax.axvline(float(cm), color=color, alpha=alpha, lw=0.75)
        n += 1
    return n


def plot_dataset_preview(
    dataset_dir: Path,
    out_dir: Path,
    bands_yaml: Path,
    *,
    n_examples: int = 3,
    seed: int = 42,
) -> list[Path]:
    """
    Превью: поглощение + метки полос (по максимуму в регионе) + метки по SMARTS/структуре.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    wn = np.asarray(z["wavenumbers"], dtype=np.float64)
    X_abs = np.asarray(z["X_absorbance_corrected"], dtype=np.float64)
    spec_ids_npz = [str(s) for s in z["spectrum_id"].tolist()]
    sid_to_idx = {s: i for i, s in enumerate(spec_ids_npz)}

    meta = pd.read_parquet(dataset_dir / "meta.parquet")
    ok_meta = meta[meta["qc_ok"] == True].copy()  # noqa: E712
    if ok_meta.empty:
        return written

    n_pick = min(int(n_examples), len(ok_meta))
    pick = ok_meta.sample(n=n_pick, random_state=int(seed)) if n_pick < len(ok_meta) else ok_meta.head(n_pick)

    labels_spec = pd.read_parquet(dataset_dir / "labels_spectrum.parquet")
    labels_str_path = dataset_dir / "labels_structure.parquet"
    labels_str = pd.read_parquet(labels_str_path) if labels_str_path.exists() else pd.DataFrame()
    labels_smarts_path = dataset_dir / "labels_structure_smarts.parquet"
    labels_smarts = pd.read_parquet(labels_smarts_path) if labels_smarts_path.exists() else pd.DataFrame()

    preview_qc: list[dict[str, object]] = []

    for plot_i, (_, row) in enumerate(pick.iterrows()):
        sid = str(row["spectrum_id"])
        if sid not in sid_to_idx:
            continue
        ab = X_abs[sid_to_idx[sid]]
        qc = validate_absorbance_spectrum(ab)
        preview_qc.append({"spectrum_id": sid, **qc})

        sub_spec = labels_spec[(labels_spec["spectrum_id"] == sid) & labels_spec["observed_peak_cm1"].notna()]
        if not labels_str.empty:
            sub_str = labels_str[(labels_str["spectrum_id"] == sid) & labels_str["observed_peak_cm1"].notna()]
        else:
            sub_str = pd.DataFrame()

        n_panels = 4 if not labels_smarts.empty else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        title = str(row.get("title", sid))
        qc_tag = "" if qc.get("ok") else f" [QC: {','.join(qc.get('issues', []))}]"
        axes[0].plot(wn, ab, "k-", lw=0.85)
        axes[0].set_ylabel("absorbance (corrected)")
        axes[0].set_title(f"{title} ({sid}){qc_tag}")
        axes[0].grid(alpha=0.2)

        axes[1].plot(wn, ab, "k-", lw=0.7)
        n_spec = _plot_peak_labels(axes[1], sub_spec, color="tab:orange")
        axes[1].set_ylabel(f"spectrum labels (n={n_spec})")
        axes[1].grid(alpha=0.2)

        axes[2].plot(wn, ab, "k-", lw=0.7)
        n_str = _plot_peak_labels(axes[2], sub_str, color="tab:blue")
        axes[2].set_ylabel(f"structure+peak (n={n_str})")
        axes[2].grid(alpha=0.2)

        if n_panels == 4:
            if not labels_smarts.empty:
                sub_smarts = labels_smarts[labels_smarts["spectrum_id"] == sid]
                peak_col = "optional_peak_cm1" if "optional_peak_cm1" in sub_smarts.columns else "observed_peak_cm1"
                sub_smarts_plot = sub_smarts[sub_smarts[peak_col].notna()].copy()
                sub_smarts_plot = sub_smarts_plot.rename(columns={peak_col: "observed_peak_cm1"})
            else:
                sub_smarts_plot = pd.DataFrame()
            axes[3].plot(wn, ab, "k-", lw=0.7)
            n_sm = _plot_peak_labels(axes[3], sub_smarts_plot, color="tab:green")
            axes[3].set_ylabel(f"SMARTS-only ref peaks (n={n_sm})")
            axes[3].set_xlim(float(np.max(wn)), float(np.min(wn)))
            axes[3].set_xlabel(r"Wavenumber (cm$^{-1}$)")
            axes[3].grid(alpha=0.2)
        else:
            axes[2].set_xlim(float(np.max(wn)), float(np.min(wn)))
            axes[2].set_xlabel(r"Wavenumber (cm$^{-1}$)")

        fig.tight_layout()
        p = out_dir / f"preview_spectrum_{plot_i}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    if preview_qc:
        (out_dir / "preview_absorbance_qc.json").write_text(
            json.dumps(preview_qc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    Y, class_names = build_multilabel_matrix(dataset_dir, spec_ids_npz, bands_yaml, label_schema="spectrum")
    counts = Y.sum(axis=0)
    order = np.argsort(-counts)
    top_n = min(40, len(class_names))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * top_n)))
    names = [class_names[i] for i in order[:top_n]]
    vals = counts[order[:top_n]]
    ax.barh(range(top_n), vals[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel("positive spectra count (labels_spectrum)")
    ax.set_title("Class balance — peaks in band regions")
    fig.tight_layout()
    p2 = out_dir / "preview_class_balance.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    written.append(p2)

    Y_str, _ = build_multilabel_matrix(dataset_dir, spec_ids_npz, bands_yaml, label_schema="structure")
    stats: dict[str, object] = {
        "n_spectra": len(spec_ids_npz),
        "n_classes": len(class_names),
        "mean_labels_spectrum_per_spectrum": float(Y.sum(axis=1).mean()),
        "mean_labels_structure_per_spectrum": float(Y_str.sum(axis=1).mean()),
    }
    smarts_path = dataset_dir / "labels_structure_smarts.parquet"
    if smarts_path.exists():
        Y_sm, _ = build_multilabel_matrix(dataset_dir, spec_ids_npz, bands_yaml, label_schema="structure_smarts")
        stats["mean_labels_structure_smarts_per_spectrum"] = float(Y_sm.sum(axis=1).mean())
    (out_dir / "preview_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return written
