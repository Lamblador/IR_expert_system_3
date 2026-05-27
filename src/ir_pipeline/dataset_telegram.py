"""Сборка telegram_arrays.npz и превью-графиков для датасета."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_pipeline.bands import load_bands
from ir_pipeline.jcamp_loader import flatten_if_link, read_jcamp_dict
from ir_pipeline.logging_utils import log
from ir_pipeline.preprocess import to_absorbance_like, wavenumbers_from_jcamp
from ir_pipeline.telegram_preprocess import (
    BOT_WAVENUMBERS,
    detect_peaks_bot,
    interpolate_bot_grid,
    jcamp_xy_to_telegram,
)


def build_telegram_arrays_from_npz(dataset_dir: Path, *, peak_threshold: float = 0.1) -> tuple[np.ndarray, list[str]]:
    """Собрать telegram tensors из spectra.npz (когда JCAMP недоступен, напр. после HF fetch)."""
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    wn = np.asarray(z["wavenumbers"], dtype=np.float64)
    Y = np.asarray(z["X_absorbance_like_interp"], dtype=np.float64)
    spec_ids = [str(s) for s in z["spectrum_id"].tolist()]
    tensors: list[np.ndarray] = []
    for i in range(len(spec_ids)):
        ab_bot = interpolate_bot_grid(wn, Y[i])
        pk = detect_peaks_bot(ab_bot, threshold=peak_threshold)
        wn_b = BOT_WAVENUMBERS.copy()
        tensors.append(np.vstack([wn_b, ab_bot, pk]).astype(np.float32))
    return np.stack(tensors, axis=0), spec_ids


def build_telegram_arrays_from_jcamp(
    meta_df: pd.DataFrame,
    raw_jcamp_dir: Path,
    *,
    peak_threshold: float = 0.1,
) -> tuple[np.ndarray, list[str]]:
    """Возвращает X_bot (N,3,L) и spectrum_ids для qc_ok строк."""
    ok = meta_df[meta_df["qc_ok"] == True].copy()  # noqa: E712
    tensors: list[np.ndarray] = []
    ids: list[str] = []
    for _, row in tqdm(ok.iterrows(), total=len(ok), desc="Telegram tensors"):
        sid = str(row["spectrum_id"])
        fp = Path(row["path"])
        if not fp.is_file():
            fp = raw_jcamp_dir / Path(row["path"]).name
        if not fp.is_file():
            continue
        try:
            d = flatten_if_link(read_jcamp_dict(fp))
            x_cm = wavenumbers_from_jcamp(np.asarray(d["x"], dtype=float), d.get("xunits"))
            y = to_absorbance_like(np.asarray(d["y"], dtype=float), d.get("yunits"))
            tg = jcamp_xy_to_telegram(x_cm, y, peak_threshold=peak_threshold)
            tensors.append(tg.tensor_3ch)
            ids.append(sid)
        except Exception as e:
            log(f"telegram tensor skip {sid}: {e}")
    if not tensors:
        raise RuntimeError("Не удалось собрать telegram tensors")
    return np.stack(tensors, axis=0).astype(np.float32), ids


def save_telegram_npz(out_dir: Path, X_bot: np.ndarray, spectrum_ids: list[str]) -> Path:
    path = out_dir / "telegram_arrays.npz"
    np.savez_compressed(
        path,
        X_bot=X_bot,
        spectrum_id=np.array(spectrum_ids, dtype=object),
        wavenumbers=BOT_WAVENUMBERS,
    )
    return path


def build_multilabel_matrix(
    dataset_dir: Path,
    spectrum_ids: list[str],
    bands_yaml: Path,
    *,
    label_schema: str = "spectrum",
) -> tuple[np.ndarray, list[str]]:
    """Y (N, C) бинарные метки: полоса присутствует, если observed_peak_cm1 задан."""
    label_file = dataset_dir / (
        "labels_spectrum.parquet" if label_schema == "spectrum" else "labels_structure.parquet"
    )
    labels = pd.read_parquet(label_file)
    bands = load_bands(bands_yaml)
    class_names = [b.band_id for b in bands]
    band_to_idx = {b: i for i, b in enumerate(class_names)}
    Y = np.zeros((len(spectrum_ids), len(class_names)), dtype=np.float32)
    sid_to_i = {s: i for i, s in enumerate(spectrum_ids)}
    pos = labels.dropna(subset=["observed_peak_cm1"])
    for _, r in pos.iterrows():
        sid = str(r["spectrum_id"])
        bid = str(r["band_id"])
        if sid not in sid_to_i or bid not in band_to_idx:
            continue
        Y[sid_to_i[sid], band_to_idx[bid]] = 1.0
    return Y, class_names


def plot_dataset_preview(
    dataset_dir: Path,
    out_dir: Path,
    bands_yaml: Path,
    *,
    n_examples: int = 3,
) -> list[Path]:
    """Примеры спектров + баланс классов."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    z = np.load(dataset_dir / "telegram_arrays.npz", allow_pickle=True)
    X_bot = z["X_bot"]
    spec_ids = [str(s) for s in z["spectrum_id"].tolist()]
    labels = pd.read_parquet(dataset_dir / "labels_spectrum.parquet")
    bands = load_bands(bands_yaml)

    for i in range(min(n_examples, len(spec_ids))):
        wn, ab, pk = X_bot[i, 0], X_bot[i, 1], X_bot[i, 2]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(wn, ab, "k-", lw=0.8)
        axes[0].set_ylabel("absorption")
        axes[0].set_title(f"Spectrum {spec_ids[i]}")
        axes[1].fill_between(wn, 0, pk * np.nanmax(ab), color="red", alpha=0.35)
        axes[1].plot(wn, ab, "k-", lw=0.6)
        axes[1].set_ylabel("peaks mask")
        sid = spec_ids[i]
        sub = labels[(labels["spectrum_id"] == sid) & labels["observed_peak_cm1"].notna()]
        for _, r in sub.head(12).iterrows():
            axes[2].axvline(float(r["observed_peak_cm1"]), color="tab:orange", alpha=0.5, lw=0.8)
        axes[2].plot(wn, ab, "k-", lw=0.6)
        axes[2].set_xlabel(r"Wavenumber (cm$^{-1}$)")
        axes[2].set_ylabel("labeled peaks")
        fig.tight_layout()
        p = out_dir / f"preview_spectrum_{i}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    Y, class_names = build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml)
    counts = Y.sum(axis=0)
    order = np.argsort(-counts)
    top_n = min(40, len(class_names))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * top_n)))
    names = [class_names[i] for i in order[:top_n]]
    vals = counts[order[:top_n]]
    ax.barh(range(top_n), vals[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel("positive spectra count")
    ax.set_title("Class balance (bands with observed peaks)")
    fig.tight_layout()
    p2 = out_dir / "preview_class_balance.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    written.append(p2)

    stats = {
        "n_spectra": len(spec_ids),
        "n_classes": len(class_names),
        "mean_labels_per_spectrum": float(Y.sum(axis=1).mean()),
    }
    (out_dir / "preview_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return written
