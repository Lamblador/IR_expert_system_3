from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_pipeline.bands import load_bands
from ir_pipeline.jcamp_loader import extract_measurement_mode, extract_sample_state, flatten_if_link, read_jcamp_dict
from ir_pipeline.preprocess import (
    intensity_bucket_from_peak,
    preprocess_to_grid,
    to_absorbance_like,
    wavenumbers_from_jcamp,
)


def _event(message: str) -> None:
    tqdm.write(f"[ir-pipeline] {message}")


def format_ir_summary(
    preds: dict[str, float],
    meta_line: str,
    bands_yaml: Path,
    wavenumbers: np.ndarray | None = None,
    y_norm: np.ndarray | None = None,
    widths: dict[str, str] | None = None,
) -> str:
    bands = {b.band_id: b for b in load_bands(bands_yaml)}
    parts = []
    for bid, cm in sorted(preds.items(), key=lambda kv: kv[1]):
        b = bands.get(bid)
        ic = (
            intensity_bucket_from_peak(y_norm, wavenumbers, cm)
            if (wavenumbers is not None and y_norm is not None)
            else "u"
        )
        w_extra = (widths or {}).get(bid, "")
        if not w_extra and bid == "oh_alcohol" and ic in {"vw", "w", "m"}:
            w_extra = "wide"
        width_txt = f", {w_extra}" if w_extra else ""
        parts.append(f"{cm:.0f} ({ic}{width_txt})")
    body = ", ".join(parts) if parts else "(no predictions)"
    return f"{meta_line}: {body}"


def plot_predictions(
    wn: np.ndarray,
    y_norm: np.ndarray,
    preds: dict[str, float],
    bands_yaml: Path,
    out_png: Path,
    title: str,
):
    bands = {b.band_id: b for b in load_bands(bands_yaml)}
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wn, y_norm, color="black", lw=1.0)
    for bid, cm in preds.items():
        b = bands.get(bid)
        color = "tab:red"
        ax.axvline(cm, color=color, ls="--", lw=1)
        if b:
            ax.axvspan(b.range_min_cm1, b.range_max_cm1, alpha=0.12, color=color)
        ax.text(cm, np.nanmax(y_norm) * 0.95, bid, rotation=90, va="top", fontsize=8)
    ax.set_xlim(np.nanmax(wn), np.nanmin(wn))
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized absorbance-like")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def predict_file_visualize(
    jcamp_path: Path,
    bands_yaml: Path,
    bundle_path: Path,
    out_dir: Path,
    torch_bundle: Path | None = None,
):
    from ir_pipeline.train_sklearn import predict_peaks, build_feature_matrix
    import joblib

    _event(f"predict start: {jcamp_path}")
    _event("reading and preprocessing JCAMP")
    d = flatten_if_link(read_jcamp_dict(jcamp_path))
    xunits = d.get("xunits")
    yunits = d.get("yunits")
    x_cm = wavenumbers_from_jcamp(np.asarray(d["x"], dtype=float), xunits)
    y_abs = to_absorbance_like(np.asarray(d["y"], dtype=float), yunits)
    grid = preprocess_to_grid(x_cm, y_abs)

    sid = "infer"

    meta_row = pd.DataFrame(
        [
            {
                "spectrum_id": sid,
                "qc_ok": True,
                "measurement_mode": extract_measurement_mode(d),
                "sample_state": extract_sample_state(d),
                "cas": str(d.get("cas registry no", "")),
                "title": str(d.get("title", "")),
            }
        ]
    )

    Xi = grid.absorbance_normalized.astype(np.float32)[None, :]

    if torch_bundle is not None:
        from ir_pipeline.torch_train import predict_torch_peaks

        _event(f"loading torch bundle: {torch_bundle}")
        preds_raw = predict_torch_peaks(torch_bundle, Xi[0])
        bands_known = {b.band_id for b in load_bands(bands_yaml)}
        preds = {k: v for k, v in preds_raw.items() if k in bands_known}
    else:
        _event("building sklearn feature row")
        feat_df, _ = build_feature_matrix(Xi, [sid], meta_row)

        _event(f"loading sklearn bundle: {bundle_path}")
        bundle = joblib.load(bundle_path)
        mode = bundle.get("mode", "spectrum")
        if mode == "spectrum_structure":
            feat_df = feat_df.copy()
            feat_df["structure_band_count"] = float(max(1, len(bundle.get("models", {}))))

        bands_order = list(bundle["models"].keys())
        _event(f"predicting peaks: models={len(bands_order)}")
        preds = predict_peaks(bundle_path, feat_df.loc[[sid]], bands_order)

    technique = str(meta_row.iloc[0]["measurement_mode"])
    meta_line = f"FT-IR ({technique}, 1/cm)"
    summary = format_ir_summary(
        preds,
        meta_line,
        bands_yaml,
        wavenumbers=grid.wavenumbers,
        y_norm=grid.absorbance_normalized,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _event("writing summary.txt")
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    _event("writing spectrum_pred.png")
    plot_predictions(grid.wavenumbers, grid.absorbance_normalized, preds, bands_yaml, out_dir / "spectrum_pred.png", title=str(jcamp_path.name))
    _event("predict done")
