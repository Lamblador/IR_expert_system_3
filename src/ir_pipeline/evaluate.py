from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ir_pipeline.train_sklearn import build_feature_matrix, load_training_arrays


def evaluate_run(dataset_dir: Path, run_dir: Path, mode: str) -> dict:
    bundle_path = run_dir / "models.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    import joblib

    bundle = joblib.load(bundle_path)

    label_file = dataset_dir / ("labels_spectrum.parquet" if mode == "spectrum" else "labels_structure.parquet")
    labels = pd.read_parquet(label_file).dropna(subset=["observed_peak_cm1"])

    X_raw, spectrum_ids, _ = load_training_arrays(dataset_dir)
    meta = pd.read_parquet(dataset_dir / "meta.parquet")
    feat_df, _valid_index = build_feature_matrix(X_raw, spectrum_ids, meta)

    if mode == "spectrum_structure" and "structure_band_count" in bundle["feature_columns"]:
        cnt = labels.groupby("spectrum_id")["band_id"].count().reindex(feat_df.index).fillna(0.0)
        feat_df = feat_df.copy()
        feat_df["structure_band_count"] = cnt.values

    cols = bundle["feature_columns"]
    models = bundle["models"]

    split_path = dataset_dir / "split.json"
    test_ids: set[str] | None = None
    if split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        cand = set(map(str, sp.get("test_ids", [])))
        test_ids = cand if len(cand) > 0 else None

    rows = []
    for band, model in models.items():
        sub = labels[labels["band_id"] == band]
        if sub.empty:
            continue
        if test_ids is not None:
            sub = sub[sub["spectrum_id"].isin(test_ids)]
        if sub.empty:
            continue
        X = feat_df.loc[sub["spectrum_id"]][cols].values
        y = sub["observed_peak_cm1"].values.astype(float)
        pred = model.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        lo = sub["region_min_cm1"].values.astype(float)
        hi = sub["region_max_cm1"].values.astype(float)
        hit = float(np.mean((pred >= lo) & (pred <= hi)))
        rows.append({"band_id": band, "mae": mae, "hit_rate_in_region": hit, "n": int(len(sub))})

    summary = {"mode": mode, "bands": rows, "eval_subset": "test_ids_from_split_json" if test_ids else "all_labeled_rows"}
    (run_dir / "eval_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
