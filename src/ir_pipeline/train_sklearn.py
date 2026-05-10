from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

Mode = Literal["spectrum", "spectrum_structure"]


def load_training_arrays(dataset_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    X = z["X"]
    spectrum_ids = [str(i) for i in z["spectrum_id"].tolist()]
    wn = np.asarray(z["wavenumbers"])
    return X, spectrum_ids, wn


def build_feature_matrix(X: np.ndarray, spectrum_ids: list[str], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Спектр + one-hot measurement_mode + sample_state."""
    meta_f = meta.copy()
    if "qc_ok" in meta_f.columns:
        meta_f = meta_f[meta_f["qc_ok"] == True]  # noqa: E712
    meta_i = meta_f.set_index("spectrum_id")
    rows = []
    valid_ids = []
    for sid, xi in zip(spectrum_ids, X):
        if sid not in meta_i.index:
            continue
        rows.append(xi)
        valid_ids.append(sid)
    Fe = np.stack(rows, axis=0)
    sub = meta_i.loc[valid_ids].reset_index()
    d1 = pd.get_dummies(sub["measurement_mode"].fillna("unknown"), prefix="mm")
    d2 = pd.get_dummies(sub["sample_state"].fillna("unknown"), prefix="st")
    tab = pd.concat([d1, d2], axis=1)
    spec = pd.DataFrame(Fe, columns=[f"f{i}" for i in range(Fe.shape[1])], index=sub["spectrum_id"])
    feat = pd.concat([spec.reset_index(drop=True), tab.reset_index(drop=True)], axis=1)
    feat.index = sub["spectrum_id"].values
    return feat, pd.Index(sub["spectrum_id"].values)


def align_feature_row(bundle_cols: list[str], feat_partial: pd.DataFrame) -> pd.DataFrame:
    """Заполняет отсутствующие dummy-колонки нулями для инференса."""
    out = pd.DataFrame(0.0, index=feat_partial.index, columns=bundle_cols)
    shared = [c for c in feat_partial.columns if c in bundle_cols]
    if shared:
        out[shared] = feat_partial[shared].values
    return out


def train_models(
    dataset_dir: Path,
    run_dir: Path,
    mode: Mode,
    train_cfg: dict,
    random_seed: int,
    train_frac: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    label_file = dataset_dir / ("labels_spectrum.parquet" if mode == "spectrum" else "labels_structure.parquet")
    if not label_file.exists():
        raise FileNotFoundError(label_file)

    X_raw, spectrum_ids, _wn = load_training_arrays(dataset_dir)
    meta = pd.read_parquet(dataset_dir / "meta.parquet")

    feat_df, valid_index = build_feature_matrix(X_raw, spectrum_ids, meta)
    labels = pd.read_parquet(label_file)
    labels = labels[labels["spectrum_id"].isin(valid_index)]
    labels = labels.dropna(subset=["observed_peak_cm1"])

    band_ids = sorted(labels["band_id"].unique().tolist())

    split_path = dataset_dir / "split.json"
    if split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(map(str, sp.get("train_ids", [])))
        test_ids = set(map(str, sp.get("test_ids", [])))
        if not test_ids:
            test_ids = train_ids
    else:
        rng = np.random.default_rng(random_seed)
        uids = np.array(sorted(valid_index.unique().tolist()))
        rng.shuffle(uids)
        split = int(max(1, round(train_frac * len(uids))))
        train_ids = set(uids[:split].tolist())
        test_ids = set(uids[split:].tolist())
        if not test_ids:
            test_ids = train_ids

    models: dict[str, RandomForestRegressor] = {}
    metrics: dict[str, float] = {}

    n_est = int(train_cfg.get("n_estimators", 200))
    max_depth = train_cfg.get("max_depth", None)
    min_leaf = int(train_cfg.get("min_samples_leaf", 1))

    extra_cols = []
    if mode == "spectrum_structure":
        # признак: число структурно размеченных полос на спектр
        cnt = labels.groupby("spectrum_id")["band_id"].count().reindex(feat_df.index).fillna(0.0)
        feat_df = feat_df.copy()
        feat_df["structure_band_count"] = cnt.values
        extra_cols.append("structure_band_count")

    for band in band_ids:
        sub = labels[labels["band_id"] == band]
        if sub.empty:
            continue
        tr_mask = sub["spectrum_id"].isin(train_ids)
        te_mask = sub["spectrum_id"].isin(test_ids)
        sub_tr = sub.loc[tr_mask]
        sub_te = sub.loc[te_mask]
        if len(sub_tr) < 5:
            continue
        X_tr = feat_df.loc[sub_tr["spectrum_id"]].values
        y_tr = sub_tr["observed_peak_cm1"].values.astype(float)
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=random_seed,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        models[band] = rf
        if len(sub_te) >= 3:
            X_te = feat_df.loc[sub_te["spectrum_id"]].values
            y_te = sub_te["observed_peak_cm1"].values.astype(float)
            pred = rf.predict(X_te)
            metrics[band] = float(mean_absolute_error(y_te, pred))

    bundle = {
        "mode": mode,
        "models": models,
        "feature_columns": feat_df.columns.tolist(),
        "band_ids": band_ids,
        "extra_note": "spectrum_structure adds structure_band_count",
    }
    joblib.dump(bundle, run_dir / "models.joblib")

    out_metrics = {"per_band_mae": metrics, "n_train_spectra": len(train_ids), "n_test_spectra": len(test_ids)}
    (run_dir / "metrics.json").write_text(json.dumps(out_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps(train_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_metrics


def predict_peaks(
    bundle_path: Path,
    feat_row: pd.DataFrame,
    bands_order: list[str],
) -> dict[str, float]:
    bundle = joblib.load(bundle_path)
    models: dict[str, RandomForestRegressor] = bundle["models"]
    cols = bundle["feature_columns"]
    Xrow = align_feature_row(cols, feat_row).values
    pred: dict[str, float] = {}
    for b in bands_order:
        m = models.get(b)
        if m is None:
            continue
        pred[b] = float(m.predict(Xrow)[0])
    return pred
