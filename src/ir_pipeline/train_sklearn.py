from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

Mode = Literal["spectrum", "spectrum_structure"]


def _event(message: str) -> None:
    tqdm.write(f"[ir-pipeline] {message}")


def _effective_rf_backend(train_cfg: dict[str, Any]) -> Literal["auto", "sklearn", "cuml"]:
    env = os.environ.get("IR_RF_BACKEND", "").strip().lower()
    if env in ("auto", "sklearn", "cuml"):
        return env  # type: ignore[return-value]
    legacy = os.environ.get("IR_USE_CUML", "").strip().lower()
    if legacy in ("0", "false", "no", "cpu", "sklearn"):
        return "sklearn"
    if legacy in ("1", "true", "yes", "gpu", "cuml"):
        return "cuml"
    raw = str(train_cfg.get("rf_backend", "auto")).strip().lower()
    if raw in ("auto", "sklearn", "cuml"):
        return raw  # type: ignore[return-value]
    return "auto"


def _cuda_device_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _try_import_cuml_rf():
    from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor

    return CuMLRandomForestRegressor


def _resolve_rf_backend(policy: Literal["auto", "sklearn", "cuml"]) -> tuple[str, Any]:
    """Возвращает ('sklearn'|'cuml', класс регрессора)."""
    if policy == "sklearn":
        return "sklearn", RandomForestRegressor

    if policy == "cuml":
        try:
            CuMLRF = _try_import_cuml_rf()
        except ImportError as e:
            raise RuntimeError(
                "Выбран rf_backend=cuml / IR_RF_BACKEND=cuml, но пакет cuml не установлен. "
                "Установите: pip install -e \".[cuml]\" или cuml-cu12 (NVIDIA CUDA, см. README)."
            ) from e
        if _cuda_device_count() < 1:
            raise RuntimeError(
                "Выбран rf_backend=cuml, но CUDA GPU недоступна (cupy: getDeviceCount()==0). "
                "Включите GPU runtime или переключите rf_backend на sklearn/auto."
            )
        return "cuml", CuMLRF

    # auto
    if _cuda_device_count() < 1:
        return "sklearn", RandomForestRegressor
    try:
        return "cuml", _try_import_cuml_rf()
    except ImportError:
        return "sklearn", RandomForestRegressor


def _maybe_configure_cuml_numpy_output(backend: str) -> None:
    if backend != "cuml":
        return
    try:
        import cuml

        cuml.set_global_output_type("numpy")
    except Exception:
        pass


def _make_rf(
    backend: str,
    rf_cls: Any,
    *,
    n_estimators: int,
    max_depth: Any,
    min_samples_leaf: int,
    random_seed: int,
) -> Any:
    if backend == "sklearn":
        return rf_cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_seed,
            n_jobs=-1,
        )
    depth = max_depth if max_depth is not None else -1
    return rf_cls(
        n_estimators=n_estimators,
        max_depth=depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=2,
        random_state=random_seed,
        max_features=1.0,
    )


def _as_numpy(pred: Any) -> np.ndarray:
    try:
        import cupy as cp

        if isinstance(pred, cp.ndarray):
            return cp.asnumpy(pred)
    except Exception:
        pass
    return np.asarray(pred)


def infer_rf_backend(bundle: dict[str, Any]) -> Literal["sklearn", "cuml"]:
    b = bundle.get("rf_backend")
    if b in ("cuml", "sklearn"):
        return b  # type: ignore[return-value]
    sample_model = next(iter(bundle.get("models", {}).values()), None)
    if sample_model is not None and "cuml" in type(sample_model).__module__:
        return "cuml"
    return "sklearn"


def rf_predict_values(backend: str, model: Any, X: np.ndarray) -> np.ndarray:
    if backend == "cuml":
        X = np.asarray(X, dtype=np.float32)
    out = model.predict(X)
    return _as_numpy(out).reshape(-1)


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
    train_cfg: dict[str, Any],
    random_seed: int,
    train_frac: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = _effective_rf_backend(train_cfg)
    backend, rf_cls = _resolve_rf_backend(policy)
    _maybe_configure_cuml_numpy_output(backend)

    _event(f"train start: dataset={dataset_dir}, run_dir={run_dir}, mode={mode}")
    if backend == "cuml":
        _event(f"RandomForest: rf_backend=cuml (RAPIDS cuML, CUDA GPU), policy={policy}")
    else:
        _event(f"RandomForest: rf_backend=sklearn (CPU), policy={policy}")

    label_file = dataset_dir / ("labels_spectrum.parquet" if mode == "spectrum" else "labels_structure.parquet")
    if not label_file.exists():
        raise FileNotFoundError(label_file)

    _event("loading spectra.npz and meta.parquet")
    X_raw, spectrum_ids, _wn = load_training_arrays(dataset_dir)
    meta = pd.read_parquet(dataset_dir / "meta.parquet")

    _event("building feature matrix")
    feat_df, valid_index = build_feature_matrix(X_raw, spectrum_ids, meta)
    _event(f"reading labels: {label_file}")
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

    models: dict[str, Any] = {}
    metrics: dict[str, float] = {}

    n_est = int(train_cfg.get("n_estimators", 200))
    max_depth = train_cfg.get("max_depth", None)
    min_leaf = int(train_cfg.get("min_samples_leaf", 1))
    _event(
        f"training RandomForest models: bands={len(band_ids)}, n_estimators={n_est}, train_ids={len(train_ids)}, test_ids={len(test_ids)}"
    )

    extra_cols = []
    if mode == "spectrum_structure":
        # признак: число структурно размеченных полос на спектр
        cnt = labels.groupby("spectrum_id")["band_id"].count().reindex(feat_df.index).fillna(0.0)
        feat_df = feat_df.copy()
        feat_df["structure_band_count"] = cnt.values
        extra_cols.append("structure_band_count")

    cast_f32 = backend == "cuml"

    for band in tqdm(band_ids, desc="Train RF per band", unit="band"):
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
        y_tr = sub_tr["observed_peak_cm1"].values.astype(np.float64 if not cast_f32 else np.float32)
        if cast_f32:
            X_tr = X_tr.astype(np.float32, copy=False)
        rf = _make_rf(
            backend,
            rf_cls,
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_seed=random_seed,
        )
        rf.fit(X_tr, y_tr)
        models[band] = rf
        if len(sub_te) >= 3:
            X_te = feat_df.loc[sub_te["spectrum_id"]].values
            if cast_f32:
                X_te = X_te.astype(np.float32, copy=False)
            y_te = sub_te["observed_peak_cm1"].values.astype(np.float64 if not cast_f32 else np.float32)
            pred = rf.predict(X_te)
            pred = _as_numpy(pred).reshape(-1)
            metrics[band] = float(mean_absolute_error(np.asarray(y_te).reshape(-1), pred))

    bundle = {
        "mode": mode,
        "models": models,
        "feature_columns": feat_df.columns.tolist(),
        "band_ids": band_ids,
        "extra_note": "spectrum_structure adds structure_band_count",
        "rf_backend": backend,
    }
    _event(f"trained models: {len(models)}; writing models.joblib")
    joblib.dump(bundle, run_dir / "models.joblib")

    out_metrics: dict[str, Any] = {"per_band_mae": metrics, "n_train_spectra": len(train_ids), "n_test_spectra": len(test_ids)}
    _event("writing metrics.json and training_config.json")
    (run_dir / "metrics.json").write_text(json.dumps(out_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps(train_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    _event("train done")
    return out_metrics


def predict_peaks(
    bundle_path: Path,
    feat_row: pd.DataFrame,
    bands_order: list[str],
) -> dict[str, float]:
    bundle = joblib.load(bundle_path)
    models: dict[str, Any] = bundle["models"]
    backend = infer_rf_backend(bundle)
    cols = bundle["feature_columns"]
    Xrow = align_feature_row(cols, feat_row).values
    pred: dict[str, float] = {}
    for b in bands_order:
        m = models.get(b)
        if m is None:
            continue
        pred[b] = float(rf_predict_values(backend, m, Xrow)[0])
    return pred
