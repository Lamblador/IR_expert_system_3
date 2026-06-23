"""Обучение IrResnet4 / KAN (multi-label) на model_inputs.npz."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import classification_report, f1_score, label_ranking_average_precision_score
from tqdm import tqdm

from ir_pipeline.dataset_preview import build_multilabel_matrix
from ir_pipeline.dataset_split import load_split_ids
from ir_pipeline.logging_utils import heartbeat, log
from ir_pipeline.measurement_context import unknown_context_vector
from ir_pipeline.models.model_factory import build_spectrum_model, count_parameters
from ir_pipeline.resnet_input import ensure_model_inputs_npz, load_model_inputs
from ir_pipeline.spectrum_augment import SpectrumAugmentor, augment_config_from_train_cfg
from ir_pipeline.train_monitor import (
    CnnTrainingMonitor,
    IrResnetTrainingPlotter,
    build_irresnet_criterion,
    build_torch_optimizer,
    build_torch_scheduler,
)
from ir_pipeline.train_sampling import (
    build_sample_weights,
    build_weighted_sampler,
    compute_class_pos_weights,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore


def is_torch_available() -> bool:
    return torch is not None


def multilabel_f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.astype(np.int32)
    yp = y_pred.astype(np.int32)
    return {
        "f1_micro": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(yt, yp, average="samples", zero_division=0)),
    }


def multilabel_lrap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(label_ranking_average_precision_score(y_true, y_prob))


def tune_prediction_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1_weighted",
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Подбор глобального порога на val."""
    if grid is None:
        grid = np.arange(0.2, 0.81, 0.05)
    best_t = 0.5
    best_s = -1.0
    for t in grid:
        pred = (y_prob > t).astype(np.int32)
        if metric == "f1_macro":
            s = float(f1_score(y_true, pred, average="macro", zero_division=0))
        else:
            s = float(f1_score(y_true, pred, average="weighted", zero_division=0))
        if s > best_s:
            best_s = s
            best_t = float(t)
    return best_t, best_s


def _eval_loader(
    model: Any,
    dl: Any,
    criterion: Any,
    dev: str,
    context_dim: int,
    forward_fn: Any,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float], float, np.ndarray, np.ndarray]:
    assert torch is not None
    model.eval()
    vl = 0.0
    vn = 0
    preds_list: list[np.ndarray] = []
    prob_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            if context_dim:
                xb, cb, yb = batch
                cb = cb.to(dev)
            else:
                xb, yb = batch
                cb = None
            xb, yb = xb.to(dev), yb.to(dev)
            logits = forward_fn(xb, cb)
            loss = criterion(logits, yb)
            vl += float(loss.item()) * len(xb)
            vn += len(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            prob_list.append(prob)
            preds_list.append((prob > threshold).astype(np.float32))
            labels_list.append(yb.cpu().numpy())
    val_loss = vl / max(vn, 1)
    y_true = np.vstack(labels_list) if labels_list else np.zeros((0, 1), dtype=np.float32)
    y_prob = np.vstack(prob_list) if prob_list else np.zeros((0, 1), dtype=np.float32)
    y_pred = np.vstack(preds_list) if preds_list else np.zeros((0, 1), dtype=np.float32)
    f1s = multilabel_f1_scores(y_true, y_pred)
    f1s["lrap"] = multilabel_lrap(y_true, y_prob)
    return val_loss, f1s, f1s["lrap"], y_true, y_prob


if torch is not None:

    class IrDataset(Dataset):
        def __init__(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            C: np.ndarray | None = None,
            *,
            training: bool = False,
            context_dropout_prob: float = 0.0,
            ctx_unknown: np.ndarray | None = None,
            augmentor: SpectrumAugmentor | None = None,
            rng: np.random.Generator | None = None,
        ):
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)
            self.C = None if C is None else C.astype(np.float32)
            self.training = training
            self.context_dropout_prob = float(context_dropout_prob)
            self.ctx_unknown = ctx_unknown
            self.augmentor = augmentor
            self.rng = rng or np.random.default_rng(0)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, i: int):
            x = self.X[i].copy()
            if self.training and self.augmentor is not None:
                x = self.augmentor.apply(x, self.rng)
            y = self.Y[i]
            if self.C is None:
                return torch.from_numpy(x), torch.from_numpy(y)
            c = self.C[i].copy()
            if self.training and self.context_dropout_prob > 0 and self.ctx_unknown is not None:
                if self.rng.random() < self.context_dropout_prob:
                    c = self.ctx_unknown.copy()
            return (
                torch.from_numpy(x),
                torch.from_numpy(c),
                torch.from_numpy(y),
            )


def _resolve_label_schema(train_cfg: dict[str, Any], label_schema: str | None) -> str:
    if label_schema:
        return label_schema
    return str(train_cfg.get("label_schema", "structure_smarts"))


def _indices_for_ids(spec_ids: list[str], id_set: set[str]) -> list[int]:
    return [i for i, s in enumerate(spec_ids) if s in id_set]


def train_irresnet_run(
    dataset_dir: Path,
    run_dir: Path,
    bands_yaml: Path,
    train_cfg: dict[str, Any],
    *,
    device: str | None = None,
    label_schema: str | None = None,
    peak_threshold: float = 0.1,
    use_measurement_context: bool | None = None,
    on_epoch_end: Callable[[int, int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    if not is_torch_available():
        raise RuntimeError("Установите torch: pip install -e '.[torch]'")
    assert torch is not None and DataLoader is not None and nn is not None

    label_schema = _resolve_label_schema(train_cfg, label_schema)
    use_ctx = bool(
        train_cfg.get("use_measurement_context", True)
        if use_measurement_context is None
        else use_measurement_context
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_model_inputs_npz(dataset_dir, peak_threshold=peak_threshold, include_context=use_ctx)
    X_in, _wn, spec_ids, X_ctx, ctx_cols = load_model_inputs(dataset_dir)
    Y, class_names = build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml, label_schema=label_schema)
    sid_to_row = {s: i for i, s in enumerate(spec_ids)}
    for i, sid in enumerate(spec_ids):
        if "_aug" in sid:
            orig = sid.split("_aug")[0]
            if orig in sid_to_row:
                Y[i] = Y[sid_to_row[orig]]
    if Y.sum() < 1:
        raise RuntimeError("Нет положительных меток для обучения")

    context_dim = int(X_ctx.shape[1]) if use_ctx and X_ctx is not None else 0
    ctx_unknown = unknown_context_vector(ctx_cols) if context_dim else None

    splits = load_split_ids(dataset_dir)
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    if not train_ids:
        rng = np.random.default_rng(int(train_cfg.get("random_seed", 42)))
        uids = np.array(sorted(set(spec_ids)))
        rng.shuffle(uids)
        split_i = int(max(1, round(float(train_cfg.get("train_frac", 0.85)) * len(uids))))
        train_ids = set(uids[:split_i].tolist())
        test_ids = set(uids[split_i:].tolist()) or train_ids
        val_ids = set()
        log("split.json отсутствует — fallback 85/15 без val")

    if not val_ids:
        val_ids = test_ids
        log("val_ids пуст — для early stop используется test (рекомендуется split v2)")

    tr_idx = _indices_for_ids(spec_ids, train_ids)
    va_idx = _indices_for_ids(spec_ids, val_ids)
    te_idx = _indices_for_ids(spec_ids, test_ids)
    if len(tr_idx) < 4:
        raise RuntimeError("Слишком мало спектров в train для IrResnet4")

    X_tr, Y_tr = X_in[tr_idx], Y[tr_idx]
    X_va, Y_va = X_in[va_idx], Y[va_idx]
    X_te, Y_te = X_in[te_idx], Y[te_idx]
    C_tr = X_ctx[tr_idx] if context_dim else None
    C_va = X_ctx[va_idx] if context_dim else None
    C_te = X_ctx[te_idx] if context_dim else None

    hidden = int(train_cfg.get("ir_hidden_size", 34))
    model_family = str(train_cfg.get("model_family", "irresnet4"))
    epochs = int(train_cfg.get("torch_epochs", 30))
    bs = int(train_cfg.get("torch_batch_size", 32))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(
        f"irresnet-train: family={model_family}, device={dev}, schema={label_schema}, "
        f"classes={len(class_names)}, train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}, "
        f"context_dim={context_dim}, hidden={hidden}"
    )

    model = build_spectrum_model(
        model_family,
        hidden_size=hidden,
        class_nums=len(class_names),
        context_dim=context_dim,
        train_cfg=train_cfg,
    ).to(dev)
    n_params, n_trainable = count_parameters(model)
    if model_family == "kan_full":
        ref = build_spectrum_model(
            "irresnet4",
            hidden_size=hidden,
            class_nums=len(class_names),
            context_dim=context_dim,
            train_cfg=train_cfg,
        )
        ref_n, _ = count_parameters(ref)
        if ref_n > 0 and n_params > ref_n * 1.15:
            log(
                f"предупреждение: kan_full params={n_params} > +15% от irresnet4 ({ref_n}); "
                "рассмотрите kan_full_hidden_size или kan_grid_size"
            )
    opt = build_torch_optimizer(model, train_cfg)
    scheduler = build_torch_scheduler(opt, train_cfg)
    class_pw = compute_class_pos_weights(
        Y_tr,
        mode=str(train_cfg.get("pos_weight_mode", "neg_pos")),
        pos_weight_scale=float(train_cfg.get("pos_weight_scale", 1.0)),
    )
    criterion = build_irresnet_criterion(Y_tr, dev, train_cfg)

    monitor = CnnTrainingMonitor.from_train_cfg(run_dir, train_cfg, title=f"train:{model_family}")
    monitor.plot_filename = "irresnet_training_curve.png"

    rng_aug = np.random.default_rng(int(train_cfg.get("random_seed", 42)))
    augmentor = None
    if train_cfg.get("augment_train"):
        augmentor = SpectrumAugmentor(augment_config_from_train_cfg(train_cfg))

    ds_tr = IrDataset(
        X_tr,
        Y_tr,
        C_tr,
        training=True,
        context_dropout_prob=float(train_cfg.get("context_dropout_prob", 0.0)),
        ctx_unknown=ctx_unknown,
        augmentor=augmentor,
        rng=rng_aug,
    )
    ds_va = IrDataset(X_va, Y_va, C_va, training=False)
    ds_te = IrDataset(X_te, Y_te, C_te, training=False)

    use_wrs = bool(train_cfg.get("use_weighted_sampler", False))
    if use_wrs:
        sw = build_sample_weights(Y_tr, class_pw)
        sampler = build_weighted_sampler(sw)
        dl_tr = DataLoader(ds_tr, batch_size=bs, sampler=sampler, drop_last=len(ds_tr) > bs)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=len(ds_tr) > bs)
    dl_va = DataLoader(ds_va, batch_size=min(bs, max(len(ds_va), 1)), shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=min(bs, max(len(ds_te), 1)), shuffle=False)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_micro": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
        "val_f1_samples": [],
        "val_lrap": [],
    }
    early_metric = str(train_cfg.get("early_stop_metric", "val_f1_weighted"))
    patience = int(train_cfg.get("early_stop_patience", 0))
    min_epochs = int(train_cfg.get("early_stop_min_epochs", 1))
    best_score = float("-inf")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    def _forward_batch(xb: torch.Tensor, cb: torch.Tensor | None) -> torch.Tensor:
        if context_dim:
            return model(xb, cb)
        return model(xb)

    def _epoch_score(f1s: dict[str, float]) -> float:
        if "lrap" in early_metric:
            return f1s.get("lrap", 0.0)
        key = early_metric.replace("val_", "")
        if key in f1s:
            return f1s[key]
        if key.startswith("f1_") and key in f1s:
            return f1s[key]
        fk = f"f1_{key}" if not key.startswith("f1_") else key
        return f1s.get(fk, f1s.get("f1_weighted", 0.0))

    stopped_early = False
    train_t0 = time.perf_counter()
    with heartbeat(60.0, "irresnet training in progress..."):
        for ep in tqdm(range(epochs), desc="IrResnet epochs", unit="epoch"):
            model.train()
            tl = 0.0
            tn = 0
            for batch in dl_tr:
                if context_dim:
                    xb, cb, yb = batch
                    cb = cb.to(dev)
                else:
                    xb, yb = batch
                    cb = None
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = criterion(_forward_batch(xb, cb), yb)
                loss.backward()
                opt.step()
                tl += float(loss.item()) * len(xb)
                tn += len(xb)
            train_loss = tl / max(tn, 1)

            val_loss, f1s, lrap, _, _ = _eval_loader(
                model, dl_va, criterion, dev, context_dim, _forward_batch
            )
            if scheduler is not None:
                scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_micro"].append(f1s["f1_micro"])
            history["val_f1_macro"].append(f1s["f1_macro"])
            history["val_f1_weighted"].append(f1s["f1_weighted"])
            history["val_f1_samples"].append(f1s["f1_samples"])
            history["val_lrap"].append(lrap)
            ep_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_weighted": f1s["f1_weighted"],
                "val_f1_macro": f1s["f1_macro"],
                "val_lrap": lrap,
            }
            monitor.update(ep + 1, epochs, ep_metrics)
            if on_epoch_end:
                on_epoch_end(ep + 1, epochs, ep_metrics)

            epoch_score = _epoch_score(f1s)
            if epoch_score > best_score:
                best_score = epoch_score
                best_val_loss = val_loss
                best_epoch = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if patience > 0 and (ep + 1) >= min_epochs and epochs_no_improve >= patience:
                log(f"early stop at epoch {ep + 1}, best={best_epoch}, {early_metric}={best_score:.4f}")
                stopped_early = True
                break

    train_wall_time_sec = time.perf_counter() - train_t0

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"best checkpoint: epoch={best_epoch}, {early_metric}={best_score:.4f}, val_loss={best_val_loss:.4f}")

    monitor.finalize()

    _, _, _, y_va_true, y_va_prob = _eval_loader(
        model, dl_va, criterion, dev, context_dim, _forward_batch, threshold=0.5
    )
    thresh, val_score_at_thresh = tune_prediction_threshold(
        y_va_true,
        y_va_prob,
        metric="f1_weighted",
    )
    if train_cfg.get("prediction_threshold") is not None:
        thresh = float(train_cfg["prediction_threshold"])

    _, test_f1s, test_lrap, y_te_true, y_te_prob = _eval_loader(
        model, dl_te, criterion, dev, context_dim, _forward_batch, threshold=thresh
    )
    y_te_pred = (y_te_prob > thresh).astype(np.float32)
    test_f1s = multilabel_f1_scores(y_te_true, y_te_pred)
    test_f1s["lrap"] = test_lrap

    test_report = classification_report(
        y_te_true,
        y_te_pred,
        target_names=class_names,
        zero_division=0,
    )
    (run_dir / "irresnet_classification_report.txt").write_text(test_report, encoding="utf-8")

    version = str(train_cfg.get("model_version", f"v0.1.0.{hidden}"))
    bundle = {
        "kind": "irresnet4_multilabel",
        "model_family": model_family,
        "model_version": version,
        "hidden_size": hidden,
        "class_names": class_names,
        "label_schema": label_schema,
        "history": history,
        "device_trained": dev,
        "context_dim": context_dim,
        "context_columns": ctx_cols if context_dim else [],
        "grid": {"min": 400.0, "max": 4000.0, "step": 2.0},
        "use_measurement_context": bool(context_dim),
        "prediction_threshold": thresh,
        "early_stop_metric": early_metric,
        "stopped_early": stopped_early,
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "kan_grid_size": train_cfg.get("kan_grid_size"),
        "kan_full_hidden_size": train_cfg.get("kan_full_hidden_size", hidden),
        "train_wall_time_sec": train_wall_time_sec,
    }
    torch.save({"model_state": model.state_dict(), "meta": bundle}, run_dir / "irresnet_bundle.pt")
    (run_dir / "irresnet_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    summary = {
        "model_family": model_family,
        "model_version": version,
        "n_classes": len(class_names),
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "train_wall_time_sec": train_wall_time_sec,
        "label_schema": label_schema,
        "best_epoch": best_epoch,
        "early_stop_metric": early_metric,
        "best_val_score": best_score,
        "best_val_loss": best_val_loss,
        "prediction_threshold": thresh,
        "val_threshold_tuning_score": val_score_at_thresh,
        "test_f1_micro": test_f1s["f1_micro"],
        "test_f1_macro": test_f1s["f1_macro"],
        "test_f1_weighted": test_f1s["f1_weighted"],
        "test_f1_samples": test_f1s["f1_samples"],
        "test_lrap": test_f1s["lrap"],
        "n_train": len(tr_idx),
        "n_val": len(va_idx),
        "n_test": len(te_idx),
        "context_dim": context_dim,
        "use_measurement_context": bool(context_dim),
        "stopped_early": stopped_early,
        "run_dir": str(run_dir),
    }
    (run_dir / "irresnet_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("irresnet-train done")
    return summary


class IrResnetTrainer:
    """Обёртка для Colab: делегирует в train_irresnet_run."""

    def __init__(
        self,
        dataset_dir: Path,
        run_dir: Path,
        bands_yaml: Path,
        train_cfg: dict[str, Any],
        **kwargs: Any,
    ):
        self.dataset_dir = dataset_dir
        self.run_dir = run_dir
        self.bands_yaml = bands_yaml
        self.train_cfg = train_cfg
        self.kwargs = kwargs
        self.plotter: IrResnetTrainingPlotter | None = None
        self.summary: dict[str, Any] | None = None

    def fit(self) -> dict[str, Any]:
        self.summary = train_irresnet_run(
            self.dataset_dir,
            self.run_dir,
            self.bands_yaml,
            self.train_cfg,
            **self.kwargs,
        )
        self.plotter = IrResnetTrainingPlotter.from_train_cfg(
            self.run_dir, self.train_cfg, title="IrResnet4"
        )
        return self.summary
