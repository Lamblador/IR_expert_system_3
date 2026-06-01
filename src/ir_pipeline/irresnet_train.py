"""Обучение IrResnet4 (multi-label) на model_inputs.npz."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ir_pipeline.dataset_preview import build_multilabel_matrix
from ir_pipeline.logging_utils import heartbeat, log
from ir_pipeline.models.ir_resnet4 import IrResnet4
from ir_pipeline.resnet_input import ensure_model_inputs_npz, load_model_inputs
from ir_pipeline.train_monitor import (
    CnnTrainingMonitor,
    build_irresnet_criterion,
    build_torch_optimizer,
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


if torch is not None:

    class IrDataset(Dataset):
        def __init__(self, X: np.ndarray, Y: np.ndarray, C: np.ndarray | None = None):
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)
            self.C = None if C is None else C.astype(np.float32)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, i: int):
            if self.C is None:
                return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])
            return (
                torch.from_numpy(self.X[i]),
                torch.from_numpy(self.C[i]),
                torch.from_numpy(self.Y[i]),
            )


def train_irresnet_run(
    dataset_dir: Path,
    run_dir: Path,
    bands_yaml: Path,
    train_cfg: dict[str, Any],
    *,
    device: str | None = None,
    label_schema: str = "structure",
    peak_threshold: float = 0.1,
    use_measurement_context: bool | None = None,
) -> dict[str, Any]:
    if not is_torch_available():
        raise RuntimeError("Установите torch: pip install -e '.[torch]'")
    assert torch is not None and DataLoader is not None and nn is not None

    use_ctx = bool(train_cfg.get("use_measurement_context", True) if use_measurement_context is None else use_measurement_context)

    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_model_inputs_npz(dataset_dir, peak_threshold=peak_threshold, include_context=use_ctx)
    X_in, _wn, spec_ids, X_ctx, ctx_cols = load_model_inputs(dataset_dir)
    Y, class_names = build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml, label_schema=label_schema)
    if Y.sum() < 1:
        raise RuntimeError("Нет положительных меток для обучения")

    context_dim = int(X_ctx.shape[1]) if use_ctx and X_ctx is not None else 0

    split_path = dataset_dir / "split.json"
    if split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(map(str, sp.get("train_ids", [])))
        test_ids = set(map(str, sp.get("test_ids", [])))
    else:
        rng = np.random.default_rng(int(train_cfg.get("random_seed", 42)))
        uids = np.array(sorted(set(spec_ids)))
        rng.shuffle(uids)
        split_i = int(max(1, round(float(train_cfg.get("train_frac", 0.85)) * len(uids))))
        train_ids = set(uids[:split_i].tolist())
        test_ids = set(uids[split_i:].tolist())
        if not test_ids:
            test_ids = train_ids

    tr_idx = [i for i, s in enumerate(spec_ids) if s in train_ids]
    te_idx = [i for i, s in enumerate(spec_ids) if s in test_ids]
    if len(tr_idx) < 4:
        raise RuntimeError("Слишком мало спектров в train для IrResnet4")

    X_tr, Y_tr = X_in[tr_idx], Y[tr_idx]
    X_te, Y_te = X_in[te_idx], Y[te_idx]
    C_tr = X_ctx[tr_idx] if context_dim else None
    C_te = X_ctx[te_idx] if context_dim else None

    hidden = int(train_cfg.get("ir_hidden_size", 34))
    epochs = int(train_cfg.get("torch_epochs", 30))
    bs = int(train_cfg.get("torch_batch_size", 32))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(
        f"irresnet-train: device={dev}, classes={len(class_names)}, "
        f"train={len(tr_idx)}, test={len(te_idx)}, context_dim={context_dim}"
    )

    model = IrResnet4(hidden_size=hidden, class_nums=len(class_names), context_dim=context_dim).to(dev)
    opt = build_torch_optimizer(model, train_cfg)
    criterion = build_irresnet_criterion(Y_tr, dev, train_cfg)
    monitor = CnnTrainingMonitor.from_train_cfg(
        run_dir,
        train_cfg,
        title="IrResnet4",
    )
    monitor.plot_filename = "irresnet_training_curve.png"
    log(
        f"optimizer={train_cfg.get('torch_optimizer', 'adamw')}, "
        f"loss={train_cfg.get('torch_loss', 'bce_with_logits')}, "
        f"live_plot={monitor.live_plot}"
    )

    ds_tr = IrDataset(X_tr, Y_tr, C_tr)
    ds_te = IrDataset(X_te, Y_te, C_te)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=len(ds_tr) > bs)
    dl_te = DataLoader(ds_te, batch_size=min(bs, len(ds_te)), shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    best_val = float("inf")
    best_state = None

    def _forward_batch(xb: torch.Tensor, cb: torch.Tensor | None) -> torch.Tensor:
        if context_dim:
            return model(xb, cb)
        return model(xb)

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

            model.eval()
            vl = 0.0
            vn = 0
            tp = fp = fn = 0.0
            with torch.no_grad():
                for batch in dl_te:
                    if context_dim:
                        xb, cb, yb = batch
                        cb = cb.to(dev)
                    else:
                        xb, yb = batch
                        cb = None
                    xb, yb = xb.to(dev), yb.to(dev)
                    logits = _forward_batch(xb, cb)
                    loss = criterion(logits, yb)
                    vl += float(loss.item()) * len(xb)
                    vn += len(xb)
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    tp += float(((pred == 1) & (yb == 1)).sum())
                    fp += float(((pred == 1) & (yb == 0)).sum())
                    fn += float(((pred == 0) & (yb == 1)).sum())
            val_loss = vl / max(vn, 1)
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_macro"].append(f1)
            monitor.update(
                ep + 1,
                epochs,
                {"train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": f1},
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    monitor.finalize()

    version = str(train_cfg.get("model_version", f"v0.1.0.{hidden}"))
    bundle = {
        "kind": "irresnet4_multilabel",
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
    }
    torch.save({"model_state": model.state_dict(), "meta": bundle}, run_dir / "irresnet_bundle.pt")
    (run_dir / "irresnet_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    summary = {
        "model_version": version,
        "n_classes": len(class_names),
        "best_val_loss": best_val,
        "final_val_f1": history["val_f1_macro"][-1] if history["val_f1_macro"] else None,
        "context_dim": context_dim,
        "run_dir": str(run_dir),
    }
    (run_dir / "irresnet_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("irresnet-train done")
    return summary
