"""Обучение IrResnet4 (multi-label) на telegram_arrays.npz."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_pipeline.dataset_telegram import build_multilabel_matrix
from ir_pipeline.logging_utils import heartbeat, log
from ir_pipeline.models.ir_resnet4 import IrResnet4

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
        def __init__(self, X: np.ndarray, Y: np.ndarray):
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, i: int):
            return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def train_irresnet_run(
    dataset_dir: Path,
    run_dir: Path,
    bands_yaml: Path,
    train_cfg: dict[str, Any],
    *,
    device: str | None = None,
    label_schema: str = "spectrum",
) -> dict[str, Any]:
    if not is_torch_available():
        raise RuntimeError("Установите torch: pip install -e '.[torch]'")
    assert torch is not None and DataLoader is not None and nn is not None

    run_dir.mkdir(parents=True, exist_ok=True)
    tg_path = dataset_dir / "telegram_arrays.npz"
    if not tg_path.exists():
        raise FileNotFoundError(f"Нет {tg_path}; пересоберите датасет (build-dataset)")

    z = np.load(tg_path, allow_pickle=True)
    X_bot = z["X_bot"]
    spec_ids = [str(s) for s in z["spectrum_id"].tolist()]
    Y, class_names = build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml, label_schema=label_schema)
    if Y.sum() < 1:
        raise RuntimeError("Нет положительных меток для обучения")

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

    X_tr, Y_tr = X_bot[tr_idx], Y[tr_idx]
    X_te, Y_te = X_bot[te_idx], Y[te_idx]

    hidden = int(train_cfg.get("ir_hidden_size", 34))
    epochs = int(train_cfg.get("torch_epochs", 30))
    bs = int(train_cfg.get("torch_batch_size", 32))
    lr = float(train_cfg.get("torch_lr", 1e-3))
    pos_weight_scale = float(train_cfg.get("pos_weight_scale", 1.0))

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"irresnet-train: device={dev}, classes={len(class_names)}, train={len(tr_idx)}, test={len(te_idx)}")

    pos = Y_tr.sum(axis=0)
    neg = len(Y_tr) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, 50.0) * pos_weight_scale, dtype=torch.float32).to(dev)

    model = IrResnet4(hidden_size=hidden, class_nums=len(class_names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    ds_tr = IrDataset(X_tr, Y_tr)
    ds_te = IrDataset(X_te, Y_te)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=len(ds_tr) > bs)
    dl_te = DataLoader(ds_te, batch_size=min(bs, len(ds_te)), shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    best_val = float("inf")
    best_state = None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with heartbeat(60.0, "irresnet training in progress..."):
        for ep in tqdm(range(epochs), desc="IrResnet epochs", unit="epoch"):
            model.train()
            tl = 0.0
            tn = 0
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
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
                for xb, yb in dl_te:
                    xb, yb = xb.to(dev), yb.to(dev)
                    logits = model(xb)
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
            log(f"epoch {ep + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={f1:.3f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    version = str(train_cfg.get("model_version", f"v0.1.0.{hidden}"))
    bundle = {
        "kind": "irresnet4_multilabel",
        "model_version": version,
        "hidden_size": hidden,
        "class_names": class_names,
        "label_schema": label_schema,
        "history": history,
        "device_trained": dev,
    }
    torch.save({"model_state": model.state_dict(), "meta": bundle}, run_dir / "irresnet_bundle.pt")
    (run_dir / "irresnet_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].legend()
    axes[0].set_title("BCE loss")
    axes[1].plot(history["val_f1_macro"], color="tab:green")
    axes[1].set_title("Val F1 (macro approx)")
    fig.tight_layout()
    fig.savefig(run_dir / "irresnet_training_curve.png", dpi=140)
    plt.close(fig)

    summary = {
        "model_version": version,
        "n_classes": len(class_names),
        "best_val_loss": best_val,
        "final_val_f1": history["val_f1_macro"][-1] if history["val_f1_macro"] else None,
        "run_dir": str(run_dir),
    }
    (run_dir / "irresnet_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("irresnet-train done")
    return summary
