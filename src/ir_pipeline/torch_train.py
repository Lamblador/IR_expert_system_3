"""Обучение многозадачной 1D CNN по позициям пиков (см⁻¹) — опционально PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore

SCALE_PEAK_CM = 4000.0


def is_torch_available() -> bool:
    return torch is not None


if torch is not None:

    class PeakMultiDataset(Dataset):
        def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray):
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)
            self.M = M.astype(np.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = torch.from_numpy(self.X[i]).unsqueeze(0)
            y = torch.from_numpy(self.Y[i])
            m = torch.from_numpy(self.M[i])
            return x, y, m

    class ConvPeakMultitask(nn.Module):
        """Компактная 1D CNN → вектор предсказаний по полосам."""

        def __init__(self, seq_len: int, n_bands: int, channels: tuple[int, ...] = (32, 64, 128)):
            super().__init__()
            c1, c2, c3 = channels
            self.net = nn.Sequential(
                nn.Conv1d(1, c1, kernel_size=7, padding=3),
                nn.BatchNorm1d(c1),
                nn.ReLU(inplace=True),
                nn.Conv1d(c1, c2, kernel_size=5, padding=2),
                nn.BatchNorm1d(c2),
                nn.ReLU(inplace=True),
                nn.Conv1d(c2, c3, kernel_size=5, padding=2),
                nn.BatchNorm1d(c3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c3, max(64, n_bands * 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(64, n_bands * 2), n_bands),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.net(x))

    def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss_elem = F.smooth_l1_loss(pred, target, reduction="none")
        denom = mask.sum().clamp_min(1.0)
        return (loss_elem * mask).sum() / denom

else:
    PeakMultiDataset = None  # type: ignore
    ConvPeakMultitask = None  # type: ignore

    def _masked_smooth_l1(pred, target, mask):  # type: ignore
        raise RuntimeError("torch не установлен")


def _load_arrays(dataset_dir: Path) -> tuple[np.ndarray, list[str]]:
    z = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    return z["X"], z["spectrum_id"].tolist()


def _labels_path(dataset_dir: Path, mode: str) -> Path:
    return dataset_dir / ("labels_spectrum.parquet" if mode == "spectrum" else "labels_structure.parquet")


def build_wide_targets(dataset_dir: Path, mode: str, spectrum_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Цели Y и маска M (1 там, где есть наблюдённый пик). Шкала Y / SCALE_PEAK_CM."""
    labels = pd.read_parquet(_labels_path(dataset_dir, mode))
    labels = labels.dropna(subset=["observed_peak_cm1"])
    wide = labels.pivot_table(index="spectrum_id", columns="band_id", values="observed_peak_cm1", aggfunc="mean")
    band_order = sorted(wide.columns.astype(str).tolist())
    wide = wide.reindex(columns=band_order)
    wide = wide.reindex([str(s) for s in spectrum_ids])
    Y_raw = wide.to_numpy(dtype=np.float64)
    M = np.isfinite(Y_raw).astype(np.float32)
    Y = np.nan_to_num(Y_raw, nan=0.0).astype(np.float64) / SCALE_PEAK_CM
    Y = Y.astype(np.float32)
    return Y, M, band_order


def train_torch_run(
    dataset_dir: Path,
    run_dir: Path,
    mode: str,
    train_cfg: dict[str, Any],
    device: str | None = None,
) -> dict[str, Any]:
    if not is_torch_available() or ConvPeakMultitask is None or PeakMultiDataset is None:
        raise RuntimeError("Установите torch: pip install -e '.[torch]'")

    assert torch is not None and DataLoader is not None

    run_dir.mkdir(parents=True, exist_ok=True)

    X_all, spec_ids_all = _load_arrays(dataset_dir)
    meta = pd.read_parquet(dataset_dir / "meta.parquet")
    meta_ok = meta[meta["qc_ok"] == True].copy()  # noqa: E712
    ok_ids = set(meta_ok["spectrum_id"].astype(str))

    idx_keep = [i for i, sid in enumerate(spec_ids_all) if sid in ok_ids]
    X = X_all[idx_keep]
    spec_ids = [spec_ids_all[i] for i in idx_keep]

    Y, M, band_order = build_wide_targets(dataset_dir, mode, spec_ids)
    if len(band_order) == 0:
        raise RuntimeError("Нет ни одной полосы с observed_peak_cm1 — пересоберите датасет.")

    split_path = dataset_dir / "split.json"
    if split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(map(str, sp["train_ids"]))
        test_ids = set(map(str, sp["test_ids"]))
    else:
        rng = np.random.default_rng(int(train_cfg.get("random_seed", 42)))
        uids = np.array(sorted(set(spec_ids)))
        rng.shuffle(uids)
        split_i = int(max(1, round(float(train_cfg.get("train_frac", 0.85)) * len(uids))))
        train_ids = set(uids[:split_i].tolist())
        test_ids = set(uids[split_i:].tolist())
        if not test_ids:
            test_ids = train_ids

    train_rows = [i for i, sid in enumerate(spec_ids) if sid in train_ids]
    test_rows = [i for i, sid in enumerate(spec_ids) if sid in test_ids]

    if len(train_rows) < 4:
        raise RuntimeError("Слишком мало спектров в train split для torch.")

    X_tr, Y_tr, M_tr = X[train_rows], Y[train_rows], M[train_rows]
    X_te, Y_te, M_te = X[test_rows], Y[test_rows], M[test_rows]

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = PeakMultiDataset(X_tr, Y_tr, M_tr)
    ds_te = PeakMultiDataset(X_te, Y_te, M_te)
    bs = int(train_cfg.get("torch_batch_size", 32))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=len(ds_tr) > bs)
    dl_te = DataLoader(ds_te, batch_size=min(bs, len(ds_te)), shuffle=False)

    seq_len = X.shape[1]
    model = ConvPeakMultitask(seq_len, len(band_order)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("torch_lr", 1e-3)), weight_decay=1e-4)
    epochs = int(train_cfg.get("torch_epochs", 30))

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae_cm": []}

    @torch.no_grad()
    def eval_epoch() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_mae_num = 0.0
        total_mae_den = 0.0
        for xb, yb, mb in dl_te:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            pred = model(xb)
            loss = _masked_smooth_l1(pred, yb, mb)
            total_loss += float(loss.item()) * len(xb)
            diff_cm = torch.abs(pred - yb) * SCALE_PEAK_CM * mb
            total_mae_num += float(diff_cm.sum().item())
            total_mae_den += float(mb.sum().item())
        n = len(ds_te)
        val_loss = total_loss / max(n, 1)
        val_mae = total_mae_num / max(total_mae_den, 1.0)
        return val_loss, val_mae

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_val = float("inf")
    best_state = None

    for _ep in range(epochs):
        model.train()
        tl_acc = 0.0
        tn = 0
        for xb, yb, mb in dl_tr:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = _masked_smooth_l1(pred, yb, mb)
            loss.backward()
            opt.step()
            tl_acc += float(loss.item()) * len(xb)
            tn += len(xb)
        train_loss = tl_acc / max(tn, 1)
        val_loss, val_mae_cm = eval_epoch()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_cm"].append(val_mae_cm)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle: dict[str, Any] = {
        "kind": "torch_conv_peak_multitask",
        "mode": mode,
        "band_order": band_order,
        "scale_peak_cm": SCALE_PEAK_CM,
        "seq_len": seq_len,
        "train_cfg": {k: train_cfg[k] for k in train_cfg if k.startswith("torch_") or k in ("random_seed", "train_frac")},
        "history": history,
        "device_trained": dev,
    }
    torch.save({"model_state": model.state_dict(), "meta": bundle}, run_dir / "torch_bundle.pt")

    (run_dir / "torch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("SmoothL1 (scaled ν)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_mae_cm"], color="tab:orange")
    axes[1].set_title("Val MAE (cm⁻¹, masked)")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(run_dir / "torch_training_curve.png", dpi=140)
    plt.close(fig)

    summary = {
        "best_val_loss_scaled": float(best_val),
        "final_val_mae_cm": float(history["val_mae_cm"][-1]) if history["val_mae_cm"] else None,
        "n_bands": len(band_order),
        "run_dir": str(run_dir),
    }
    (run_dir / "torch_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def predict_torch_peaks(bundle_path: Path, spectrum_normalized: np.ndarray) -> dict[str, float]:
    """Инференс по сохранённому torch_bundle.pt → словарь band_id → ν см⁻¹."""
    if not is_torch_available() or ConvPeakMultitask is None:
        raise RuntimeError("torch не установлен")

    assert torch is not None

    try:
        ck = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(bundle_path, map_location="cpu")
    meta = ck["meta"]
    band_order: list[str] = meta["band_order"]
    seq_len: int = int(meta["seq_len"])
    scale: float = float(meta["scale_peak_cm"])

    x = np.asarray(spectrum_normalized, dtype=np.float32).reshape(-1)
    if x.shape[0] != seq_len:
        raise ValueError(f"Длина спектра {x.shape[0]} != ожидаемой {seq_len}")

    model = ConvPeakMultitask(seq_len, len(band_order))
    model.load_state_dict(ck["model_state"])
    model.eval()

    with torch.no_grad():
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        pred = model(t).numpy().flatten()
    nu = pred * scale
    return {b: float(nu[i]) for i, b in enumerate(band_order)}
