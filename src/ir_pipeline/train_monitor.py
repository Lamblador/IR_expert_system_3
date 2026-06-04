"""Мониторинг обучения CNN / IrResnet: хвост логов, live-графики (Colab), финальный PNG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ir_pipeline.logging_utils import log


def is_colab_runtime() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def live_plot_default(train_cfg: dict[str, Any]) -> bool:
    if "live_training_plot" in train_cfg:
        return bool(train_cfg["live_training_plot"])
    return is_colab_runtime()


class CnnTrainingMonitor:
    """После каждой эпохи: лог последних N значений, опционально clear_output + график."""

    def __init__(
        self,
        run_dir: Path,
        *,
        tail: int = 5,
        live_plot: bool = False,
        title: str = "CNN training",
        plot_filename: str = "live_training_curve.png",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tail = max(1, int(tail))
        self.live_plot = live_plot
        self.title = title
        self.plot_filename = plot_filename
        self.series: dict[str, list[float]] = {}
        self._display_handle = None

    @classmethod
    def from_train_cfg(cls, run_dir: Path, train_cfg: dict[str, Any], *, title: str) -> CnnTrainingMonitor:
        return cls(
            run_dir,
            tail=int(train_cfg.get("train_log_tail", 5)),
            live_plot=live_plot_default(train_cfg),
            title=title,
            plot_filename=str(train_cfg.get("live_plot_filename", "live_training_curve.png")),
        )

    def update(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.series.setdefault(key, []).append(float(value))

        summary = " | ".join(f"{k}={self._fmt(k, v)}" for k, v in metrics.items())
        log(f"[{self.title}] epoch {epoch}/{total_epochs}: {summary}")

        for key, values in self.series.items():
            tail_vals = values[-self.tail :]
            formatted = ", ".join(self._fmt(key, v) for v in tail_vals)
            log(f"  last {len(tail_vals)} {key}: {formatted}")

        self._write_history_json()
        if self.live_plot:
            self._clear_and_plot(epoch, total_epochs)
        else:
            self._save_static_plot()

    def finalize(self) -> Path:
        path = self._save_static_plot()
        log(f"[{self.title}] training curves → {path}")
        return path

    def _fmt(self, key: str, value: float) -> str:
        if "f1" in key.lower():
            return f"{value:.3f}"
        if "loss" in key.lower() or "mae" in key.lower():
            return f"{value:.4f}"
        return f"{value:.4g}"

    def _write_history_json(self) -> None:
        path = self.run_dir / "live_history.json"
        path.write_text(json.dumps(self.series, indent=2), encoding="utf-8")

    def _clear_and_plot(self, epoch: int, total_epochs: int) -> None:
        try:
            from IPython.display import clear_output, display
        except ImportError:
            self._save_static_plot()
            return

        import matplotlib.pyplot as plt

        clear_output(wait=True)
        fig = self._build_figure(f"{self.title} (epoch {epoch}/{total_epochs})")
        out = self.run_dir / self.plot_filename
        fig.savefig(out, dpi=140, bbox_inches="tight")
        display(fig)
        plt.show(block=False)
        plt.close(fig)

    def _save_static_plot(self, fig=None) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = self.run_dir / self.plot_filename
        if fig is None:
            if not self.series:
                return out
            fig = self._build_figure(self.title)
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
        else:
            fig.savefig(out, dpi=140, bbox_inches="tight")
        return out

    def _build_figure(self, title: str):
        import matplotlib.pyplot as plt

        keys = list(self.series.keys())
        n = len(keys)
        if n == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_title(title)
            return fig

        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(title, fontsize=11)
        for i, key in enumerate(keys):
            ax = axes[i // ncols][i % ncols]
            ys = self.series[key]
            xs = list(range(1, len(ys) + 1))
            ax.plot(xs, ys, marker="o", ms=3, lw=1.2)
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.tight_layout()
        return fig


def build_torch_optimizer(model: Any, train_cfg: dict[str, Any]) -> Any:
    """AdamW (по умолчанию), Adam или SGD — ключ torch_optimizer в yaml."""
    import torch

    name = str(train_cfg.get("torch_optimizer", "adamw")).lower().strip()
    lr = float(train_cfg.get("torch_lr", 1e-3))
    wd = float(train_cfg.get("torch_weight_decay", 1e-4))
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(train_cfg.get("torch_momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_torch_scheduler(optimizer: Any, train_cfg: dict[str, Any]) -> Any | None:
    """StepLR как в оригинале (step 75, gamma 0.2)."""
    import torch

    sched = str(train_cfg.get("torch_scheduler") or "").lower().strip()
    if sched in {"", "none", "null"}:
        return None
    if sched == "steplr":
        step = int(train_cfg.get("torch_scheduler_step_size", 75))
        gamma = float(train_cfg.get("torch_scheduler_gamma", 0.2))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    log(f"неизвестный torch_scheduler={sched}, scheduler отключён")
    return None


# Alias for Colab notebooks
IrResnetTrainingPlotter = CnnTrainingMonitor


def build_irresnet_criterion(
    Y_tr: Any,
    device: str,
    train_cfg: dict[str, Any],
) -> Any:
    """BCEWithLogitsLoss (по умолчанию) с pos_weight."""
    import numpy as np
    import torch
    import torch.nn as nn

    loss_name = str(train_cfg.get("torch_loss", "bce_with_logits")).lower().strip()
    pos_weight_scale = float(train_cfg.get("pos_weight_scale", 1.0))
    pos = Y_tr.sum(axis=0)
    neg = len(Y_tr) - pos
    pw = torch.tensor(
        np.clip(neg / np.maximum(pos, 1.0), 1.0, 50.0) * pos_weight_scale,
        dtype=torch.float32,
    ).to(device)

    if loss_name in {"bce", "bce_logits", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if loss_name == "focal":
        # упрощённый focal на logits — через BCE с модификатором не делаем; fallback
        log("torch_loss=focal не реализован, используется bce_with_logits")
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    log(f"неизвестный torch_loss={loss_name}, используется bce_with_logits")
    return nn.BCEWithLogitsLoss(pos_weight=pw)
