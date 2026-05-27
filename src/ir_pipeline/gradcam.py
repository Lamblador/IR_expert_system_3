"""Grad-CAM для IrResnet4 (1D спектр)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ir_pipeline.logging_utils import log
from ir_pipeline.models.ir_resnet4 import IrResnet4


class _ActivationHook:
    def __init__(self, layer: torch.nn.Module):
        self.activation = None
        self.hook = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activation = out.detach().clone()

    def remove(self):
        self.hook.remove()


class _GradientHook:
    def __init__(self, layer: torch.nn.Module):
        self.gradient = None
        self.hook = layer.register_backward_hook(self._hook)

    def _hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach().clone()

    def remove(self):
        self.hook.remove()


def compute_cam(
    model: IrResnet4,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    layer = model.cam_target_layer()
    act_h = _ActivationHook(layer)
    grad_h = _GradientHook(layer)
    model.train()
    out = model(input_tensor)
    loss = out[0, class_idx]
    model.zero_grad()
    loss.backward(retain_graph=True)
    acts = act_h.activation
    grads = grad_h.gradient
    act_h.remove()
    grad_h.remove()
    weights = grads.mean(dim=2, keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()
    model.eval()
    return cam


def interpolate_cam_to_wavenumbers(cam: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    from scipy.interpolate import interp1d

    model_wn = np.linspace(500, 4100, len(cam))
    f = interp1d(model_wn, cam, kind="linear", bounds_error=False, fill_value=0.0)
    return f(wavenumbers)


def run_cam_examples(
    bundle_path: Path,
    dataset_dir: Path,
    out_dir: Path,
    *,
    n_examples: int = 3,
    confidence_threshold: float = 0.5,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(bundle_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    class_names: list[str] = meta["class_names"]
    hidden = int(meta["hidden_size"])
    model = IrResnet4(hidden_size=hidden, class_nums=len(class_names))
    model.load_state_dict(ck["model_state"])
    model.eval()

    z = np.load(dataset_dir / "telegram_arrays.npz", allow_pickle=True)
    X = z["X_bot"]
    wn = np.asarray(z["wavenumbers"], dtype=float)
    written: list[Path] = []

    for i in range(min(n_examples, len(X))):
        x = torch.from_numpy(X[i : i + 1]).float()
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0]
        pred_idx = (probs > confidence_threshold).nonzero(as_tuple=False).flatten().tolist()
        if not pred_idx:
            pred_idx = [int(probs.argmax())]

        ab = X[i, 1]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wn, ab, color="gray", lw=1.0)
        for j, cidx in enumerate(pred_idx[:8]):
            cam = compute_cam(model, x, cidx)
            cam_i = interpolate_cam_to_wavenumbers(cam, wn)
            cam_i = cam_i / (cam_i.max() + 1e-9)
            ax.fill_between(wn, 0, cam_i * np.nanmax(ab) * 0.9, alpha=0.25, label=class_names[cidx][:24])
            ax.axvline(wn[int(np.argmax(cam_i))], color="tab:red", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(wn.max(), wn.min())
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.set_title(f"Grad-CAM example {i} ({len(pred_idx)} classes)")
        if pred_idx:
            ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        p = out_dir / f"cam_example_{i}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)
        log(f"CAM saved: {p}")

    return written
