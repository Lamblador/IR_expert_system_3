"""Фабрика спектральных моделей M0/M1/M2."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ir_pipeline.models.ir_kan_hybrid import IrKanHybrid
from ir_pipeline.models.ir_kan_net import IrKanNet
from ir_pipeline.models.ir_resnet4 import IrResnet4

MODEL_FAMILIES = ("irresnet4", "kan_hybrid", "kan_full")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_spectrum_model(
    model_family: str,
    *,
    hidden_size: int,
    class_nums: int,
    context_dim: int = 0,
    train_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    cfg = train_cfg or {}
    family = str(model_family).lower().strip()
    grid = int(cfg.get("kan_grid_size", 5 if family == "kan_hybrid" else 3))
    spline_order = int(cfg.get("kan_spline_order", 3))
    head_hidden = int(cfg.get("kan_head_hidden", 200))

    if family in ("irresnet4", "m0"):
        return IrResnet4(hidden_size=hidden_size, class_nums=class_nums, context_dim=context_dim)
    if family in ("kan_hybrid", "m1"):
        return IrKanHybrid(
            hidden_size=hidden_size,
            class_nums=class_nums,
            context_dim=context_dim,
            kan_grid_size=grid,
            kan_spline_order=spline_order,
            head_hidden=head_hidden,
        )
    if family in ("kan_full", "m2", "kan_net"):
        full_hidden = int(cfg.get("kan_full_hidden_size", hidden_size))
        return IrKanNet(
            hidden_size=full_hidden,
            class_nums=class_nums,
            context_dim=context_dim,
            kan_grid_size=grid,
            kan_spline_order=spline_order,
            head_hidden=head_hidden,
        )
    raise ValueError(f"Неизвестный model_family={model_family!r}; ожидается one of {MODEL_FAMILIES}")
