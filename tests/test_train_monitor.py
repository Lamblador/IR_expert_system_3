from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ir_pipeline.train_monitor import build_torch_optimizer


def test_build_torch_optimizer_adamw():
    model = torch.nn.Linear(4, 2)
    opt = build_torch_optimizer(model, {"torch_optimizer": "adamw", "torch_lr": 1e-2})
    assert isinstance(opt, torch.optim.AdamW)


def test_build_torch_optimizer_sgd():
    model = torch.nn.Linear(4, 2)
    opt = build_torch_optimizer(model, {"torch_optimizer": "sgd", "torch_lr": 0.1, "torch_momentum": 0.8})
    assert isinstance(opt, torch.optim.SGD)
