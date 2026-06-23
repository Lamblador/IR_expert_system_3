"""Тесты KAN-моделей и фабрики."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ir_pipeline.models import build_spectrum_model, count_parameters


@pytest.mark.parametrize("family", ["irresnet4", "kan_hybrid", "kan_full"])
def test_spectrum_model_forward_shapes(family: str) -> None:
    model = build_spectrum_model(
        family,
        hidden_size=34,
        class_nums=17,
        context_dim=12,
        train_cfg={"kan_grid_size": 3, "kan_full_hidden_size": 34},
    )
    x = torch.randn(2, 3, 1801)
    ctx = torch.randn(2, 12)
    logits = model(x, ctx)
    assert logits.shape == (2, 17)
    n_total, n_train = count_parameters(model)
    assert n_total > 0
    assert n_train == n_total


def test_irresnet4_no_context() -> None:
    model = build_spectrum_model("irresnet4", hidden_size=34, class_nums=5, context_dim=0)
    x = torch.randn(1, 3, 1801)
    out = model(x)
    assert out.shape == (1, 5)


def test_kan_hybrid_encode_spectrum() -> None:
    model = build_spectrum_model("kan_hybrid", hidden_size=34, class_nums=5, context_dim=0)
    x = torch.randn(1, 3, 1801)
    enc = model.encode_spectrum(x)
    assert enc.ndim == 2
    assert enc.shape[0] == 1
