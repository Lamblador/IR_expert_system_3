"""IrKanNet (M2): ConvKAN backbone + KAN-голова."""

from __future__ import annotations

import torch
import torch.nn as nn

from ir_pipeline.models.ir_encoder import KanResNetEncoder
from ir_pipeline.models.kan_layers import KANHead


class IrKanNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 34,
        class_nums: int = 17,
        context_dim: int = 0,
        *,
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        head_hidden: int = 200,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = int(context_dim)
        self.class_nums = int(class_nums)
        self.kan_grid_size = kan_grid_size
        self.encoder = KanResNetEncoder(hidden_size, grid_size=kan_grid_size)
        self.do1 = nn.Dropout1d(0.5)
        self.head = KANHead(
            self.encoder.flat_dim + self.context_dim,
            head_hidden,
            class_nums,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

    def encode_spectrum(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_spectrum(batch)

    def forward(self, batch: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        batch = self.encode_spectrum(batch)
        batch = self.do1(batch)
        if self.context_dim > 0:
            if context is None:
                raise ValueError(f"Ожидается context размерности {self.context_dim}")
            batch = torch.cat([batch, context], dim=1)
        return self.head(batch)

    def cam_target_layer(self) -> nn.Module:
        return self.encoder.cam_target_layer()
