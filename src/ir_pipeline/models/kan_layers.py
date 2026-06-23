"""KAN-слои для 1D спектральных моделей (B-spline на рёбрах)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _b_splines(x: torch.Tensor, grid: torch.Tensor, spline_order: int) -> torch.Tensor:
    """B-spline basis: x (B, in), grid (in, G) -> (B, in, n_basis)."""
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        left = grid[:, : -(k + 1)]
        right_l = grid[:, k:-1]
        right_r = grid[:, k + 1 :]
        left_r = grid[:, 1 : -(k)]
        denom_l = right_l - left
        denom_r = right_r - left_r
        term_l = torch.where(denom_l > 0, (x - left) / denom_l, torch.zeros_like(x))
        term_r = torch.where(denom_r > 0, (right_r - x) / denom_r, torch.zeros_like(x))
        bases = term_l * bases[:, :, :-1] + term_r * bases[:, :, 1:]
    return bases


class KANLinear(nn.Module):
    """KAN-линейный слой: learnable spline на каждом ребре + базовая SiLU-ветка."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous(), persistent=False)
        n_basis = grid_size + spline_order
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.base_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(x_flat), self.base_weight * self.scale_base, self.base_bias)
        bases = _b_splines(x_flat, self.grid, self.spline_order)
        spline = torch.einsum("oid,bin->bo", self.spline_weight * self.scale_spline, bases)
        out = base + spline
        return out.reshape(*orig_shape[:-1], self.out_features)


class KANHead(nn.Module):
    """Две KANLinear с dropout (замена fc + fc1 в IrResnet4)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc = KANLinear(in_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order)
        self.do = nn.Dropout(dropout)
        self.fc1 = KANLinear(hidden_dim, out_dim, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.do(self.fc(x)))


class ConvKAN1d(nn.Module):
    """1D свёртка: sum_i phi_i(x_i) по окну ядра (KAN на патче)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        grid_size: int = 3,
        spline_order: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        patch_dim = in_channels * kernel_size
        self.kan = KANLinear(patch_dim, out_channels, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding:
            x = F.pad(x, (self.padding, self.padding))
        patches = x.unfold(2, self.kernel_size, self.stride)
        b, _c, l_out, _k = patches.shape
        flat = patches.permute(0, 2, 1, 3).reshape(b, l_out, -1)
        out = self.kan(flat)
        return out.permute(0, 2, 1)


class KanBasicBlock(nn.Module):
    """Residual block с ConvKAN1d вместо Conv1d."""

    def __init__(
        self,
        in_channels: int,
        downsample: bool = False,
        *,
        grid_size: int = 3,
    ):
        super().__init__()
        self.downsample_flag = downsample
        self.grid_size = grid_size
        if downsample:
            self.conv1 = ConvKAN1d(
                in_channels // 2,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                grid_size=grid_size,
            )
        else:
            self.conv1 = ConvKAN1d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                grid_size=grid_size,
            )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = ConvKAN1d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            grid_size=grid_size,
        )
        self.bn2 = nn.BatchNorm1d(in_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, stride=2),
                nn.BatchNorm1d(in_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample_flag:
            identity = self.downsample(x)
        return F.relu(out + identity)
