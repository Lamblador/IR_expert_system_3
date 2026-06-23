"""Общий 1D ResNet-энкодер для IrResnet4 и KAN-вариантов."""

from __future__ import annotations

import torch
import torch.nn as nn

from ir_pipeline.models.kan_layers import KanBasicBlock


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, downsample: bool = False):
        super().__init__()
        self.downsample_flag = downsample
        if downsample:
            self.conv1 = nn.Conv1d(in_channels // 2, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, stride=2),
                nn.BatchNorm1d(in_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample_flag:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _resnet_stages(hidden_size: int, block_cls: type, **block_kw) -> tuple[nn.Module, ...]:
    h = hidden_size
    layer1 = nn.Sequential(block_cls(h, **block_kw), block_cls(h, **block_kw), block_cls(h, **block_kw))
    layer3 = nn.Sequential(
        block_cls(h * 2, downsample=True, **block_kw),
        block_cls(h * 2, **block_kw),
        block_cls(h * 2, **block_kw),
    )
    layer5 = nn.Sequential(
        block_cls(h * 4, downsample=True, **block_kw),
        block_cls(h * 4, **block_kw),
        block_cls(h * 4, **block_kw),
    )
    layer7 = nn.Sequential(
        block_cls(h * 8, downsample=True, **block_kw),
        block_cls(h * 8, **block_kw),
        block_cls(h * 8, **block_kw),
    )
    return layer1, layer3, layer5, layer7


class ResNetEncoder(nn.Module):
    """CNN-энкодер IrResnet4: (B,3,L) -> (B, flat_dim)."""

    def __init__(self, hidden_size: int = 34):
        super().__init__()
        self.hidden_size = hidden_size
        h = hidden_size
        self.conv1 = nn.Conv1d(3, h, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(h)
        self.relu = nn.ReLU()
        self.layer1, self.layer3, self.layer5, self.layer7 = _resnet_stages(h, BasicBlock)
        self.max3 = nn.MaxPool1d(3, 2, 0)
        self.flatten = nn.Flatten()
        self.flat_dim = h * 8 * 56

    def encode_spectrum(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.relu(self.bn1(self.conv1(batch)))
        batch = self.layer1(batch)
        batch = self.layer3(batch)
        batch = self.layer5(batch)
        batch = self.max3(batch)
        batch = self.layer7(batch)
        return self.flatten(batch)

    def cam_target_layer(self) -> nn.Module:
        return self.layer7[-1]


class KanResNetEncoder(nn.Module):
    """ConvKAN backbone: (B,3,L) -> (B, flat_dim)."""

    def __init__(self, hidden_size: int = 34, *, grid_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        h = hidden_size
        self.conv1 = ConvKAN1dStem(3, h, kernel_size=3, stride=2, padding=1, grid_size=grid_size)
        self.bn1 = nn.BatchNorm1d(h)
        self.layer1, self.layer3, self.layer5, self.layer7 = _resnet_stages(
            h, KanBasicBlock, grid_size=grid_size
        )
        self.max3 = nn.MaxPool1d(3, 2, 0)
        self.flatten = nn.Flatten()
        self.flat_dim = h * 8 * 56

    def encode_spectrum(self, batch: torch.Tensor) -> torch.Tensor:
        batch = torch.relu(self.bn1(self.conv1(batch)))
        batch = self.layer1(batch)
        batch = self.layer3(batch)
        batch = self.layer5(batch)
        batch = self.max3(batch)
        batch = self.layer7(batch)
        return self.flatten(batch)

    def cam_target_layer(self) -> nn.Module:
        return self.layer7[-1]


class ConvKAN1dStem(nn.Module):
    """Первая свёртка KanNet (отдельно от kan_layers.ConvKAN1d для ясности импорта)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        grid_size: int = 3,
    ):
        super().__init__()
        from ir_pipeline.models.kan_layers import ConvKAN1d

        self.conv = ConvKAN1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            grid_size=grid_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
