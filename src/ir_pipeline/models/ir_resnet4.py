"""1D ResNet для multi-label классификации ИК-спектров (3 канала, 400–4000 см⁻¹)."""

from __future__ import annotations

import torch
import torch.nn as nn


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


class IrResnet4(nn.Module):
    """
    Вход: (B, 3, L), L=1801 для сетки 400–4000 см⁻¹, шаг 2.
    Опционально context (B, C): one-hot техники измерения + фазы образца, конкатенируется перед FC.
    """

    def __init__(self, hidden_size: int = 34, class_nums: int = 17, context_dim: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = int(context_dim)
        h = hidden_size
        self.conv1 = nn.Conv1d(3, h, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(h)
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(BasicBlock(h), BasicBlock(h), BasicBlock(h))
        self.layer3 = nn.Sequential(
            BasicBlock(h * 2, downsample=True),
            BasicBlock(h * 2),
            BasicBlock(h * 2),
        )
        self.layer5 = nn.Sequential(
            BasicBlock(h * 4, downsample=True),
            BasicBlock(h * 4),
            BasicBlock(h * 4),
        )
        self.max3 = nn.MaxPool1d(3, 2, 0)
        self.layer7 = nn.Sequential(
            BasicBlock(h * 8, downsample=True),
            BasicBlock(h * 8),
            BasicBlock(h * 8),
        )
        self.flatten = nn.Flatten()
        self.do1 = nn.Dropout1d(0.5)
        flat_dim = h * 8 * 56
        self.fc = nn.Linear(flat_dim + self.context_dim, 200)
        self.do2 = nn.Dropout1d(0.2)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(200, class_nums)

    def encode_spectrum(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.relu(self.bn1(self.conv1(batch)))
        batch = self.layer1(batch)
        batch = self.layer3(batch)
        batch = self.layer5(batch)
        batch = self.max3(batch)
        batch = self.layer7(batch)
        return self.flatten(batch)

    def forward(self, batch: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        batch = self.encode_spectrum(batch)
        batch = self.do1(batch)
        if self.context_dim > 0:
            if context is None:
                raise ValueError(f"Ожидается context размерности {self.context_dim}")
            batch = torch.cat([batch, context], dim=1)
        batch = self.fc(batch)
        batch = self.do2(batch)
        batch = self.relu1(batch)
        return self.fc1(batch)

    def cam_target_layer(self) -> nn.Module:
        return self.layer7[-1]
