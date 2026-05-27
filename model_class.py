from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * torch.sigmoid(attention)


class SpatialAttention3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg, max_values], dim=1)
        return x * torch.sigmoid(self.conv(attention))


class CBAM3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention3D(channels)
        self.spatial = SpatialAttention3D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.attention = CBAM3D(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class LTAResNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_outputs: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, num_outputs)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(in_channels, out_channels, stride=stride)]
        layers.extend(BasicBlock3D(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_classification_model(num_outputs: int = 1) -> LTAResNet:
    return LTAResNet(num_outputs=num_outputs)
