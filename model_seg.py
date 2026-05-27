from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        current = in_channels
        for feature in features:
            self.down_blocks.append(ConvBlock(current, feature))
            self.pools.append(nn.MaxPool3d(kernel_size=2))
            current = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose3d(current, feature, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(feature * 2, feature))
            current = feature

        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.head(x)


def build_segmentation_model() -> UNet3D:
    return UNet3D()
