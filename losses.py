from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs * probs, dim=dims) + torch.sum(targets * targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """Lightweight boundary-aware proxy using 3D gradient consistency."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        loss = 0.0
        for dim in (2, 3, 4):
            pred_grad = torch.diff(probs, dim=dim)
            target_grad = torch.diff(targets, dim=dim)
            loss = loss + F.l1_loss(pred_grad, target_grad)
        return loss / 3.0


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, boundary_weight: float = 0.3) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.boundary_weight * self.boundary(logits, targets)


class WeightedBCELoss(nn.Module):
    def __init__(self, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([positive_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight.to(logits.device))
