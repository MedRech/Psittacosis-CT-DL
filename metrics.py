from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    intersection = torch.sum(preds * targets).item()
    denominator = torch.sum(preds).item() + torch.sum(targets).item()
    return (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def voxel_sensitivity(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    return (tp + 1e-6) / (tp + fn + 1e-6)


def voxel_ppv(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    return (tp + 1e-6) / (tp + fp + 1e-6)


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y_true = targets.detach().cpu().numpy().reshape(-1)
    y_pred = (probs >= 0.5).astype(np.int64)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc,
    }
