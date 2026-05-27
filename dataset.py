from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_volume(path: str | Path) -> torch.Tensor:
    array = np.load(path).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {array.shape} from {path}")
    array = (array - array.mean()) / (array.std() + 1e-6)
    return torch.from_numpy(array).unsqueeze(0)


class SegmentationDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        self.items = pd.read_csv(csv_path)
        required = {"image", "mask"}
        if not required.issubset(self.items.columns):
            raise ValueError(f"Segmentation CSV must contain columns: {sorted(required)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.items.iloc[index]
        image = _load_volume(row["image"])
        mask = np.load(row["mask"]).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return {"image": image, "mask": mask}


class ClassificationDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        self.items = pd.read_csv(csv_path)
        required = {"image", "label"}
        if not required.issubset(self.items.columns):
            raise ValueError(f"Classification CSV must contain columns: {sorted(required)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.items.iloc[index]
        image = _load_volume(row["image"])
        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return {"image": image, "label": label}
