from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ClassificationDataset, SegmentationDataset
from metrics import classification_metrics, dice_score, voxel_ppv, voxel_sensitivity
from model_class import build_classification_model
from model_seg import build_segmentation_model


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config: dict) -> torch.device:
    requested = config.get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def evaluate_seg(config: dict, checkpoint: str | Path) -> None:
    device = resolve_device(config)
    model = build_segmentation_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    loader = DataLoader(SegmentationDataset(config["val_csv"]), batch_size=1, shuffle=False, num_workers=config["num_workers"])
    model.eval()
    dice_values = []
    sensitivity_values = []
    ppv_values = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="seg test"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            dice_values.append(dice_score(logits, masks))
            sensitivity_values.append(voxel_sensitivity(logits, masks))
            ppv_values.append(voxel_ppv(logits, masks))
    print({
        "dice": float(np.mean(dice_values)),
        "voxel_sensitivity": float(np.mean(sensitivity_values)),
        "voxel_ppv": float(np.mean(ppv_values)),
    })


def evaluate_class(config: dict, checkpoint: str | Path) -> None:
    device = resolve_device(config)
    model = build_classification_model(config.get("num_classes", 1)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    loader = DataLoader(ClassificationDataset(config["val_csv"]), batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="class test"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits_list.append(model(images).cpu())
            labels_list.append(labels.cpu())
    print(classification_metrics(torch.cat(logits_list), torch.cat(labels_list)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["seg", "class"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.task == "seg":
        evaluate_seg(config, args.checkpoint)
    else:
        evaluate_class(config, args.checkpoint)


if __name__ == "__main__":
    main()
