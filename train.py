from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ClassificationDataset, SegmentationDataset
from losses import SegmentationLoss, WeightedBCELoss
from metrics import classification_metrics, dice_score
from model_class import build_classification_model
from model_seg import build_segmentation_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config: dict) -> torch.device:
    requested = config.get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def train_seg(config: dict) -> None:
    device = resolve_device(config)
    model = build_segmentation_model().to(device)
    loss_fn = SegmentationLoss(config["dice_weight"], config["boundary_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_loader = DataLoader(SegmentationDataset(config["train_csv"]), batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(SegmentationDataset(config["val_csv"]), batch_size=1, shuffle=False, num_workers=config["num_workers"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"seg train {epoch}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_scores = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"seg val {epoch}"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                val_scores.append(dice_score(model(images), masks))

        mean_dice = float(np.mean(val_scores))
        print(f"epoch={epoch} train_loss={train_loss / max(len(train_loader), 1):.4f} val_dice={mean_dice:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), checkpoint_dir / "seg_best.pt")


def train_class(config: dict) -> None:
    device = resolve_device(config)
    model = build_classification_model(config.get("num_classes", 1)).to(device)
    loss_fn = WeightedBCELoss(config["positive_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_loader = DataLoader(ClassificationDataset(config["train_csv"]), batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(ClassificationDataset(config["val_csv"]), batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"class train {epoch}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"class val {epoch}"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits_list.append(model(images).cpu())
                labels_list.append(labels.cpu())

        metrics = classification_metrics(torch.cat(logits_list), torch.cat(labels_list))
        print(f"epoch={epoch} train_loss={train_loss / max(len(train_loader), 1):.4f} val_auc={metrics['auc']:.4f} val_acc={metrics['accuracy']:.4f}")
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), checkpoint_dir / "class_best.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["seg", "class"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    if args.task == "seg":
        train_seg(config)
    else:
        train_class(config)


if __name__ == "__main__":
    main()
