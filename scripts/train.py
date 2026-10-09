#!/usr/bin/env python3
"""
RetinaCare-AI - EfficientNet-B2 Training Script
==============================================
Trains a real PyTorch transfer learning model for Diabetic Retinopathy 5-class grading.
Tracks Quadratic Weighted Kappa (QWK) on validation set and saves best model checkpoint.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ai.datasets.retina_dataset import create_dataloader
from ai.evaluation.metrics import calculate_metrics
from ai.models.efficientnet import EfficientNetB2_DR
from ai.preprocessing.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-Train")


def compute_class_weights(df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """Compute balanced class weights to handle severe DR class imbalance."""
    counts = df["diagnosis"].value_counts().to_dict()
    total = len(df)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        # Smoothed inverse frequency weighting
        w = total / (num_classes * max(cnt, 1))
        weights.append(w)
    # Normalize weights so mean is 1.0
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    use_amp = (device.type == "cuda") and (scaler is not None)

    for images, targets, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_metrics = calculate_metrics(all_targets, all_preds)
    return epoch_loss, epoch_metrics["qwk"]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(all_targets, all_preds)
    return val_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B2 for Diabetic Retinopathy")
    parser.add_argument("--train-csv", type=str, default="data/processed/train_split.csv")
    parser.add_argument("--val-csv", type=str, default="data/processed/val_split.csv")
    parser.add_argument("--images-dir", type=str, default="data/processed/images")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"RetinaCare-AI EfficientNet-B2 Training Pipeline")
    logger.info(f"  Device        : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"  Train CSV     : {args.train_csv}")
    logger.info(f"  Val CSV       : {args.val_csv}")
    logger.info(f"  Images Dir    : {args.images_dir}")
    logger.info(f"  Epochs        : {args.epochs}")
    logger.info(f"  Batch Size    : {args.batch_size}")
    logger.info(f"  Learning Rate : {args.lr}")
    logger.info("=" * 70)

    # Check data paths
    if not os.path.exists(args.train_csv) or not os.path.exists(args.val_csv):
        logger.error(
            f"Split files not found! Please run 'python scripts/prepare_dataset.py' first."
        )
        sys.exit(1)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    logger.info(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    # Create DataLoaders
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform = get_val_transforms(image_size=args.image_size)

    train_loader = create_dataloader(
        df=train_df,
        images_dir=args.images_dir,
        transform=train_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = create_dataloader(
        df=val_df,
        images_dir=args.images_dir,
        transform=val_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # Calculate class weights for loss function
    class_weights = compute_class_weights(train_df, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build Model
    logger.info("Initializing EfficientNet-B2 with ImageNet pretrained weights...")
    model = EfficientNetB2_DR(num_classes=5, pretrained=True, dropout_rate=0.4).to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Tracking
    best_val_qwk = -1.0
    history = []
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_qwk = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_loss, val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        val_qwk = val_metrics["qwk"]
        val_acc = val_metrics["accuracy"]
        ref_sens = val_metrics["referable_dr"]["sensitivity"]
        ref_spec = val_metrics["referable_dr"]["specificity"]
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train QWK: {train_qwk:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val QWK: {val_qwk:.4f}, Val Acc: {val_acc*100:.1f}% | "
            f"Ref Sens: {ref_sens*100:.1f}%, Ref Spec: {ref_spec*100:.1f}%"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_qwk": round(train_qwk, 4),
            "val_loss": round(val_loss, 4),
            "val_qwk": round(val_qwk, 4),
            "val_accuracy": round(val_acc, 4),
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        # Checkpoint if QWK improved
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "architecture": "EfficientNetB2_DR",
                "num_classes": 5,
            }, best_ckpt_path)
            logger.info(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val QWK: {best_val_qwk:.4f})")

    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Training Complete in {total_time/60:.2f} minutes.")
    logger.info(f"Best Validation QWK: {best_val_qwk:.4f}")
    logger.info(f"Best model saved at: {best_ckpt_path}")
    logger.info("=" * 70)

    # Save training history JSON
    history_file = os.path.join(args.results_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history log to: {history_file}")


if __name__ == "__main__":
    main()
