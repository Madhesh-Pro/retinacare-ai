"""
Targeted Fine-Tuning for Sharp Multi-Grade Diabetic Retinopathy Classification
==============================================================================
Trains classifier head and top convolutional blocks of EfficientNet-B2 with
stratified class sampling for distinct class separation across all 5 DR grades.
"""

import os
import sys
from pathlib import Path

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {device}")

    train_df = pd.read_csv("data/processed/train_split.csv")
    val_df = pd.read_csv("data/processed/val_split.csv")

    image_size = 384
    batch_size = 16
    epochs = 6

    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)

    train_loader = create_dataloader(
        df=train_df,
        images_dir="data/processed/images",
        transform=train_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = create_dataloader(
        df=val_df,
        images_dir="data/processed/images",
        transform=val_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = EfficientNetB2_DR(num_classes=5, pretrained=True, dropout_rate=0.3).to(device)

    # Freeze bottom stages to accelerate CPU training and preserve low-level edges
    for name, param in model.backbone.features[:6].named_parameters():
        param.requires_grad = False

    # Top features and classifier head are trainable
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_qwk = -1.0
    best_acc = 0.0
    best_ckpt = "checkpoints/best_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.detach().cpu().numpy())

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val"):
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        metrics = calculate_metrics(val_targets, val_preds)
        qwk = metrics["qwk"]
        acc = metrics["accuracy"]

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Val QWK: {qwk:.4f}, Val Acc: {acc*100:.1f}%")

        if qwk > best_qwk or acc > best_acc:
            best_qwk = max(best_qwk, qwk)
            best_acc = max(best_acc, acc)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_qwk": qwk,
                "val_accuracy": acc,
                "architecture": "EfficientNetB2_DR",
                "num_classes": 5,
            }, best_ckpt)
            print(f"  --> Saved improved checkpoint to {best_ckpt}")

    print(f"Fine-tuning complete! Best Val QWK: {best_qwk:.4f}, Best Val Acc: {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()
