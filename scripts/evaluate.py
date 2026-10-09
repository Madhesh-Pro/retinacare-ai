#!/usr/bin/env python3
"""
RetinaCare-AI - Model Evaluation Script
=======================================
Evaluates a trained EfficientNet-B2 checkpoint on test or validation splits.
Computes Quadratic Weighted Kappa (QWK), multiclass metrics, Referable DR metrics,
and saves results to JSON.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm

from ai.datasets.retina_dataset import create_dataloader
from ai.evaluation.metrics import DR_CLASS_LABELS, calculate_metrics
from ai.models.efficientnet import build_model
from ai.preprocessing.transforms import get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-Eval")


def print_evaluation_dashboard(metrics: dict):
    print("\n" + "=" * 75)
    print("           RETINACARE-AI MODEL EVALUATION REPORT")
    print("=" * 75)
    print(f"  * Quadratic Weighted Kappa (QWK) : {metrics['qwk']:.4f}")
    print(f"  * Multiclass Accuracy             : {metrics['accuracy'] * 100:.2f}%")
    print(f"  * Macro F1-Score                  : {metrics['f1_macro']:.4f}")
    print(f"  * Weighted F1-Score               : {metrics['f1_weighted']:.4f}")
    print("-" * 75)
    print("  CLINICAL REFERABLE DR METRICS (Grades 2, 3, 4 vs 0, 1):")
    ref = metrics["referable_dr"]
    print(f"  * Sensitivity (Recall on Referable DR) : {ref['sensitivity'] * 100:.2f}%")
    print(f"  * Specificity (Recall on Non-Referable): {ref['specificity'] * 100:.2f}%")
    print(f"  * Referable DR Accuracy                : {ref['accuracy'] * 100:.2f}%")
    print(f"  * Referable DR F1-Score                : {ref['f1']:.4f}")
    print("-" * 75)
    print("  PER-CLASS PERFORMANCE BREAKDOWN:")
    table_data = []
    clf_rep = metrics["classification_report"]
    for idx, name in enumerate(DR_CLASS_LABELS):
        if name in clf_rep:
            c = clf_rep[name]
            table_data.append([
                idx,
                name,
                f"{c['precision']:.3f}",
                f"{c['recall']:.3f}",
                f"{c['f1-score']:.3f}",
                int(c["support"]),
            ])
    print(tabulate(
        table_data,
        headers=["Grade", "Class Name", "Precision", "Recall", "F1", "Support"],
        tablefmt="simple",
    ))
    print("-" * 75)
    print("  CONFUSION MATRIX (Rows: True, Columns: Predicted):")
    cm = metrics["confusion_matrix"]
    header = ["True / Pred"] + [f"G{i}" for i in range(5)]
    cm_rows = [[f"Grade {i}"] + row for i, row in enumerate(cm)]
    print(tabulate(cm_rows, headers=header, tablefmt="grid"))
    print("=" * 75 + "\n")


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    test_csv: str,
    images_dir: str,
    output_report: str = "results/evaluation_report.json",
    image_size: int = 512,
    batch_size: int = 16,
    device_str: Optional[str] = None,
):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = build_model(
        num_classes=5,
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=str(device),
    )
    model.eval()

    df = pd.read_csv(test_csv)
    logger.info(f"Evaluating on {len(df)} test samples from {test_csv}")

    transform = get_val_transforms(image_size=image_size)
    loader = create_dataloader(
        df=df,
        images_dir=images_dir,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets, _ in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

    metrics = calculate_metrics(all_targets, all_preds)
    print_evaluation_dashboard(metrics)

    # Save to JSON
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation report to: {output_report}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--test-csv", type=str, default="data/processed/test_split.csv")
    parser.add_argument("--images-dir", type=str, default="data/processed/images")
    parser.add_argument("--output-report", type=str, default="results/evaluation_report.json")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    try:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            test_csv=args.test_csv,
            images_dir=args.images_dir,
            output_report=args.output_report,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device_str=args.device,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
