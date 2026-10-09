#!/usr/bin/env python3
"""
RetinaCare-AI - Grad-CAM Visual Explainability CLI
==================================================
Generates and saves visual explainability heatmaps for an input fundus image,
highlighting the microvascular lesion regions driving the AI model's decision.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai.evaluation.metrics import DR_CLASS_LABELS
from ai.explainability.gradcam import GradCAM, overlay_gradcam
from ai.models.efficientnet import build_model
from ai.preprocessing.normalization import preprocess_fundus_image
from ai.preprocessing.transforms import get_inference_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-GradCAM")


def generate_gradcam_visual(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    output_path: str = "results/gradcam_explanation.png",
    target_class: Optional[int] = None,
    image_size: int = 512,
    device_str: Optional[str] = None,
):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Read and preprocess raw image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Preprocess fundus image (circle crop + CLAHE/Ben Graham)
    preprocessed_rgb = preprocess_fundus_image(raw_rgb, target_size=image_size, use_ben_graham=True)

    # Prepare tensor
    transform = get_inference_transforms(image_size=image_size)
    tensor = transform(image=preprocessed_rgb)["image"].unsqueeze(0).to(device)

    # Load Model
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        model = build_model(num_classes=5, pretrained=False, checkpoint_path=checkpoint_path, device=str(device))
    else:
        logger.warning("No checkpoint specified or file not found. Using pretrained EfficientNet-B2 backbone.")
        model = build_model(num_classes=5, pretrained=True, device=str(device))

    # Initialize Grad-CAM on last conv layer
    target_layer = model.get_last_conv_layer()
    gradcam_engine = GradCAM(model=model, target_layer=target_layer)

    # Generate Heatmap
    heatmap, pred_class, confidence, _ = gradcam_engine.generate_cam(tensor, target_class=target_class)
    gradcam_engine.remove_hooks()

    # Overlay
    overlay_rgb = overlay_gradcam(preprocessed_rgb, heatmap, alpha=0.45)

    # Plot 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Original Retinal Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(preprocessed_rgb)
    axes[1].set_title("Preprocessed (ROI + Illumination)", fontsize=13)
    axes[1].axis("off")

    class_name = DR_CLASS_LABELS[pred_class]
    referable_tag = "[REFERABLE DR]" if pred_class >= 2 else "[Non-Referable]"
    title_text = f"Grad-CAM Attention: Grade {pred_class} - {class_name}\nConfidence: {confidence*100:.1f}% | {referable_tag}"

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(title_text, fontsize=13, color="darkred" if pred_class >= 2 else "darkgreen")
    axes[2].axis("off")

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Grad-CAM visualization successfully saved to: {output_path}")
    logger.info(f"  * Predicted Class : Grade {pred_class} ({class_name})")
    logger.info(f"  * Confidence      : {confidence * 100:.2f}%")
    logger.info(f"  * Referable DR    : {'YES (Refer to ophthalmologist)' if pred_class >= 2 else 'NO'}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for Retinal Fundus Image")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input retinal image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model weights checkpoint")
    parser.add_argument("--output", type=str, default="results/gradcam_explanation.png", help="Output visualization path")
    parser.add_argument("--target-class", type=int, default=None, help="Target class index (0-4) or None for top prediction")
    parser.add_argument("--image-size", type=int, default=512, help="Image resolution")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cuda/cpu)")
    args = parser.parse_args()

    try:
        generate_gradcam_visual(
            image_path=args.image_path,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            target_class=args.target_class,
            image_size=args.image_size,
            device_str=args.device,
        )
    except Exception as e:
        logger.exception(f"Grad-CAM generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
