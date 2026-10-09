#!/usr/bin/env python3
"""
RetinaCare-AI - Fundus Image Prediction Script
==============================================
Runs real inference on a single retinal fundus image or directory of images.
Outputs predicted DR severity grade (0-4), referability status, class probabilities,
and clinical recommendation.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tabulate import tabulate

from ai.inference.predictor import DRPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-Predict")


def print_prediction_report(img_name: str, result):
    print("\n" + "=" * 70)
    print(f"      RETINACARE-AI CLINICAL DIAGNOSIS REPORT: {img_name}")
    print("=" * 70)
    print(f"  * Predicted DR Grade    : Grade {result.predicted_grade} ({result.class_name})")
    print(f"  * Model Confidence      : {result.confidence * 100:.2f}%")
    
    if result.is_referable:
        print(f"  * Referable DR Status   : [REFERABLE - CLINICAL ATTENTION REQUIRED]")
    else:
        print(f"  * Referable DR Status   : [NON-REFERABLE - ROUTINE FOLLOW-UP]")

    print("-" * 70)
    print("  MULTI-CLASS PROBABILITY DISTRIBUTION:")
    table_data = []
    for cls_name, prob in result.probabilities.items():
        bar_len = int(prob * 25)
        bar_str = "[" + "=" * bar_len + " " * (25 - bar_len) + "]"
        table_data.append([cls_name, f"{prob * 100:>6.2f}%", bar_str])
    print(tabulate(table_data, headers=["Class", "Probability", "Distribution"], tablefmt="simple"))

    print("-" * 70)
    print(f"  CLINICAL RECOMMENDATION:\n  {result.clinical_recommendation}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Diabetic Retinopathy for Retinal Images")
    parser.add_argument("--image-path", type=str, required=True, help="Path to retinal image or folder of images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model weights checkpoint")
    parser.add_argument("--output-json", type=str, default=None, help="Save predictions to JSON file")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cuda/cpu)")
    args = parser.parse_args()

    input_path = Path(args.image_path)
    if not input_path.exists():
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)

    logger.info("Initializing RetinaCare-AI DRPredictor...")
    predictor = DRPredictor(checkpoint_path=args.checkpoint, device=args.device)

    results_list = []
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ]

    for img_file in image_files:
        try:
            res = predictor.predict(str(img_file), generate_gradcam=False)
            print_prediction_report(img_file.name, res)
            results_list.append({
                "image_file": img_file.name,
                "predicted_grade": res.predicted_grade,
                "class_name": res.class_name,
                "is_referable": res.is_referable,
                "confidence": res.confidence,
                "probabilities": res.probabilities,
                "clinical_recommendation": res.clinical_recommendation,
            })
        except Exception as e:
            logger.error(f"Failed to run inference on {img_file.name}: {e}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2)
        logger.info(f"Saved prediction results to: {args.output_json}")


if __name__ == "__main__":
    main()
