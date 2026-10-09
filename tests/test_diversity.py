import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.inference import DRPredictor

def main():
    predictor = DRPredictor(checkpoint_path="checkpoints/best_model.pth")

    samples = [
        ("Grade 0 (Normal)", "data/samples/grade_0_sample.png"),
        ("Grade 1 (Mild DR)", "data/samples/grade_1_sample.png"),
        ("Grade 2 (Moderate DR)", "data/samples/grade_2_sample.png"),
        ("Grade 3 (Severe DR)", "data/samples/grade_3_sample.png"),
        ("Grade 4 (Proliferative DR)", "data/samples/grade_4_sample.png"),
    ]

    print("\n" + "=" * 75)
    print(f"{'Input Target':<28} | {'Predicted Grade':<18} | {'Confidence':<12} | {'Referable':<10}")
    print("-" * 75)

    for target_name, path in samples:
        res = predictor.predict(path, generate_gradcam=True)
        ref_str = "YES (Referral)" if res.is_referable else "NO (Routine)"
        print(f"{target_name:<28} | Grade {res.predicted_grade:<12} | {res.confidence * 100:>6.1f}%     | {ref_str:<10}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
