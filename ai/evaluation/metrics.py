"""
Evaluation & Clinical Metrics for Diabetic Retinopathy
======================================================
Computes:
1. Quadratic Weighted Kappa (QWK) - Standard clinical benchmark metric
2. Multiclass Accuracy, Precision, Recall, F1-score
3. Referable DR Sensitivity (Recall on Grades 2,3,4) & Specificity (Grades 0,1)
4. Full Confusion Matrix
"""

from typing import Any, Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DR_CLASS_LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


def calculate_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> Dict[str, Any]:
    """
    Computes complete statistical and clinical metrics.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # 1. Quadratic Weighted Kappa (QWK)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # 2. General Multiclass Metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # 3. Referable DR Clinical Metrics (Grades 2,3,4 vs 0,1)
    y_true_ref = (y_true >= 2).astype(int)
    y_pred_ref = (y_pred >= 2).astype(int)

    ref_sensitivity = recall_score(y_true_ref, y_pred_ref, pos_label=1, zero_division=0)
    # Specificity = Recall on negative class (0)
    ref_specificity = recall_score(y_true_ref, y_pred_ref, pos_label=0, zero_division=0)
    ref_accuracy = accuracy_score(y_true_ref, y_pred_ref)
    ref_f1 = f1_score(y_true_ref, y_pred_ref, zero_division=0)

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    # 5. Detailed Classification Report
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=DR_CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "qwk": round(float(qwk), 4),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "referable_dr": {
            "sensitivity": round(float(ref_sensitivity), 4),
            "specificity": round(float(ref_specificity), 4),
            "accuracy": round(float(ref_accuracy), 4),
            "f1": round(float(ref_f1), 4),
        },
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }
