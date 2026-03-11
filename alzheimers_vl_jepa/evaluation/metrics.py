"""
metrics.py
----------
Classification metrics for multi-class Alzheimer's stage prediction.
Uses scikit-learn — no GPU needed for metric computation.

Available:
  - accuracy
  - per-class precision, recall, F1
  - macro-averaged F1
  - confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import List, Optional


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute a full suite of classification metrics.

    Args:
        y_true:       Ground-truth label list  (ints)
        y_pred:       Predicted label list      (ints)
        class_names:  Human-readable class names for the report
        verbose:      If True, print the full classification report

    Returns:
        dict with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    f1        = f1_score(y_true, y_pred,        average="macro", zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    if verbose:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true, y_pred,
            target_names=class_names or [str(i) for i in range(cm.shape[0])],
            zero_division=0,
        ))
        print(f"Macro F1 Score : {f1:.4f}")
        print(f"Accuracy       : {accuracy:.4f}")
        print("=" * 60 + "\n")

    return {
        "accuracy":         float(accuracy),
        "precision":        float(precision),
        "recall":           float(recall),
        "f1":               float(f1),
        "confusion_matrix": cm,
    }
