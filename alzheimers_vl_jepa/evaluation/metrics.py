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
    roc_auc_score,
)
from typing import List, Optional


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
    y_probs: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute a full suite of classification metrics.

    Args:
        y_true:       Ground-truth label list  (ints)
        y_pred:       Predicted label list      (ints)
        class_names:  Human-readable class names for the report
        verbose:      If True, print the full classification report
        y_probs:      Optional [N, C] predicted probabilities for ROC AUC (one-vs-rest).

    Returns:
        dict with keys: accuracy, precision, recall, f1, confusion_matrix;
        plus auc_macro_ovr / auc_per_class_ovr when y_probs is provided.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    auc_macro: Optional[float] = None
    auc_per_class: Optional[np.ndarray] = None
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=np.float64)
        n_classes = y_probs.shape[1]
        try:
            auc_macro = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            )
            auc_per_class = roc_auc_score(
                y_true,
                y_probs,
                multi_class="ovr",
                average=None,
                labels=np.arange(n_classes),
            )
        except ValueError as e:
            if verbose:
                print(f"[metrics] ROC AUC (ovr) skipped: {e}")

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

    if verbose and auc_macro is not None and auc_per_class is not None:
        names = class_names or [str(i) for i in range(len(auc_per_class))]
        print("=" * 60)
        print("ROC AUC (one-vs-rest)")
        print("=" * 60)
        for i, name in enumerate(names):
            if i < len(auc_per_class):
                print(f"  {name:25s}: {float(auc_per_class[i]):.4f}")
        print(f"  {'Macro AUC':25s}: {auc_macro:.4f}")
        print("=" * 60 + "\n")

    out = {
        "accuracy":         float(accuracy),
        "precision":        float(precision),
        "recall":           float(recall),
        "f1":               float(f1),
        "confusion_matrix": cm,
    }
    if auc_macro is not None:
        out["auc_macro_ovr"] = auc_macro
    if auc_per_class is not None:
        out["auc_per_class_ovr"] = [float(x) for x in auc_per_class]
    return out
