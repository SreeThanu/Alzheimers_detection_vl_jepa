"""
evaluation_plots.py
-------------------
Extra figures for multi-class model evaluation (ROC, PR, per-class bars).
Safe to import in notebooks (does not force matplotlib Agg).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)


def plot_roc_curves_one_vs_rest(
    y_true: Sequence[int],
    y_probs: np.ndarray,
    class_names: List[str],
    figsize: Tuple[float, float] = (7.5, 6.5),
    title: str = "ROC curves (one-vs-rest)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    One ROC curve per class using predicted probability for that class.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    n_classes = y_probs.shape[1]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_pr_curves_one_vs_rest(
    y_true: Sequence[int],
    y_probs: np.ndarray,
    class_names: List[str],
    figsize: Tuple[float, float] = (7.5, 6.5),
    title: str = "Precision–recall curves (one-vs-rest)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    One precision–recall curve per class (good for imbalanced classes).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    n_classes = y_probs.shape[1]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
        ap = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP = {ap:.3f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_per_class_metrics_bars(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    figsize: Tuple[float, float] = (9, 4.5),
    title: str = "Per-class precision, recall, F1",
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart for precision / recall / F1 per class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(class_names)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n), zero_division=0
    )

    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, prec, width, label="Precision", color="#4C72B0")
    ax.bar(x, rec, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1, width, label="F1", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_confidence_histogram(
    y_probs: np.ndarray,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (9, 4),
    title: str = "Max softmax confidence (correct vs incorrect)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Histogram of prediction confidence (max probability) for correct vs wrong predictions.
    """
    y_probs = np.asarray(y_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    conf = y_probs.max(axis=1)
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(0, 1, 31)
    ax.hist(conf[correct], bins=bins, alpha=0.65, label=f"Correct (n={correct.sum()})", color="#2ca02c")
    ax.hist(conf[~correct], bins=bins, alpha=0.65, label=f"Wrong (n={(~correct).sum()})", color="#d62728")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_class_distribution_comparison(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    figsize: Tuple[float, float] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Side-by-side bar counts: true label distribution vs predicted label distribution."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(class_names)
    true_counts = np.bincount(y_true, minlength=n)
    pred_counts = np.bincount(y_pred, minlength=n)

    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, true_counts, width, label="True labels", color="#4C72B0")
    ax.bar(x + width / 2, pred_counts, width, label="Predicted labels", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Class distribution: true vs predicted", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax
