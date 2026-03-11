"""
visualization.py
----------------
Plotting utilities for:
  - Training / validation loss and accuracy curves
  - Confusion matrix heatmap

Output files are saved as PNG to experiments/results/.
All plotting is non-interactive (backend=Agg) so it works on headless systems.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for laptops and servers
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_training_history(
    history: dict,
    save_path: str = "experiments/results/training_history.png",
):
    """
    Plot training and validation loss + accuracy curves side by side.

    Args:
        history:   Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Output PNG file path
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # ── Loss ─────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336", linewidth=2, linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#4CAF50", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="#FF9800", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "experiments/results/confusion_matrix.png",
    title: str = "Confusion Matrix",
):
    """
    Plot a colour-coded confusion matrix heatmap.

    Args:
        cm:           Confusion matrix array [n_classes, n_classes]
        class_names:  List of class label strings
        save_path:    Output PNG file path
        title:        Plot title
    """
    n = cm.shape[0]
    class_names = class_names or [str(i) for i in range(n)]

    # Normalise for percentage annotations
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Annotate cells with raw count + percentage
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                fontsize=8, color=color,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_sample_images(
    images,          # List of PIL Images or tensors [C,H,W]
    labels: list,
    class_names: list,
    save_path: str = "experiments/results/sample_images.png",
    n_cols: int = 4,
):
    """
    Display a grid of sample MRI images with their class labels.
    Optionally used in notebooks for quick sanity checks.
    """
    import math
    import torch

    n = len(images)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()

    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i]
        if isinstance(img, torch.Tensor):
            # Denormalize and convert to HWC numpy
            from data.preprocessing import denormalize
            img = denormalize(img).permute(1, 2, 0).numpy()
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(class_names[lbl], fontsize=9)
        ax.axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("MRI Sample Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
