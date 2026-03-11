"""
evaluate.py
-----------
Load a trained model checkpoint and run a full evaluation on the test set.
Outputs:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix plot saved to experiments/results/
  - Returns metrics dict for programmatic use

Usage:
    python evaluation/evaluate.py
    # Or imported by main.py / notebooks
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from utils.config import load_config
from utils.helpers import set_seed, get_device
from data.dataset_loader import build_dataloaders, CLASS_NAMES
from data.preprocessing import get_val_transform
from models.vl_jepa_model import VLJEPAModel
from evaluation.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix


def evaluate_model(
    model: torch.nn.Module = None,
    test_loader=None,
    device: torch.device = None,
    cfg: dict = None,
    checkpoint_path: str = None,
) -> dict:
    """
    Run evaluation on the test set.

    Can be called:
      (a) Directly with an already-loaded model + test_loader
      (b) With no arguments → loads config, checkpoint, and test data automatically

    Returns:
        metrics dict (accuracy, precision, recall, f1, confusion_matrix)
    """
    # ── Load config if not provided ─────────────────────────────────
    if cfg is None:
        cfg = load_config()

    if device is None:
        set_seed(cfg["project"]["seed"])
        device = get_device(cfg["device"])

    # ── Build test loader if not provided ───────────────────────────
    if test_loader is None:
        data_root = cfg["paths"]["data_root"]
        val_tf    = get_val_transform(cfg["dataset"]["image_size"])
        _, _, test_loader, vocab_size = build_dataloaders(
            data_root       = data_root,
            train_transform = val_tf,
            val_transform   = val_tf,
            batch_size      = cfg["training"]["batch_size"],
            num_workers     = cfg["training"]["num_workers"],
            seed            = cfg["project"]["seed"],
            max_seq_len     = cfg["text_encoder"]["max_seq_len"],
        )
    else:
        vocab_size = None  # Will be inferred from model (already loaded)

    # ── Load model if not provided ───────────────────────────────────
    if model is None:
        ckpt_path = checkpoint_path or os.path.join(
            cfg["paths"]["checkpoint_dir"],
            cfg["training"]["checkpoint_name"],
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Train the model first with: python main.py"
            )
        # We need vocab_size to build the model — get it from a temp dataset
        val_tf = get_val_transform(cfg["dataset"]["image_size"])
        _, _, _, vocab_size = build_dataloaders(
            data_root       = cfg["paths"]["data_root"],
            train_transform = val_tf,
            val_transform   = val_tf,
            batch_size      = 1,
            num_workers     = 0,
            seed            = cfg["project"]["seed"],
            max_seq_len     = cfg["text_encoder"]["max_seq_len"],
        )
        model = VLJEPAModel(
            vocab_size     = vocab_size,
            embedding_dim  = cfg["image_encoder"]["embedding_dim"],
            projection_dim = cfg["vl_jepa"]["projection_dim"],
            num_classes    = cfg["vl_jepa"]["num_classes"],
            dropout        = cfg["vl_jepa"]["dropout"],
            use_text       = cfg["vl_jepa"]["use_text_branch"],
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(val loss: {ckpt['val_loss']:.4f})")

    model = model.to(device)
    model.eval()

    # ── Inference loop ─────────────────────────────────────────────
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, tokens, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            tokens = tokens.to(device)
            out    = model(images, tokens)
            preds  = out["logits"].argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── Metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_preds, class_names=CLASS_NAMES, verbose=True)

    # ── Confusion matrix plot ───────────────────────────────────────
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=cm_path,
    )
    print(f"Confusion matrix saved → {cm_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
