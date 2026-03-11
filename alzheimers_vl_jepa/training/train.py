"""
train.py
--------
Entry-point script for training.
1. Load configuration
2. Build transforms and dataloaders
3. Instantiate VLJEPAModel
4. Run Trainer.fit()
5. Save training history plots

Usage:
    python training/train.py
    # Or via main.py (recommended)
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import load_config
from utils.helpers import set_seed, get_device
from data.dataset_loader import build_dataloaders
from data.preprocessing import get_train_transform, get_val_transform
from models.vl_jepa_model import VLJEPAModel
from training.trainer import Trainer
from utils.visualization import plot_training_history


def run_training(cfg=None):
    """
    Main training function.

    Args:
        cfg: Config dict (if None, loads from config files)

    Returns:
        history dict with loss/acc curves
    """
    if cfg is None:
        cfg = load_config()

    # ── Reproducibility ─────────────────────────────────────────────
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["device"])
    print(f"Using device: {device}")

    # ── Dataset root (parent that contains AugmentedAlzheimerDataset/, OriginalDataset/, etc.)
    data_root  = cfg["paths"]["data_root"]   # e.g. "data/raw"

    # ── Transforms ──────────────────────────────────────────────────
    image_size = cfg["dataset"]["image_size"]
    train_tf   = get_train_transform(image_size)
    val_tf     = get_val_transform(image_size)

    # ── DataLoaders — leakage-free stratified 70/15/15 split ────────
    train_loader, val_loader, test_loader, vocab_size = build_dataloaders(
        data_root       = data_root,
        train_transform = train_tf,
        val_transform   = val_tf,
        batch_size      = cfg["training"]["batch_size"],
        train_frac      = 0.70,
        val_frac        = 0.15,
        num_workers     = cfg["training"]["num_workers"],
        seed            = cfg["project"]["seed"],
        max_seq_len     = cfg["text_encoder"]["max_seq_len"],
    )

    # ── Model ────────────────────────────────────────────────────────
    model = VLJEPAModel(
        vocab_size      = vocab_size,
        embedding_dim   = cfg["image_encoder"]["embedding_dim"],
        projection_dim  = cfg["vl_jepa"]["projection_dim"],
        num_classes     = cfg["vl_jepa"]["num_classes"],
        dropout         = cfg["vl_jepa"]["dropout"],
        use_text        = cfg["vl_jepa"]["use_text_branch"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        device           = device,
        lr               = cfg["training"]["learning_rate"],
        weight_decay     = cfg["training"]["weight_decay"],
        epochs           = cfg["training"]["epochs"],
        checkpoint_dir   = cfg["paths"]["checkpoint_dir"],
        checkpoint_name  = cfg["training"]["checkpoint_name"],
        use_amp          = cfg["training"]["use_amp"],
        early_stopping_patience = cfg["training"]["early_stopping"]["patience"],
    )

    history = trainer.fit()

    # ── Save training curves ─────────────────────────────────────────
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved → {plot_path}")

    return history, model, test_loader, device, cfg


if __name__ == "__main__":
    run_training()
