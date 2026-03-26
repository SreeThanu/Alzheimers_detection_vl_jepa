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

import torch

from utils.config import load_config
from utils.helpers import set_seed, get_device
from data.dataset_loader import build_dataloaders, CLASS_NAMES, SimpleTokenizer
from data.preprocessing import get_train_transform, get_val_transform
from models.vl_jepa_model import VLJEPAModel
from training.trainer import Trainer
from utils.visualization import plot_training_history


def compute_inverse_frequency_class_weights(
    train_loader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Derive ``CrossEntropyLoss`` class weights from the **training** split only.

    1. Count samples per class: ``n_c``.
    2. Inverse frequency: ``w_c = 1 / n_c`` (``n_c >= 1`` via clamp).
    3. Normalize so weights sum to ``num_classes`` (mean 1.0).

    Returns:
        Float tensor ``[num_classes]`` on ``device``.
    """
    ds = train_loader.dataset
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for i in range(len(ds)):
        _, _, y = ds[i]
        yi = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
        counts[yi] += 1.0

    inv = 1.0 / counts.clamp(min=1.0)
    inv = inv * (float(num_classes) / inv.sum())
    return inv.to(device=device, dtype=torch.float32)


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

    # ── Dataset root from config (parent of OriginalDataset/, etc.)
    data_root = cfg["paths"]["data_root"]
    ds_cfg    = cfg["dataset"]

    # ── Transforms ──────────────────────────────────────────────────
    image_size = ds_cfg["image_size"]
    train_tf   = get_train_transform(image_size)
    val_tf     = get_val_transform(image_size)

    # ── DataLoaders — OriginalDataset-only + stratified split when enabled in YAML ──
    train_loader, val_loader, test_loader, vocab_size = build_dataloaders(
        data_root       = data_root,
        train_transform = train_tf,
        val_transform   = val_tf,
        batch_size      = cfg["training"]["batch_size"],
        train_frac      = float(ds_cfg.get("train_frac", 0.70)),
        val_frac        = float(ds_cfg.get("val_frac", 0.15)),
        num_workers     = cfg["training"]["num_workers"],
        seed            = cfg["project"]["seed"],
        max_seq_len     = cfg["text_encoder"]["max_seq_len"],
        class_prompts   = ds_cfg.get("class_prompts"),
        use_original_dataset_only=bool(ds_cfg.get("use_original_dataset_only", False)),
    )

    # ── Model ────────────────────────────────────────────────────────
    vlj = cfg["vl_jepa"]
    use_txt = bool(vlj.get("use_text_branch", True))
    cache_txt = bool(vlj.get("cache_text_embeddings", False)) and use_txt
    class_tokens = None
    if cache_txt:
        prompts = ds_cfg["class_prompts"]
        mx = cfg["text_encoder"]["max_seq_len"]
        tok = SimpleTokenizer(prompts, max_seq_len=mx)
        class_tokens = torch.stack([tok.encode(prompts[name]) for name in CLASS_NAMES])

    model = VLJEPAModel(
        vocab_size               = vocab_size,
        embedding_dim            = cfg["image_encoder"]["embedding_dim"],
        projection_dim           = vlj["projection_dim"],
        num_classes              = vlj["num_classes"],
        dropout                  = vlj["dropout"],
        use_text                 = use_txt,
        cache_text_embeddings    = cache_txt,
        class_token_ids          = class_tokens,
        use_attention_fusion     = bool(vlj.get("use_attention_fusion", False)) and use_txt,
        fusion_dropout           = float(vlj.get("fusion_dropout", 0.0)),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    tr_cfg = cfg["training"]
    n_cls = int(cfg["vl_jepa"]["num_classes"])
    use_cw = bool(tr_cfg.get("use_class_weights", False))
    class_weights = (
        compute_inverse_frequency_class_weights(train_loader, n_cls, device)
        if use_cw
        else None
    )
    if use_cw and class_weights is not None:
        print(f"Class weights (inverse freq, normalized): {class_weights.cpu().tolist()}")

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        device           = device,
        lr               = tr_cfg["learning_rate"],
        weight_decay     = tr_cfg["weight_decay"],
        epochs           = tr_cfg["epochs"],
        checkpoint_dir   = cfg["paths"]["checkpoint_dir"],
        checkpoint_name  = tr_cfg["checkpoint_name"],
        use_amp          = tr_cfg["use_amp"],
        early_stopping_patience = tr_cfg["early_stopping"]["patience"],
        class_weights    = class_weights,
    )

    history = trainer.fit()

    # ── Temperature scaling (validation NLL → T); for inference only ──
    eval_cfg = cfg.get("evaluation", {})
    if bool(eval_cfg.get("temperature_scaling", False)):
        from utils.calibration import TemperatureScaler, save_temperature

        model.eval()
        scaler = TemperatureScaler.fit_from_validation(
            model, val_loader, device=device
        )
        out_dir = cfg["paths"].get("outputs_dir", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        t_path = os.path.join(out_dir, "calibration_temperature.json")
        save_temperature(t_path, scaler.temperature)
        print(
            f"Temperature scaling: T = {scaler.temperature:.4f} saved → {t_path} "
            "(applied at evaluation inference only)"
        )

    # ── Save training curves ─────────────────────────────────────────
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved → {plot_path}")

    return history, model, test_loader, device, cfg


if __name__ == "__main__":
    run_training()
