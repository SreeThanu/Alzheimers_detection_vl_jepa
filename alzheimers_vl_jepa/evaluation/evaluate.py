"""
evaluate.py
-----------
Load a trained model checkpoint and run a full evaluation on the test set.
Outputs:
  - Classification report (accuracy, precision, recall, F1)
  - ROC AUC one-vs-rest (per-class + macro) when probabilities are available
  - Per-sample predictions CSV under ``paths.outputs_dir``
  - Confusion matrix plot saved to experiments/results/
  - Returns metrics dict for programmatic use

Usage:
    python evaluation/evaluate.py
    # Or imported by main.py / notebooks
"""

import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.config import load_config
from utils.helpers import set_seed, get_device
from data.dataset_loader import build_dataloaders, CLASS_NAMES
from data.preprocessing import get_val_transform
from models.vl_jepa_model import VLJEPAModel
from evaluation.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix
from utils.gradcam import save_gradcam_overlay
from utils.calibration import load_temperature, apply_temperature


def _resolve_sample_path(dataset, index_in_loader_order: int) -> str:
    """Best-effort image path for CSV (indexed datasets expose ``get_image_path``)."""
    getter = getattr(dataset, "get_image_path", None)
    if callable(getter):
        return getter(index_in_loader_order)
    return ""


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
        metrics dict (accuracy, precision, recall, f1, confusion_matrix, optional AUC keys)
    """
    if cfg is None:
        cfg = load_config()

    if device is None:
        set_seed(cfg["project"]["seed"])
        device = get_device(cfg["device"])

    num_classes = int(cfg["vl_jepa"]["num_classes"])

    # ── Build test loader if not provided ───────────────────────────
    if test_loader is None:
        data_root = cfg["paths"]["data_root"]
        ds_cfg    = cfg["dataset"]
        val_tf    = get_val_transform(ds_cfg["image_size"])
        _, _, test_loader, vocab_size = build_dataloaders(
            data_root       = data_root,
            train_transform = val_tf,
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
    else:
        vocab_size = None

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
        val_tf = get_val_transform(cfg["dataset"]["image_size"])
        ds_cfg = cfg["dataset"]
        _, _, _, vocab_size = build_dataloaders(
            data_root       = cfg["paths"]["data_root"],
            train_transform = val_tf,
            val_transform   = val_tf,
            batch_size      = 1,
            num_workers     = 0,
            seed            = cfg["project"]["seed"],
            max_seq_len     = cfg["text_encoder"]["max_seq_len"],
            class_prompts   = ds_cfg.get("class_prompts"),
            train_frac      = float(ds_cfg.get("train_frac", 0.70)),
            val_frac        = float(ds_cfg.get("val_frac", 0.15)),
            use_original_dataset_only=bool(ds_cfg.get("use_original_dataset_only", False)),
        )
        vlj = cfg["vl_jepa"]
        ds_prompts = cfg["dataset"]["class_prompts"]
        use_txt = bool(vlj.get("use_text_branch", True))
        cache_txt = bool(vlj.get("cache_text_embeddings", False)) and use_txt
        class_tokens = None
        if cache_txt:
            from data.dataset_loader import SimpleTokenizer

            mx = cfg["text_encoder"]["max_seq_len"]
            tok = SimpleTokenizer(ds_prompts, max_seq_len=mx)
            class_tokens = torch.stack([tok.encode(ds_prompts[n]) for n in CLASS_NAMES])

        model = VLJEPAModel(
            vocab_size            = vocab_size,
            embedding_dim         = cfg["image_encoder"]["embedding_dim"],
            projection_dim        = vlj["projection_dim"],
            num_classes           = num_classes,
            dropout               = vlj["dropout"],
            use_text              = use_txt,
            cache_text_embeddings = cache_txt,
            class_token_ids       = class_tokens,
            use_attention_fusion  = bool(vlj.get("use_attention_fusion", False)) and use_txt,
            fusion_dropout        = float(vlj.get("fusion_dropout", 0.0)),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(val loss: {ckpt['val_loss']:.4f})"
        )

    model = model.to(device)
    model.eval()
    if getattr(model, "cache_text_embeddings", False):
        model.update_text_embedding_cache()

    outputs_dir = cfg["paths"].get("outputs_dir", "outputs")
    t_path = os.path.join(outputs_dir, "calibration_temperature.json")
    temperature = load_temperature(t_path)
    if temperature is None:
        temperature = 1.0
    else:
        print(f"Inference temperature scaling: T = {temperature:.4f} (from {t_path})")

    test_ds = test_loader.dataset
    all_preds: list = []
    all_labels: list = []
    all_probs: list = []
    all_conf: list = []
    all_paths: list = []
    offset = 0

    # ── Inference loop ─────────────────────────────────────────────
    with torch.no_grad():
        for images, tokens, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            tokens = tokens.to(device)
            labels_t = labels.to(device)
            out = model(images, tokens, labels=labels_t)
            logits = apply_temperature(out["logits"], temperature)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            bsz = labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu().numpy())
            all_conf.extend(conf.cpu().tolist())
            for j in range(bsz):
                all_paths.append(_resolve_sample_path(test_ds, offset + j))
            offset += bsz

    y_probs = np.vstack(all_probs) if all_probs else None

    # ── Metrics (existing + optional AUC) ────────────────────────────
    metrics = compute_metrics(
        all_labels,
        all_preds,
        class_names=CLASS_NAMES,
        verbose=True,
        y_probs=y_probs,
    )

    # ── Per-sample CSV ─────────────────────────────────────────────
    os.makedirs(outputs_dir, exist_ok=True)
    pred_csv = os.path.join(outputs_dir, "predictions.csv")
    prob_cols = [f"prob_class_{c}" for c in range(num_classes)]
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["image_path", "true_label", "predicted_label", "confidence"]
            + prob_cols
        )
        for i in range(len(all_labels)):
            ti, pi = all_labels[i], all_preds[i]
            row = [
                all_paths[i],
                CLASS_NAMES[ti],
                CLASS_NAMES[pi],
                f"{all_conf[i]:.6f}",
            ]
            row += [f"{y_probs[i, c]:.6f}" for c in range(num_classes)]
            w.writerow(row)
    print(f"Per-sample predictions saved → {pred_csv}")

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

    # ── Optional Grad-CAM (last conv of image encoder) ─────────────
    n_gc = int(cfg.get("evaluation", {}).get("gradcam_num_samples", 0))
    if n_gc > 0:
        gradcam_dir = os.path.join(outputs_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)
        ds_mean = tuple(cfg["dataset"].get("mean", [0.485, 0.456, 0.406]))
        ds_std = tuple(cfg["dataset"].get("std", [0.229, 0.224, 0.225]))
        mx_len = cfg["text_encoder"]["max_seq_len"]
        saved = 0
        model.eval()
        for images, _tokens, labels in test_loader:
            for b in range(images.size(0)):
                if saved >= n_gc:
                    break
                img_b = images[b : b + 1].to(device)
                tgt = int(labels[b].item())
                out_path = os.path.join(
                    gradcam_dir, f"sample_{saved:04d}_class{tgt}_{CLASS_NAMES[tgt]}.png"
                )
                save_gradcam_overlay(
                    model,
                    img_b,
                    target_class=tgt,
                    save_path=out_path,
                    device=device,
                    mean=ds_mean,
                    std=ds_std,
                    max_seq_len=mx_len,
                )
                saved += 1
            if saved >= n_gc:
                break
        print(f"Grad-CAM overlays saved → {gradcam_dir}/ ({saved} files)")

    return metrics


if __name__ == "__main__":
    evaluate_model()
