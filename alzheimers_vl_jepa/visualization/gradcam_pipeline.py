"""
gradcam_pipeline.py
-------------------
High-level helpers to load a trained VL-JEPA checkpoint and export Grad-CAM
overlays for visualization (used by the notebook and optional CLI).

Relies on ``utils.gradcam`` for heatmap computation on the last conv layer
of ``LightweightCNNEncoder``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset_loader import CLASS_NAMES, SimpleTokenizer, build_dataloaders
from data.preprocessing import get_val_transform
from models.vl_jepa_model import VLJEPAModel
from utils.config import load_config
from utils.gradcam import generate_gradcam, overlay_heatmap, save_gradcam_overlay, tensor_to_rgb_uint8
from utils.paths import find_checkpoint_for_inference, resolve_project_path
from utils.vl_jepa_checkpoint import apply_checkpoint_arch_to_model_kwargs


def load_vl_jepa_from_checkpoint(
    cfg: Dict,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[VLJEPAModel, torch.device]:
    """
    Build ``VLJEPAModel`` from merged config and load ``model_state`` weights.

    Args:
        cfg: Merged config from ``load_config``.
        checkpoint_path: Path to ``.pt`` file; default ``checkpoint_dir/checkpoint_name``.
        device: Target device; default from ``cfg['device']`` via ``get_device``.

    Returns:
        (model, device) with model in eval mode and text cache updated if enabled.
    """
    from utils.helpers import get_device

    if device is None:
        device = get_device(cfg["device"])

    ckpt = find_checkpoint_for_inference(cfg, checkpoint_path, allow_fallback=True)
    state = torch.load(ckpt, map_location=device)
    sd = state["model_state"]

    # Need vocab_size: build minimal loaders (same as evaluation script)
    val_tf = get_val_transform(cfg["dataset"]["image_size"])
    ds_cfg = cfg["dataset"]
    _, _, _, vocab_size = build_dataloaders(
        data_root=resolve_project_path(cfg["paths"]["data_root"]),
        train_transform=val_tf,
        val_transform=val_tf,
        batch_size=2,
        train_frac=float(ds_cfg.get("train_frac", 0.70)),
        val_frac=float(ds_cfg.get("val_frac", 0.15)),
        num_workers=0,
        seed=cfg["project"]["seed"],
        max_seq_len=cfg["text_encoder"]["max_seq_len"],
        class_prompts=ds_cfg.get("class_prompts"),
        use_original_dataset_only=bool(ds_cfg.get("use_original_dataset_only", False)),
    )

    prompts = ds_cfg["class_prompts"]
    mx = cfg["text_encoder"]["max_seq_len"]
    tok = SimpleTokenizer(prompts, max_seq_len=mx)
    class_tokens_stacked = torch.stack([tok.encode(prompts[n]) for n in CLASS_NAMES])

    model = VLJEPAModel(
        **apply_checkpoint_arch_to_model_kwargs(cfg, sd, vocab_size, class_tokens_stacked)
    )
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    if getattr(model, "cache_text_embeddings", False):
        model.update_text_embedding_cache()

    print(f"Loaded checkpoint: {ckpt} (epoch {state.get('epoch', '?')})")
    return model, device


def build_test_loader_for_gradcam(
    cfg: Dict,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """Build test split DataLoader with val transforms (no augmentation)."""
    ds_cfg = cfg["dataset"]
    val_tf = get_val_transform(ds_cfg["image_size"])
    _, _, test_loader, _ = build_dataloaders(
        data_root=resolve_project_path(cfg["paths"]["data_root"]),
        train_transform=val_tf,
        val_transform=val_tf,
        batch_size=batch_size,
        train_frac=float(ds_cfg.get("train_frac", 0.70)),
        val_frac=float(ds_cfg.get("val_frac", 0.15)),
        num_workers=num_workers,
        seed=cfg["project"]["seed"],
        max_seq_len=cfg["text_encoder"]["max_seq_len"],
        class_prompts=ds_cfg.get("class_prompts"),
        use_original_dataset_only=bool(ds_cfg.get("use_original_dataset_only", False)),
    )
    return test_loader


def export_gradcam_batch(
    model: VLJEPAModel,
    test_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    output_dir: str | Path,
    num_samples: int = 12,
    target_mode: Literal["true_label", "predicted"] = "true_label",
    alpha: float = 0.45,
    file_prefix: str = "sample",
) -> List[str]:
    """
    Save Grad-CAM overlay PNGs for the first ``num_samples`` test images.

    Args:
        model: Trained ``VLJEPAModel`` in eval mode.
        test_loader: Test ``DataLoader`` (images, tokens, labels).
        cfg: Config (for mean/std and ``max_seq_len``).
        device: Torch device.
        output_dir: Directory for PNG files.
        num_samples: Maximum number of overlays to write.
        target_mode: ``true_label`` explains the ground-truth class logit;
            ``predicted`` explains the argmax prediction (may differ from label).
        alpha: Heatmap blend strength for ``overlay_heatmap``.

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_mean = tuple(cfg["dataset"].get("mean", [0.485, 0.456, 0.406]))
    ds_std = tuple(cfg["dataset"].get("std", [0.229, 0.224, 0.225]))
    mx_len = cfg["text_encoder"]["max_seq_len"]

    saved_paths: List[str] = []
    n = 0

    for images, tokens, labels in test_loader:
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)

        bsz = images.size(0)
        target_classes: List[int] = []

        if target_mode == "predicted":
            with torch.no_grad():
                out_d = model(images, tokens, labels=labels)
                preds = out_d["logits"].argmax(dim=1)
            for j in range(bsz):
                target_classes.append(int(preds[j].item()))
        else:
            for j in range(bsz):
                target_classes.append(int(labels[j].item()))

        for j in range(bsz):
            if n >= num_samples:
                break
            img_b = images[j : j + 1]
            tgt = target_classes[j]
            true_y = int(labels[j].item())

            path = out / f"{file_prefix}_{n:04d}_explain_cls{tgt}_{CLASS_NAMES[tgt]}_true{true_y}.png"
            save_gradcam_overlay(
                model,
                img_b.detach().cpu(),
                target_class=tgt,
                save_path=str(path),
                device=device,
                mean=ds_mean,
                std=ds_std,
                max_seq_len=mx_len,
                alpha=alpha,
            )
            saved_paths.append(str(path.resolve()))
            n += 1

        if n >= num_samples:
            break

    print(f"Wrote {len(saved_paths)} Grad-CAM overlays → {out}")
    return saved_paths


def gradcam_arrays_for_display(
    model: torch.nn.Module,
    image_1chw: torch.Tensor,
    target_class: int,
    device: torch.device,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    max_seq_len: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (rgb_uint8, heatmap_01, overlay_uint8) for a single ``[1,3,H,W]`` tensor.
    """
    img = image_1chw.to(device)
    heatmap = generate_gradcam(
        model,
        img,
        target_class=target_class,
        device=device,
        max_seq_len=max_seq_len,
    )
    rgb = tensor_to_rgb_uint8(image_1chw, mean=mean, std=std)
    overlay = overlay_heatmap(rgb, heatmap, alpha=0.45)
    return rgb, heatmap, overlay


def run_gradcam_cli() -> None:
    """Optional: ``python -m visualization.gradcam_pipeline`` from project root."""
    import argparse

    parser = argparse.ArgumentParser(description="Export Grad-CAM overlays from checkpoint")
    parser.add_argument("--config", default=None, help="Config directory (default: configs/)")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint .pt path")
    parser.add_argument("--out", default="outputs/gradcam_manual", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["true_label", "predicted"],
        default="true_label",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, device = load_vl_jepa_from_checkpoint(cfg, checkpoint_path=args.checkpoint)
    loader = build_test_loader_for_gradcam(cfg, batch_size=8, num_workers=0)
    export_gradcam_batch(
        model,
        loader,
        cfg,
        device,
        args.out,
        num_samples=args.num_samples,
        target_mode=args.mode,
    )


if __name__ == "__main__":
    run_gradcam_cli()
