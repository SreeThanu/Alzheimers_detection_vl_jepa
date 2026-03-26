"""
gradcam.py
----------
Grad-CAM for the lightweight CNN image encoder (last convolutional layer).

Hooks the final ``Conv2d`` in ``model.image_encoder.features`` (output of the last
``ConvBlock`` before BN/ReLU/Pool), backprops the target class logit, and builds
a spatial heatmap overlaid on the input MRI.

Example::

    from utils.gradcam import generate_gradcam, save_gradcam_overlay
    # x: [1, 3, H, W] on device, model.eval()
    cam = generate_gradcam(model, x, target_class=2, device=device)
    save_gradcam_overlay(model, x, target_class=2, save_path="outputs/gradcam/explain.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_gradcam_target_conv(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the last convolutional layer used for Grad-CAM.

    For ``LightweightCNNEncoder`` this is the first submodule of the last
    ``ConvBlock`` (``Conv2d``), whose output is [B, 256, 8, 8] for 128×128 input.
    """
    enc = model.image_encoder
    last_block = enc.features[-1]
    conv = last_block.block[0]
    if not isinstance(conv, torch.nn.Conv2d):
        raise TypeError(
            f"Expected last ConvBlock's first layer to be Conv2d, got {type(conv)}"
        )
    return conv


def _class_token_row(
    class_index: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Token ids [1, L] for the default prompt of ``CLASS_NAMES[class_index]``."""
    from data.dataset_loader import CLASS_NAMES, DEFAULT_CLASS_PROMPTS, SimpleTokenizer

    prompts = DEFAULT_CLASS_PROMPTS
    tok = SimpleTokenizer(prompts, max_seq_len=max_seq_len)
    name = CLASS_NAMES[int(class_index)]
    row = tok.encode(prompts[name]).unsqueeze(0).to(device)
    return row


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: Optional[torch.device] = None,
    max_seq_len: int = 16,
    token_ids: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for one image and a chosen class index.

    Args:
        model: ``VLJEPAModel`` (or any module with ``image_encoder`` as above).
        image_tensor: ``[1, 3, H, W]`` (typically normalized like training).
        target_class: Class index ``k``; gradients are taken from ``logits[0, k]``.
        device: Run device; default: ``image_tensor.device`` or model device.
        max_seq_len: Used if ``token_ids`` is built from default prompts.
        token_ids: Optional ``[1, L]`` long tensor; if None, uses default class prompt
            for ``target_class`` (aligned with the explained class).

    Returns:
        Heatmap ``[H_in, W_in]`` float32 in ``[0, 1]``, upsampled to input resolution.
    """
    if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
        raise ValueError("image_tensor must have shape [1, 3, H, W]")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    conv = get_gradcam_target_conv(model)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out: torch.Tensor) -> None:
        activations.append(out)

    def bwd_hook(_m, _grad_in, grad_out: Tuple[torch.Tensor, ...]) -> None:
        if grad_out[0] is not None:
            gradients.append(grad_out[0])

    h_fwd = conv.register_forward_hook(fwd_hook)
    h_bwd = conv.register_full_backward_hook(bwd_hook)

    img = image_tensor.to(device).detach().requires_grad_(True)

    if token_ids is None:
        token_ids = _class_token_row(target_class, max_seq_len, device)
    else:
        token_ids = token_ids.to(device)

    labels = torch.tensor([target_class], dtype=torch.long, device=device)

    try:
        model.zero_grad(set_to_none=True)
        out = model(img, token_ids, labels=labels)
        logits = out["logits"]
        score = logits[0, int(target_class)]
        score.backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

    # [1, C, h, w] → per-channel maps for batch item 0
    act = activations[0][0]
    grad = gradients[0][0]
    # Global average pooling of gradients per channel → importance weights
    weights = grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
    cam = (weights * act).sum(dim=0)  # [h, w]
    cam = F.relu(cam)

    cam_np = cam.detach().float().cpu().numpy()
    cmin, cmax = float(cam_np.min()), float(cam_np.max())
    if cmax > cmin:
        cam_np = (cam_np - cmin) / (cmax - cmin)
    else:
        cam_np = np.zeros_like(cam_np, dtype=np.float32)

    _, _, h, w = img.shape
    cam_t = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
    return cam_up.squeeze().numpy().astype(np.float32)


def tensor_to_rgb_uint8(
    image_1chw: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize CHW tensor to uint8 RGB [H, W, 3] for overlay."""
    t = image_1chw.detach().cpu().float().squeeze(0).numpy().transpose(1, 2, 0)
    m = np.array(mean, dtype=np.float32)
    s = np.array(std, dtype=np.float32)
    for c in range(3):
        t[..., c] = t[..., c] * s[c] + m[c]
    t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return t


def overlay_heatmap(
    rgb_uint8: np.ndarray,
    heatmap_01: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend jet colormap heatmap over ``rgb_uint8`` [H,W,3]."""
    import matplotlib.pyplot as plt

    try:
        cmap = plt.colormaps["jet"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("jet")
    colored = cmap(heatmap_01)[..., :3]
    colored = (colored * 255.0).astype(np.uint8)
    base = np.asarray(rgb_uint8, dtype=np.uint8)
    out = (alpha * colored.astype(np.float32) + (1.0 - alpha) * base.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def save_gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    save_path: str,
    device: Optional[torch.device] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    max_seq_len: int = 16,
) -> None:
    """Run ``generate_gradcam`` and write a jet-overlay PNG to ``save_path``."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = generate_gradcam(
        model, image_tensor, target_class, device=device, max_seq_len=max_seq_len
    )
    rgb = tensor_to_rgb_uint8(image_tensor, mean=mean, std=std)
    overlay = overlay_heatmap(rgb, heatmap)
    Image.fromarray(overlay).save(path)
