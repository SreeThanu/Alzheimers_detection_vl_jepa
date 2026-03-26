"""
calibration.py
--------------
Post-hoc temperature scaling for multi-class probabilities (Guo et al., 2017).

Learns a single scalar ``T > 0`` on a validation set by minimizing NLL
(cross-entropy) on ``logits / T``. Apply **only at inference** (evaluation /
deployment), not during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class TemperatureScaler:
    """
    Holds a scalar temperature and optional fit logic.

    Usage:
        scaler = TemperatureScaler.fit_from_validation(model, val_loader, device)
        calibrated_logits = scaler.scale_logits(raw_logits)
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return ``logits / T`` (same shape as ``logits``)."""
        return logits / self.temperature

    @classmethod
    def fit_from_validation(
        cls,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> "TemperatureScaler":
        """
        Collect logits on the validation loader (no grad through the model),
        then optimize ``T`` to minimize NLL: ``CrossEntropyLoss(logits / T, y)``.
        """
        model.eval()
        logit_chunks: list = []
        label_chunks: list = []

        with torch.no_grad():
            for images, tokens, labels in tqdm(
                val_loader, desc="  Calib (val logits)", leave=False
            ):
                images = images.to(device)
                tokens = tokens.to(device)
                labels = labels.to(device)
                out = model(images, tokens, labels=labels)
                logit_chunks.append(out["logits"].detach().cpu())
                label_chunks.append(labels.detach().cpu())

        if not logit_chunks:
            raise RuntimeError("TemperatureScaler: empty validation loader")

        logits = torch.cat(logit_chunks, dim=0).to(device)
        y = torch.cat(label_chunks, dim=0).to(device)

        t = cls._optimize_temperature(logits, y, max_iter=max_iter, lr=lr)
        return cls(temperature=t)

    @staticmethod
    def _optimize_temperature(
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """LBFGS on ``log_T`` so ``T = exp(log_T)`` stays positive."""
        logits = logits.detach().to(labels.device).float()
        labels = labels.long()

        log_t = nn.Parameter(torch.zeros(1, device=logits.device, dtype=torch.float32))
        opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            t = log_t.exp().clamp(min=1e-3, max=100.0)
            loss = F.cross_entropy(logits / t, labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(log_t.exp().clamp(min=1e-3, max=100.0).detach().cpu().item())


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Functional: ``logits / temperature`` (for use when you only have ``T``)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def save_temperature(path: str, temperature: float) -> None:
    """Persist ``{"temperature": T}`` JSON next to other ``outputs/`` artifacts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"temperature": float(temperature)}, f, indent=2)


def load_temperature(path: str) -> Optional[float]:
    """Load ``T`` from JSON; return ``None`` if missing."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("temperature", 1.0))


"""
Example (evaluation-only inference)
-------------------------------------

.. code-block:: python

    from utils.calibration import load_temperature, apply_temperature
    import torch.nn.functional as F

    T = load_temperature("outputs/calibration_temperature.json") or 1.0
    with torch.no_grad():
        out = model(images, tokens, labels=labels)
        logits = apply_temperature(out["logits"], T)
        probs = F.softmax(logits, dim=1)
"""
