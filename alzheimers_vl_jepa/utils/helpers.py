"""
helpers.py
----------
General-purpose utility functions:
  - Reproducibility: fix random seeds across Python / NumPy / PyTorch
  - Device detection: auto-select CUDA > MPS (Apple Silicon) > CPU
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Fix all random seeds for reproducibility.
    Must be called BEFORE any dataset creation or model initialisation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (small performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(preference: str = "auto") -> torch.device:
    """
    Return the best available torch.device.

    Priority order (when preference == 'auto'):
      CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    Args:
        preference: 'auto' | 'cuda' | 'mps' | 'cpu'

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available — falling back to CPU.")
        return torch.device("cpu")

    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available — falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Convert seconds to mm:ss string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
