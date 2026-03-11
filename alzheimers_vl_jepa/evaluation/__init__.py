"""
evaluation/__init__.py
-----------------------
Exposes evaluation utilities.
"""

from .metrics import compute_metrics
from .evaluate import evaluate_model

__all__ = ["compute_metrics", "evaluate_model"]
