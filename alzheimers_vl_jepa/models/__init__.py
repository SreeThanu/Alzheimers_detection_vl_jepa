"""
models/__init__.py
------------------
Exposes the top-level model and its submodules for easy import.
"""

from .image_encoder import LightweightCNNEncoder
from .text_encoder import TextEncoder
from .vl_jepa_model import VLJEPAModel

__all__ = ["LightweightCNNEncoder", "TextEncoder", "VLJEPAModel"]
