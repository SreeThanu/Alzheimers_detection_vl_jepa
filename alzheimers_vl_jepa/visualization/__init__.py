"""Visualization utilities (Grad-CAM pipeline, figures)."""

from .gradcam_pipeline import (
    build_test_loader_for_gradcam,
    export_gradcam_batch,
    gradcam_arrays_for_display,
    load_vl_jepa_from_checkpoint,
)

__all__ = [
    "build_test_loader_for_gradcam",
    "export_gradcam_batch",
    "gradcam_arrays_for_display",
    "load_vl_jepa_from_checkpoint",
]
