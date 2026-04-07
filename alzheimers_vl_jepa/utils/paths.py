"""
paths.py
--------
Resolve paths from ``configs/*.yaml`` relative to the **project root**
(``alzheimers_vl_jepa/``), not the process current working directory.

This fixes notebooks started from ``notebooks/`` where relative paths like
``experiments/checkpoints/best_model.pt`` would otherwise resolve under
``notebooks/experiments/...``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


def project_root() -> Path:
    """Directory containing ``configs/``, ``data/``, ``experiments/``, etc."""
    return Path(__file__).resolve().parent.parent


def resolve_project_path(relative_or_abs: str) -> str:
    """
    Resolve a config path (e.g. ``data/raw``, ``experiments/checkpoints``).

    Absolute paths are returned unchanged (normalized). Relative paths are
    anchored to :func:`project_root`.
    """
    p = Path(relative_or_abs)
    if p.is_absolute():
        return str(p)
    return str((project_root() / p).resolve())


def resolve_checkpoint_path(cfg: Dict, checkpoint_path: Optional[str] = None) -> str:
    """
    Full path to the training checkpoint ``.pt`` file.

    If ``checkpoint_path`` is omitted, uses
    ``paths.checkpoint_dir`` + ``training.checkpoint_name`` under the project root.

    If ``checkpoint_path`` is relative, we try **project root** first, then
    **current working directory** (so ``../experiments/...`` from ``notebooks/``
    still works).
    """
    if checkpoint_path:
        p = Path(checkpoint_path)
        if p.is_absolute():
            return str(p)
        cand_root = (project_root() / p).resolve()
        if cand_root.is_file():
            return str(cand_root)
        cand_cwd = (Path.cwd() / p).resolve()
        if cand_cwd.is_file():
            return str(cand_cwd)
        return str(cand_root)

    return str(
        (project_root() / cfg["paths"]["checkpoint_dir"] / cfg["training"]["checkpoint_name"]).resolve()
    )


def list_checkpoints_in_dir(cfg: Dict) -> List[str]:
    """Absolute paths of ``*.pt`` files under ``paths.checkpoint_dir`` (project root)."""
    d = project_root() / cfg["paths"]["checkpoint_dir"]
    if not d.is_dir():
        return []
    return sorted(str(p.resolve()) for p in d.glob("*.pt"))


def find_checkpoint_for_inference(
    cfg: Dict,
    checkpoint_path: Optional[str] = None,
    *,
    allow_fallback: bool = True,
) -> str:
    """
    Resolve a checkpoint path for loading a trained model.

    1. If ``checkpoint_path`` is set, use :func:`resolve_checkpoint_path` (must exist
       unless a fallback is allowed and the directory has other ``*.pt`` files).
    2. Else use the default ``checkpoint_dir`` / ``checkpoint_name`` from config.
    3. If that file is missing and ``allow_fallback`` is True, use the **newest**
       ``*.pt`` under ``checkpoint_dir`` (by modification time).

    Returns:
        Absolute path to an existing ``.pt`` file.

    Raises:
        FileNotFoundError: No usable checkpoint found (includes hints and directory listing).
    """
    primary = resolve_checkpoint_path(cfg, checkpoint_path)
    if os.path.isfile(primary):
        return primary

    ckpt_dir = project_root() / cfg["paths"]["checkpoint_dir"]
    candidates = list_checkpoints_in_dir(cfg)

    if allow_fallback and candidates:
        # Prefer newest file if default name is missing
        by_mtime = sorted(
            candidates,
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        chosen = by_mtime[0]
        print(
            f"[paths] Default checkpoint not found:\n  {primary}\n"
            f"[paths] Using newest available: {chosen}"
        )
        return chosen

    hint = (
        f"No checkpoint found.\n"
        f"  Expected (primary): {primary}\n"
        f"  Project root: {project_root()}\n"
        f"  Checkpoint directory: {ckpt_dir}\n"
    )
    if candidates:
        hint += f"  Found .pt files (set CHECKPOINT_PATH or rename to best_model.pt): {candidates}\n"
    else:
        hint += (
            f"  No .pt files in that directory. Train first from the project root:\n"
            f"    python main.py --mode train\n"
        )
    hint += (
        "If you edited code, restart the Jupyter kernel so imports pick up changes.\n"
    )
    raise FileNotFoundError(hint)
