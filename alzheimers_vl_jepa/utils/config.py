"""
config.py
---------
Loads and merges all YAML config files into a single flat dict.
This avoids heavy config libraries (Hydra / OmegaConf) while keeping
things readable.

Usage:
    from utils.config import load_config
    cfg = load_config()          # loads defaults
    cfg = load_config("configs") # explicit config directory
"""

import os
import yaml


# Default config directory (relative to project root)
_DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "configs"
)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str = None) -> dict:
    """
    Load and merge all YAML config files.

    Merges (in order, later files override):
      config.yaml          → project-level settings
      model_config.yaml    → model architecture
      training_config.yaml → training hyperparameters
      dataset_config.yaml  → dataset settings

    Returns:
        Flat merged dict accessible with cfg["key"]["subkey"]
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    config_dir = os.path.abspath(config_dir)

    filenames = [
        "config.yaml",
        "model_config.yaml",
        "training_config.yaml",
        "dataset_config.yaml",
    ]

    merged = {}
    for fname in filenames:
        path = os.path.join(config_dir, fname)
        if os.path.exists(path):
            part = _load_yaml(path)
            merged = _deep_merge(merged, part)
        else:
            print(f"[config] Warning: {path} not found — skipping.")

    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
