"""
main.py
-------
Top-level entry point for the Alzheimer's VL-JEPA project.

Modes:
  --mode train    → train the model
  --mode evaluate → evaluate a saved checkpoint
  --mode both     → train then evaluate (default)

Usage:
    python main.py
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode both --config configs/config.yaml
"""

import argparse
import os
import sys

# ── Ensure project root is on the Python path ────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import load_config
from utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alzheimer's VL-JEPA — Train & Evaluate"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "both"],
        default="both",
        help="Which pipeline to run (default: both)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config directory (default: configs/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load merged config
    cfg = load_config(args.config)

    print("\n" + "=" * 60)
    print(f"  Project : {cfg['project']['name']}")
    print(f"  Mode    : {args.mode}")
    print("=" * 60 + "\n")

    model       = None
    test_loader = None
    device      = None

    if args.mode in ("train", "both"):
        from training.train import run_training
        history, model, test_loader, device, cfg = run_training(cfg)
        print("\nTraining complete!\n")

    if args.mode in ("evaluate", "both"):
        from evaluation.evaluate import evaluate_model
        metrics = evaluate_model(
            model       = model,        # None if mode == 'evaluate'
            test_loader = test_loader,  # None if mode == 'evaluate'
            device      = device,       # None if mode == 'evaluate'
            cfg         = cfg,
        )
        print(f"\nFinal Test Accuracy : {metrics['accuracy']:.4f}")
        print(f"Final Macro F1      : {metrics['f1']:.4f}\n")

    print("Done. Outputs saved to experiments/")


if __name__ == "__main__":
    main()
