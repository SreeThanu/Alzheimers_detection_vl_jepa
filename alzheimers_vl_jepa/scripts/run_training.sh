#!/usr/bin/env bash
# ================================================================
# run_training.sh
# Convenience wrapper to start the training pipeline
# ================================================================

set -e

echo "=== Alzheimer's VL-JEPA Training ==="
echo "Starting training with default config..."
echo ""

# Activate a virtual environment if one exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv/"
fi

# Run training
python main.py --mode both

echo ""
echo "Training done! Check experiments/ for:"
echo "  checkpoints/best_model.pt"
echo "  results/training_history.png"
echo "  results/confusion_matrix.png"
