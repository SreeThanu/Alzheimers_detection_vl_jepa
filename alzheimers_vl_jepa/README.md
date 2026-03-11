# Alzheimer's Disease Detection — VL-JEPA

> Lightweight Vision-Language JEPA-style model for Alzheimer's stage classification from MRI images.  
> Optimised to run on a personal laptop (Apple M2 / 8 GB RAM).

---

## Project Overview

This project classifies MRI brain scans into four Alzheimer's disease stages:

| Class | Description |
|---|---|
| `NonDemented` | No signs of dementia |
| `VeryMildDemented` | Very mild cognitive impairment |
| `MildDemented` | Mild cognitive impairment |
| `ModerateDemented` | Significant brain atrophy |

**Architecture**: A lightweight CNN Image Encoder is aligned with a Text Encoder (simple embedding lookup) using JEPA-style contrastive projection heads. A classification head outputs the final prediction.

```
MRI Image ──► CNN Encoder ──► Img Projection ─┐
                                               ├─► Fuse ──► Classifier ──► Stage
Text Prompt ─► Txt Encoder ──► Txt Projection ─┘         (InfoNCE Loss)
```

---

## Dataset Setup

Download from Kaggle: **"Augmented Alzheimer MRI Dataset"**

### Option A — Kaggle CLI (Recommended)
```bash
pip install kaggle
# Put your kaggle.json API key at ~/.kaggle/kaggle.json
bash scripts/download_dataset.sh
```

### Option B — Manual Download
1. Go to: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
2. Download and unzip
3. Place files at:
```
alzheimers_vl_jepa/data/raw/augmented_alzheimer_mri_dataset/
  train/
    MildDemented/
    ModerateDemented/
    NonDemented/
    VeryMildDemented/
  test/
    ...same structure...
```

---

## Installation

```bash
# 1. Clone or navigate into the project
cd alzheimers_vl_jepa

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
# Train and evaluate (default)
python main.py

# Train only
python main.py --mode train

# Evaluate a saved checkpoint only
python main.py --mode evaluate

# Or use the convenience script
bash scripts/run_training.sh
```

Training outputs are saved to:
```
experiments/
  checkpoints/best_model.pt
  results/training_history.png
  results/confusion_matrix.png
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_dataset_exploration.ipynb` | Visualize MRI samples, class counts, image stats |
| `02_training_demo.ipynb` | Run training, plot loss/accuracy curves |
| `03_model_evaluation.ipynb` | Load checkpoint, compute metrics, plot confusion matrix |

```bash
cd notebooks
jupyter notebook
```

---

## Configuration

All hyperparameters live in `configs/`:

| File | Controls |
|---|---|
| `config.yaml` | Project paths and device |
| `model_config.yaml` | Encoder dims, dropout |
| `training_config.yaml` | LR, batch size, early stopping |
| `dataset_config.yaml` | Image size, augmentations, class prompts |

---

## Laptop Optimisation Notes

- **Image size**: 128×128 (configurable) — small enough for fast CPU training
- **Batch size**: 16 — fits in 8 GB RAM
- **Mixed precision**: enabled on MPS (Apple Silicon) and CUDA
- **Num workers**: 2 (set to 0 in notebooks for safety)
- **Model size**: ~500K parameters — trains in minutes per epoch on CPU
- **Early stopping**: prevents wasted compute (default patience = 7)

---

## Requirements

```
torch ≥ 2.0
torchvision ≥ 0.15
numpy, pandas, scikit-learn
matplotlib, tqdm, Pillow, PyYAML
```

No external NLP libraries (CLIP, HuggingFace, etc.) are required.
