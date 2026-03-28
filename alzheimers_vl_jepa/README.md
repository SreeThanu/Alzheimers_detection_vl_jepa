# Vision–Language JEPA-Style Learning for Multi-Stage Alzheimer’s Detection from Structural MRI

---

## 2. Abstract

Alzheimer’s disease (AD) remains a leading cause of dementia worldwide; early and accurate staging from non-invasive neuroimaging supports clinical decision-making and trial stratification. Deep learning has shown promise for automated analysis of structural magnetic resonance imaging (MRI), yet many systems rely on image-only convolutional networks with limited semantic grounding and weak calibration under class imbalance. This work presents a **vision–language Joint-Embedding Predictive Architecture (VL-JEPA)–inspired** framework that couples a lightweight convolutional image encoder with a compact text encoder over **clinically motivated class prompts**, fuses modalities via **learned projection heads** and optional **cross-attention**, and optimizes a **composite objective** combining supervised cross-entropy with **contrastive alignment** between image and text embeddings. Training incorporates **inverse-frequency class weighting**, **cosine learning-rate scheduling**, **mixed-precision optimization** where supported, and **post-hoc temperature scaling** on a held-out validation split for improved probability calibration. Evaluation reports **accuracy, macro-averaged precision/recall/F1, multi-class ROC–AUC (one-vs-rest), and confusion structure**; **Grad-CAM** overlays provide spatial attribution for clinical interpretability. On a publicly distributed **axial MRI slice classification benchmark** with four cognitive-severity categories, the pipeline is designed for **reproducible experimentation** on consumer hardware (CPU/MPS/CUDA). *Reported numerical scores are run-dependent; after training, aggregate metrics are emitted under `experiments/results/` and should be inserted into manuscripts as the authoritative results block.*

**Keywords:** Alzheimer’s disease; structural MRI; vision–language learning; JEPA-style embeddings; interpretability; calibration.

---

## 3. Introduction

### 3.1 Background

Alzheimer’s disease is characterized by progressive neurodegeneration and cognitive decline. Structural MRI captures macroscopic patterns—hippocampal and cortical atrophy, ventricular enlargement—that correlate with disease stage. Computational pathology from MRI aims to reduce inter-rater variability and surface subtle patterns early in the disease course.

### 3.2 Importance of Early and Fine-Grained Detection

Staging beyond binary “demented vs. not” (e.g., cognitively normal vs. very mild vs. mild vs. moderate impairment) supports prognostic modeling and enrichment for therapeutic trials. Multi-class MRI classification is challenging due to **inter-scan variability**, **class imbalance**, and **limited labeled cohorts** relative to natural-image benchmarks.

### 3.3 Limitations of Conventional Deep Learning Pipelines

Standard CNN baselines often (i) ignore complementary semantic supervision available in clinical ontologies and radiology lexicons, (ii) exhibit **miscalibrated softmax outputs** under shift and imbalance, and (iii) offer limited **post-hoc explanations** aligned with radiological review. Self-supervised and joint-embedding methods partially address representation quality but frequently omit explicit **language grounding** at decision time.

### 3.4 Motivation

We investigate a **vision–language joint embedding** with **JEPA-style projection heads** and optional **cross-modal fusion**, combining discriminative training with **prompt-based textual semantics** per class. This design seeks improved **semantic structure** of the embedding space, **interpretability** via Grad-CAM, and **calibrated** predictions via temperature scaling—while retaining a **lightweight encoder** suitable for research prototyping.

---

## 4. Objectives

### 4.1 Primary Goals

1. **Multi-class classification** of structural MRI slices into four ordered severity categories (see §5).
2. **Joint image–text representation learning** with supervised and alignment losses.
3. **Rigorous evaluation** with standard classification metrics and ROC-based analysis.

### 4.2 Secondary Goals

1. **Interpretability:** class-discriminative saliency maps (Grad-CAM) for qualitative validation.
2. **Calibration:** scalar temperature scaling fit on validation logits.
3. **Reproducibility:** unified configuration, fixed seeds, and scripted train/eval entry points.

---

## 5. Dataset Description

### 5.1 Source and Acquisition Assumptions

This codebase targets a **public, slice-based MRI classification corpus** distributed for research (e.g., community benchmarks derived from T1-weighted acquisitions). *The repository does not redistribute raw data; users must obtain the dataset under its license and place it under the configured root.* For manuscript preparation, authors may additionally benchmark on **ADNI** or other curated cohorts using the same preprocessing and model interface—report cohort identifiers, acquisition parameters, and ethics approvals per venue requirements.

### 5.2 Classes and Label Semantics

The implemented label space contains **four classes** (ordinal cognitive severity):

| Index | Class name           | Semantic description (abbrev.)   |
|------:|----------------------|----------------------------------|
| 0     | `NonDemented`        | No dementia / cognitively intact |
| 1     | `VeryMildDemented`   | Very mild cognitive decline      |
| 2     | `MildDemented`       | Mild dementia                    |
| 3     | `ModerateDemented`   | Moderate dementia                |

Class prompts used by the text encoder are specified in `configs/dataset_config.yaml` (full natural-language sentences per class).

### 5.3 Sample Counts and Splits

Image counts are **dataset-dependent**. The data loader supports:

- **Stratified splitting** of pooled images (configurable `train_frac`, `val_frac`, remainder test), or  
- **Explicit directories** `train/` and `test/` under `paths.data_root` for **leakage-free** evaluation (validation carved only from `train/`).

**Table 1.** *Summary of dataset characteristics (fill after download; counts are illustrative placeholders).*

| Characteristic              | Value / note                                      |
|----------------------------|---------------------------------------------------|
| Modality                   | Structural MRI (slices rendered as RGB)           |
| Input resolution (config)  | 128 × 128 pixels                                  |
| Number of classes          | 4                                                 |
| Train / val / test split   | Stratified 70% / 15% / 15% *or* explicit train/test |
| Class imbalance            | Typically present; mitigated via inverse-frequency weights |

### 5.4 Preprocessing

1. **Resize** to \(128 \times 128\) (configurable).
2. **Grayscale-to-RGB** replication where source images are single-channel.
3. **Normalization** using ImageNet mean/std (transfer-learning convention; configurable).
4. **Training augmentations (optional):** random horizontal flip, small rotation (\(\pm 10^\circ\)), mild brightness/contrast jitter.
5. **Validation/test:** deterministic resize and normalization only.

---

## 6. Methodology

### 6.1 Pipeline Overview

**End-to-end flow:**

1. **Configuration merging** (`configs/`) → unified hyperparameters.
2. **Data ingestion** (`data/dataset_loader.py`) → stratified or explicit-split DataLoaders; tokenized class prompts.
3. **Preprocessing** (`data/preprocessing.py`) → augmentation (train) / deterministic transforms (val/test).
4. **Model** (`models/vl_jepa_model.py`) → image + text branches, projections, fusion, classifier.
5. **Training** (`training/trainer.py`) → composite loss, AMP, early stopping, checkpointing.
6. **Evaluation** (`evaluation/evaluate.py`) → metrics, confusion matrix, optional Grad-CAM and temperature scaling.

### 6.2 Model Architecture

#### 6.2.1 Image Encoder

**LightweightCNNEncoder** (`models/image_encoder.py`): a **four-block** convolutional stack (Conv–BatchNorm–ReLU–MaxPool) with channel progression \(3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256\), **global average pooling**, and a linear projection to **embedding dimension** \(d=256\) (default). *Rationale:* parameter efficiency and stable optimization on small medical datasets compared to Vision Transformers on limited hardware.

#### 6.2.2 Text Encoder

**TextEncoder** (`models/text_encoder.py`): embedding bag over a **small vocabulary** derived from a deterministic tokenizer over class prompts; outputs \(d\)-dimensional embeddings aligned with the image branch.

#### 6.2.3 Projection and Fusion

- **ProjectionHead:** two-layer MLP (ReLU, dropout) mapping embeddings through a **bottleneck** (`projection_dim`, default 128) back to \(d\), in the spirit of JEPA/CLIP-style joint embedding learning.
- **Fusion:** configurable **element-wise sum** (`SumFusion`) or **cross-attention** (`CrossAttentionFusion` in `models/fusion.py`) with **queries from image features** and **keys/values from text features**, followed by a linear classifier over \(C=4\) classes.

#### 6.2.4 Optional Text Embedding Cache

For evaluation efficiency, **cached projected text embeddings** per class may be used when enabled (`cache_text_embeddings`), reducing redundant text-forward passes while preserving gradient paths during training as implemented.

### 6.3 Training Strategy

#### 6.3.1 Loss Functions

- **Primary:** **weighted cross-entropy** (optional **inverse-frequency class weights** estimated on the training split).
- **Auxiliary:** **contrastive alignment** between image and text projections (weight `contrastive_weight` in `Trainer`; default non-zero in code—see `training/trainer.py`).

#### 6.3.2 Optimizer and Scheduling

- **Optimizer:** Adam (configurable to SGD).
- **Weight decay:** L2 regularization (default \(10^{-4}\)).
- **Learning rate:** base LR \(10^{-3}\) with **cosine** decay (alternatives: step, none).

#### 6.3.3 Regularization and Stopping

- Dropout on encoder/projections/classifier (configurable).
- **Early stopping** on validation loss with patience and minimum improvement threshold.

#### 6.3.4 Train–Validation Split

Matches `dataset_config.yaml`: stratified **70% / 15% / 15%** when using single-root mode, or **train-directory split** (holdout fraction \(1 - \texttt{train\_frac}\) of train folder) when using explicit `train_dir` / `test_dir` in notebooks—**test images never leak** into training in explicit mode.

### 6.4 Experimental Setup

| Item            | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Framework       | PyTorch (\(\geq 2.0\)), torchvision, scikit-learn            |
| Precision       | Automatic mixed precision when enabled (CUDA/MPS paths)      |
| Hardware        | CPU, Apple MPS, or NVIDIA CUDA (device auto or config)       |
| Reproducibility | Global seed in `configs/config.yaml` (`project.seed`)      |

---

## 7. Results and Evaluation

*The following subsections define the **evaluation protocol**. Populate numeric cells from your training run logs and `experiments/results/`.*

### 7.1 Metrics

| Metric                         | Definition (multi-class)                                      |
|--------------------------------|---------------------------------------------------------------|
| Accuracy                       | Fraction of correct top-1 predictions                       |
| Precision / Recall / F1        | Macro average across classes unless noted otherwise           |
| ROC–AUC                        | One-vs-rest; macro average reported when applicable           |
| Confusion matrix               | Class-wise error structure                                    |

### 7.2 Performance Table (Baseline vs. Proposed)

**Table 2.** *Template for paper-ready comparison. Replace dashed entries with measured values.*

| Method                         | Accuracy | Macro-F1 | Macro ROC–AUC | Params / note        |
|--------------------------------|----------|----------|---------------|----------------------|
| CNN image-only (ablation)      | —        | —        | —             | `use_text_branch: false` |
| VL-JEPA (sum fusion)           | —        | —        | —             | `use_attention_fusion: false` |
| **VL-JEPA (cross-attention)**  | **—**    | **—**    | **—**         | **default fusion**   |

### 7.3 Visualizations (Describe)

- **Confusion matrix:** saved under `experiments/results/`; inspect class confusion between adjacent severity levels.
- **ROC curves:** per-class OvR curves when probabilities available (evaluation script).
- **Training curves:** loss and accuracy vs. epoch via `utils/visualization.plot_training_history`.

---

## 8. Model Interpretability

### 8.1 Grad-CAM

**Gradient-weighted Class Activation Mapping** produces a coarse spatial heatmap of input regions that most influence the predicted class logit. Implementation: `utils/gradcam.py`; overlays may be exported for the first \(N\) test samples (`evaluation.gradcam_num_samples`).

### 8.2 Clinical Relevance

Attribution maps support **qualitative alignment** with expected neuroanatomy (e.g., medial temporal structures); they do **not** constitute standalone clinical evidence and must be interpreted by qualified personnel.

### 8.3 Observed Patterns (Qualitative)

*Document in the paper:* whether saliency concentrates on hippocampal/medial temporal regions vs. diffuse patterns; compare across severity classes and failure cases.

---

## 9. Comparison with Existing Methods

| Family                     | Typical approach                         | Contrast with this work                                      |
|----------------------------|------------------------------------------|--------------------------------------------------------------|
| CNN baselines              | ResNet/EfficientNet fine-tuning          | We add **language grounding** + **alignment loss**         |
| Transfer learning          | ImageNet-pretrained encoders             | Our default encoder is **lightweight CNN**; backbone swappable |
| Self-supervised / JEPA     | Image-only latent prediction             | We use **paired text semantics** at train and optionally eval |
| Vision–language (CLIP-like) | Large-scale web pretraining            | We use **small vocabulary prompts** and **medical class priors** |

*Quantitative comparison requires running matched baselines on the same split—fill Table 2 accordingly.*

---

## 10. Ablation Study

**Table 3.** *Suggested ablations (implemented or configurable in this repository).*

| Configuration                    | Change                                      | Expected analysis                          |
|----------------------------------|---------------------------------------------|--------------------------------------------|
| No text branch                   | `use_text_branch: false`                    | Impact of vision–language grounding        |
| Sum vs. attention fusion         | `use_attention_fusion`                      | Capacity vs. overfitting trade-off         |
| No class weights                 | `use_class_weights: false`                | Effect under imbalance                     |
| No temperature scaling           | `temperature_scaling: false` in eval        | Calibration (ECE—extend metrics if needed)   |
| Contrastive weight \(= 0\)       | set `contrastive_weight` to 0 in `Trainer` | Role of alignment term                     |

---

## 11. Limitations

1. **Data:** Public slice benchmarks may not represent full 3D volume distribution; **site/scanner shift** is not modeled.
2. **Labels:** Slice-level labels may not reflect subject-level diagnosis; **longitudinal** modeling is out of scope.
3. **Model capacity:** Lightweight CNN may underfit complex morphometry; **pretrained backbones** are optional extensions.
4. **Text branch:** Prompts are hand-crafted; **prompt engineering sensitivity** not exhaustively studied.
5. **Generalization:** External validation on independent cohorts (e.g., ADNI vs. local hospital) is **required** for clinical claims.

---

## 12. Future Work

- **Multimodal fusion:** integrate demographics, cognitive scores, and fluid biomarkers.
- **Volume-level aggregation:** 3D CNNs or slice-sequence transformers with attention pooling.
- **Clinical deployment:** latency optimization, ONNX/TensorRT export, and integration with PACS.
- **Uncertainty:** deep ensembles or Bayesian last layers for epistemic uncertainty.
- **Fairness:** subgroup analysis across age, sex, and scanner protocol.

---

## 13. Reproducibility

### 13.1 Installation

```bash
cd alzheimers_vl_jepa
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 13.2 Requirements

See `requirements.txt` (PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, tqdm, Pillow, PyYAML). *Install a CUDA-enabled PyTorch build if using GPU.*

### 13.3 Data Layout

Place data under `data/raw/` per `configs/config.yaml` (`paths.data_root`). Supported layouts include explicit `train/` and `test/` class subfolders.

### 13.4 Training

```bash
python main.py --mode train
# or
python training/train.py
```

### 13.5 Evaluation / Inference

```bash
python main.py --mode evaluate
# or
python evaluation/evaluate.py
```

Full train-then-eval:

```bash
python main.py --mode both
```

Jupyter notebooks under `notebooks/` mirror the workflow for interactive experimentation.

---

## 14. Contributions

1. **Method:** A **vision–language JEPA-style** formulation with **dual projection heads**, optional **cross-attention fusion**, and **contrastive alignment** for MRI-based Alzheimer’s **multi-class** staging.
2. **Training practice:** **Class reweighting**, **cosine schedule**, **AMP**, **early stopping**, and **temperature scaling** for calibration-oriented evaluation.
3. **Interpretability:** **Grad-CAM** tooling for spatial attribution.
4. **Open implementation:** Modular configs and scripts suitable for **extension** to stronger backbones and external cohorts.

---

## 15. References

1. Jack, C. R., Jr., et al. “Tracking pathophysiological processes in Alzheimer’s disease: an updated hypothetical model of dynamic biomarkers.” *The Lancet Neurology* (2013).  
2. Litjens, G., et al. “A survey on deep learning in medical image analysis.” *Medical Image Analysis* (2017).  
3. Baevski, A., et al. “Data2vec: A general framework for self-supervised learning in speech, vision and language.” *ICML* (2022).  
4. Assran, M., et al. “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” *ICML* (2023).  
5. Radford, A., et al. “Learning Transferable Visual Models From Natural Language Supervision.” *ICML* (2021).  
6. Selvaraju, R. R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.” *ICCV* (2017).  
7. Guo, C., et al. “On Calibration of Modern Neural Networks.” *ICML* (2017).  
8. *Replace with ADNI citation and dataset-specific references when using those cohorts.*

---

## Citation (software)

If you use this repository in academic work, please cite the project URL and commit hash, and describe modifications to data and models in your paper’s experimental section.

---

*Document version aligned with repository configuration files (`configs/`, `models/`, `training/`, `evaluation/`). Update Tables 1–3 with empirical results before camera-ready submission.*
