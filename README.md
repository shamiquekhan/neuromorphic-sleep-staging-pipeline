# NeuroSleep

**A compact deep learning pipeline for five-stage sleep classification from polysomnography signals.**

> End-to-end sleep-stage classification using EEG, EOG, and EMG signals with a sub-100K parameter model designed for edge deployment.

[![Tests](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline/actions/workflows/tests.yml)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Final Results](#final-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training](#training)
- [Adaptation Methods](#adaptation-methods)
- [Evaluation](#evaluation)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Deployment](#deployment)
- [Live Resources](#live-resources)
- [Reproducibility](#reproducibility)
- [Team](#team)
- [License](#license)

---

## Overview

NeuroSleep is a lightweight sleep-stage classification system that scores five sleep stages (Wake, N1, N2, N3, REM) from polysomnography (PSG) signals. The model processes 300 seconds of context (10 × 30-second epochs) and classifies each epoch using 4-channel input:

| Channel | Signal | Purpose |
|---------|--------|---------|
| Fpz-Cz | EEG | Frontal-central brain activity |
| Pz-Oz | EEG | Parietal-occipital brain activity |
| EOG | EOG | Eye movement detection |
| EMG | EMG | Muscle tone for REM detection |

**Key Properties:**
- **99,477 parameters** — ~400 KB when serialized
- **300s context window** — captures temporal sleep architecture
- **Edge-ready** — designed for resource-constrained deployment (MCUs, wearables)
- **5-class classification** — Wake, N1, N2, N3, REM per AASM criteria

---

## Final Results

### 100-Subject Benchmark (Authoritative)

**Dataset:** Sleep-EDF Expanded (92 subjects, 10-fold subject-level CV)
**Seeds:** 3 seeds × 10 folds = 30 folds per method
**Model:** Improved Student — Full Fine-Tuning (99,477 params)

| Metric | Value |
|--------|-------|
| **Accuracy** | **87.7% ± 2.7%** |
| **Cohen's Kappa** | **0.763 ± 0.043** |
| **Macro F1** | **0.730 ± 0.037** |
| **Weighted F1** | **89.0% ± 2.1%** |
| **MGm** | **0.797 ± 0.040** |
| **Parameters** | **99,477** |

### Per-Class Performance (Full Fine-Tuning, 30 Folds)

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.964 ± 0.016 | 0.995 ± 0.003 | 0.936 ± 0.031 |
| N1 | **0.445 ± 0.061** | 0.330 ± 0.065 | 0.712 ± 0.070 |
| N2 | 0.768 ± 0.041 | 0.878 ± 0.043 | 0.687 ± 0.063 |
| N3 | 0.681 ± 0.114 | 0.562 ± 0.135 | 0.896 ± 0.075 |
| REM | 0.771 ± 0.078 | 0.772 ± 0.077 | 0.785 ± 0.118 |

### Adaptation Method Comparison

| Model | Trainable Params | Accuracy | κ | Macro F1 |
|-------|----------------:|---------:|----:|---------:|
| Frozen Base | 0 (0%) | 87.1% ± 3.6% | 0.738 ± 0.077 | 0.673 ± 0.074 |
| LoRA CNN+Head (r=8) | 1,448 (1.43%) | 83.6% ± 3.7% | 0.693 ± 0.057 | 0.674 ± 0.045 |
| **Full Fine-Tuning** | **99,477 (100%)** | **87.7% ± 2.7%** | **0.763 ± 0.043** | **0.730 ± 0.037** |

> **Key finding:** Full fine-tuning achieves the strongest overall and stage-balanced performance. LoRA CNN+Head uses 68.7× fewer trainable parameters while retaining 95.4% of full FT accuracy.

---

## Architecture

```
PSG Input (Fpz-Cz, Pz-Oz, EOG, EMG)  [B, 10, 4, 3000]
      ↓
Multi-Resolution Stem (2 parallel Conv1d branches)
  ├── Short kernel (25) — fast transients
  └── Long kernel (200) — slow oscillations
      ↓
Concatenation  [B, 10, 16, ~500]
      ↓
Depthwise-Separable CNN (2 blocks)
  ├── Block 0: DW Conv1d(16→16) + PW Conv1d(16→32) + BN
  └── Block 1: DW Conv1d(32→32) + PW Conv1d(32→32) + BN
      ↓
Adaptive Average Pooling → [B, 10, 32, 8]
      ↓
Parametric Gabor Feature Extraction (8 learnable filters)
  ├── Learnable frequency (0.5–30 Hz)
  ├── Learnable sigma (bandwidth)
  └── Projection: Linear(8→16)
      ↓
Feature Fusion → [B, 10, 272]  (32×8 + 16)
      ↓
2-Layer GRU (hidden=64, 300s context)
      ↓
Linear(64→5) + Softmax
      ↓
Wake / N1 / N2 / N3 / REM
```

### Module Breakdown

| Module | Parameters | Description |
|--------|------------|-------------|
| Stem (Short + Long) | 7,232 | Multi-resolution feature extraction |
| Encoder (2 blocks) | 2,304 | Depthwise-separable CNN |
| Gabor FEB | 144 | Parametric spectral features |
| GRU | 89,856 | Temporal context modeling |
| Head | 325 | 5-class classification |
| **Total** | **99,477** | |

### Input/Output Specification

| Property | Value |
|----------|-------|
| Input Shape | `[batch, 10, 4, 3000]` |
| Output Shape | `[batch, 10, 5]` |
| Sampling Rate | 100 Hz |
| Epoch Length | 30 seconds (3,000 samples) |
| Sequence Length | 10 epochs (300 seconds) |
| Channels | 4 (Fpz-Cz, Pz-Oz, EOG, EMG) |

---

## Repository Structure

```
neurosleep/
│
├── src/sleep_staging/                    # Core Python package
│   ├── models/
│   │   └── improved_student.py           # ImprovedStudent architecture (99,477 params)
│   ├── adaptation/
│   │   └── lora.py                       # LoRA implementation (LoRALinear, LoRAConv1d)
│   ├── inference/
│   │   └── predictor.py                  # Inference engine with batch processing
│   ├── data/
│   │   ├── loader.py                     # Cached subject loading (.npz files)
│   │   ├── labels.py                     # Stage mapping, canonical subject list
│   │   └── dataset.py                    # EEG dataset handling
│   ├── preprocessing/
│   │   ├── filtering.py                  # Bandpass + notch filters
│   │   └── quality.py                    # Signal quality control
│   ├── training/
│   │   └── cross_dataset.py              # SequenceDataset, class weights, training loop
│   ├── evaluation/
│   │   └── metrics.py                    # Accuracy, κ, F1, per-class metrics
│   ├── visualization/
│   │   ├── hypnogram.py                  # Hypnogram plotting
│   │   └── signals.py                    # Signal visualization
│   ├── config.py                         # StudentConfig, PreprocessingConfig, paths
│   └── utils/                            # Utility functions
│
├── app/                                  # Streamlit dashboard (Swiss design)
│   ├── streamlit_app.py                  # Main entry point
│   ├── components.py                     # Reusable UI components
│   ├── state.py                          # Session state management
│   └── pages/
│       ├── 01_Dashboard.py               # Main dashboard with inference
│       ├── 02_Signal_Viewer.py           # Raw signal visualization
│       ├── 03_Sleep_Night_Explorer.py    # Hypnogram explorer
│       └── 04_Model_Information.py       # Architecture & results
│
├── scripts/                              # CLI entry points
│   ├── train_adaptation.py               # Frozen / LoRA / Full FT benchmark
│   ├── run_100_subject_benchmark.py      # 100-subject from-scratch training
│   ├── aggregate_adaptation_results.py   # Fold-level multi-seed aggregation
│   ├── protocol_fingerprint.py           # Pre-run consistency verification
│   ├── evaluate_final_model.py           # Evaluate final checkpoint
│   ├── run_4fold_simple.py               # Development 4-fold CV
│   ├── download_sleep_edf_expanded.py    # Dataset download
│   ├── prepare_dataset.py                # Data preparation
│   ├── smoke_test_lora.py                # LoRA verification tests
│   └── audit_100_subjects.py             # Dataset quality audit
│
├── configs/                              # YAML configuration files
│   ├── full_100_subject.yaml             # 100-subject benchmark config
│   ├── final.yaml                        # 15-subject development config
│   └── benchmark_canonical.yaml          # Canonical benchmark config
│
├── artifacts/                            # Model checkpoints
│   └── final/
│       └── student_full_finetuned.pt     # Authoritative final checkpoint
│
├── results/                              # Evaluation results
│   ├── 100_subject_adaptation/           # Adaptation comparison study
│   │   ├── final/                        # 3-seed aggregate (authoritative)
│   │   │   ├── aggregate_metrics.json    # Mean ± std across 30 folds
│   │   │   ├── overall_comparison.csv    # Overall metrics
│   │   │   ├── per_class_comparison.csv  # Per-stage metrics
│   │   │   └── FINAL_ADAPTATION_RESULTS.md
│   │   ├── frozen/                       # Frozen base results
│   │   ├── lora_r8_enc.0.pw_enc.1.pw_head/  # LoRA CNN+Head results
│   │   └── full_finetune/                # Full FT results
│   ├── full_100_subject/                 # From-scratch benchmark (seed 42)
│   ├── audit/                            # Protocol verification
│   │   ├── benchmark_comparison.json     # Original vs adaptation comparison
│   │   ├── benchmark_side_by_side.json   # Side-by-side config comparison
│   │   └── protocol_fingerprint_seed42.json  # Reference fingerprint
│   └── final/                            # 15-subject development results
│
├── data/
│   ├── manifests/
│   │   ├── canonical_subject_folds_92subj.json  # 10-fold CV splits
│   │   ├── canonical_subject_folds.json         # 4-fold development splits
│   │   └── sleep_edf_expanded.json              # Subject metadata
│   └── cache/
│       └── sleep_edf/                    # Cached .npz files (92 subjects)
│
├── tests/                                # Pytest test suite
│   ├── test_lora.py                      # LoRA wrapping tests
│   ├── test_lora_conv1d.py              # Conv1d LoRA tests
│   ├── test_model_new.py                # Model architecture tests
│   ├── test_evaluation.py               # Metrics tests
│   └── test_checkpoint.py               # Checkpoint loading tests
│
├── docs/                                 # Documentation
│   ├── results.md                        # Authoritative results document
│   ├── LIMITATIONS.md                    # Per-class analysis & limitations
│   ├── architecture.md                   # Architecture deep dive
│   ├── dataset.md                        # Dataset documentation
│   ├── methodology.md                    # Training methodology
│   └── team.md                           # Team contributions
│
├── notebooks/                            # Jupyter notebooks (analysis)
├── deployment/                           # Deployment artifacts
│   ├── app.py                            # Deployment app
│   └── config/inference.yaml             # Inference config
│
├── hf_model_card.md                      # Hugging Face model card
├── MODEL_REPORT.md                       # Detailed model report
├── GUIDE.md                              # Project guide
├── requirements.txt                      # Python dependencies
└── pyproject.toml                        # Package configuration
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline.git
cd neuromorphic-sleep-staging-pipeline

# Create conda environment (recommended)
conda create -n neurosleep python=3.11
conda activate neurosleep

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Run Inference

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sleep_staging.models.improved_student import ImprovedStudent

# Download checkpoint
path = hf_hub_download(
    repo_id="shamique/Light-Weight-Neuromorphic-Sleep-Stage-Model",
    filename="student_full_finetuned.safetensors",
)

# Load model
model = ImprovedStudent()
model.load_state_dict(load_file(path, device="cpu"))
model.eval()

# Run inference on preprocessed PSG data
# Input: [batch, 10, 4, 3000] — 10 epochs, 4 channels, 3000 samples @ 100Hz
x = torch.randn(1, 10, 4, 3000)  # replace with real data

with torch.inference_mode():
    logits = model(x)              # [1, 10, 5]
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)   # [1, 10]

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
for i in range(10):
    print(f"Epoch {i}: {STAGE_NAMES[preds[0, i].item()]} ({probs[0, i, preds[0, i]].item():.2%})")
```

### Evaluate the Final Model

```bash
python scripts/evaluate_final_model.py
```

### Run 100-Subject Benchmark

```bash
# Full fine-tuning (seed 42)
python scripts/train_adaptation.py --mode full_finetune --seed 42 --device cuda

# Frozen base
python scripts/train_adaptation.py --mode frozen --seed 42 --device cuda

# LoRA CNN+Head
python scripts/train_adaptation.py --mode lora --targets enc.0.pw,enc.1.pw,head --rank 8 --alpha 16 --seed 42 --device cuda

# Aggregate all seeds
python scripts/aggregate_adaptation_results.py
```

---

## Dataset

### Sleep-EDF Expanded

- **Source:** PhysioNet Sleep-EDF Expanded database
- **Total downloaded:** 100 subjects
- **Excluded:** 8 wake-only subjects (no sleep stages)
- **Final cohort:** 92 subjects
- **Channels:** Fpz-Cz, Pz-Oz (EEG), EOG, EMG
- **Sampling rate:** 100 Hz
- **Epoch length:** 30 seconds

### Class Distribution

| Stage | Percentage | Description |
|-------|------------|-------------|
| Wake | ~68% | Awake state |
| N1 | ~4.6% | Light sleep (transitional) |
| N2 | ~17% | Intermediate sleep |
| N3 | ~6% | Deep sleep (slow-wave) |
| REM | ~6% | Rapid eye movement |

### Cross-Validation Splits

- **Method:** 10-fold subject-level CV
- **Fold assignment:** Canonical (canonical_subject_folds_92subj.json)
- **No data leakage:** Each subject appears in exactly one fold's test set
- **All-position supervision:** Every epoch in the 10-epoch window is supervised

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Epochs | 20 |
| Early Stopping Patience | 5 |
| Batch Size | 32 |
| Scheduler | Cosine Annealing |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | True (CUDA AMP) |
| Class Weights | N1=2×, REM=2× |

### Reproducibility

```bash
# Save protocol fingerprint
python scripts/protocol_fingerprint.py --seed 42 --mode full_finetune --save-reference

# Verify protocol matches
python scripts/protocol_fingerprint.py --seed 43 --mode full_finetune
```

The fingerprint includes:
- Dataset manifest hash (SHA-256)
- Checkpoint hash
- Config hash
- Git commit
- PyTorch/CUDA version
- GPU name
- All training hyperparameters

---

## Adaptation Methods

### Frozen Base

- Load pre-trained checkpoint
- Freeze all parameters
- Evaluate directly on test folds
- **0 trainable parameters**

### LoRA (Low-Rank Adaptation)

```python
from sleep_staging.adaptation.lora import LoRAConfig, apply_lora

lora_config = LoRAConfig(
    rank=8,
    alpha=16,
    target_modules=["enc.0.pw", "enc.1.pw", "head"],
    dropout=0.05,
)
model = apply_lora(model, lora_config)
# trainable params: 1,448 || all params: 99,477 || trainable%: 1.43%
```

### Full Fine-Tuning

- Load pre-trained checkpoint
- Train all 99,477 parameters
- Best overall performance

### Comparison

| Method | Params | Accuracy | κ | Macro F1 | Accuracy Retention |
|--------|-------:|---------:|----:|---------:|-------------------:|
| Frozen | 0 | 87.1% | 0.738 | 0.673 | 99.3% |
| LoRA CNN+Head | 1,448 | 83.6% | 0.693 | 0.674 | 95.4% |
| **Full FT** | **99,477** | **87.7%** | **0.763** | **0.730** | **100%** |

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Cohen's κ | Inter-rater agreement (chance-corrected) |
| Macro F1 | Unweighted mean of per-class F1 |
| Weighted F1 | Support-weighted mean of per-class F1 |
| MGm | Geometric mean of per-class recalls |

### Per-Stage Analysis

**N1 (F1=0.445):** Most challenging stage due to:
- Brief, transitional nature (1-7 minutes)
- Low prevalence (~4.6% of epochs)
- Physiological overlap with Wake and N2

**Strengths:**
- High Wake F1 (0.964) — excellent awake detection
- Strong N2 detection (0.768) — light sleep well-distinguished
- Balanced performance across all stages

---

## Streamlit Dashboard

The project includes a Swiss-design Streamlit dashboard with 4 pages:

| Page | Description |
|------|-------------|
| **Dashboard** | Main inference interface with real-time prediction |
| **Signal Viewer** | Raw PSG signal visualization |
| **Sleep Night Explorer** | Hypnogram exploration |
| **Model Information** | Architecture, results, reproducibility |

### Launch

```bash
streamlit run app/streamlit_app.py
```

The dashboard automatically loads results from `results/100_subject_adaptation/final/aggregate_metrics.json`.

---

## Deployment

### Edge Deployment

The model is designed for resource-constrained devices:

| Property | Value |
|----------|-------|
| Parameters | 99,477 |
| Model Size | ~400 KB (FP32) |
| Inference Time | <10ms per epoch (CPU) |
| Memory | <10 MB |

### Hugging Face

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

path = hf_hub_download(
    repo_id="shamique/Light-Weight-Neuromorphic-Sleep-Stage-Model",
    filename="student_full_finetuned.safetensors",
)
```

---

## Live Resources

| Resource | Link |
|----------|------|
| **Interactive Demo** | [Hugging Face Space](https://huggingface.co/spaces/shamiquekhan/neurosleep-demo) |
| **Model** | [Hugging Face Model Hub](https://huggingface.co/shamique/Light-Weight-Neuromorphic-Sleep-Stage-Model) |
| **Reproducibility** | [Kaggle Notebook](https://www.kaggle.com/shamiquekhan/neurosleep-final) |
| **Source** | [GitHub](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) |

---

## Reproduction

### Full Benchmark

```bash
# 1. Verify protocol
python scripts/protocol_fingerprint.py --seed 42 --mode full_finetune --save-reference

# 2. Run all 3 modes × 3 seeds
for seed in 42 43 44; do
  python scripts/train_adaptation.py --mode frozen --seed $seed --device cuda
  python scripts/train_adaptation.py --mode lora --targets enc.0.pw,enc.1.pw,head --rank 8 --alpha 16 --seed $seed --device cuda
  python scripts/train_adaptation.py --mode full_finetune --seed $seed --device cuda
done

# 3. Aggregate results
python scripts/aggregate_adaptation_results.py
```

### Expected Output

```
FINAL ADAPTATION BENCHMARK — ALL SEEDS AGGREGATED
======================================================================
  Model                  Params     Accuracy            κ     Macro F1
----------------------------------------------------------------------
  Frozen                      0 0.8706±0.0362 0.7378±0.0773 0.6725±0.0740
  LoRA CNN+Head           1,448 0.8361±0.0366 0.6934±0.0572 0.6736±0.0454
  Full Fine-Tuning       99,477 0.8766±0.0267 0.7632±0.0427 0.7302±0.0367
======================================================================
```

---

## Configuration

The final model configuration is in `configs/full_100_subject.yaml`:

```yaml
model:
  name: ImprovedStudent
  params: 99477

data:
  dataset: Sleep-EDF Expanded
  subjects: 92
  channels: [Fpz-Cz, Pz-Oz, EOG, EMG]
  sampling_rate: 100
  epoch_seconds: 30

training:
  optimizer: AdamW
  lr: 3e-4
  weight_decay: 1e-4
  epochs: 20
  early_stopping_patience: 5
  batch_size: 32
  class_weights:
    N1: 2.0
    REM: 2.0
  mixed_precision: true

evaluation:
  method: 10-fold subject-level CV
  seeds: [42, 43, 44]
  sequence_length: 10
  stride: 5
  supervision: all_position
```

---

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance, Streamlit Dashboard |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

> Commits consolidated by Shamique Khan for repo hygiene. See [docs/team.md](docs/team.md) for individual contributions.

---

## License

CC-BY-4.0 (Creative Commons Attribution 4.0 International)

See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@project{neurosleep_2026,
  title={NeuroSleep: Light-Weight Sleep Stage Scoring},
  author={Kaushik, P. and Vora, S. and Bhatt, S. and Khan, S. and Lone, A.J.},
  year={2026},
  institution={VIT Bhopal University}
}
```
