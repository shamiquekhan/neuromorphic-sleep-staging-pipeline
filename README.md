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

### Primary Benchmark: Person-Level 10-Fold CV (EXP-BENCH-PERSON)

**Dataset:** Sleep-EDF Expanded — 92-record eligible cohort (52 persons), 10-fold person-level CV
**Seeds:** 3 seeds × 10 folds = 30 folds per method
**Model:** Improved Student — Trained from Scratch (99,477 params)
**Protocol:** Causal unique-epoch evaluation (stride=1, last-epoch supervision)

| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
| **Accuracy** | **87.30% ± 0.33%** | [85.80%, 88.79%] |
| **Cohen's Kappa** | **0.738 ± 0.010** | [0.691, 0.785] |
| **Macro F1** | **0.724 ± 0.005** | [0.693, 0.755] |
| **Weighted F1** | **0.880 ± 0.003** | [0.860, 0.900] |
| **MGm** | **0.772 ± 0.006** | [0.721, 0.823] |
| **Parameters** | **99,477** | |

### Per-Class Performance (Person-Level, 30 Folds)

| Stage | F1 | 95% CI |
|-------|-----|--------|
| Wake | 0.960 ± 0.026 | [0.950, 0.970] |
| N1 | **0.445 ± 0.087** | [0.413, 0.478] |
| N2 | 0.733 ± 0.163 | [0.672, 0.794] |
| N3 | 0.700 ± 0.112 | [0.658, 0.742] |
| REM | 0.782 ± 0.116 | [0.739, 0.826] |

### Historical Exhibition Result (EXP-EXHIBITION-15SUBJ)

> **Note:** This is a historical fixed 70/15/15 subject split under the legacy all-position protocol. Not the primary research benchmark.

| Metric | Value |
|--------|-------|
| **Accuracy** | **88.64%** |
| **Cohen's Kappa** | **0.7725** |
| **Macro F1** | **0.7186** |
| **Weighted F1** | **0.8953** |
| **Test Subjects** | 15 (held-out) |
| **Protocol** | Legacy all-position (stride=5) |

### Quarantined: Legacy Record-Level Benchmark

> **Warning:** The legacy "92-subject" benchmark used record-level folds with person-level leakage (SC4ss1/SC4ss2 = same person in train and test). Results are **record-level estimates only**, not person-generalization.

| Metric | Value (Quarantined) |
|--------|---------------------|
| Accuracy | 87.66% ± 2.22% |
| Cohen's κ | 0.763 ± 0.043 |
| Macro F1 | 0.730 ± 0.037 |

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
│   │   ├── improved_student.py           # ImprovedStudent architecture (99,477 params)
│   │   └── improved_teacher.py           # ImprovedTeacher for distillation
│   ├── adaptation/
│   │   └── lora.py                       # LoRA implementation
│   ├── inference/
│   │   └── predictor.py                  # Inference engine
│   ├── data/
│   │   ├── loader.py                     # Cached subject loading
│   │   ├── labels.py                     # Stage mapping
│   │   └── dataset.py                    # Dataset handling
│   ├── preprocessing/
│   │   ├── filtering.py                  # Bandpass + notch filters
│   │   └── quality.py                    # Signal quality control
│   ├── training/
│   │   └── cross_dataset.py              # SequenceDataset, training loop
│   ├── evaluation/
│   │   └── metrics.py                    # Accuracy, κ, F1, per-class metrics
│   ├── visualization/
│   │   ├── hypnogram.py                  # Hypnogram plotting
│   │   └── signals.py                    # Signal visualization
│   ├── config.py                         # StudentConfig, PreprocessingConfig
│   └── utils/                            # Utility functions
│
├── app/                                  # Streamlit dashboard
│   ├── streamlit_app.py                  # Main entry point
│   ├── components.py                     # Reusable UI components
│   ├── state.py                          # Session state management
│   └── pages/
│       ├── 01_Dashboard.py               # Main dashboard
│       ├── 02_Signal_Viewer.py           # Raw signal visualization
│       ├── 03_Sleep_Night_Explorer.py    # Hypnogram explorer
│       └── 04_Model_Information.py       # Architecture & results
│
├── scripts/                              # CLI entry points
│   ├── run_person_level_benchmark.py     # Primary benchmark training
│   ├── summarize_person_benchmark.py     # Fold-level aggregation
│   ├── protocol_fingerprint.py           # Pre-run consistency verification
│   ├── verify_protocol.py                # Protocol integrity check
│   ├── generate_person_folds.py          # Person-level fold generation
│   ├── audit_repository.py               # Repository consistency audit
│   ├── download_sleep_edf_expanded.py    # Dataset download
│   ├── prepare_dataset.py                # Data preparation
│   └── quarantine/                       # Legacy scripts (archived)
│       ├── run_100_subject_benchmark.py
│       ├── train_adaptation.py
│       └── aggregate_adaptation_results.py
│
├── configs/                              # YAML configuration files
│   ├── experiments/
│   │   ├── exhibition_15subj.yaml        # Historical exhibition (EXP-EXHIBITION-15SUBJ)
│   │   └── person_level_cv.yaml          # Primary benchmark (EXP-BENCH-PERSON)
│   └── model/
│       └── improved_student.yaml         # Model architecture spec
│
├── artifacts/                            # Model checkpoints
│   ├── exhibition/
│   │   └── EXP-EXHIBITION-15SUBJ/
│   │       └── seed-42/
│   │           ├── teacher_improved_best.pt
│   │           ├── student_best.pt
│   │           └── provenance.json
│   ├── research/
│   │   └── EXP-BENCH-PERSON/
│   └── quarantine/                       # Legacy checkpoints
│       └── student_full_finetuned_generic.pt
│
├── results/                              # Evaluation results
│   ├── exhibition/
│   │   └── EXP-EXHIBITION-15SUBJ/        # Historical exhibition results
│   ├── research/
│   │   └── EXP-BENCH-PERSON/             # Primary benchmark results
│   └── quarantine/                       # Legacy/contaminated results
│       ├── legacy_benchmark_92subj/
│       └── legacy_adaptation/
│
├── data/
│   ├── manifests/
│   │   ├── exhibition_15subj_v1.json     # Exhibition split manifest
│   │   ├── person_folds_52subj.json      # Person-level 10-fold CV (52 persons)
│   │   ├── person_groups.json            # SC4ss1/SC4ss2 person mapping
│   │   └── sleep_edf.csv                 # Subject metadata
│   └── cache/
│       └── sleep_edf/                    # Cached .npz files
│
├── tests/                                # Pytest test suite
│   ├── test_model_contract.py            # Model I/O contract
│   ├── test_sequence_dataset.py          # Sequence dataset tests
│   ├── test_lora.py                      # LoRA wrapping tests
│   ├── test_lora_conv1d.py              # Conv1d LoRA tests
│   ├── test_checkpoint.py               # Checkpoint loading tests
│   ├── test_evaluation.py               # Metrics tests
│   └── test_seed.py                      # Deterministic execution
│
├── docs/                                 # Documentation
│   ├── EXPERIMENTS.md                    # Experiment registry (authoritative)
│   ├── RESULTS.md                        # Validated metrics (authoritative)
│   ├── REPRODUCIBILITY.md                # Reproduction guide
│   ├── adaptation.md                     # Quarantined adaptation study
│   ├── architecture.md                   # Architecture deep dive
│   ├── dataset.md                        # Dataset documentation
│   ├── methodology.md                    # Training methodology
│   └── team.md                           # Team contributions
│
├── notebooks/                            # Jupyter notebooks (canonical pipeline)
│   ├── 01_data_import_and_dataset_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_model_architecture_and_training.ipynb
│   ├── 05_evaluation_and_benchmarking.ipynb
│   └── 06_lora_adaptation_and_evaluation.ipynb (historical)
│
├── deployment/                           # Deployment artifacts
│   ├── app.py
│   └── config/inference.yaml
│
├── hf_model_card.md                      # Hugging Face model card
├── MODEL_REPORT.md                       # Detailed model report
├── GUIDE.md                              # Project guide
├── ROADMAP.md                            # Development roadmap
├── requirements.txt                      # Python dependencies
├── requirements-lock.txt                 # Locked dependencies
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

### Evaluate the Final Model (Exhibition)

```bash
python scripts/verify_protocol.py
# Then run Notebook 05 for full evaluation
```

### Run Person-Level Benchmark (Primary)

```bash
# Verify protocol first
python scripts/verify_protocol.py

# Single fold, seed 42
python scripts/run_person_level_benchmark.py --fold 0 --seed 42

# All folds, seed 42
for fold in {0..9}; do
    python scripts/run_person_level_benchmark.py --fold $fold --seed 42
done

# Multi-seed (seeds 42, 43, 44)
for seed in 42 43 44; do
    for fold in {0..9}; do
        python scripts/run_person_level_benchmark.py --fold $fold --seed $seed
    done
done

# Summarize results
python scripts/summarize_person_benchmark.py --results-dir results/research/EXP-BENCH-PERSON
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

- **Method:** 10-fold person-level CV (whole persons, both nights)
- **Fold assignment:** Person-level (person_folds_52subj.json)
- **Validation:** Fixed 5 persons (both nights), stratified by age decade
- **No person leakage:** Each person appears in exactly one fold's test set
- **Training supervision:** All-position (stride=5, training signal only)
- **Evaluation protocol:** Causal unique-epoch (stride=1, last-epoch only)

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
# Verify protocol integrity
python scripts/verify_protocol.py

# Save protocol fingerprint (for person-level benchmark)
python scripts/protocol_fingerprint.py --seed 42 --mode full_finetune --save-reference

# Verify protocol matches
python scripts/protocol_fingerprint.py --seed 43 --mode full_finetune
python scripts/protocol_fingerprint.py --seed 44 --mode full_finetune
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

## Adaptation Methods (Quarantined)

> **Note:** The adaptation study (Frozen / LoRA / Full-FT) used a contaminated base checkpoint and record-level folds with person-level leakage. Results are retained in `docs/adaptation.md` for internal comparison only and must not be reported as valid person-generalization estimates.

For historical reference, the quarantined scripts are in `scripts/quarantine/`.

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
| Aasir Jaffer Lone | Model Evaluation & Performance |
|Prachi Kamboj| Project Documentation |

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
  author={Kaushik, P. and Vora, S. and Bhatt, S. and Khan, S. and Lone, A.J. and Kamboj, P.},
  year={2026},
  institution={VIT Bhopal University}
}
```
