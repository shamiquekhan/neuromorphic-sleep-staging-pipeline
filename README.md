# Neuromorphic Sleep Stage Scoring

**A Compact Deep Learning Pipeline for Five-Stage Sleep Classification from PSG Signals**

> End-to-end sleep-stage classification using EEG, EOG, and EMG signals with a sub-100K parameter model designed for edge deployment, with parameter-efficient LoRA adaptation.

## Official Final Result (15-Subject Expanded Dataset)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **87.5% ± 3.2%** |
| **Cohen's Kappa** | **0.763 ± 0.043** |
| **Macro F1** | **0.721 ± 0.050** |
| **Parameters** | **99,477** |
| **Dataset** | **15 subjects, Sleep-EDF Expanded (PhysioNet)** |

### Per-Class F1 (15-Subject 4-Fold CV)

| Stage | F1 | Recall |
|-------|-----|--------|
| Wake | 0.961 ± 0.007 | 0.924 |
| **N1** | **0.720 ± 0.086** | 0.564 |
| N2 | 0.845 ± 0.050 | 0.735 |
| N3 | 0.955 ± 0.017 | 0.914 |
| REM | 0.918 ± 0.058 | 0.852 |

### N1 Improvement Over Baseline

| Configuration | N1 F1 | N1 Recall | Macro F1 |
|---------------|-------|-----------|----------|
| Frozen (4 subjects) | 0.000 | 0.000 | 0.411 |
| Improved (4 subjects) | 0.324 ± 0.019 | 0.405 ± 0.035 | 0.730 |
| **Expanded (15 subjects)** | **0.720 ± 0.086** | **0.564** | **0.721** |

**Key finding:** N1 F1 improved from 0.000 → 0.324 (training fix) → 0.720 (subject diversity). Subject diversity is the primary N1 bottleneck.

## Previous Results (4-Subject Baseline)

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.34% |
| Cohen's Kappa | 0.7551 |
| Macro F1 | 0.6259 |

## LoRA Adaptation Results

**LoRA r=8 achieved 90.66% ± 3.59% held-out-subject test accuracy and κ = 0.8092 ± 0.0663 across four folds, with only 552 trainable parameters (0.55% of the 99,477-parameter base model).**

| Method | Trainable | Accuracy | κ | Macro F1 |
|---|---:|---:|---:|---:|
| Frozen Base | 0 | 79.92% ± 5.65% | 0.5001 ± 0.1329 | 0.4106 ± 0.0775 |
| Full Fine-Tuning | 99,477 | 93.04% ± 0.87% | 0.8595 ± 0.0092 | 0.7379 ± 0.0390 |
| **LoRA r=8** | **552** | **90.66% ± 3.59%** | **0.8092 ± 0.0663** | **0.6464 ± 0.0603** |
| **Expanded (15 subj)** | **99,477** | **87.5% ± 3.2%** | **0.763 ± 0.043** | **0.721 ± 0.050** |

## Project Overview

This system classifies 30-second sleep epochs into five AASM stages (Wake, N1, N2, N3, REM) from multi-channel polysomnography recordings. The pipeline covers signal preprocessing, artifact quality control, deep learning with knowledge distillation, and edge-oriented deployment with parameter-efficient LoRA adaptation.

### Architecture

```
PSG Input (EEG + EOG + EMG)
      ↓
Multi-Resolution Stem
      ↓
Depthwise-Separable CNN
      ↓
Parametric Gabor FEB
      ↓
2-Layer GRU (300s context)
      ↓
5-Class Softmax
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Evaluate the Final Model

```bash
python scripts/verify_model.py
```

### Run LoRA Cross-Validation

```bash
python scripts/run_lora_cv.py
```

### Run Expanded Dataset Training

```bash
python scripts/download_sleep_edf_expanded.py  # Download 183 subjects
python scripts/preprocess_sleep_edf_expanded.py  # Preprocess all
```

## Repository Structure

```
├── src/sleep_staging/    Modular Python package
│   ├── models/           ImprovedStudent architecture
│   ├── adaptation/       LoRA adaptation
│   ├── inference/        Prediction engine
│   ├── data/             Dataset loading, labels, harmonization
│   ├── preprocessing/    Signal filtering and QC
│   ├── training/         Cross-dataset training utilities
│   ├── evaluation/       Metrics
│   └── visualization/    Signal and hypnogram plots
├── app/                  Streamlit dashboard
├── scripts/              CLI entry points
├── notebooks/            5 exhibition notebooks
├── tests/                Pytest test suite (33 tests)
├── artifacts/            Checkpoint and LoRA adapters
│   ├── baseline/         Frozen config and metrics
│   ├── lora/             Trained adapters (r=2, r=4, r=8)
│   ├── cross_val_16subj/  15-subject CV model checkpoints
│   └── expanded_5subjects/  5-subject model
├── results/              CV results and per-class metrics
│   ├── cross_val_16subj.json  15-subject 4-fold CV results
│   └── expanded_5subjects/    5-subject experiment results
└── configs/              YAML configuration files
```

## Team

| Member | Responsibility |
|--------|---------------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

## License

MIT
