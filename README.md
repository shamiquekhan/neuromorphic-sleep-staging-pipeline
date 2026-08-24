# Neuromorphic Sleep Stage Scoring

**A Compact Deep Learning Pipeline for Five-Stage Sleep Classification from PSG Signals**

> End-to-end sleep-stage classification using EEG, EOG, and EMG signals with a sub-100K parameter model designed for edge deployment, with parameter-efficient LoRA adaptation.

## Official Final Result (Frozen Base)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **87.34%** |
| **Cohen's Kappa** | **0.7551** |
| **Macro F1** | **0.6259** |
| **Parameters** | **99,477** |
| **CPU Latency** | **8.5 ms/batch** |

### Per-Class F1

| Stage | F1 |
|-------|-----|
| Wake (W) | 0.9693 |
| N1 | 0.2006 |
| N2 | 0.8162 |
| N3 | 0.7849 |
| REM | 0.3586 |

## LoRA Adaptation Results

**LoRA r=8 achieved 90.66% ± 3.59% held-out-subject test accuracy and κ = 0.8092 ± 0.0663 across four folds, with only 552 trainable parameters (0.55% of the 99,477-parameter base model). Validation accuracy used for checkpoint selection peaked at 87.27% (κ = 0.7175).**

| Method | Trainable | Accuracy | κ | Macro F1 |
|---|---:|---:|---:|---:|
| Frozen Base | 0 | 79.92% ± 5.65% | 0.5001 ± 0.1329 | 0.4106 ± 0.0775 |
| Full Fine-Tuning | 99,477 | 93.04% ± 0.87% | 0.8595 ± 0.0092 | 0.7379 ± 0.0390 |
| **LoRA r=8** | **552** | **90.66% ± 3.59%** | **0.8092 ± 0.0663** | **0.6464 ± 0.0603** |

### Multi-Seed Confirmation (LoRA r=8)

| Seed | Fold 1 κ | Fold 2 κ | Fold 3 κ | Fold 4 κ | Mean κ |
|---|---:|---:|---:|---:|---:|
| 42 | 0.8748 | 0.8771 | 0.7706 | 0.7004 | 0.8057 |
| 43 | 0.8798 | 0.8717 | 0.7670 | 0.7363 | 0.8137 |
| 44 | 0.8748 | 0.8595 | 0.7681 | 0.7386 | 0.8102 |
| **Overall** | | | | | **0.8099 ± 0.0657** |

### Per-Class F1 (Frozen vs LoRA r=8)

| Class | Frozen F1 | LoRA F1 | Δ |
|---|---:|---:|---:|
| Wake | 0.8883 | 0.9771 | +0.0888 |
| N1 | 0.0000 | 0.0000 | 0.0000 |
| N2 | 0.6115 | 0.8437 | +0.2322 |
| N3 | 0.5531 | 0.8320 | +0.2788 |
| REM | 0.0000 | 0.5759 | +0.5759 |

### Engineering Verification

| Metric | Value |
|---|---|
| Base latency | 6.25 ms/batch |
| LoRA unmerged | 5.66 ms/batch |
| LoRA merged | 5.62 ms/batch |
| Merge diff | 0.00e+00 (PASS) |
| Adapter reload | 0.00e+00 (PASS) |

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

### Run Inference

```bash
python scripts/predict.py --checkpoint artifacts/student_improved_best.pt
```

## Repository Structure

```
├── src/sleep_staging/    Modular Python package
│   ├── models/           ImprovedStudent architecture
│   ├── adaptation/       LoRA adaptation
│   ├── inference/        Prediction engine
│   ├── data/             Dataset loading
│   ├── preprocessing/    Signal filtering and QC
│   ├── evaluation/       Metrics
│   └── visualization/    Signal and hypnogram plots
├── app/                  Streamlit dashboard
├── scripts/              CLI entry points
├── notebooks/            5 exhibition notebooks
├── tests/                Pytest test suite (33 tests)
├── artifacts/            Checkpoint and LoRA adapters
│   ├── baseline/         Frozen config and metrics
│   └── lora/             Trained adapters (r=2, r=4, r=8)
├── results/              CV results and per-class metrics
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
