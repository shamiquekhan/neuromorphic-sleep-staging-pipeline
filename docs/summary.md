# Project Summary — Neuromorphic Sleep Stage Scoring

## One-Line Summary

A compact deep-learning system that classifies 30-second sleep epochs into five AASM stages (Wake, N1, N2, N3, REM) from EEG, EOG, and EMG signals, using only 99,477 parameters for edge deployment.

---

## Project at a Glance

| Property | Value |
|----------|-------|
| **Title** | Neuromorphic Sleep Stage Scoring |
| **Venue** | VIT Bhopal University |
| **Final Model** | Improved Student |
| **Parameters** | 99,477 |
| **Test Accuracy** | 87.34% |
| **Cohen's Kappa** | 0.7551 |
| **Macro F1** | 0.6259 |
| **CPU Latency** | 8.5 ms/batch |
| **Checkpoint** | `artifacts/student_improved_best.pt` |

---

## Problem Statement

Sleep staging is clinically important but requires expert analysis of multi-channel polysomnography (PSG) recordings. Manual scoring takes 2–4 hours per study and requires specialized training. This project develops an automated system to classify five sleep stages from physiological signals.

---

## Solution Overview

```
Raw PSG Signals (EEG + EOG + EMG)
        │
        ▼
┌─────────────────────────────────┐
│ Signal Preprocessing            │
│ • 0.5–35 Hz bandpass            │
│ • 50 Hz notch                   │
│ • Z-score normalization         │
│ • Artifact QC                   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ Model Architecture              │
│ • Multi-Resolution Stem         │
│ • Depthwise-Separable CNN       │
│ • Parametric Gabor FEB          │
│ • 2-Layer GRU (300s context)    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 5-Class Sleep Stage Prediction  │
│ Wake, N1, N2, N3, REM          │
└─────────────────────────────────┘
```

---

## Key Components

### 1. Signal Preprocessing
- Bandpass filtering (0.5–35 Hz) removes noise
- Notch filtering (50 Hz) eliminates power-line interference
- Z-score normalization standardizes signals
- Artifact QC flags ~2% of epochs

### 2. Multi-Resolution Stem
- Parallel short (250ms) and long (1s) receptive fields
- Captures patterns at different temporal scales
- Combines local waveform morphology with slow oscillations

### 3. Depthwise-Separable Convolution
- Reduces parameters by ~77% vs standard convolution
- Maintains feature extraction capability
- Enables edge deployment

### 4. Parametric Gabor Feature Extraction
- 8 learnable Gabor filters (0.5–30 Hz)
- Captures frequency-localized patterns
- Compact parameterization (16 learnable params)

### 5. GRU Temporal Modeling
- 2-layer GRU with 64 hidden units
- Models dependencies across 10 consecutive epochs
- Provides 300 seconds of temporal context

---

## Official Result

```
┌─────────────────────────────────────────┐
│     NEUROMORPHIC SLEEP STAGE SCORING    │
├─────────────────────────────────────────┤
│                                         │
│     87.34%                              │
│     TEST ACCURACY                       │
│                                         │
│     0.7551                              │
│     COHEN'S κ                           │
│                                         │
│     99,477                              │
│     PARAMETERS                          │
│                                         │
│     8.5 ms/batch                        │
│     CPU LATENCY                         │
│                                         │
└─────────────────────────────────────────┘
```

### Per-Class F1 Scores

| Stage | F1 | Notes |
|-------|-----|-------|
| Wake | 0.97 | Excellent |
| N1 | 0.20 | Challenging |
| N2 | 0.82 | Good |
| N3 | 0.78 | Good |
| REM | 0.36 | Moderate |

---

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

---

## Repository Structure

```
sleep/
├── configs/           YAML configuration files
├── src/               Modular Python package
│   ├── data/          Dataset loading
│   ├── preprocessing/ Signal filtering
│   ├── models/        ImprovedStudent architecture
│   ├── training/      Loss functions & trainer
│   ├── evaluation/    Metrics & visualization
│   └── inference/     Prediction utilities
├── scripts/           CLI entry points
├── notebooks/         5 exhibition notebooks
├── demo/              Streamlit dashboard
├── tests/             Pytest test suite
├── artifacts/         Final trained checkpoint
├── results/           Official evaluation results
└── docs/              Project documentation
```

---

## Quick Start

```bash
# Activate environment
conda activate sleep

# Run tests
python -m pytest tests/ -v

# Run demo
streamlit run demo/demo.py

# Run inference
python scripts/infer.py --checkpoint artifacts/student_improved_best.pt --input demo/sample_inputs/sample_epoch.npz
```

---

## Key Innovations

1. **Sub-100K parameters:** Achieves 87.34% accuracy with only 99,477 parameters
2. **Multi-resolution feature capture:** Parallel stems capture patterns at different timescales
3. **Learnable frequency filters:** Parametric Gabor filters adapt to sleep-specific frequencies
4. **Efficient architecture:** Depthwise-separable convolutions enable edge deployment
5. **Temporal context:** 300-second GRU context models sleep-stage transitions

---

## Deployment Readiness

| Property | Value |
|----------|-------|
| Model size (FP32) | ~400 KB |
| Input size | 120 KB per epoch |
| CPU latency | 8.5 ms/batch |
| Real-time capable | Yes |
| Edge deployment | Yes (ARM Cortex-M7+) |

---

## Exhibition Story

1. **Problem:** Sleep staging requires expert analysis
2. **Solution:** Automated classification using deep learning
3. **Innovation:** Compact architecture with multi-resolution features
4. **Result:** 87.34% accuracy with 99,477 parameters
5. **Impact:** Edge-deployable for clinical assistance

---

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/architecture.md` | Detailed model design |
| `docs/dataset.md` | Data source and preprocessing |
| `docs/deployment.md` | Edge deployment guide |
| `docs/exhibition.md` | Demo script and poster layout |
| `docs/methodology.md` | Research approach |
| `docs/results.md` | Official metrics |
| `docs/team.md` | Team roles |
| `docs/index.md` | Documentation index |

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
