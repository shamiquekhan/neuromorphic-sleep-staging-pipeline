# Neuromorphic Sleep Stage Scoring

**A Compact Deep Learning Pipeline for Five-Stage Sleep Classification from PSG Signals**

> End-to-end sleep-stage classification using EEG, EOG, and EMG signals with a sub-100K parameter model designed for edge deployment.

## Official Final Result

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

## Project Overview

This system classifies 30-second sleep epochs into five AASM stages (Wake, N1, N2, N3, REM) from multi-channel polysomnography recordings. The pipeline covers signal preprocessing, artifact quality control, deep learning with knowledge distillation, and edge-oriented deployment.

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

### Run the Demo

```bash
streamlit run demo/demo.py
```

### Evaluate the Final Model

```bash
python scripts/evaluate.py --checkpoint artifacts/student_improved_best.pt
```

### Run Inference

```bash
python scripts/infer.py --checkpoint artifacts/student_improved_best.pt --input demo/sample_inputs/sample_epoch.npz
```

## Repository Structure

```
├── configs/           YAML configuration files
├── src/               Modular Python package
│   ├── data/          Dataset loading and manifest
│   ├── preprocessing/ Signal filtering and QC
│   ├── models/        ImprovedStudent architecture
│   ├── training/      Loss functions and trainer
│   ├── evaluation/    Metrics and visualization
│   └── inference/     Prediction utilities
├── scripts/           CLI entry points
├── notebooks/         5 exhibition notebooks
├── demo/              Streamlit dashboard
├── tests/             Pytest test suite
├── artifacts/         Final trained checkpoint
└── results/           Official evaluation results
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
