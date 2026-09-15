# Project Summary — Neuromorphic Sleep Stage Scoring

## One-Line Summary

A compact deep-learning system that classifies 30-second sleep epochs into five AASM stages (Wake, N1, N2, N3, REM) from EEG, EOG, and EMG signals, using only 99,477 parameters for edge deployment — trained end-to-end by the five-notebook pipeline in this repository.

---

## Project at a Glance

| Property | Value |
|----------|-------|
| **Title** | Neuromorphic Sleep Stage Scoring |
| **Venue** | VIT Bhopal University |
| **Final Model** | Improved Student (distilled from Improved Teacher) |
| **Parameters** | 99,477 |
| **Primary Benchmark** (person-level CV, 3 seeds × 10 folds) | **87.30% ± 0.33% acc · κ 0.738 · macro-F1 0.724** |
| **Notebook Pipeline** (15 held-out test subjects) | **88.64% acc · κ 0.773 · macro-F1 0.719** |
| **CPU Latency** | ~8.9 ms/batch |
| **Dataset** | Sleep-EDF Expanded (92 records / 52 persons) |
| **Checkpoint** | `artifacts/final/student_full_finetuned.pt` |

---

## Problem Statement

Sleep staging is clinically important but requires expert analysis of multi-channel polysomnography (PSG) recordings. Manual scoring takes 2–4 hours per study and requires specialized training. This project develops an automated system to classify five sleep stages from physiological signals under a strict sub-100K parameter budget for edge/wearable deployment.

---

## Solution Overview

```
Raw PSG Signals (EEG + EOG + EMG)
        │
        ▼
┌─────────────────────────────────┐
│ Notebook 01 — Manifest & Split   │
│ 100 PSG/hypnogram pairs,         │
│ subject-level 70/15/15 split     │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Notebook 02 — Preprocessing      │
│ 0.5–35 Hz bandpass, 50 Hz notch, │
│ 30-s epochs, z-score, QC flags   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Notebook 03 — EDA               │
│ class balance, spectra,         │
│ transition structure            │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Notebook 04 — Training          │
│ focal-loss Teacher → distilled  │
│ Student (per-epoch logs)        │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Notebook 05 — Evaluation        │
│ 15 held-out subjects, metrics,  │
│ confusion matrix, latency       │
└─────────────────────────────────┘
```

---

## Key Results

| Benchmark | Accuracy | Cohen's κ | Macro F1 | Evidence |
|-----------|---------:|----------:|---------:|----------|
| **Person-level primary** (10-fold CV over 52 persons, seeds 42/43/44) | 87.30% ± 0.33% | 0.738 ± 0.010 | 0.724 ± 0.005 | `results/benchmark_person_level/` |
| **Notebook pipeline** (single split, 15 test subjects) | 88.64% | 0.773 | 0.719 | `notebooks/05_evaluation_and_benchmarking.ipynb` |

Per-class F1 (notebook pipeline): Wake 0.970 · N2 0.791 · REM 0.707 · N3 0.685 · N1 0.440.

---

## Architecture

```
PSG Input (4 channels × 10 epochs × 3000 samples)
        ↓
Multi-Resolution Stem (kernel 25 / kernel 200)
        ↓
Depthwise-Separable CNN (2 blocks)
        ↓
Parametric Gabor Features (8 learnable filters)
        ↓
Feature Fusion
        ↓
2-Layer GRU (hidden 64 — 89,856 of 99,477 params)
        ↓
5-class head → Wake / N1 / N2 / N3 / REM
```

---

## Repository Structure

```
notebooks/          THE pipeline (01→06), outputs embedded
src/sleep_staging/  Installable package (models, LoRA, data, evaluation, inference)
scripts/            CLI entry points (benchmark, folds, summarize, verify)
app/                Streamlit dashboard
tests/              92 passing tests
docs/               results.md is the numbers source of truth
artifacts/          Final checkpoints (student + teacher)
results/            Primary benchmark evidence + notebook result
configs/            benchmark_person_level.yaml (primary)
data/               Manifests (tracked) + cache/raw (local only)
deployment/         Docker deployment
huggingface/        Hub model card + demo space assets
```

---

## Quick Commands

```bash
pip install -r requirements.txt && pip install -e .

# Run the full pipeline (notebooks 01→05)
jupyter nbconvert --to notebook --execute notebooks/01_data_import_and_dataset_collection.ipynb --inplace
# ...continue through 05

# Dashboard
streamlit run app/streamlit_app.py

# Tests
python -m pytest tests/ -v
```

---

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |
