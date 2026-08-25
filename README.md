# NeuroSleep

**A compact deep learning pipeline for five-stage sleep classification from polysomnography signals.**

> End-to-end sleep-stage classification using EEG, EOG, and EMG signals with a sub-100K parameter model designed for edge deployment.

## Final Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **93.0% ± 1.0%** |
| **Cohen's Kappa** | **0.861 ± 0.027** |
| **Macro F1** | **0.794 ± 0.036** |
| **Weighted F1** | **0.935 ± 0.007** |
| **Parameters** | **99,477** |
| **Dataset** | **15 subjects, Sleep-EDF Expanded (PhysioNet)** |

### Per-Class F1 (4-Fold Subject-Level CV)

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.983 ± 0.011 | 1.000 | 0.967 |
| N1 | **0.682 ± 0.090** | 1.000 | 0.552 |
| N2 | 0.912 ± 0.044 | 1.000 | 0.848 |
| N3 | 0.958 ± 0.016 | 1.000 | 0.912 |
| REM | 0.966 ± 0.017 | 1.000 | 0.930 |

### Fold Breakdown

| Fold | Test Subject | Accuracy | Kappa | Macro F1 |
|------|-------------|----------|-------|----------|
| 1 | SC4001 | 92.2% | 0.822 | 0.749 |
| 2 | SC4002 | 92.0% | 0.854 | 0.779 |
| 3 | SC4011 | 94.6% | 0.896 | 0.849 |
| 4 | SC4012 | 93.0% | 0.871 | 0.799 |

## Architecture

```
PSG Input (EEG + EOG + EMG, 4 channels)
      ↓
Multi-Resolution Stem
      ↓
Depthwise-Separable CNN
      ↓
Parametric Gabor Feature Extraction
      ↓
2-Layer GRU (300s context)
      ↓
5-Class Softmax
```

**99,477 parameters** — designed for edge deployment on resource-constrained devices.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Evaluate the Final Model

```bash
python scripts/evaluate_final_model.py
```

### Run 4-Fold Cross-Validation

```bash
python scripts/run_4fold_simple.py
```

## Repository Structure

```
├── src/sleep_staging/        Python package
│   ├── models/               ImprovedStudent architecture
│   ├── inference/            Prediction engine
│   ├── data/                 Dataset loading, labels
│   ├── preprocessing/        Signal filtering and QC
│   ├── training/             Cross-dataset training utilities
│   ├── evaluation/           Metrics
│   └── visualization/        Signal and hypnogram plots
├── app/                      Streamlit dashboard (Swiss design)
├── scripts/                  CLI entry points
├── notebooks/                Analysis notebooks
├── tests/                    Pytest test suite
├── artifacts/                Checkpoints and configs
│   ├── final/                Authoritative final checkpoint
│   └── cross_val_16subj/     CV fold checkpoints
├── results/                  Evaluation results
│   └── final/                Authoritative final metrics
├── configs/                  YAML configuration files
├── data/                     Dataset manifests and cache
└── docs/                     Documentation
```

## Live Resources

| Resource | Link |
|----------|------|
| **Interactive Demo** | [Hugging Face Space](https://huggingface.co/spaces/shamiquekhan/neurosleep-demo) |
| **Model** | [Hugging Face Model Hub](https://huggingface.co/shamiquekhan/neuromorphic-sleep-staging) |
| **Reproducibility** | [Kaggle Notebook](https://www.kaggle.com/shamiquekhan/neurosleep-final) |
| **Source** | [GitHub](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) |

## Configuration

The final model configuration is in `configs/final.yaml`:

- **Model:** Improved Student (99,477 params)
- **Data:** 15 subjects, Sleep-EDF Expanded
- **Training:** 15 epochs, AdamW, lr=3e-4, N1/REM weighting (2x)
- **Evaluation:** 4-fold subject-level cross-validation

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

## License

CC-BY-4.0 (Creative Commons Attribution 4.0 International)
