# NeuroSleep

**A compact deep learning pipeline for five-stage sleep classification from polysomnography signals.**

> End-to-end sleep-stage classification using EEG, EOG, and EMG signals with a sub-100K parameter model designed for edge deployment.

## Final Results (100-Subject Benchmark)

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

### Per-Class F1 (Full Fine-Tuning, 30 Folds)

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.964 ± 0.016 | 0.995 | 0.936 |
| N1 | **0.445 ± 0.061** | 0.330 | 0.712 |
| N2 | 0.768 ± 0.041 | 0.878 | 0.687 |
| N3 | 0.681 ± 0.114 | 0.562 | 0.896 |
| REM | 0.771 ± 0.078 | 0.772 | 0.785 |

### Adaptation Method Comparison

| Model | Trainable Params | Accuracy | κ | Macro F1 |
|-------|----------------:|---------:|----:|---------:|
| Frozen Base | 0 (0%) | 87.1% ± 3.6% | 0.738 ± 0.077 | 0.673 ± 0.074 |
| LoRA CNN+Head (r=8) | 1,448 (1.43%) | 83.6% ± 3.7% | 0.693 ± 0.057 | 0.674 ± 0.045 |
| **Full Fine-Tuning** | **99,477 (100%)** | **87.7% ± 2.7%** | **0.763 ± 0.043** | **0.730 ± 0.037** |

> **Key finding:** Full fine-tuning achieves the strongest overall and stage-balanced performance. LoRA CNN+Head uses 68.7× fewer trainable parameters while retaining 95.4% of full FT accuracy.

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
│   ├── adaptation/           LoRA parameter-efficient adaptation
│   ├── evaluation/           Metrics
│   └── visualization/        Signal and hypnogram plots
├── app/                      Streamlit dashboard (Swiss design)
├── scripts/                  CLI entry points
├── notebooks/                Analysis notebooks
├── tests/                    Pytest test suite
├── artifacts/                Checkpoints and configs
│   └── final/                Authoritative final checkpoint
├── results/                  Evaluation results
│   ├── full_100_subject/     100-subject benchmark (from scratch)
│   ├── 100_subject_adaptation/  Adaptation comparison study
│   │   └── final/            3-seed aggregate results
│   └── final/                15-subject development results
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

The final model configuration is in `configs/full_100_subject.yaml`:

- **Model:** Improved Student (99,477 params)
- **Data:** 92 subjects, Sleep-EDF Expanded (8 wake-only excluded)
- **Training:** 20 epochs, AdamW, lr=3e-4, N1/REM weighting (2x), mixed precision
- **Evaluation:** 10-fold subject-level CV, 3 seeds (42, 43, 44)

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
