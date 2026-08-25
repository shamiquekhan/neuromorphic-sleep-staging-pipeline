# NeuroSleep — Project Guide

## Overview

NeuroSleep is a neuromorphic sleep stage scoring pipeline that classifies 30-second polysomnography epochs into five AASM stages (Wake, N1, N2, N3, REM).

## Final Model

| Property | Value |
|----------|-------|
| Model | Improved Student |
| Parameters | 99,477 |
| Accuracy | 93.0% ± 1.0% |
| Cohen's Kappa | 0.861 ± 0.027 |
| Macro F1 | 0.794 ± 0.036 |
| Dataset | 15 subjects, Sleep-EDF Expanded |

## Key Files

| File | Description |
|------|-------------|
| `configs/final.yaml` | Model configuration |
| `artifacts/final/student_full_finetuned.pt` | Trained checkpoint |
| `results/final/final_metrics.json` | Evaluation results |
| `scripts/evaluate_final_model.py` | Evaluation script |
| `app/streamlit_app.py` | Dashboard |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app/streamlit_app.py

# Evaluate model
python scripts/evaluate_final_model.py
```

## Repository Structure

```
src/sleep_staging/     Core package
app/                   Streamlit dashboard
scripts/               CLI tools
notebooks/             Analysis notebooks
tests/                 Test suite
artifacts/             Checkpoints
results/               Evaluation results
configs/               YAML configs
data/                  Dataset manifests
docs/                  Documentation
```

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

## License

CC-BY-4.0
