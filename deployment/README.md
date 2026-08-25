# NeuroSleep — Deployment

Deployment artifacts for the NeuroSleep sleep stage scoring system.

## Quick Start

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker

```bash
docker build -t neurosleep .
docker run -p 7860:7860 neurosleep
```

## Architecture

```
deployment/app.py
    ↓
sleep_staging.inference.predictor
    ↓
sleep_staging.models.student
    ↓
student_full_finetuned.pt (99,477 params)
    ↓
5-stage probabilities
```

## Final Model

| Metric | Value |
|--------|-------|
| Accuracy | 93.0% ± 1.0% |
| Cohen's Kappa | 0.861 ± 0.027 |
| Macro F1 | 0.794 ± 0.036 |
| Parameters | 99,477 |

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit application |
| `Dockerfile` | Docker build |
| `requirements.txt` | Dependencies |
| `config/inference.yaml` | Model contract |
| `../src/sleep_staging/` | Core package |
| `../artifacts/final/` | Checkpoint |
