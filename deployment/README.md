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

## Model Checkpoint

> **Evidence status:** the bundled checkpoint
> (`student_full_finetuned.pt`) is the **15-subject development-era
> model** (EXP-DEV-15SUBJ). The metrics below are its *development
> benchmark* results and must **not** be quoted as the project's
> primary generalization result. The primary benchmark
> (EXP-BENCH-92SUBJ, 92-subject cohort, from scratch: 87.66% ± 2.22%,
> κ 0.762, macro-F1 0.728) is documented in
> [`docs/results.md`](../docs/results.md).

| Metric | Value |
|--------|-------|
| Accuracy (15-subject dev benchmark) | 93.0% ± 1.0% |
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
