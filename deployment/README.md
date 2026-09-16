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
student_99477_best.pt (99,477 params)
    ↓
5-stage probabilities
```

## Model Checkpoint

> **Evidence status:** the bundled checkpoint
> (`student_99477_best.pt`) is the **deployed standalone model**
> (notebooks 01→05, supervised class-weighted cross-entropy, seed 42,
> exhibition 70/15/15 subject split): 90.57% accuracy, κ 0.808,
> macro-F1 0.749 on 15 held-out test subjects. Under the stricter
> person-level causal protocol (EXP-BENCH-PERSON, seeds 42/43/44) the
> same architecture reports 87.30% ± 0.33% / κ 0.738 ± 0.010 — see
> [`docs/RESULTS.md`](../docs/RESULTS.md).

| Metric | Value |
|--------|-------|
| Accuracy (15-subject holdout) | 90.57% |
| Cohen's Kappa | 0.8080 |
| Macro F1 | 0.7490 |
| CPU latency (measured) | 6.2 ms/batch |
| Parameters | 99,477 |

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit application |
| `Dockerfile` | Docker build |
| `requirements.txt` | Dependencies |
| `config/inference.yaml` | Model contract |
| `../src/sleep_staging/` | Core package |
| `../artifacts/standalone_99k/` | Checkpoint |
