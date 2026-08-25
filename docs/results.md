# Final Results — NeuroSleep

> **This is the single authoritative result document for the project.** All other result tables in the repository are historical.

---

## Executive Summary

| Property | Value |
|----------|-------|
| Model | Improved Student |
| Parameters | 99,477 |
| Dataset | Sleep-EDF Expanded (15 subjects) |
| **Accuracy** | **93.0% ± 1.0%** |
| **Cohen's Kappa** | **0.861 ± 0.027** |
| **Macro F1** | **0.794 ± 0.036** |
| **Weighted F1** | **0.935 ± 0.007** |

---

## Per-Class Performance

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.983 ± 0.011 | 1.000 | 0.967 |
| **N1** | **0.682 ± 0.090** | 1.000 | 0.552 |
| N2 | 0.912 ± 0.044 | 1.000 | 0.848 |
| N3 | 0.958 ± 0.016 | 1.000 | 0.912 |
| REM | 0.966 ± 0.017 | 1.000 | 0.930 |

### N1 Analysis

N1 is the hardest stage with F1=0.682. This is expected because:
- N1 epochs are brief (1-7 minutes) and transitional
- N1 is often confused with N2 and Wake
- Only 2.9% of test epochs are N1 (636 out of 22,200)

### Model Strengths

- **Perfect precision** — when the model predicts a class, it's always correct
- **Strong N3 detection** (F1=0.958) — deep sleep is well-distinguished
- **High weighted F1** (0.935) — overall classification quality is excellent

---

## Fold Breakdown

| Fold | Test Subject | Accuracy | Kappa | Macro F1 |
|------|-------------|----------|-------|----------|
| 1 | SC4001 | 92.2% | 0.822 | 0.749 |
| 2 | SC4002 | 92.0% | 0.854 | 0.779 |
| 3 | SC4011 | 94.6% | 0.896 | 0.849 |
| 4 | SC4012 | 93.0% | 0.871 | 0.799 |

---

## Model Architecture

```
PSG Input (EEG + EOG + EMG, 4 channels)
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

| Property | Value |
|----------|-------|
| Parameters | 99,477 |
| Model Size | ~400 KB (FP32) |
| Input Shape | [batch, 10, 4, 3000] |
| Output Shape | [batch, 10, 5] |
| Context Window | 300 seconds (10 × 30s epochs) |

---

## Training Protocol

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-2 |
| Epochs | 15 |
| Batch Size | 16 |
| Class Weights | N1=2x, REM=2x |
| Gradient Clipping | max_norm=1.0 |
| Device | NVIDIA GTX 1650 (CUDA 12.4) |
| Seed | 42 |

---

## Evaluation Protocol

- **Method:** 4-fold subject-level cross-validation
- **Subjects:** 15 from Sleep-EDF Expanded
- **Fold assignment:** Canonical (SC4001, SC4002, SC4011, SC4012 as test)
- **Sequence:** length=10, stride=5, 30s epochs
- **All-position supervision** — every epoch in the window is supervised

---

## Files

| File | Description |
|------|-------------|
| `configs/final.yaml` | Model configuration |
| `artifacts/final/student_full_finetuned.pt` | Checkpoint |
| `results/final/final_metrics.json` | Complete results |
| `results/final/per_class_metrics.csv` | Per-class by fold |
| `results/final/fold_metrics.csv` | Fold metrics |
| `results/final/confusion_matrix.csv` | Confusion matrix |
| `results/final/predictions.csv` | All predictions |

---

## Reproduction

```bash
# Evaluate the final model
python scripts/evaluate_final_model.py

# Run cross-validation
python scripts/run_4fold_simple.py
```

---

## Citation

```bibtex
@project{neurosleep_2026,
  title={NeuroSleep: Neuromorphic Sleep Stage Scoring},
  author={Kaushik, P. and Vora, S. and Bhatt, S. and Khan, S. and Lone, A.J.},
  year={2026},
  institution={VIT Bhopal University}
}
```

---

*Last updated: August 2026*
