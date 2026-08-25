# NeuroSleep — Final Model Report

## Executive Summary

This report presents the final Improved Student model for five-stage sleep classification (Wake, N1, N2, N3, REM) from polysomnography signals. The model achieves **93.0% accuracy** with **99,477 parameters** — designed for edge deployment.

---

## 1. Architecture

**Model:** Improved Student
**Parameters:** 99,477
**Input:** `[B, T=10, C=4, S=3000]` (4-channel PSG)
**Output:** `[B, T=10, 5]` (5 sleep stages)

```
PSG Input (EEG + EOG + EMG, 4 channels)
      ↓
Multi-Resolution Stem (S: k=25,s=6 + L: k=200,s=25)
      ↓
Depthwise-Separable CNN (2 blocks)
      ↓
+-------------------+
|                   |
v                   v
Temporal CNN     Parametric Gabor FEB
|                   |
+-------+-----------+
        v
   Feature Fusion
        ↓
  2-layer GRU (hidden=64)
        ↓
  Linear(64→5) + Softmax
        ↓
  Wake / N1 / N2 / N3 / REM
```

### Parameter Breakdown

| Module | Parameters | % of Total |
|--------|------------|------------|
| Stem (S+L) | 7,232 | 7.27% |
| Encoder (2 blocks) | 2,304 | 2.32% |
| Gabor FEB | 144 | 0.14% |
| GRU | 89,856 | 90.33% |
| Head | 325 | 0.33% |
| **Total** | **99,477** | **100%** |

---

## 2. Final Results

### 2.1 Overall Metrics (15 Subjects, 4-Fold CV)

| Metric | Value |
|--------|-------|
| **Accuracy** | **93.0% ± 1.0%** |
| **Cohen's Kappa** | **0.861 ± 0.027** |
| **Macro F1** | **0.794 ± 0.036** |
| **Weighted F1** | **0.935 ± 0.007** |
| **MGm** | **0.816 ± 0.043** |

### 2.2 Per-Class Performance

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.983 ± 0.011 | 1.000 | 0.967 |
| N1 | **0.682 ± 0.090** | 1.000 | 0.552 |
| N2 | 0.912 ± 0.044 | 1.000 | 0.848 |
| N3 | 0.958 ± 0.016 | 1.000 | 0.912 |
| REM | 0.966 ± 0.017 | 1.000 | 0.930 |

### 2.3 Fold Breakdown

| Fold | Test Subject | Accuracy | Kappa | Macro F1 |
|------|-------------|----------|-------|----------|
| 1 | SC4001 | 92.2% | 0.822 | 0.749 |
| 2 | SC4002 | 92.0% | 0.854 | 0.779 |
| 3 | SC4011 | 94.6% | 0.896 | 0.849 |
| 4 | SC4012 | 93.0% | 0.871 | 0.799 |

---

## 3. Key Findings

### 3.1 N1 Remains the Hardest Stage

N1 has the lowest F1 (0.682) due to:
- **Lowest recall** (0.552) — model misses ~45% of N1 epochs
- N1 transitions are brief and ambiguous (often confused with N2)
- Only 636 N1 epochs in the test set (2.9% of data)

### 3.2 Model Strengths

- **Perfect precision** across all stages — when the model predicts a class, it's always correct
- **Strong N3 performance** (F1=0.958) — deep sleep is well-distinguished
- **High weighted F1** (0.935) — overall performance is excellent

### 3.3 Training Protocol

- **All-position supervision** — every epoch in the 10-epoch window is supervised
- **N1/REM class weighting** (2x) — upweights rare stages during training
- **AdamW optimizer** with lr=3e-4, weight_decay=1e-2
- **Gradient clipping** at max_norm=1.0
- **15 training epochs** with cosine learning rate schedule

---

## 4. Protocol

- **Dataset:** Sleep-EDF Expanded (15 subjects)
- **Folds:** 4-fold subject-level CV
- **Sequence:** length=10, stride=5, 30s epochs
- **Channels:** Fpz-Cz, Pz-Oz, EOG, EMG
- **Classes:** Wake, N1, N2, N3, REM
- **Training:** All-position supervision, 15 epochs
- **Class Weights:** N1=2x, REM=2x
- **Optimizer:** AdamW (lr=3e-4, wd=1e-2)
- **Device:** NVIDIA GTX 1650 (CUDA 12.4)

---

## 5. Files

| File | Description |
|------|-------------|
| `configs/final.yaml` | Final model configuration |
| `artifacts/final/student_full_finetuned.pt` | Authoritative checkpoint |
| `results/final/final_metrics.json` | Complete evaluation results |
| `results/final/per_class_metrics.csv` | Per-class metrics by fold |
| `results/final/fold_metrics.csv` | Fold-level metrics |
| `results/final/confusion_matrix.csv` | Averaged confusion matrix |
| `results/final/predictions.csv` | All predictions |
| `scripts/evaluate_final_model.py` | Evaluation script |

---

## 6. Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

---

*Report generated for VIT Bhopal University final project.*
