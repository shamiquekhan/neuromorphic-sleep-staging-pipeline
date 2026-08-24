# Official Final Results — Neuromorphic Sleep Stage Scoring

> **Important:** This document contains the **single authoritative result** for the project exhibition. All other result tables in the repository are historical and should not be presented as final project outcomes.

---

## Executive Summary

| Property | Value |
|----------|-------|
| Model | Improved Student |
| Parameters | 99,477 |
| Test Accuracy | 87.34% |
| Cohen's Kappa | 0.7551 |
| Macro F1 | 0.6259 |
| Weighted F1 | 0.8653 |
| Macro Geometric Mean | 0.5371 |
| CPU Latency | 8.5 ms/batch |

---

## Detailed Metrics

### Overall Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test Accuracy | 87.34% | Correctly classified 87.34% of all epochs |
| Cohen's Kappa | 0.7551 | Substantial agreement beyond chance |
| Macro F1 | 0.6259 | Average precision-recall across all classes |
| Weighted F1 | 0.8653 | Performance weighted by class frequency |
| Macro Geometric Mean | 0.5371 | Geometric mean of per-class F1 scores |

### Per-Class Performance

| Sleep Stage | F1 Score | Precision | Recall | Support |
|-------------|----------|-----------|--------|---------|
| Wake (W) | 0.9693 | 0.97 | 0.97 | High |
| N1 | 0.2006 | 0.22 | 0.19 | Low |
| N2 | 0.8162 | 0.82 | 0.81 | High |
| N3 | 0.7849 | 0.79 | 0.78 | Medium |
| REM | 0.3586 | 0.37 | 0.35 | Medium |

### Model Characteristics

| Property | Value |
|----------|-------|
| Architecture | Multi-Res Stem + DWSep CNN + Gabor FEB + GRU |
| Parameters | 99,477 |
| Model Size (FP32) | ~400 KB |
| Input Shape | [batch, 10, 4, 3000] |
| Output Shape | [batch, 10, 5] |
| CPU Latency | 8.5 ms/batch |
| Context Window | 300 seconds (10 epochs × 30s) |

---

## Comparison with Historical Models

> **Note:** These are historical benchmarks, not competing final results. Only the Improved Student is the official project model.

| Model | Parameters | Accuracy | Kappa | Macro F1 |
|-------|-----------|----------|-------|----------|
| Baseline Teacher | 303,789 | 79.12% | 0.6233 | 0.5263 |
| Improved Teacher | 193,197 | 79.28% | 0.6019 | 0.5545 |
| Baseline Student | 567,749 | 90.25% | 0.8123 | 0.6420 |
| **Improved Student** | **99,477** | **87.34%** | **0.7551** | **0.6259** |

### Key Observations

1. **Parameter efficiency:** Improved Student achieves 87.34% with only 99,477 parameters (17% of Baseline Student's size)
2. **Accuracy trade-off:** 2.91% accuracy reduction for 82.5% parameter reduction
3. **Edge-readiness:** Sub-100K parameters enable MCU deployment

---

## Result Interpretation

### Strengths

1. **High Wake detection (F1=0.9693):** The model excels at identifying awake states, which is clinically important for sleep study analysis
2. **Strong N2 classification (F1=0.8162):** The most common sleep stage is well-handled
3. **Good N3 detection (F1=0.7849):** Deep sleep is reliably identified
4. **Compact model:** 99,477 parameters enable edge deployment

### Limitations

1. **N1 classification (F1=0.2006):** Light sleep is challenging due to:
   - Brief duration (typically 1-7 minutes)
   - Transitional nature (between Wake and N2)
   - Subtle EEG features
   - Class imbalance (N1 is rare)

2. **REM classification (F1=0.3586):** Rapid eye movement sleep is difficult due to:
   - EEG similarity to Wake/N1
   - Reliance on EOG/EMG for differentiation
   - Variable REM characteristics

3. **Class imbalance impact:** Macro F1 (0.6259) is lower than Weighted F1 (0.8653) because minority classes (N1, REM) have lower performance

---

## LoRA Adaptation Results

**LoRA r=8 achieved 90.66% ± 3.59% held-out-subject test accuracy and κ = 0.8092 ± 0.0663 across four folds, with only 552 trainable parameters (0.55% of the 99,477-parameter base model). Validation accuracy used for checkpoint selection peaked at 87.27% (κ = 0.7175).**

| Method | Trainable | Accuracy | κ | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Frozen Base | 0 | 79.92% ± 5.65% | 0.5001 ± 0.1329 | 0.4106 ± 0.0775 | 0.7462 ± 0.0689 |
| Full Fine-Tuning | 99,477 | 93.04% ± 0.87% | 0.8595 ± 0.0092 | 0.7379 ± 0.0390 | 0.9243 ± 0.0106 |
| **LoRA r=8** | **552** | **90.66% ± 3.59%** | **0.8092 ± 0.0663** | **0.6464 ± 0.0603** | **0.8927 ± 0.0387** |

### Per-Class F1 (Frozen vs LoRA r=8)

| Class | Frozen F1 | LoRA F1 | Δ |
|---|---:|---:|---:|
| Wake | 0.8883 | 0.9771 | +0.0888 |
| N1 | 0.0000 | 0.0000 | 0.0000 |
| N2 | 0.6115 | 0.8437 | +0.2322 |
| N3 | 0.5531 | 0.8320 | +0.2788 |
| REM | 0.0000 | 0.5759 | +0.5759 |

### Engineering Verification

| Metric | Value |
|---|---|
| Base latency | 6.25 ms/batch |
| LoRA unmerged | 5.66 ms/batch |
| LoRA merged | 5.62 ms/batch |
| Merge diff | 0.00e+00 (PASS) |
| Adapter reload | 0.00e+00 (PASS) |

### Terminology

- **Validation accuracy (87.27%)**: Used for early stopping/checkpoint selection during training on 20% of training data. Not the final result.
- **4-fold held-out-subject CV test accuracy (90.66% ± 3.59%)**: Final result on held-out subjects. This is the reported metric.

---

## Exhibition Presentation

### Primary Result Block

Use this exact block across all exhibition materials:

```
┌─────────────────────────────────────────┐
│     NEUROMORPHIC SLEEP STAGE SCORING    │
├─────────────────────────────────────────┤
│                                         │
│     87.34%                              │
│     TEST ACCURACY                       │
│                                         │
│     0.7551                              │
│     COHEN'S κ                           │
│                                         │
│     99,477                              │
│     PARAMETERS                          │
│                                         │
│     8.5 ms/batch                        │
│     CPU LATENCY                         │
│                                         │
└─────────────────────────────────────────┘
```

### Per-Class F1 Visualization

```
Wake  ████████████████████████████████  0.97
N1    ████                              0.20
N2    ████████████████████████████      0.82
N3    ██████████████████████████        0.78
REM   ███████                           0.36
```

### Talking Points

1. **Problem:** "Sleep staging requires expert analysis of multi-channel PSG recordings. We built an automated system to classify five sleep stages."

2. **Solution:** "Our model uses a compact architecture with multi-resolution features, depthwise-separable convolutions, and temporal sequence modeling."

3. **Result:** "The final model achieves 87.34% accuracy with only 99,477 parameters — small enough for edge deployment."

4. **Impact:** "This demonstrates that efficient deep learning can automate sleep classification while remaining practical for resource-constrained devices."

---

## Reproducibility

To reproduce this result:

```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py

# 2. Train model
python scripts/train.py --config configs/training.yaml

# 3. Evaluate
python scripts/evaluate.py --checkpoint artifacts/student_improved_best.pt
```

Expected output:
```
Final Official Result — Improved Student
  test_accuracy: 0.8734
  cohen_kappa: 0.7551
  macro_f1: 0.6259
  weighted_f1: 0.8653
  macro_gmean: 0.5371
```

---

## Citation

```bibtex
@project{neuromorphic_sleep_2026,
  title={Neuromorphic Sleep Stage Scoring},
  author={Kaushik, P. and Vora, S. and Bhatt, S. and Khan, S. and Lone, A.J.},
  year={2026},
  institution={VIT Bhopal University},
  note={Exhibition project}
}
```

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
