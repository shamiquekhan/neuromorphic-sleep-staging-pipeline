# Final Results — NeuroSleep

> **This is the single authoritative result document for the project.** All other result tables in the repository are historical.

---

## Executive Summary

| Property | Value |
|----------|-------|
| Model | Improved Student — Full Fine-Tuning |
| Parameters | 99,477 |
| Dataset | Sleep-EDF Expanded (92 subjects) |
| **Accuracy** | **87.7% ± 2.7%** |
| **Cohen's Kappa** | **0.763 ± 0.043** |
| **Macro F1** | **0.730 ± 0.037** |
| **Weighted F1** | **89.0% ± 2.1%** |
| **MGm** | **0.797 ± 0.040** |
| Evaluation | 10-fold subject-level CV, 3 seeds (42, 43, 44) |

---

## Per-Class Performance (Full Fine-Tuning, 30 Folds)

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.964 ± 0.016 | 0.995 ± 0.003 | 0.936 ± 0.031 |
| **N1** | **0.445 ± 0.061** | 0.330 ± 0.065 | 0.712 ± 0.070 |
| N2 | 0.768 ± 0.041 | 0.878 ± 0.043 | 0.687 ± 0.063 |
| N3 | 0.681 ± 0.114 | 0.562 ± 0.135 | 0.896 ± 0.075 |
| REM | 0.771 ± 0.078 | 0.772 ± 0.077 | 0.785 ± 0.118 |

### N1 Analysis

N1 is the hardest stage with F1=0.445. This is expected because:
- N1 epochs are brief (1-7 minutes) and transitional
- N1 is often confused with N2 and Wake
- N1 is only ~4.6% of the dataset

### Model Strengths

- **High Wake F1** (0.964) — Wake classification is excellent
- **Strong N2 detection** (F1=0.768) — light sleep well-distinguished
- **Balanced performance** — Macro F1 (0.730) indicates good stage balance

---

## Adaptation Method Comparison (30 Folds)

| Model | Trainable Params | Accuracy | κ | Macro F1 | Weighted F1 | MGm |
|-------|----------------:|---------:|----:|---------:|------------:|----:|
| Frozen Base | 0 (0%) | 87.1% ± 3.6% | 0.738 ± 0.077 | 0.673 ± 0.074 | 87.1% ± 4.2% | 0.668 ± 0.102 |
| LoRA CNN+Head (r=8) | 1,448 (1.43%) | 83.6% ± 3.7% | 0.693 ± 0.057 | 0.674 ± 0.045 | 85.5% ± 3.3% | 0.744 ± 0.062 |
| **Full Fine-Tuning** | **99,477 (100%)** | **87.7% ± 2.7%** | **0.763 ± 0.043** | **0.730 ± 0.037** | **89.0% ± 2.1%** | **0.797 ± 0.040** |

### Per-Stage F1 Comparison

| Stage | Frozen | LoRA CNN+Head | Full FT |
|-------|-------:|--------------:|--------:|
| Wake | 0.964 ± 0.020 | 0.945 ± 0.024 | 0.964 ± 0.016 |
| N1 | 0.345 ± 0.082 | 0.357 ± 0.041 | **0.445 ± 0.061** |
| N2 | 0.753 ± 0.055 | 0.721 ± 0.044 | **0.768 ± 0.041** |
| N3 | 0.629 ± 0.114 | 0.668 ± 0.110 | **0.681 ± 0.114** |
| REM | 0.672 ± 0.157 | 0.675 ± 0.117 | **0.771 ± 0.078** |

### Parameter Efficiency

- LoRA CNN+Head uses **68.7× fewer trainable parameters** (1,448 vs 99,477)
- LoRA CNN+Head retains **95.4% of full FT accuracy** (83.6% / 87.7%)
- LoRA CNN+Head retains **90.9% of full FT κ** (0.693 / 0.763)
- Frozen base already transfers well at **99.3% of full FT accuracy**

### Key Findings

1. **Full fine-tuning achieves the strongest overall and stage-balanced performance**
2. **Frozen base transfers surprisingly well** — only 0.6% accuracy drop vs full FT
3. **LoRA provides parameter-efficient adaptation** but does not match full FT
4. **Full FT improves every stage** over frozen base, especially N1 (+0.10 F1) and REM (+0.10 F1)

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

## Training Protocol (100-Subject)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Epochs | 20 (early stopping patience=5) |
| Batch Size | 32 |
| Class Weights | N1=2x, REM=2x |
| Gradient Clipping | max_norm=1.0 |
| Scheduler | Cosine Annealing |
| Mixed Precision | True (CUDA) |
| Device | NVIDIA GTX 1650 (CUDA 12.4) |
| Seeds | 42, 43, 44 |

---

## Evaluation Protocol (100-Subject)

- **Method:** 10-fold subject-level cross-validation
- **Subjects:** 92 from Sleep-EDF Expanded (8 wake-only excluded)
- **Fold assignment:** Canonical (canonical_subject_folds_92subj.json)
- **Sequence:** length=10, stride=5, 30s epochs
- **All-position supervision** — every epoch in the window is supervised
- **Reproducibility:** 3 seeds × 10 folds = 30 folds per method

---

## Files

### 100-Subject Benchmark (Authoritative)

| File | Description |
|------|-------------|
| `configs/full_100_subject.yaml` | 100-subject configuration |
| `artifacts/final/student_full_finetuned.pt` | Checkpoint |
| `results/100_subject_adaptation/final/aggregate_metrics.json` | 3-seed aggregate |
| `results/100_subject_adaptation/final/overall_comparison.csv` | Overall metrics |
| `results/100_subject_adaptation/final/per_class_comparison.csv` | Per-class metrics |
| `results/100_subject_adaptation/final/FINAL_ADAPTATION_RESULTS.md` | Full report |

### 15-Subject Development (Historical)

| File | Description |
|------|-------------|
| `configs/final.yaml` | Development configuration |
| `results/final/final_metrics.json` | Development results |

---

## Reproduction

```bash
# Run 100-subject adaptation benchmark (all 3 modes, 1 seed)
python scripts/train_adaptation.py --mode frozen --seed 42 --device cuda
python scripts/train_adaptation.py --mode lora --targets enc.0.pw,enc.1.pw,head --rank 8 --alpha 16 --seed 42 --device cuda
python scripts/train_adaptation.py --mode full_finetune --seed 42 --device cuda

# Aggregate all 3 seeds
python scripts/aggregate_adaptation_results.py

# Verify protocol consistency
python scripts/protocol_fingerprint.py --seed 42 --mode full_finetune --save-reference
python scripts/protocol_fingerprint.py --seed 43 --mode full_finetune
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
