# LoRA Adaptation Results

## Training & Validation

The LoRA r=8 adapter was trained with 20% of the training data held out for validation. Early stopping used validation kappa.

**Validation performance during training:**

| Epoch | Val Accuracy | Val κ |
|------:|-------------:|------:|
| 1 | 85.15% | 0.6686 |
| 2 | 86.36% | 0.6927 |
| 3 | 85.45% | 0.6710 |
| **4** | **87.27%** | **0.7175** |
| 5 | 86.67% | 0.6989 |
| 6 | 85.76% | 0.6779 |
| 7 | 87.27% | 0.7156 |

**Best validation accuracy: 87.27% (κ = 0.7175) at epoch 4.**

## Held-Out Subject Cross-Validation (Final Result)

The final evaluation uses subject-level held-out cross-validation: train on 3 subjects, test on 1. This is the scientifically reported result.

### 4-Fold CV Summary

| Method | Trainable | Accuracy | κ | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Frozen Base | 0 | 79.92% ± 5.65% | 0.5001 ± 0.1329 | 0.4106 ± 0.0775 | 0.7462 ± 0.0689 |
| Full Fine-Tuning | 99,477 | 93.04% ± 0.87% | 0.8595 ± 0.0092 | 0.7379 ± 0.0390 | 0.9243 ± 0.0106 |
| LoRA r=2 | 138 | 89.94% ± 3.95% | 0.7904 ± 0.0743 | 0.6316 ± 0.0750 | 0.8842 ± 0.0427 |
| LoRA r=4 | 276 | 89.92% ± 3.23% | 0.7904 ± 0.0587 | 0.6323 ± 0.0570 | 0.8845 ± 0.0349 |
| **LoRA r=8** | **552** | **90.66% ± 3.59%** | **0.8092 ± 0.0663** | **0.6464 ± 0.0603** | **0.8927 ± 0.0387** |

### Multi-Seed Confirmation (LoRA r=8)

| Seed | Fold 1 κ | Fold 2 κ | Fold 3 κ | Fold 4 κ | Mean κ |
|---|---:|---:|---:|---:|---:|
| 42 | 0.8748 | 0.8771 | 0.7706 | 0.7004 | 0.8057 |
| 43 | 0.8798 | 0.8717 | 0.7670 | 0.7363 | 0.8137 |
| 44 | 0.8748 | 0.8595 | 0.7681 | 0.7386 | 0.8102 |
| **Overall** | | | | | **0.8099 ± 0.0657** |

### Per-Class F1 (Frozen vs LoRA r=8)

| Class | Frozen F1 | LoRA F1 | Δ |
|---|---:|---:|---:|
| Wake | 0.8883 | 0.9771 | +0.0888 |
| N1 | 0.0000 | 0.0000 | 0.0000 |
| N2 | 0.6115 | 0.8437 | +0.2322 |
| N3 | 0.5531 | 0.8320 | +0.2788 |
| REM | 0.0000 | 0.5759 | +0.5759 |

## Engineering Verification

| Metric | Value |
|---|---|
| Base latency | 6.25 ms/batch |
| LoRA unmerged | 5.66 ms/batch |
| LoRA merged | 5.62 ms/batch |
| Merge diff | 0.00e+00 (PASS) |
| Adapter reload | 0.00e+00 (PASS) |

## Key Conclusion

**LoRA r=8 achieves 94.2% of full fine-tuning's κ (0.8099 vs 0.8595) while training only 0.55% of the parameters (552 vs 99,477).**

## Important Distinction

- **Validation accuracy (87.27%)**: Used for early stopping during training on 20% of training data.
- **CV test accuracy (90.66% ± 3.59%)**: Final result on held-out subjects. This is the reported metric.
