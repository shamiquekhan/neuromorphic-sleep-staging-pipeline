# 100-Subject Adaptation Benchmark — Final Comparison

**Dataset:** Sleep-EDF Expanded (92 subjects, 10-fold subject-level CV)
**Seed:** 42
**Base Checkpoint:** `artifacts/final/student_full_finetuned.pt`

## Overall Metrics Comparison

| Model | Trainable Params | Accuracy | κ | Macro F1 | Weighted F1 | MGm |
|-------|----------------:|---------:|----:|---------:|------------:|----:|
| **Frozen Base** | 0 (0%) | 87.1% ± 3.6% | 0.738 ± 0.077 | 0.672 ± 0.074 | 0.871 ± 0.042 | 0.668 ± 0.102 |
| **LoRA Head (r=8)** | 552 (0.55%) | 83.1% ± 3.8% | 0.682 ± 0.065 | 0.659 ± 0.053 | 0.850 ± 0.036 | 0.724 ± 0.070 |
| **LoRA CNN+Head (r=8)** | 1,448 (1.43%) | 83.5% ± 3.7% | 0.692 ± 0.057 | 0.673 ± 0.046 | 0.855 ± 0.033 | 0.744 ± 0.062 |
| **Full Fine-Tuning** | 99,477 (100%) | 87.3% ± 3.0% | 0.758 ± 0.047 | 0.726 ± 0.037 | 0.887 ± 0.025 | 0.794 ± 0.040 |

## Key Findings

### 1. Full Fine-Tuning Achieves Best Overall Performance
- **Accuracy:** 87.3% (best)
- **κ:** 0.758 (best)
- **Macro F1:** 0.726 (best)
- **Weighted F1:** 0.887 (best)

### 2. Frozen Base Transfers Surprisingly Well
- Only 0.2% accuracy drop vs full fine-tuning
- Demonstrates strong pretrained representations
- But significantly lower Macro F1 (0.672 vs 0.726)

### 3. LoRA Underperforms Full Fine-Tuning
- LoRA Head: 4.2% accuracy drop vs full FT
- LoRA CNN+Head: 3.8% accuracy drop vs full FT
- Both LoRA variants trail frozen base on accuracy
- But LoRA shows better minority-class recall (higher MGm)

### 4. Parameter Efficiency Analysis

| Model | Trainable Params | Efficiency Ratio | Accuracy Retention |
|-------|----------------:|------------------:|------------------:|
| Full FT | 99,477 | 1.0× | 100.0% |
| LoRA CNN+Head | 1,448 | 68.7× fewer | 95.6% (83.5/87.3) |
| LoRA Head | 552 | 180.2× fewer | 95.2% (83.1/87.3) |
| Frozen | 0 | ∞ | 99.7% (87.1/87.3) |

**Interpretation:** LoRA CNN+Head uses 68.7× fewer trainable parameters while retaining approximately 95.6% of full FT accuracy. For κ: LoRA CNN+Head retains 91.3% (0.692/0.758).

### 5. Minority Class Performance (N1)

| Model | N1 F1 | N1 Recall |
|-------|------:|----------:|
| Frozen | ~0.32 | ~0.35 |
| LoRA Head | ~0.35 | ~0.60 |
| LoRA CNN+Head | ~0.36 | ~0.65 |
| Full FT | ~0.46 | ~0.72 |

**Finding:** LoRA variants improve N1 recall over frozen base, but full FT achieves best N1 F1.

## Conclusion

> **Full fine-tuning remains the optimal choice for maximum five-stage sleep classification performance.** LoRA provides a parameter-efficient alternative, using only 1.4% of trainable parameters while retaining approximately 95.6% of full FT accuracy. The frozen base demonstrates that the pretrained Improved Student learns strong transferable representations.

## Recommendation

- **Production Model:** Full Fine-Tuning (99,477 params)
- **Research Contribution:** LoRA CNN+Head as parameter-efficient adaptation benchmark
- **Deployment:** Use full FT model for maximum accuracy; LoRA for resource-constrained adaptation scenarios

---

*Generated from 100-subject adaptation benchmark, seed 42, 10-fold subject-level CV*
