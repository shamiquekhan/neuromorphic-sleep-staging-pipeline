# 100-Subject Adaptation Benchmark — Final Results

**Dataset:** Sleep-EDF Expanded (92 subjects, 10-fold subject-level CV)
**Seeds:** 42, 43, 44 (30 folds per method)
**Base Checkpoint:** `artifacts/final/student_full_finetuned.pt`

## Overall Metrics (Mean ± Std Across 30 Folds)

| Model | Trainable Params | Accuracy | κ | Macro F1 | Weighted F1 | MGm |
|-------|----------------:|---------:|----:|---------:|------------:|----:|
| Frozen | 0 | 0.8706 ± 0.0362 | 0.7378 ± 0.0773 | 0.6725 ± 0.0740 | 0.8708 ± 0.0425 | 0.6680 ± 0.1025 |
| LoRA CNN+Head | 1,448 | 0.8361 ± 0.0366 | 0.6934 ± 0.0572 | 0.6736 ± 0.0454 | 0.8561 ± 0.0323 | 0.7448 ± 0.0596 |
| Full Fine-Tuning | 99,477 | 0.8766 ± 0.0267 | 0.7632 ± 0.0427 | 0.7302 ± 0.0367 | 0.8898 ± 0.0230 | 0.7960 ± 0.0384 |

## Per-Stage F1 (Mean ± Std Across 30 Folds)

| Stage | Frozen | LoRA CNN+Head | Full Fine-Tuning |
|-------|-------:|--------------:|-----------------:|
| Wake | 0.9640 ± 0.0200 | 0.9465 ± 0.0235 | 0.9659 ± 0.0135 |
| N1 | 0.3449 ± 0.0815 | 0.3582 ± 0.0418 | 0.4504 ± 0.0598 |
| N2 | 0.7526 ± 0.0554 | 0.7200 ± 0.0429 | 0.7725 ± 0.0383 |
| N3 | 0.6286 ± 0.1142 | 0.6635 ± 0.1110 | 0.6832 ± 0.1119 |
| REM | 0.6722 ± 0.1567 | 0.6798 ± 0.1126 | 0.7788 ± 0.0719 |

## Per-Stage Precision (Mean ± Std)

| Stage | Frozen | LoRA CNN+Head | Full Fine-Tuning |
|-------|-------:|--------------:|-----------------:|
| Wake | 0.9682 ± 0.0329 | 0.9904 ± 0.0131 | 0.9945 ± 0.0043 |
| N1 | 0.3611 ± 0.0811 | 0.2507 ± 0.0411 | 0.3354 ± 0.0632 |
| N2 | 0.7794 ± 0.0708 | 0.8625 ± 0.0439 | 0.8796 ± 0.0408 |
| N3 | 0.4889 ± 0.1282 | 0.5492 ± 0.1380 | 0.5688 ± 0.1359 |
| REM | 0.7452 ± 0.0991 | 0.6412 ± 0.0646 | 0.7745 ± 0.0558 |

## Per-Stage Recall (Mean ± Std)

| Stage | Frozen | LoRA CNN+Head | Full Fine-Tuning |
|-------|-------:|--------------:|-----------------:|
| Wake | 0.9609 ± 0.0241 | 0.9073 ± 0.0410 | 0.9393 ± 0.0256 |
| N1 | 0.3460 ± 0.1047 | 0.6497 ± 0.0737 | 0.7099 ± 0.0615 |
| N2 | 0.7302 ± 0.0599 | 0.6194 ± 0.0505 | 0.6913 ± 0.0543 |
| N3 | 0.9198 ± 0.0475 | 0.8739 ± 0.0771 | 0.8886 ± 0.0782 |
| REM | 0.6375 ± 0.1811 | 0.7460 ± 0.1715 | 0.7943 ± 0.1124 |

## Parameter Efficiency

- LoRA CNN+Head: 1,448 / 99,477 = **1.43%** of trainable parameters
- Accuracy retention: 0.8361 / 0.8766 = **95.4%**
- κ retention: 0.6934 / 0.7632 = **90.9%**
- Frozen accuracy retention: 0.8706 / 0.8766 = **99.3%**

## Conclusion

> **Full fine-tuning is the preferred model for the NeuroSleep project** because it provides the strongest overall and stage-balanced performance on the 100-subject adaptation benchmark.

---

*Generated from 3 seeds × 10 folds = 30 folds per method*