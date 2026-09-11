# Archived Experiment — 15-Subject Development Benchmark

> **STATUS: DEVELOPMENT BENCHMARK — NOT the primary/generalization result.**
>
> This experiment was the project's original final-project benchmark (August 2026).
> It was superseded by the 92-subject eligible-cohort benchmark
> (see [`docs/results.md`](../results.md)). It is preserved here for project
> history and must **never** be quoted as the final generalization performance.

## Historical Result

| Property | Value |
|----------|-------|
| Subjects | 15 (Sleep-EDF Expanded) |
| Folds | 4-fold subject-level CV (4-subject rotating test set, 11 additional subjects for train/val) |
| **Accuracy** | **93.0% ± 1.0%** |
| **Cohen's κ** | **0.861 ± 0.027** |
| **Macro F1** | **0.794 ± 0.036** |
| Weighted F1 | 0.935 ± 0.007 |
| MGm | 0.816 ± 0.043 |
| Parameters | 99,477 |
| Epochs | 15 |
| Batch size | 16 |
| Weight decay | 1e-2 |

## Fold Breakdown

| Fold | Test Subject | Accuracy | κ | Macro F1 |
|------|-------------|----------|-------|----------|
| 1 | SC4001 | 92.2% | 0.822 | 0.749 |
| 2 | SC4002 | 92.0% | 0.854 | 0.779 |
| 3 | SC4011 | 94.6% | 0.896 | 0.849 |
| 4 | SC4012 | 93.0% | 0.871 | 0.799 |

## Training Subjects of the Base Checkpoint

`artifacts/final/student_full_finetuned.pt` (the checkpoint produced by this
development benchmark) was trained on the following 15 subjects:

```
SC4001 SC4002 SC4011 SC4012 SC4022
SC4031 SC4032 SC4041 SC4042 SC4051
SC4052 SC4061 SC4062 SC4071 SC4072
```

> **PROTOCOL WARNING:** These 15 subjects are part of the 92-subject evaluation
> cohort. Any later experiment that uses this checkpoint as a "frozen base" while
> evaluating on the 92-subject folds is **subject-contaminated**
> (12 of the 15 appear in eval test folds, 3 in eval validation folds).
> See [`docs/adaptation.md`](../adaptation.md) and
> [`scripts/verify_protocol.py`](../../scripts/verify_protocol.py).

## Why It Was Superseded

1. **Small cohort** — 15 subjects cannot support a generalization claim.
2. **Optimistic estimate** — 4 folds over a small, partly overlapping subject
   pool produced an optimistic accuracy (93.0%) that did not survive the
   92-subject benchmark.
3. **Contamination** — the checkpoint from this benchmark later leaked into
   the 92-subject evaluation folds (see warning above).

## Files (Preserved)

| File | Description |
|------|-------------|
| `configs/development_15_subject.yaml` | Configuration (renamed from `final.yaml`) |
| `results/final/` | Raw fold metrics, confusion matrices, predictions |
| `results/final/final_metrics.json` | *Overwritten in a later commit — no longer reflects this benchmark; treat `fold_metrics.csv` + fold JSONs as the raw evidence* |
| `artifacts/final/student_full_finetuned.pt` | Checkpoint (15-subject era) |

## Original LoRA Development Study (Same Era)

An earlier LoRA study on this cohort compared Full FT, LoRA r=2/4/8 (head only)
under 4-fold CV — preserved in `results/LORA_RESULTS.md`. It reported
LoRA r=8 (552 params) at 90.66% ± 3.59%. Same 15-subject era; **not**
comparable to the 92-subject benchmark numbers.

---

*Archived September 2026. Primary benchmark: 92-subject eligible cohort,
10-fold subject-level CV (`docs/results.md`).*
