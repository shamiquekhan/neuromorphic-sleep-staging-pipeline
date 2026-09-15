# NeuroSleep — Experiment Registry
## Authoritative Source of Experiment Definitions

This document is the **single source of truth** for all experiment definitions, protocols, and their statuses. All other documentation must reference this file.

---

## Experiment Index

| Experiment ID | Name | Status | Description | Config | Results |
|---------------|------|--------|-------------|--------|---------|
| **EXP-BENCH-PERSON** | Person-Level 10-Fold CV Benchmark | **PRIMARY** | Person-level 10-fold CV over 52 persons, causal unique-epoch protocol | `configs/experiments/person_level_cv.yaml` | `results/research/EXP-BENCH-PERSON/` |
| **EXP-EXHIBITION-15SUBJ** | Exhibition 15-Subject Holdout | **HISTORICAL** | Fixed 70/15/15 subject split, legacy all-position protocol | `configs/experiments/exhibition_15subj.yaml` | `results/exhibition/EXP-EXHIBITION-15SUBJ/` |

---

## EXP-BENCH-PERSON (Primary Benchmark)

### Overview
- **Dataset:** Sleep-EDF Expanded — 92-record eligible cohort from 100 downloaded records (8 wake-only excluded); 52 persons (40 two-night, 12 one-night)
- **Split:** Person-level 10-fold CV (whole persons assigned to folds, never single nights)
- **Validation:** Fixed 5 persons (both nights), stratified by age decade
- **Protocol:** Causal unique-epoch evaluation (stride=1, last-epoch supervision)
- **Seeds:** 42, 43, 44 (30 folds total)
- **Training:** From scratch, 20 epochs, batch=32, lr=3e-4, wd=1e-4
- **Loss:** Weighted cross-entropy (N1=2.0, REM=2.0)
- **Model:** Improved Student (99,477 params)

### Results (30 folds across 3 seeds)

| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
| Accuracy | 87.30% ± 0.33% | [85.80%, 88.79%] |
| Cohen's κ | 0.738 ± 0.010 | [0.691, 0.785] |
| Macro F1 | 0.724 ± 0.005 | [0.693, 0.755] |
| Weighted F1 | 0.880 ± 0.003 | [0.860, 0.900] |
| MGm | 0.772 ± 0.006 | [0.721, 0.823] |

### Per-Class F1 (Mean ± Std)

| Stage | F1 Mean ± Std | 95% CI |
|-------|---------------|--------|
| Wake | 0.960 ± 0.026 | [0.950, 0.970] |
| N1 | 0.445 ± 0.087 | [0.413, 0.478] |
| N2 | 0.733 ± 0.163 | [0.672, 0.794] |
| N3 | 0.700 ± 0.112 | [0.658, 0.742] |
| REM | 0.782 ± 0.116 | [0.739, 0.826] |

### Artifacts
- Checkpoints: `artifacts/research/EXP-BENCH-PERSON/` (per fold per seed)
- Provenance: `results/research/EXP-BENCH-PERSON/provenance.json`

---

## EXP-EXHIBITION-15SUBJ (Historical Exhibition)

### Overview
- **Dataset:** Sleep-EDF Expanded — 100 records (52 unique persons)
- **Split:** Fixed 70/15/15 subject-level split (70 train, 15 val, 15 test)
- **Protocol:** Legacy all-position evaluation (stride=5, all positions scored)
- **Seed:** 42
- **Training:** Teacher (focal loss) → Student (distillation), 20 epochs each, batch=16
- **Model:** Improved Student (99,477 params) distilled from Improved Teacher (193,197 params)

### Results (15 held-out test subjects)

| Metric | Value |
|--------|-------|
| Accuracy | 88.64% |
| Cohen's κ | 0.7725 |
| Macro F1 | 0.7186 |
| Weighted F1 | 0.8953 |
| Macro Geometric Mean | 0.7655 |

### Per-Class F1

| Stage | F1 |
|-------|-----|
| Wake | 0.970 |
| N1 | 0.440 |
| N2 | 0.791 |
| N3 | 0.685 |
| REM | 0.707 |

### Status: HISTORICAL
> This experiment uses the **legacy all-position protocol** which scores overlapping sequence windows. The primary research benchmark (EXP-BENCH-PERSON) uses the stricter causal unique-epoch protocol. Results are not directly comparable.

### Artifacts
- Teacher: `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/teacher_improved_best.pt`
- Student: `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/student_best.pt`
- Provenance: `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/provenance.json`

---

## Quarantined Experiments

| Experiment | Reason | Location |
|------------|--------|----------|
| Legacy 92-Subject Record-Level CV | Person-level leakage in 10/10 folds (SC4ss1/SC4ss2 = same person) | `results/quarantine/legacy_benchmark_92subj/` |
| Adaptation Study (Frozen/LoRA/Full-FT) | Contaminated base checkpoint + record-level folds | `results/quarantine/legacy_adaptation/` |

---

## Protocol Definitions

### Legacy All-Position (`legacy_all_position`)
- Sequence length: 10 epochs
- Training stride: 5
- Evaluation stride: 5
- Supervision: All positions (every epoch in window gets a loss)
- Used by: EXP-EXHIBITION-15SUBJ

### Causal Unique-Epoch (`causal_unique_epoch`)
- Sequence length: 10 epochs
- Training stride: 5 (all-position for training signal only)
- Evaluation stride: 1
- Supervision: Last epoch only (exactly one prediction per unique scored epoch)
- Person-level: Whole persons held out (no same-person train/test)
- Used by: EXP-BENCH-PERSON

---

## Version History

| Date | Change |
|------|--------|
| 2026-09-15 | Initial authoritative registry created. Split legacy and primary benchmarks. |