# NeuroSleep — Experiment Registry
## Authoritative Source of Experiment Definitions

This document is the **single source of truth** for all experiment definitions, protocols, and their statuses. All other documentation must reference this file.

---

## Experiment Index

| Experiment ID | Name | Status | Description | Config | Results |
|---------------|------|--------|-------------|--------|---------|
| **EXP-BENCH-PERSON** | Person-Level 10-Fold CV Benchmark | **PRIMARY** | Person-level 10-fold CV over 52 persons, causal unique-epoch protocol | `configs/experiments/person_level_cv.yaml` | `results/research/EXP-BENCH-PERSON/` |
| **EXP-STANDALONE-99K** | Standalone Notebooks 01→05 Run (supervised CE) | **DEPLOYED** | Fixed 70/15/15 subject split, all-position protocol, supervised class-weighted CE | notebooks 04/05, seed 42 | `results/standalone_99k/` |

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

## EXP-STANDALONE-99K (Standalone Notebook Run — Deployed Checkpoint)

### Overview
- **Dataset:** Sleep-EDF Expanded — exhibition manifest, 70 train / 15 val / 15 test subjects
- **Split:** Fixed 70/15/15 subject-level split (`data/manifests/exhibition_15subj_v1.json`)
- **Protocol:** All-position (stride 5, every position supervised)
- **Seed:** 42
- **Training:** Improved Student from scratch, supervised class-weighted cross-entropy, 20 epochs, batch 16, AdamW 3e-4
- **Model:** Improved Student (99,477 params)

### Results (15 held-out test subjects, 74,860 epochs)

| Metric | Value |
|--------|-------|
| Accuracy | 90.57% |
| Cohen's κ | 0.8080 |
| Macro F1 | 0.7490 |
| Weighted F1 | 0.9115 |
| Macro Geometric Mean | 0.7808 |
| CPU latency (measured) | 6.2 ms/batch |

### Per-Class F1

| Stage | F1 |
|-------|-----|
| Wake | 0.978 |
| N1 | 0.477 |
| N2 | 0.823 |
| N3 | 0.705 |
| REM | 0.762 |

### Artifacts
- Checkpoint: `artifacts/standalone_99k/student_99477_best.pt` (best epoch 18, val κ 0.8274)
- Results: `results/standalone_99k/` (history, metrics, confusion matrix, plots)
- Dashboard/deployment result row: `results/final/notebook_pipeline_result.csv`

---

## Quarantined Experiments

| Experiment | Reason | Location |
|------------|--------|----------|
| Legacy 92-Subject Record-Level CV | Person-level leakage in 10/10 folds (SC4ss1/SC4ss2 = same person) | evidence removed; recorded in `docs/results.md` |
| Adaptation Study (Frozen/LoRA/Full-FT) | Contaminated base checkpoint + record-level folds | evidence removed; recorded in `docs/adaptation.md` |

---

## Protocol Definitions

### Legacy All-Position (`legacy_all_position`)
- Sequence length: 10 epochs
- Training stride: 5
- Evaluation stride: 5
- Supervision: All positions (every epoch in window gets a loss)
- Used by: EXP-STANDALONE-99K

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
| 2026-09-17 | Added EXP-STANDALONE-99K (standalone notebooks 01→05 supervised run, deployable checkpoint `artifacts/standalone_99k/student_99477_best.pt`). |
| 2026-09-17 | Removed the historical distilled-student exhibition experiment (EXP-EXHIBITION-15SUBJ), teacher architecture, and its artifacts — the repository ships a single architecture. |