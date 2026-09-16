# NeuroSleep — Model Report

## Executive Summary

NeuroSleep investigates compact multi-resolution convolutional-recurrent
sleep staging under a strict parameter budget. The final deployable model
— the **Improved Student** (99,477 parameters) — is trained from scratch
with supervised class-weighted cross-entropy in the notebook pipeline
(Notebooks 01→05), and evaluated on held-out
test subjects.

> **Cohort correction (Sept 2026):** PhysioNet's SC records are two
> nights per person (`SC4ss1`/`SC4ss2` = same person). The "92-subject"
> cohort is **92 records from 52 persons**. The legacy record-level folds
> leaked at person level in 10/10 folds, so all earlier numbers are
> **record-level estimates**. The primary benchmark (EXP-BENCH-PERSON)
> trains and evaluates under person-level 10-fold CV over 52 persons.

Current status:

- **Person-level benchmark (EXP-BENCH-PERSON, primary, seeds 42/43/44,
  30 folds):** accuracy **87.30% ± 0.33%**, κ **0.738 ± 0.010**,
  macro-F1 **0.724 ± 0.005** — the honest person-generalization estimate
- **Standalone notebook run (deployed checkpoint, single 70/15/15
  subject split, supervised CE):** accuracy **90.57%**, κ **0.808**,
  macro-F1 **0.749** on 15 held-out test subjects — per-epoch training
  logs in Notebook 04; checkpoint `artifacts/standalone_99k/student_99477_best.pt`
- **Adaptation study (Frozen / LoRA / Full-FT):** quarantined —
  contaminated base checkpoint and record-level folds; retained in
  `docs/adaptation.md` as a like-for-like internal comparison only

## Model

| Property | Value |
|----------|-------|
| Architecture | Multi-resolution stem → depthwise-separable CNN → parametric Gabor FEB → 2-layer GRU → 5-class head |
| Parameters | 99,477 (~400 KB FP32) |
| Input | 10 × 30 s epochs × 4 channels @ 100 Hz (300 s context) |
| Output | Per-epoch probabilities over {Wake, N1, N2, N3, REM} |
| Training | Supervised class-weighted cross-entropy (from scratch), AdamW 3e-4, cosine schedule with 10% warmup, 20 epochs |
| Deployment | CPU inference 6.2 ms per 10-epoch batch (measured); checkpoint `artifacts/standalone_99k/student_99477_best.pt` |

### Parameter budget

| Module | Parameters |
|--------|-----------:|
| Multi-Resolution Stem | 7,232 |
| Depthwise-Separable Encoder | 2,304 |
| Gabor Feature Extraction | 144 |
| GRU (2 layers, hidden 64) | 89,856 |
| Head | 325 |
| **Total** | **99,477** |

## Training Pipeline (Notebooks)

1. **01 — Data import:** 100 PSG/hypnogram pairs matched by subject;
   subject-level 70/15/15 split (no leakage, seed 42).
2. **02 — Preprocessing:** 0.5–35 Hz bandpass + 50 Hz notch,
   30-s epochs, AASM harmonization, z-score normalization, QC flags;
   232,219 epochs cached.
3. **03 — EDA:** class imbalance (~68% Wake), artifact burden, per-stage
   spectral fingerprints, 87.5% self-transition probability — motivating
   multi-scale features + temporal context.
4. **04 — Training:** Improved Student from scratch, supervised
   class-weighted cross-entropy. **Per-epoch logs**
   (train loss, val κ/acc/macro-F1); best-κ checkpointing; best val κ
   0.8274 @ epoch 18/20.
5. **05 — Evaluation:** 15 held-out subjects, 74,860 scored epochs —
   **90.57% accuracy, κ 0.808, macro-F1 0.749, weighted-F1 0.912**,
   CPU latency 6.2 ms/batch (measured).

## Evaluation Results

### Primary benchmark (person-level CV)

| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
| Accuracy | 87.30% ± 0.33% | [85.80%, 88.79%] |
| Cohen's κ | 0.738 ± 0.010 | [0.691, 0.785] |
| Macro F1 | 0.724 ± 0.005 | [0.693, 0.755] |
| Weighted F1 | 0.880 ± 0.003 | [0.860, 0.900] |
| MGm | 0.772 ± 0.006 | [0.721, 0.823] |

### Standalone notebook run (deployed checkpoint, single split, 15 test subjects)

| Metric | Value |
|--------|-------|
| Accuracy | 90.57% |
| Cohen's κ | 0.8080 |
| Macro F1 | 0.7490 |
| Weighted F1 | 0.9115 |
| Macro geometric mean | 0.7808 |
| F1 (Wake / N1 / N2 / N3 / REM) | 0.978 / 0.477 / 0.823 / 0.705 / 0.762 |
| CPU latency | 6.2 ms/batch (measured) |

### Known limitations

- **N1 remains the weakest class** (F1 0.477) — consistent with the
  literature; N1 is transitional, rare (~4.6%), and visually ambiguous.
- N3 recall is high but precision moderate; N2/N3 boundary confusion
  dominates remaining errors.
- Single-cohort dataset (Sleep-EDF); cross-dataset validation (SHHS)
  is future work.

## Reproducibility

- Notebooks 01→05 execute deterministically (seed 42) in well under an
  hour on one GTX-1650-class GPU including preprocessing; Notebook 04
  alone trains in ~18 minutes.
- Package code is covered by 92 passing tests (`pytest tests/`).
- Protocol integrity gates: `scripts/verify_protocol.py`,
  `scripts/protocol_fingerprint.py`.
- Historical evidence hierarchy and contamination record:
  `docs/results.md`, `docs/adaptation.md`, `docs/LIMITATIONS.md`.

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

## License

CC-BY-4.0. Research prototype — not a medical device.
