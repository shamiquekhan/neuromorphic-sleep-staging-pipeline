# NeuroSleep — Model Report

## Executive Summary

NeuroSleep investigates compact multi-resolution convolutional-recurrent
sleep staging under a strict parameter budget. The final deployable model
— the **NeuroSleep Model** (99,477 parameters) — is trained from scratch
with supervised class-weighted cross-entropy in the notebook pipeline
(Notebooks 01→05), and evaluated on held-out
test subjects.

> **Cohort correction (Sept 2026):** PhysioNet's SC records are two
> nights per person (`SC4ss1`/`SC4ss2` = same person). The "92-subject"
> cohort is **92 records from 52 persons**. The legacy record-level folds
> leaked at person level in 10/10 folds, so all earlier numbers are
> **record-level estimates**. The historical benchmark
> (EXP-BENCH-PERSON) trained and evaluated under person-level 10-fold
> CV over 52 persons (archived).

Current status:

- **Final submission run (EXP-FULL-AUG30, project freeze Sept 2026,
  complete corpus):** accuracy **90.48%**, κ **0.8283**, macro-F1
  **0.7899** (weighted **0.9089**) on 16 held-out test subjects (7,220
  stride-10 windows = 72,200 epochs) from the full Sleep-EDF Expanded
  corpus — 197 recordings / 100 subjects, person-level 70/15/15 split
  (seed 42), train-only augmentation, batch 8, early-stopped at 17/30
  epochs on best validation macro-F1; ends with `FINAL PROTOCOL AUDIT
  PASSED` (Notebook 05)
- **Fit diagnosis (train/val/test gap):** **92.50% / 87.37% / 90.48%**
  (train/val/test accuracy, identical stride-10 protocol) — train−val
  gap **+5.12 pp**; N1 is weak even on training data (F1 0.627) →
  mild, controlled generalization gap (label ambiguity, not
  memorization) → `results/final/fit_diagnosis.json`
- **Standalone notebook run (deployed checkpoint, single 70/15/15
  subject split, supervised CE):** accuracy **90.57%**, κ **0.808**,
  macro-F1 **0.749** on 15 held-out test subjects — per-epoch training
  logs in `results/standalone_99k/training_history.csv`;
  checkpoint `artifacts/standalone_99k/student_99477_best.pt`
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
| Training | Supervised class-weighted cross-entropy (from scratch), AdamW 3e-4, cosine schedule with 10% warmup; final protocol: batch 8, ≤30 epochs with early stopping (patience 5) on validation macro-F1, train-only augmentation |
| Deployment | CPU inference 6.2 ms per 10-epoch batch (measured, standalone); 8.94 ms per 5-minute window (EXP-FULL-AUG30); checkpoints `artifacts/standalone_99k/student_99477_best.pt` and `artifacts/final/EXP-FULL-AUG30_seed42.pt` (promoted: `artifacts/neurosleep_model_best.pt`) |

### Parameter budget

| Module | Parameters |
|--------|-----------:|
| Multi-Resolution Stem | 7,232 |
| Depthwise-Separable Encoder | 1,904 |
| Gabor Feature Extraction (incl. projection) | 160 |
| GRU (2 layers, hidden 64) | 89,856 |
| Head | 325 |
| **Total** | **99,477** |

## Training Pipeline (Notebooks)

Final protocol (complete corpus, Sept 2026 freeze):

1. **01 — Data import:** complete Sleep-EDF Expanded corpus — 394 EDF
   files (197 PSG/hypnogram pairs, 100 subjects across the age-effects
   and sleep-telemetry cohorts), SHA-1-verified against the official MNE
   record tables; per-cohort person-level 70/15/15 split (69 / 15 / 16
   subjects, no leakage, seed 42).
2. **02 — Preprocessing:** 0.5–35 Hz bandpass (50 Hz notch recorded as
   metadata but skipped — it equals Nyquist at 100 Hz), 30-s epochs,
   AASM harmonization, z-score normalization, QC flags; **457,652
   epochs** cached in `data/cache_full/` (mmap-able per-recording layout).
3. **03 — EDA:** class imbalance, artifact burden, per-cohort
   distributions, per-stage spectral fingerprints, stage-transition
   structure — motivating multi-scale features + temporal context.
4. **04 — Training:** NeuroSleep Model from scratch, supervised
   class-weighted cross-entropy, train-only augmentation (amplitude
   0.90–1.10×, noise σ 0.005–0.03, temporal masking, channel dropout).
   **Per-epoch logs**; best-validation-macro-F1 checkpointing; best val
   macro-F1 0.7645 @ epoch 12, early-stopped at 17/30.
5. **05 — Evaluation:** 16 held-out test subjects (7,220
   stride-10 windows = 72,200 scored epochs) — **90.48% accuracy,
   κ 0.8283, macro-F1 0.7899, weighted-F1 0.9089**; CPU latency 8.94
   ms/window; fit diagnosis (train/val/test gap) written to
   `results/final/fit_diagnosis.json`; verified checkpoint promoted to
   `artifacts/neurosleep_model_best.pt`.

## Evaluation Results

### Final Submission Result — EXP-FULL-AUG30 (person-level holdout, 16 test subjects)

| Metric | Value |
|--------|-------|
| Accuracy | 90.48% |
| Cohen's κ | 0.8283 |
| Macro F1 | 0.7899 |
| Weighted F1 | 0.9089 |
| Macro geometric mean | 0.7911 |
| F1 (Wake / N1 / N2 / N3 / REM) | 0.980 / 0.532 / 0.832 / 0.763 / 0.842 |
| Recall (Wake / N1 / N2 / N3 / REM) | 0.969 / 0.647 / 0.834 / 0.717 / 0.827 |
| Best validation | Macro F1 0.7645 @ epoch 12 (early-stopped 17/30) |
| CPU latency | 8.94 ms per 5-minute window (measured) |
| Checkpoint | `artifacts/final/EXP-FULL-AUG30_seed42.pt` → promoted to `artifacts/neurosleep_model_best.pt` |
| Evidence | `results/final/final_metrics.json`, `results/final/fit_diagnosis.json` |

> Protocol note: person-level 70/15/15 holdout (seed 42), gap-aware
> sequence windows (stride 5 train / 10 eval, each test epoch scored
> once), probabilities calibrated to sum to 1, metrics generated only
> from the held-out test subjects. Not directly comparable to the
> historical tiers below (different splits and evaluation semantics).

**Fit diagnosis — train/val/test gap** (frozen checkpoint, identical
stride-10 protocol, augmentation off; `results/final/fit_diagnosis.json`):

| Split | Windows | Accuracy | Macro F1 | N1 F1 |
|-------|--------:|---------:|---------:|------:|
| Train | 31,285 | 92.50% | 0.8328 | 0.627 |
| Val | 7,008 | 87.37% | 0.7645 | 0.521 |
| Test | 7,220 | 90.48% | 0.7899 | 0.532 |

Gaps: train−val **+5.12 pp**, train−test **+2.01 pp**, test above val by
3.11 pp. Weakest train class N1 (F1 0.627) — weak on training data too,
so residual errors are label ambiguity, not memorization (mild,
controlled generalization gap).

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

- **N1 remains the weakest class** (EXP-FULL-AUG30 F1 0.532, recall 0.647;
  standalone F1 0.477) — consistent with the literature; N1 is
  transitional, rare, and visually ambiguous. The full corpus plus
  augmentation lifted N1 recall from 0.55 to 0.65.
- N3 precision (0.80) trails its recall (0.73 recall → N3→N2 confusion
  dominates remaining errors).
- Single-database dataset (Sleep-EDF, two cohorts); cross-dataset
  validation (SHHS) is future work.

## Reproducibility

- Notebooks 01→05 execute deterministically (seed 42) on one
  GTX-1650-class GPU. Final-corpus wall-clock: download ~4 h
  (network-dependent; ~8.2 GB), preprocessing ~2.5 h, training ~1.5 h
  (17 epochs, early-stopped), evaluation minutes.
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
