# Results — NeuroSleep

> **Single authoritative results document.** All numbers below are
> regenerated from canonical result artifacts by
> `scripts/generate_results_doc.py`.
> Never hand-edit a number; re-run the generator instead.

---

## Evidence Status (read this first)

The repository contains several tiers of evidence. Only the
**Primary** tier is citable as the project's generalization result.

> **P0 finding (Sept 2026):** `SC4ss1`/`SC4ss2` record pairs are **two
> nights of the same person** (PhysioNet sleep-edfx README: "ss is the
> subject number, and N is the night"; confirmed by SC-subjects.xls and
> EDF headers). The "92-subject" cohort is **92 records from 52
> persons**. The legacy record-level folds put a test record's
> same-person mate in the train set in **10/10 folds** — every number
> produced on them is a *record-level* estimate, not person
> generalization. The historical benchmark (EXP-BENCH-PERSON) uses
> person-level folds (`person_folds_52subj.json`); the runner refuses
> leaky folds by default.

> **Protocol fix (Sept 2026, applied):** the legacy evaluation
> protocol (stride-5 windows over subject-concatenated arrays,
> all-position supervision) double-counted epochs, allowed
> subject-seam windows, and spliced contexts across dropped-epoch gaps.
> The canonical historical protocol is **causal, one prediction per
> unique epoch**, subject-safe and gap-safe, enforced by
> `tests/test_sequence_dataset.py` and documented in
> `docs/evaluation_protocol.md`. All pre-fix results are quarantined
> (evidence removed in the Sept 2026 cleanup).

| Tier | Experiment | Status | Evidence |
|------|-----------|--------|----------|
| **Primary** | EXP-FULL-AUG30 — final full-corpus training, person-level 70/15/15 holdout (seed 42), stride-10 all-position evaluation | **Complete (seed 42): 90.48% / κ 0.8283** | `results/final/final_metrics.json` |
| Historical | EXP-BENCH-PERSON — from-scratch, person-level 10-fold CV over 52 persons, fixed (causal unique-epoch) protocol | Complete (seeds 42/43/44, 30 folds): 87.30% / κ 0.738 | `results/research/EXP-BENCH-PERSON/` |
| Deployed | EXP-STANDALONE-99K — notebooks 01→05, supervised CE, exhibition 70/15/15 subject split, all-position protocol | Complete (seed 42): 90.57% / κ 0.808 | `results/standalone_99k/`, `artifacts/standalone_99k/student_99477_best.pt` |
| Superseded | EXP-BENCH-92SUBJ — from-scratch, record-level folds (person-leaky), legacy protocol | 87.66% ± 2.22% is a **record-level** estimate only | recorded here (evidence removed in the Sept 2026 cleanup) |
| Quarantined | EXP-ADAPT-* — Frozen / LoRA / Full-FT from the 15-record-era base checkpoint | **Contaminated** — base checkpoint's training records overlap 12 eval test folds and 3 validation folds; frozen baseline inflated ~+2.5pp | `docs/adaptation.md` (evidence removed in the Sept 2026 cleanup) |
| Archived | EXP-DEV-15SUBJ — 15-record development benchmark (93.0%) | Historical only; small cohort; do not cite as final | `docs/archive/development_15_subject.md` |

**Canonical cohort phrasing (historical tiers):** *"92-record eligible
cohort (52 persons) from 100 downloaded Sleep-EDF Expanded records
(8 wake-only excluded)."* Never write "92-subject evaluation" — the
evaluation cohort is 52 persons / 92 records.

---

## Final Submission Run — EXP-FULL-AUG30 (person-level 70/15/15 holdout)

> **This is the current authoritative result.** Notebooks 01→05 on the
> complete Sleep-EDF Expanded v1.0.0 corpus (197
> recordings / 100 subjects), person-level 70/15/15
> split (seed 42): train 69 / val
> 15 / test 16 subjects.
> Improved Student (99,477 params), 10×30 s context,
> batch 8, ≤30 epochs with early
> stopping (patience 5) on best
> validation **Macro F1**, train-only augmentation, AdamW
> 0.0003 + cosine-warmup. All metrics are computed
> from the **held-out test subjects only** (16
> subjects, 7,220 stride-10 windows
> = 72,200 labels); the notebook chain ends with
> `FINAL PROTOCOL AUDIT PASSED`.

**Headline:** NeuroSleep achieved **90.48% accuracy**,
**0.8283 Cohen's kappa**, and **0.7899 macro F1** on a held-out
person-level test set of 16 subjects from the
complete Sleep-EDF Expanded corpus, using the
99,477-parameter Improved Student model.

**Evaluation semantics:** test evaluation used non-overlapping
stride-10 windows, with each test epoch scored once.

### Test metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9048 |
| Cohen's κ | 0.8283 |
| Macro F1 | 0.7899 |
| Weighted F1 | 0.9089 |
| Macro Geometric Mean | 0.7911 |
| CPU latency (measured) | 8.94 ms per 5-minute window |
| Best validation | Macro F1 0.7645 @ epoch 12 (early-stopped at 17/30) |

### Per-class results (held-out test subjects)

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Wake | 0.991 | 0.969 | 0.980 |
| N1 | 0.451 | 0.647 | 0.532 |
| N2 | 0.831 | 0.834 | 0.832 |
| N3 | 0.816 | 0.717 | 0.763 |
| REM | 0.858 | 0.827 | 0.842 |

- **Protocol:** subject-level 70/15/15 (seed 42, per cohort) · evaluation stride
  10 (all positions, non-overlapping) · train
  stride 5 · checkpoint selection
  `val_macro_f1` · augmentation train-only
- **Reproducibility:** seed 42, manifest
  `data/manifests/sleep_edf_full.csv` (SHA-256 `45ddd5774d7a7370…`),
  checkpoint `artifacts/final/EXP-FULL-AUG30_seed42.pt`
- **Config:** `results/final/experiment_config.json` ·
  **History:** `results/final/training_history.csv` ·
  **Predictions:** `results/final/predictions.csv`

### Fit diagnosis — train/val/test gap (overfitting check)

The frozen checkpoint scored on clean stride-10 windows of all three splits
(augmentation off, no retraining; Notebook 05 §14,
`results/final/fit_diagnosis.json`):

| Split | Windows | Labels | Accuracy | κ | Macro F1 | N1 F1 |
|-------|--------:|-------:|---------:|----:|---------:|------:|
| Train | 31,285 | 312,850 | 0.9250 | 0.8653 | 0.8328 | 0.627 |
| Val | 7,008 | 70,080 | 0.8737 | 0.7782 | 0.7645 | 0.521 |
| Test | 7,220 | 72,200 | 0.9048 | 0.8283 | 0.7899 | 0.532 |

- **Gaps:** train−val **+5.12 pp**,
  train−test **+2.01 pp**,
  val−test **-3.11 pp**
- **Weakest class on train:** N1 (F1 0.627) — weak on the
  training split too, so its errors are label ambiguity, not memorization
- **Verdict:** Mild, controlled generalization gap (5.12 pp train-val): train accuracy 0.9250 is not saturated and the weakest train class (N1, F1 0.627) is weak even on training data — errors are dominated by label ambiguity, not memorization. Test (0.9048) exceeding val (0.8737) confirms no systematic degradation on unseen subjects.

---

## Historical Benchmark — EXP-BENCH-PERSON (from-scratch, seeds 42/43/44, fixed protocol, complete)

> **Historical benchmark (retained for protocol comparison — not the
> submission result; see EXP-FULL-AUG30 above).** Produced under the
> post-audit protocol: subject-safe sequence windows (no window spans
> two subjects or a dropped-epoch gap) and causal unique-epoch
> evaluation (stride-1, last-epoch supervision — exactly one prediction
> per scored epoch). See `docs/evaluation_protocol.md`.

- **Protocol:** 10-fold **person-level** CV — whole persons (both
  nights) assigned to folds; 5 fixed validation persons; folds
  stratified by age decade (cohort spans 25–101 yr); 52 persons /
  92 records
- **Sequence construction:** subject-safe windows, stride 5
  (training only), all-position supervision (training signal only)
- **Evaluation:** causal, stride 1, last-epoch supervision; every
  scored epoch predicted exactly once with (subject, epoch)
  provenance; epochs whose 5-min context spans a dropped epoch are
  excluded as unscoreable (798 across the 10 test folds)
- **Initialization:** from scratch (random init)
- **Seeds:** 42, 43, 44 (30 trained folds total) — cross-seed
  agreement is tight (accuracy SD across seed means:
  0.33pp)
- **Environment:** pinned in `results/benchmark_person_level/env.json`
  (torch 2.6.0+cu124, CUDA 12.4, GTX 1650, git SHA, protocol-file
  checksums)

### Overall metrics (mean of per-seed means; pooled fold-level 95% CI, n = 30)

| Metric | Seeds 42 / 43 / 44 | Mean ± SD(seeds) | 95% CI (pooled) |
|--------|--------------------|------------------|-----------------|
| Accuracy | 87.55 / 87.42 / 86.92% | **87.30% ± 0.33%** | [85.80, 88.79]% |
| Cohen's κ | 0.745 / 0.743 / 0.726 | **0.738 ± 0.010** | [0.691, 0.785] |
| Macro F1 | 0.728 / 0.726 / 0.719 | **0.724 ± 0.005** | [0.693, 0.755] |
| Weighted F1 | 88.27 / 88.16 / 87.65% | 88.03% ± 0.33% | [86.01, 90.05]% |
| MGm | 0.777 / 0.774 / 0.766 | 0.772 ± 0.006 | [0.721, 0.823] |

### Per-class F1 (mean of seed means; pooled 95% CI)

| Stage | F1 | 95% CI |
|-------|----|--------|
| Wake | 0.960 | [0.950, 0.970] |
| **N1** | **0.445** | [0.413, 0.478] |
| N2 | 0.733 | [0.672, 0.794] |
| N3 | 0.700 | [0.658, 0.742] |
| REM | 0.782 | [0.739, 0.826] |

### Interpretation

- The honest person-generalization estimate for this historical tier
  under the fixed protocol is **87.30%
  (κ 0.738)**, stable across three seeds. The
  near-coincidence with the superseded record-level number (87.66%) is
  *not* evidence of equivalence: the fixed protocol is stricter
  (unique-epoch scoring, no subject-seam windows, no spliced-gap
  contexts) while person-level folds are honest — the biases
  partially offset.
- Fold variance dominates seed variance (fold-level SD ~4pp vs
  seed-level ~0.3pp): person-level test groups are small
  (4–5 persons) and demographically heterogeneous. Fold 10 collapsed
  on N2 recall (0.17) — an honest hard-fold data point, retained
  rather than hidden.
- N1 remains the bottleneck (F1
  0.445, precision ~0.31 vs recall
  ~0.69 — the model over-predicts N1 relative to its ~4.6% base
  rate). This is a core research direction, not a cosmetic issue.
- Per-fold evidence: `results/research/EXP-BENCH-PERSON/fold_XX/`
  (seed 42) and `fold_XX_seed43/` / `fold_XX_seed44/`
  (metrics, confusion matrices, training history);
  cross-seed aggregate in `summary_multiseed.json`.
  Per-fold prediction dumps and fold checkpoints are regenerable and
  not tracked (see `.gitignore`).

---

## Historical (pre-protocol-fix) Benchmarks

Both runs below used the legacy protocol (stride-5 windows over
subject-concatenated arrays, all-position supervision) and are
quarantined (evidence removed in the Sept 2026 cleanup). They are
retained for like-for-like protocol comparison only.

### EXP-BENCH-PERSON, legacy protocol (person-level folds, seed 42)

Accuracy 85.47% ± 3.99%, κ 0.705 ± 0.128, macro F1 0.697 ± 0.082.
Note: the legacy *all-position* protocol scored most epochs twice and
included spliced-gap contexts, so these numbers are not directly
comparable to the fixed-protocol benchmark above.

### EXP-BENCH-92SUBJ (superseded: record-level folds, person-leaky, legacy protocol)

87.66% ± 2.22% — a **record-level** estimate only (each test record's
same-person mate was in the train pool in 10/10 folds). Per-class F1:
Wake 0.967 · N1 0.452 · N2 0.767 · N3 0.688 · REM 0.764.

---

## Quarantined — Adaptation Study (EXP-ADAPT-*)

These numbers are **retained for internal, like-for-like comparison
only** and must not be cited as generalization results. The base
checkpoint was trained on 15 records, 12 of which appear in the
evaluation test folds (see `docs/adaptation.md` for the full overlap
analysis). Additionally these runs used the record-level (person-leaky)
folds.

| Regime | Trainable params | Accuracy | κ | Macro F1 |
|--------|-----------------:|---------:|----:|---------:|
| 2A Frozen | 0 | 87.1% ± 3.6% | 0.738 ± 0.077 | 0.673 ± 0.074 |
| 2B LoRA CNN+Head (r=8, α=16) | 1,448 (1.43%) | 83.6% ± 3.7% | 0.693 ± 0.057 | 0.674 ± 0.045 |
| 2C Full FT | 99,477 (100%) | 87.7% ± 2.7% | 0.763 ± 0.043 | 0.730 ± 0.037 |

Reading notes:

- The Frozen/Full-FT numbers are inflated by ~+2.5pp by record overlap
  and further inflated by person-level leakage of the folds themselves.
- **Honest LoRA reading:** the tested CNN+Head configuration trains
  68.7× fewer parameters than full FT but produced *lower* accuracy and
  substantially lower macro-F1 (0.674 vs 0.730) in this (contaminated)
  run. The scientific claim — whether low-rank adaptation can
  compensate for a completely frozen GRU (90.3% of parameters) —
  remains **open** until re-run on a leak-free base checkpoint over
  person-level folds (`docs/adaptation.md` §4).

---

## Standalone Notebook Run — EXP-STANDALONE-99K (deployed checkpoint)

End-to-end run of notebooks 01→05 on the exhibition 70/15/15 subject
split (seed 42): supervised class-weighted
cross-entropy, 20 epochs, batch
16, AdamW 0.0003,
all-position protocol (not comparable to the fixed-protocol
historical benchmark above).

```
Accuracy    = 90.57%   (74,860 test epochs, 15 held-out subjects)
Cohen's κ   = 0.8080
Macro F1    = 0.7490
Weighted F1 = 0.9115
MGm         = 0.7808
Per-class F1 (W/N1/N2/N3/REM) = 0.978 / 0.477 / 0.823 / 0.705 / 0.762
Best validation κ = 0.8274 @ epoch 18/20
CPU latency = 6.2 ms/batch (measured, input [1,10,4,3000])
```

Evidence: `results/standalone_99k/`, checkpoint
`artifacts/standalone_99k/student_99477_best.pt`.

---

## Archived — 15-Record Development Benchmark (EXP-DEV-15SUBJ)

93.0% ± 1.0% accuracy, κ 0.861, macro-F1 0.794, 4-fold CV over 15
records. Superseded by the benchmarks above. Full record:
`docs/archive/development_15_subject.md`.

---

## Model Architecture (context for all results)

99,477 parameters — multi-resolution stem (7,232),
depthwise-separable encoder (1,904), parametric Gabor filters incl.
projection (160), 2-layer GRU hidden 64 (89,856 — 90.3% of the
parameter budget), linear head (325).
Input `[B, 10, 4, 3000]` @ 100 Hz; output per-epoch 5-class logits over
a 300-second context.

---

## Reproduction

```bash
# Final submission run (canonical pipeline — Notebooks 01→05)
#   NB01  manifests + person-level 70/15/15 split (seed 42)
#   NB04  trains EXP-FULL-AUG30 -> artifacts/final/EXP-FULL-AUG30_seed42.pt
#   NB05  evaluates held-out test subjects + FINAL PROTOCOL AUDIT
#         -> results/final/final_metrics.json (+ predictions, confusion
#            matrix, per-class metrics, experiment_config.json)

# Historical CV benchmark: fold manifest + CI summarizer
python scripts/generate_person_folds.py
python scripts/summarize_person_benchmark.py

# Regenerate this document from canonical artifacts
python scripts/generate_results_doc.py

# Verify protocol integrity
python scripts/verify_protocol.py

# Repository hygiene checks
python scripts/audit_repository.py
```

---

*Generated by `scripts/generate_results_doc.py` from canonical result
artifacts. Last regenerated: 29 September 2026.*
