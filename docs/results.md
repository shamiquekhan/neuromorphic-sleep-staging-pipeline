# Results — NeuroSleep

> **Single authoritative results document.** All numbers below are
> regenerated from raw fold evidence by `scripts/summarize_benchmark.py`.
> Never hand-edit a number; re-run the summarizer instead.

---

## Evidence Status (read this first)

The repository contains four tiers of evidence. Only the first is citable
as a generalization result.

> **P0 finding (Sept 2026):** `SC4ss1`/`SC4ss2` record pairs are **two
> nights of the same person** (PhysioNet sleep-edfx README: "ss is the
> subject number, and N is the night"; confirmed by SC-subjects.xls and
> EDF headers). The "92-subject" cohort is **92 records from 52
> persons**. The legacy record-level folds put a test record's
> same-person mate in the train set in **10/10 folds** — every number
> produced on them is a *record-level* estimate, not person
> generalization. The primary benchmark (EXP-BENCH-PERSON) uses
> person-level folds (`person_folds_52subj.json`); the runner refuses
> leaky folds by default.

> **Protocol fix (Sept 2026, applied):** the legacy evaluation
> protocol (stride-5 windows over subject-concatenated arrays,
> all-position supervision) double-counted epochs, allowed
> subject-seam windows, and spliced contexts across dropped-epoch gaps.
> The canonical protocol is now **causal, one prediction per unique
> epoch**, subject-safe and gap-safe, enforced by
> `tests/test_sequence_dataset.py` and documented in
> `docs/evaluation_protocol.md`. All pre-fix results are quarantined
> (evidence removed in the Sept 2026 cleanup). The primary benchmark
> below was **re-run under the fixed protocol**.

| Tier | Experiment | Status | Evidence |
|------|-----------|--------|----------|
| **Primary** | EXP-BENCH-PERSON — from-scratch, person-level 10-fold CV over 52 persons, fixed (causal unique-epoch) protocol | **Complete (seeds 42/43/44, 30 folds)** | `results/research/EXP-BENCH-PERSON/` |
| **Deployed** | EXP-STANDALONE-99K — notebooks 01→05, supervised CE, exhibition 70/15/15 split, all-position protocol | Complete (seed 42): 90.57% / κ 0.808 | `results/standalone_99k/`, `artifacts/standalone_99k/student_99477_best.pt` |
| Superseded | EXP-BENCH-92SUBJ — from-scratch, record-level folds (person-leaky), legacy protocol | 87.66% ± 2.22% is a **record-level** estimate only | recorded here (evidence removed in the Sept 2026 cleanup) |
| Quarantined | EXP-ADAPT-* — Frozen / LoRA / Full-FT from the 15-record-era base checkpoint | **Contaminated** — base checkpoint's training records overlap 12 eval test folds and 3 validation folds; frozen baseline inflated ~+2.5pp | `docs/adaptation.md` (evidence removed in the Sept 2026 cleanup) |
| Archived | EXP-DEV-15SUBJ — 15-record development benchmark (93.0%) | Historical only; small cohort; do not cite as final | `docs/archive/development_15_subject.md`, `results/final/` |

**Canonical cohort phrasing:** *"92-record eligible cohort (52 persons)
from 100 downloaded Sleep-EDF Expanded records (8 wake-only excluded)."*
Never write "92-subject evaluation" — the evaluation cohort is 52
persons / 92 records.

---

## Primary Benchmark — EXP-BENCH-PERSON (from-scratch, seeds 42/43/44, fixed protocol, complete)

> **This is the current authoritative result.** Produced under the
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
  agreement is tight (accuracy SD across seed means: 0.33pp)
- **Environment:** pinned in `results/benchmark_person_level/env.json`
  (torch 2.6.0+cu124, CUDA 12.4, GTX 1650, git SHA, protocol-file
  checksums)

### Overall metrics (mean of per-seed means; pooled fold-level 95% CI, n = 30)

| Metric | Seeds 42 / 43 / 44 | Mean ± SD(seeds) | 95% CI (pooled) |
|--------|--------------------|------------------|-----------------|
| **Accuracy** | 87.55 / 87.42 / 86.92% | **87.30% ± 0.33%** | [85.80, 88.79]% |
| **Cohen's κ** | 0.745 / 0.743 / 0.726 | **0.738 ± 0.010** | [0.691, 0.785] |
| **Macro F1** | 0.728 / 0.726 / 0.719 | **0.724 ± 0.005** | [0.693, 0.755] |
| Weighted F1 | 0.883 / 0.882 / 0.877 | 0.880 ± 0.003 | [0.860, 0.901] |
| MGm | 0.777 / 0.774 / 0.766 | 0.772 ± 0.006 | [0.721, 0.823] |

### Per-class F1 (mean of seed means; pooled 95% CI)

| Stage | F1 | 95% CI |
|-------|----|--------|
| Wake | 0.960 | [0.950, 0.970] |
| N1 | **0.445** | [0.413, 0.478] |
| N2 | 0.733 | [0.672, 0.794] |
| N3 | 0.700 | [0.658, 0.742] |
| REM | 0.782 | [0.739, 0.826] |

### Interpretation

- The honest person-generalization estimate under the fixed protocol is
  **87.30% (κ 0.738)**, stable across three seeds. The near-coincidence
  with the superseded record-level number (87.66%) is *not* evidence of
  equivalence: the fixed protocol is stricter (unique-epoch scoring, no
  subject-seam windows, no spliced-gap contexts) while person-level
  folds are honest — the biases partially offset.
- Fold variance dominates seed variance (fold-level SD ~4pp vs
  seed-level ~0.3pp): person-level test groups are small (4–5 persons)
  and demographically heterogeneous. Fold 10 collapsed on N2 recall
  (0.17) — an honest hard-fold data point, retained rather than hidden.
- N1 remains the bottleneck (F1 0.445, precision ~0.31 vs recall
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
comparable to the primary benchmark above.

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
split (seed 42): supervised class-weighted cross-entropy,
20 epochs, batch 16, AdamW 3e-4, all-position
protocol (not comparable to the causal unique-epoch primary above).

```
Accuracy   = 90.57%   (74,860 test epochs, 15 held-out subjects)
Cohen's κ  = 0.8080
Macro F1   = 0.7490
Weighted F1 = 0.9115
Per-class F1 (W/N1/N2/N3/REM) = 0.978 / 0.477 / 0.823 / 0.705 / 0.762
Best validation κ = 0.8274 @ epoch 18/20
CPU latency = 6.2 ms/batch (measured, input [1,10,4,3000])
```

Evidence: `results/standalone_99k/`, checkpoint
`artifacts/standalone_99k/student_99477_best.pt`, dashboard row
`results/final/notebook_pipeline_result.csv`.

---

## Archived — 15-Record Development Benchmark (EXP-DEV-15SUBJ)

93.0% ± 1.0% accuracy, κ 0.861, macro-F1 0.794, 4-fold CV over 15
records. Superseded by the person-level benchmark. Full record:
`docs/archive/development_15_subject.md`.

---

## Model Architecture (context for all results)

99,477 parameters — multi-resolution stem (7,232), depthwise-separable
encoder (1,904), parametric Gabor filters incl. projection (160), 2-layer GRU hidden 64
(89,856 — 90.3% of the parameter budget), linear head (325).
Input `[B, 10, 4, 3000]` @ 100 Hz; output per-epoch 5-class logits over
a 300-second context.

---

## Reproduction

```bash
# Generate person-level folds (idempotent, seeded)
python scripts/build_person_groups.py
python scripts/generate_person_folds.py

# Primary benchmark (person-level; runner refuses leaky folds)
python scripts/run_100_subject_benchmark.py --seed 42 --device cuda \
    --folds-manifest data/manifests/person_folds_52subj.json \
    --output-dir results/benchmark_person_level

# Regenerate tables + CIs from fold evidence
python scripts/summarize_benchmark.py --results-dir results/benchmark_person_level

# Verify protocol integrity (person-level checks included)
python scripts/verify_protocol.py
```

To re-run the superseded record-level benchmark (research purposes
only), pass `--allow-record-level` explicitly; outputs are labeled
record-level.

---

*Numbers generated by `scripts/summarize_benchmark.py` from fold
summaries. Last regenerated: September 2026.*
