# Evaluation Protocol

This document is the single source of truth for how NeuroSleep numbers
are produced. It reflects the post-audit protocol (Sept 2026), which
supersedes the stride-5 / all-position protocol used by all results
predating the boundary/protocol fixes.

## Sequence construction (P0.1 — subject-safe)

**Rule: a training or evaluation window never spans two subjects, and
never spans an unlabeled-epoch gap.**

The legacy pipeline concatenated all subjects
(``load_subjects`` → ``np.concatenate``) and then windowed over the
concatenation. The final epochs of one subject could therefore share a
window with the first epochs of the next — a synthetic "context" like
`patient A REM → patient B N2` that no GRU should ever see. On a real
two-subject pair (SC4001 + SC4002) the legacy construction produced
1,094 windows of which 1 crossed the subject seam; the subject-safe
dataset produces 1,093.

Additionally, preprocessing drops epochs annotated `?` (unknown) or
movement time *before* caching. Consecutive cached rows can therefore
be temporally discontiguous even within one subject. Caches produced by
``scripts/preprocess_sleep_edf_expanded.py`` now store
``orig_epoch_idx`` (the epoch's position on the raw 30 s annotation
grid), and the datasets refuse to build windows across such gaps.
Legacy caches without this key still run, but the benchmark prints an
explicit warning that gap protection is unavailable.

Implementation: ``src/sleep_staging/data/sequence_dataset.py``
(``SubjectSequenceDataset``). Machine-checked by
``tests/test_sequence_dataset.py`` (boundary, gap, and seam tests) —
these tests are mandatory before any benchmark run.

## Evaluation protocol (P0.6 — causal, one prediction per epoch)

**Rule: every scored epoch receives exactly one prediction, produced
from the 5 minutes of context immediately preceding it.**

| Aspect            | Legacy (pre-fix)                  | Current (canonical)              |
| ----------------- | --------------------------------- | -------------------------------- |
| Window stride     | 5 epochs                         | 1 epoch                         |
| Supervision       | all 10 positions                 | last epoch only                  |
| Epochs scored     | most epochs scored 2×             | every epoch exactly once         |
| Metric unit       | window-position pairs            | unique scored epochs             |
| Provenance        | none                             | (subject_id, epoch_index)        |
| First scoreable epoch | epoch 0 (with padded context) | epoch 9 (first with full context) |

The first `seq_len - 1` epochs of each subject cannot be scored under a
causal protocol (they lack the full 5-minute context) and are excluded
from evaluation. This matches the streaming/edge deployment scenario:
classify the current epoch from the context that ends at it.

Implementation: ``src/sleep_staging/evaluation/protocol.py``
(``evaluate_causal``). The per-epoch uniqueness guarantee
(``_assert_unique_epochs``) raises at runtime if it is ever violated,
and result artifacts carry full ``(subject_id, epoch_index, true,
pred, per-class probabilities)`` rows for every scored epoch.

### Why training still uses stride-5 all-position windows

Training windows are subject-safe but use stride 5 with all-position
supervision: this is a training *signal* choice (more supervision per
window, 2× fewer windows to traverse), not a reporting protocol. No
metric is ever computed on overlapping windows. Training and reporting
protocols are now separate by construction: train windows come from
``SubjectSequenceDataset``, reported predictions from
``CausalEvalDataset``.

## Splits

All primary numbers come from 10-fold **person-level** cross
validation over the 92-record / 52-person eligible cohort
(``data/manifests/person_folds_52subj.json``): whole persons (both
nights) are assigned to train/test, with 5 fixed validation persons per
fold. The legacy record-level folds leaked at person level —
``SC4ss1``/``SC4ss2`` are two nights of the same person — and are kept
only for documented, labeled comparison. See
``docs/RESULTS.md`` for the split history and
``scripts/verify_protocol.py`` for the machine checks.

## Run logs

Every benchmark/adaptation/CV run tees its full console output into
``<results_dir>/run_logs/<name>.log`` (handled by
``sleep_staging.utils.runlog.tee_run_log`` — wired into
``train_adaptation.py`` and the development-era CV runner). These logs are committed with the results
they produced: they contain the per-epoch training curves, the
gap-exclusion notes, the person-disjointness guard output, and the
per-fold test summaries — the human-readable counterpart of
``metrics.json``. Transient ``logs/`` at the repo root is git-ignored;
only ``results/**/run_logs/`` is tracked.

## Checklist before citing any number

1. ``python -m pytest tests/ -q`` — all green (includes boundary/gap
   uniqueness tests).
2. ``python scripts/verify_protocol.py`` — folds and checkpoint
   provenance pass.
3. Result artifact contains ``predictions.csv`` with one row per
   ``(subject_id, epoch_index)`` and no duplicates.
4. The result is labeled with protocol version
   (``causal_unique_epoch``) and split level (``person``).
5. ``run_logs/`` beside the artifact contains the run's console log
   (per-epoch curves, guard output) — evidence of how the numbers were
   produced.
