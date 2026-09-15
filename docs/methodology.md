# Methodology — NeuroSleep

## Research Approach

NeuroSleep investigates compact multi-resolution convolutional-recurrent
sleep staging under a strict parameter budget, and evaluates whether
parameter-efficient low-rank adaptation can approach full fine-tuning
performance while substantially reducing the number of trainable
parameters.

The methodology centers on three commitments:

1. **Subject-level separation** — every reported metric comes from
   subject-level cross-validation folds; no subject's data is split
   across train/validation/test.
2. **Evidence regeneration, not hand-editing** — all numbers in the
   documentation regenerate from raw fold evidence via
   `scripts/summarize_benchmark.py`; the regeneration order is
   CODE → CONFIG → EXPERIMENT → RAW RESULTS → AGGREGATION → RESULT
   TABLES → MODEL_REPORT → README → HF MODEL CARD.
3. **Machine-checkable protocol integrity** —
   `scripts/verify_protocol.py` fails the run if folds, configs, or
   base-checkpoint subject disjointness are violated.

---

## Experimental Matrix

| ID | Experiment | Status |
|----|-----------|--------|
| EXP-DEV-15SUBJ | 15-record development benchmark, 4-fold CV | Archived (historical) |
| EXP-BENCH-92SUBJ | 92-record from-scratch benchmark, record-level folds | Complete (seed 42) — **superseded** (person leakage) |
| **EXP-BENCH-PERSON** | Person-level from-scratch benchmark, 10-fold CV over 52 persons | **Complete** (seed 42): 85.47% ± 3.99% — **primary** |
| EXP-ADAPT-FROZEN / -LORA-R8-CNNHEAD / -FULLFT | Three adaptation regimes from one base checkpoint | Quarantined (contaminated base + record-level folds) — re-run pending |
| EXP-LORA-R{2,4,16}-CNNHEAD | LoRA rank ablation | Pending leak-free base + person-level folds |
| EXP-LORA-TARGET-* | LoRA target-module matrix (head / CNN / Gabor / GRU) | Pending LoRA-GRU support |
| EXP-CROSS-DATASET | Cross-dataset validation (e.g., SHHS) | Planned |

**Records ≠ persons (P0).** SC4ss1/SC4ss2 are two nights of the same
person (PhysioNet sleep-edfx README; SC-subjects.xls; EDF headers). The
92-record cohort is 52 persons. All splits — and the adaptation base
checkpoint's training set — must be disjoint at the *person* level.
Machine-enforced by `scripts/verify_protocol.py` and the benchmark
runner's refusal guard.

Definitions the matrix depends on:

- **From-scratch training:** random initialization → all parameters
  trained (the primary benchmark).
- **Full fine-tuning:** pretrained base checkpoint → all 99,477
  parameters unfrozen → task adaptation.
- **LoRA:** base weights frozen; low-rank A/B adapters trainable in
  selected projection layers.
- **Frozen:** base weights frozen; no training; evaluation only.

---

## Phase 1 — Data Acquisition & Governance

**Source:** PhysioNet Sleep-EDF Expanded (SC cassette study).

- 100 records downloaded → 8 wake-only records excluded →
  **92-record eligible cohort — 52 persons** (SC4ss1/SC4ss2 = same
  person's two nights; 40 two-night, 12 one-night persons).
- Channels: EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG @ 100 Hz.
- PSG–hypnogram pairing validation, duplicate record detection,
  channel availability checks.
- Person-level fold assignment:
  `data/manifests/person_folds_52subj.json` (10 folds over persons,
  5 fixed validation persons, stratified by age decade).
- Person groups: `data/manifests/person_groups.json`
  (`scripts/build_person_groups.py`).
- QC flags are computed and cached per epoch by Notebook 02 (see `data/cache/cache_index.csv`).

## Phase 2 — Signal Preprocessing

| Step | Setting | Rationale |
|------|---------|-----------|
| Bandpass | 4th-order Butterworth, 0.5–35 Hz | Preserve sleep-relevant frequencies, remove drift |
| Notch | 50 Hz, Q=30 | Power-line interference |
| Normalization | z-score per channel | Amplitude invariance |
| QC flags | clipping / flatline / NaN | Stored; not used to filter training |

Deliverable: cached per-subject `.npz` epochs (`data/cache/sleep_edf/`).

## Phase 3 — Model Development

**Improved Student** (99,477 parameters):

| Module | Parameters | Share |
|--------|-----------:|------:|
| Multi-Resolution Stem (parallel Conv1d, kernels 25/200) | 7,232 | 7.27% |
| Depthwise-Separable Encoder (2 blocks) | 2,304 | 2.32% |
| Parametric Gabor Filters (8 learnable, 0.5–30 Hz) | 144 | 0.14% |
| 2-Layer GRU (hidden 64, 300-s context) | 89,856 | 90.33% |
| Linear Head | 325 | 0.33% |

Design principles: multi-scale temporal receptive fields; depthwise-
separable compute for edge deployment; learnable spectral features;
recurrent temporal modeling. The architecture is a conventional
differentiable CNN–GRU network designed for edge constraints — see the
positioning note in [`MODEL_REPORT.md`](../MODEL_REPORT.md) §1.

**Training objective:** class-weighted cross-entropy (N1 2×, REM 2×)
over all 10 positions of each subject-safe 10-epoch window. All-position
supervision is a *training signal only* — all reported metrics come from
the causal one-prediction-per-epoch protocol (see
[`evaluation_protocol.md`](evaluation_protocol.md)).

> Historical note: an early teacher-distillation phase existed during
> development. The final architecture and all reported results use the
> student model trained directly; the teacher is not part of the
> reported pipeline.

## Phase 4 — Benchmarks (record-level superseded; person-level primary)

- **EXP-BENCH-PERSON (primary, in progress):** 10-fold **person-level**
  CV over 52 persons — whole persons (both nights) per fold role, 5
  fixed validation persons, stratified by age decade
  (`configs/benchmark_person_level.yaml`)
- **EXP-BENCH-92SUBJ (superseded):** identical training on record-level
  folds (`configs/benchmark_person_level.yaml`); seeds 42/43/44 numbers are
  record-level estimates
- From-scratch initialization per fold
- AdamW, lr 3e-4, weight decay 1e-4, ≤20 epochs, early stopping
  patience 5 (checkpoint selection on validation macro-F1), batch 32,
  cosine schedule, grad clip 1.0, CUDA AMP
- Metrics: accuracy, Cohen's κ, macro/weighted F1, MGm, per-class
  precision/recall/F1, confusion matrices — computed per fold,
  aggregated over folds (the fold is the unit of analysis), with 95% CIs
- The runner refuses person-leaky folds unless `--allow-record-level` is
  passed explicitly

Current evidence: person-level seed 42 complete — **85.47% ± 3.99%,
κ 0.705, macro-F1 0.697** (primary); record-level seed 42 complete but
superseded — see [`results.md`](results.md) for numbers, evidence
tiers, and the single-seed caveat.

## Phase 5 — Parameter-Efficient Adaptation (EXP-ADAPT-*)

Three regimes, all from the **same base checkpoint**, on the same folds:

| Regime | Base weights | Trainable | Optimized |
|--------|--------------|-----------|-----------|
| 2A Frozen | frozen | 0 | — (eval only) |
| 2B LoRA | frozen | 1,448 (r=8, CNN+Head) | low-rank A/B factors |
| 2C Full FT | unfrozen | 99,477 | all parameters |

**Person-level requirement (critical):**

```
base_checkpoint_training_RECORDS ∩ evaluation_records        = ∅
base_checkpoint_training_PERSONS ∩ evaluation_PERSONS          = ∅
```

(SC4ss1/SC4ss2 = same person, two nights.) The legacy runs violated
record-disjointness (12 test / 3 validation overlaps) *and* used
person-leaky folds; they are quarantined. The protocol, contamination
record, and re-run procedure are documented in
[`adaptation.md`](adaptation.md). Planned follow-ups:
rank ablation (r = 2/4/8/16), target-module matrix (head / CNN /
Gabor / **GRU** — the key open question, since the GRU holds 90.3% of
parameters), alpha and dropout ablations, paired fold-level statistics
(Wilcoxon signed-rank, effect sizes, bootstrap CIs).

## Phase 6 — Evaluation & Deployment

- Per-fold evaluation artifacts (`metrics.json`,
  `training_history.csv`, `predictions.csv`, `confusion_matrix.csv`)
  make every result independently auditable.
- Protocol fingerprints (dataset manifest hash, checkpoint hash, config
  hash, git commit, PyTorch/CUDA, GPU, hyperparameters) recorded per
  run; `scripts/verify_protocol.py` enforces them.
- Streamlit dashboard for demonstration; ONNX/INT8 edge deployment is
  future work (see `ROADMAP.md`).

---

## Reproducibility

| Aspect | Implementation |
|--------|---------------|
| Seeds | Seeded Python/NumPy/Torch; primary benchmark protocol uses seeds 42/43/44 (43/44 pending) |
| Folds | Canonical, manifest-pinned subject-level folds |
| Configs | One YAML per experiment ID under `configs/` |
| Integrity | `scripts/verify_protocol.py` gate |
| Aggregation | `scripts/summarize_benchmark.py` (never hand-edit numbers) |

## Ethical Considerations

1. **Not a medical device** — research prototype, no clinical validation.
2. Validation limited to healthy-subject Sleep-EDF data.
3. Should supplement, not replace, expert scoring.
4. Uses publicly available, de-identified data.
5. Claims in documentation are tied to the evidence hierarchy, not
   inflated by superseded results.

---

*Last updated: September 2026 — regenerated as part of the
documentation-consistency repair; see `ROADMAP.md` for experiment
status.*
