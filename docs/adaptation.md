# Parameter-Efficient Adaptation — Protocol

This document defines the adaptation protocol, its subject-disjointness
requirement, the contamination found in the legacy runs, and the correct
way to re-run it.

---

## 1. The Three Regimes

All three regimes start from the **same base checkpoint** and are evaluated
on the same 10 subject-level folds (92-subject eligible cohort):

| Regime | Base weights | Trainable params | Optimized |
|--------|--------------|-----------------|-----------|
| **2A Frozen** | frozen | 0 | nothing (evaluation only) |
| **2B LoRA** | frozen | 1,448 (primary: r=8, CNN+Head) | low-rank A/B factors |
| **2C Full fine-tuning** | unfrozen | 99,477 | all parameters |

"Full fine-tuning" = pretrained base checkpoint + all parameters unfrozen.
"From-scratch training" = random initialization (the primary benchmark).
The initialization point is part of the experiment definition and must be
reported with every number.

The LoRA method itself is specified in [`docs/lora.md`](lora.md).

---

## 2. Subject-Disjointness Requirement (Critical)

The adaptation benchmark is only valid if:

```
base_checkpoint_training_records ∩ evaluation_test_records      = ∅
base_checkpoint_training_records ∩ evaluation_validation_records = ∅
```

If this does not hold, the frozen baseline and every adaptation number are
inflated, because the "pretrained" model has already seen the evaluation
records during its original training.

### 2.1 Person-level requirement (P0 addendum, Sept 2026)

**Record-disjointness is not sufficient.** `SC4ss1`/`SC4ss2` are two
nights of the **same person** (PhysioNet sleep-edfx README: "ss is the
subject number, and N is the night"; verified against SC-subjects.xls
and the EDF headers of every downloaded record). Therefore:

```
base_checkpoint_training_PERSONS ∩ evaluation_PERSONS = ∅
```

must also hold, where person = `SC4ss`. A base checkpoint trained on
SC4002 (validation) while SC4001 (test) is evaluated is
person-contaminated even though no record overlaps.

The same requirement applies to the evaluation folds themselves: the
the legacy record-level folds (removed) put a
test record's same-person mate in the train pool in 10/10 folds. All
adaptation runs must use `person_folds_52subj.json`
(`scripts/generate_person_folds.py`). `scripts/verify_protocol.py`
enforces both record- and person-level disjointness.

### Why this matters concretely

The per-fold frozen results make the inflation measurable. Folds whose test
sets contained records the base checkpoint was trained on scored
**87.6%** on average, while the two folds with zero overlap scored **85.1%**
— a **+2.5pp** contamination inflation on the frozen baseline, propagating
into all three regimes. This is on top of the person-level fold leakage
(§2.1), which inflates further.

---

## 3. Contamination Finding (Quarantined)

The legacy adaptation runs (August 2026; evidence removed in the Sept 2026 cleanup)
used the base checkpoint produced by the 15-subject development benchmark
(`artifacts/final/student_full_finetuned.pt`).

That checkpoint was trained on 15 subjects:

```
SC4001 SC4002 SC4011 SC4012 SC4022 SC4031 SC4032 SC4041 SC4042 SC4051
SC4052 SC4061 SC4062 SC4071 SC4072
```

Overlap with the 92-subject evaluation folds:

| Overlap type | Count | Subjects |
|--------------|-------|----------|
| Eval **test** subjects | **12** | SC4001, SC4011, SC4012, SC4022, SC4031, SC4032, SC4041, SC4051, SC4052, SC4061, SC4062, SC4071 |
| Eval **validation** subjects | **3** | SC4002, SC4042, SC4072 |

**Consequence:** the legacy adaptation numbers (Frozen 87.1%, LoRA 83.6%,
Full-FT 87.7%) are quarantined as contaminated evidence. They:

- must **not** be cited as generalization results,
- were preserved in `results/100_subject_adaptation/` (removed in the Sept 2026 cleanup; renamed
  documentation header) and in the archive notes,
- can still be used *internally* as a like-for-like relative comparison
  between the three regimes (all equally inflated by the same base), with
  explicit caveats.

The README previously quoted the contaminated Full-FT number
(87.7% ± 2.7%) as the primary result. This is corrected: the primary
result is the **person-level from-scratch benchmark**
(EXP-BENCH-PERSON, seed 42 complete: 85.47% ± 3.99%, κ 0.705,
macro-F1 0.697 — see [`docs/results.md`](results.md)).
The record-level 92-record benchmark (87.66% ± 2.22%) is itself only a
record-level estimate — its folds leak at person level.

---

## 4. Correct Protocol

### Step 1 — Produce a leak-free base checkpoint

Train a base checkpoint on evaluation **validation records only**, or on
any record set whose **persons** (SC4ss prefix) are disjoint from every
test and validation fold's persons. Record the training record list
alongside the checkpoint. Note: the 5 validation *persons* of
`person_folds_52subj.json` include both nights where available — use
whole persons, never single nights.

### Step 2 — Verify disjointness

```bash
python scripts/verify_protocol.py
```

This checks the protocol fingerprint, config consistency, and — critically —
that the base checkpoint's training records **and persons** are disjoint
from all evaluation test and validation records/persons. It fails the run
otherwise.

### Step 3 — Run the three regimes

All adaptation runs use the **person-level folds**
(`person_folds_52subj.json`), not the legacy record-level manifest:

```bash
python scripts/train_adaptation.py --mode frozen      --seed 42 --device cuda --folds-manifest data/manifests/person_folds_52subj.json
python scripts/train_adaptation.py --mode lora --targets enc.0.pw,enc.1.pw,head \
                                    --rank 8 --alpha 16 --seed 42 --device cuda --folds-manifest data/manifests/person_folds_52subj.json
python scripts/train_adaptation.py --mode full_finetune --seed 42 --device cuda --folds-manifest data/manifests/person_folds_52subj.json
```

Repeat for seeds 43, 44. (Note: frozen evaluation is deterministic — its
per-seed results are identical up to wall-clock; report it as 10 unique
fold evaluations, not 30.)

### Step 4 — Aggregate + statistics

```bash
# (aggregate_adaptation_results.py removed — quarantined evidence no longer stored)
python scripts/statistical_analysis.py
```

Paired comparisons (same folds across regimes) use the fold/person as the
unit of analysis with Wilcoxon signed-rank tests, effect sizes, and 95% CIs.

---

## 5. Legacy Evidence Map

| Directory | Content | Status |
|-----------|---------|--------|
| Historical adaptation runs | Frozen / LoRA / Full-FT from contaminated base | **Quarantined** (evidence removed in cleanup) |
| `results/LORA_RESULTS.md` + `results/lora_*.json` | 15-subject-era LoRA study (4-fold, head-only) | Archived (small cohort) |
| `results/benchmark_person_level/` | From-scratch person-level benchmark, seeds 42/43/44 | **Clean, primary** |
| `results/final/` | 15-subject development benchmark raw folds | Archived (see `docs/archive/development_15_subject.md`) |

---

## 6. Regeneration Checklist

- [ ] Train leak-free base checkpoint (validation subjects only)
- [ ] `scripts/verify_protocol.py` passes
- [ ] Frozen × 10 folds (one seed suffices — deterministic)
- [ ] LoRA primary config × 3 seeds
- [ ] Full FT × 3 seeds
- [x] quarantined evidence documented (files removed in cleanup)
- [ ] `statistical_analysis.py` (paired tests on fold level)
- [ ] Update `docs/results.md` and README from the regenerated aggregates
