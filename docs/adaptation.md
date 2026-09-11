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
base_checkpoint_training_subjects ∩ evaluation_test_subjects      = ∅
base_checkpoint_training_subjects ∩ evaluation_validation_subjects = ∅
```

If this does not hold, the frozen baseline and every adaptation number are
inflated, because the "pretrained" model has already seen the evaluation
subjects during its original training.

### Why this matters concretely

The per-fold frozen results make the inflation measurable. Folds whose test
sets contained subjects the base checkpoint was trained on scored
**87.6%** on average, while the two folds with zero overlap scored **85.1%**
— a **+2.5pp** contamination inflation on the frozen baseline, propagating
into all three regimes.

---

## 3. Contamination Finding (Quarantined)

The legacy adaptation runs (`results/100_subject_adaptation/`, August 2026)
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
- are preserved only in `results/100_subject_adaptation/` (renamed
  documentation header) and in the archive notes,
- can still be used *internally* as a like-for-like relative comparison
  between the three regimes (all equally inflated by the same base), with
  explicit caveats.

The README previously quoted the contaminated Full-FT number
(87.7% ± 2.7%) as the primary result. This is corrected: the primary
result is the **from-scratch** 92-subject benchmark (seed 42 complete;
seeds 43/44 pending — see [`docs/results.md`](results.md)).

---

## 4. Correct Protocol

### Step 1 — Produce a leak-free base checkpoint

Train a base checkpoint on the evaluation **validation subjects only**
(the 9 fixed validation subjects of `canonical_subject_folds_92subj.json`),
or on any subject set disjoint from every test fold. Record the training
subject list alongside the checkpoint.

### Step 2 — Verify disjointness

```bash
python scripts/verify_protocol.py
```

This checks the protocol fingerprint, config consistency, and — critically —
that the base checkpoint's training subjects are disjoint from all
evaluation test and validation subjects. It fails the run otherwise.

### Step 3 — Run the three regimes

```bash
python scripts/train_adaptation.py --mode frozen      --seed 42 --device cuda
python scripts/train_adaptation.py --mode lora --targets enc.0.pw,enc.1.pw,head \
                                   --rank 8 --alpha 16 --seed 42 --device cuda
python scripts/train_adaptation.py --mode full_finetune --seed 42 --device cuda
```

Repeat for seeds 43, 44. (Note: frozen evaluation is deterministic — its
per-seed results are identical up to wall-clock; report it as 10 unique
fold evaluations, not 30.)

### Step 4 — Aggregate + statistics

```bash
python scripts/aggregate_adaptation_results.py
python scripts/statistical_analysis.py
```

Paired comparisons (same folds across regimes) use the fold/subject as the
unit of analysis with Wilcoxon signed-rank tests, effect sizes, and 95% CIs.

---

## 5. Legacy Evidence Map

| Directory | Content | Status |
|-----------|---------|--------|
| `results/100_subject_adaptation/` | Frozen / LoRA / Full-FT on 92-subject folds from contaminated base | **Quarantined** (contaminated base) |
| `results/LORA_RESULTS.md` + `results/lora_*.json` | 15-subject-era LoRA study (4-fold, head-only) | Archived (small cohort) |
| `results/benchmark_92_subject/` | From-scratch benchmark, seed 42 | **Clean, primary** |
| `results/final/` | 15-subject development benchmark raw folds | Archived (see `docs/archive/development_15_subject.md`) |

---

## 6. Regeneration Checklist

- [ ] Train leak-free base checkpoint (validation subjects only)
- [ ] `scripts/verify_protocol.py` passes
- [ ] Frozen × 10 folds (one seed suffices — deterministic)
- [ ] LoRA primary config × 3 seeds
- [ ] Full FT × 3 seeds
- [ ] `aggregate_adaptation_results.py`
- [ ] `statistical_analysis.py` (paired tests on fold level)
- [ ] Update `docs/results.md` and README from the regenerated aggregates
