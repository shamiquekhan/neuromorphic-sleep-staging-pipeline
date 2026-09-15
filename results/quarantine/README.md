# Quarantine Directory

This directory contains artifacts, results, and scripts that are **quarantined** and must not be used for primary results or reporting.

## Contents

### `legacy_benchmark_92subj/`
- **What:** Record-level 92-subject benchmark results (10-fold CV)
- **Why quarantined:** The legacy `canonical_subject_folds_92subj.json` treats SC4ss1/SC4ss2 as independent subjects. In 10/10 folds, a test record's same-person mate appears in the training set. This is **person-level leakage**.
- **Status:** All metrics from these folds are **record-level estimates**, not person-generalization estimates.
- **Reference:** See `docs/adaptation.md` and `scripts/verify_protocol.py` for details.

### `legacy_adaptation/`
- **What:** Frozen/LoRA/Full-FT adaptation study results
- **Why quarantined:** Base checkpoint (`student_full_finetuned.pt`) trained on person-level validation persons that appear as test records in the evaluation folds. Documented contamination.
- **Status:** Retained only for like-for-like internal comparison. Must not be reported as valid adaptation results.

### `student_full_finetuned_generic.pt`
- **What:** Generic checkpoint with ambiguous name and no provenance
- **Why quarantined:** Filename `student_full_finetuned.pt` does not encode experiment ID, seed, or fold. Superseded by `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/student_best.pt` with full provenance.

### `scripts/quarantine/run_100_subject_benchmark.py`
- **What:** Legacy benchmark script using contaminated folds
- **Why quarantined:** Uses record-level folds with person leakage. Superseded by person-level benchmark configuration.

---

## Active Artifacts (Use These Instead)

| Experiment | Artifacts | Results |
|------------|-----------|---------|
| **EXP-EXHIBITION-15SUBJ** (historical) | `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/` | `results/exhibition/EXP-EXHIBITION-15SUBJ/` |
| **EXP-BENCH-PERSON** (primary) | `artifacts/research/EXP-BENCH-PERSON/` | `results/research/EXP-BENCH-PERSON/` |

## Verification

Run protocol verification:
```bash
python scripts/verify_protocol.py
```

This will confirm which checkpoints/folds are clean vs contaminated.