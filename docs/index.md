# Documentation Index — NeuroSleep

> The canonical project pipeline is the notebook series
> `notebooks/01` → `06`. These documents describe and evidence it.

## Documentation Structure

```
docs/
├── index.md             # This file
├── RESULTS.md           # Authoritative results (EXP-BENCH-PERSON + standalone 99k)
├── EXPERIMENTS.md       # Authoritative experiment registry
├── REPRODUCIBILITY.md   # Reproduction guide
├── methodology.md       # Experimental protocol & research approach
├── adaptation.md        # Three-regime adaptation protocol + contamination record
├── lora.md              # LoRA mathematics, targets, guarantees
├── architecture.md      # Architecture deep dive
├── dataset.md           # Dataset documentation
├── deployment.md        # Edge deployment guide
├── LIMITATIONS.md       # Per-class analysis & limitations
├── evaluation_protocol.md # Protocol definitions
├── exhibition.md        # Exhibition/demo guide
├── team.md              # Team roles
└── archive/
    └── development_15_subject.md  # 15-subject dev benchmark (historical)
```

## Evidence Hierarchy (applies to every document)

1. **Primary:** person-level from-scratch benchmark (EXP-BENCH-PERSON,
   52 persons, seeds 42/43/44) — `RESULTS.md`,
   `results/research/EXP-BENCH-PERSON/`
2. **Standalone notebook pipeline:** single 70/15/15 subject split,
   supervised CE (seed 42) — `results/standalone_99k/`
3. **Quarantined:** adaptation study — `adaptation.md`

## Documentation by Audience

### For Researchers

| Document | Description |
|----------|-------------|
| `RESULTS.md` | Authoritative metrics with 95% CIs and evidence status |
| `EXPERIMENTS.md` | Authoritative experiment registry |
| `methodology.md` | Experimental design, regimes, disjointness requirement |
| `adaptation.md` / `lora.md` | Adaptation protocol, LoRA math and ablations |
| `architecture.md` | Design decisions and parameter analysis |
| `dataset.md` | Cohort, preprocessing, data contract |

### For Engineers

| Document | Description |
|----------|-------------|
| `architecture.md` | I/O specs and components |
| `deployment.md` | Export, quantization, edge deployment |
| `dataset.md` | Data formats and cache structure |
| `methodology.md` | Training configuration and reproducibility |

### For Exhibition

| Document | Description |
|----------|-------------|
| `exhibition.md` | Demo script, poster layout, Q&A |
| `RESULTS.md` | Authoritative result block for display |
| `team.md` | Team roles |

## Quick Reference

### Evidence Status

> **P0:** SC4ss1/SC4ss2 = same person's two nights. The 92-record
> cohort is **52 persons**. Legacy record-level folds leaked at person
> level (10/10 folds); person-level folds are now primary.

| Tier | Experiment | Status |
|------|-----------|--------|
| Primary | EXP-BENCH-PERSON (person-level CV, 52 persons, seeds 42/43/44) | **Complete**: 87.30% ± 0.33% |
| Standalone | Notebooks 01→05 supervised run (seed 42) | Complete: 90.57% (κ 0.808) |
| Quarantined | EXP-ADAPT-* | Contaminated base + leaky folds |
| Archived | EXP-DEV-15SUBJ (93.0%) | Historical |

### Primary Result (Person-Level, 30 folds / 3 seeds)

```
Improved Student — from scratch, person-level folds (52 persons)
Accuracy   = 87.30% ± 0.33%  (95% CI [85.80, 88.79]%)
Kappa      = 0.738 ± 0.010   (95% CI [0.691, 0.785])
Macro F1   = 0.724 ± 0.005   (95% CI [0.693, 0.755])
Weighted F1 = 0.880 ± 0.003  (95% CI [0.860, 0.900])
Parameters = 99,477
```

> Do **not** cite the archived 15-record result (93.0%) or the
> quarantined adaptation numbers as final.

### Run Commands

```bash
# Verify protocol integrity (record + person level)
python scripts/verify_protocol.py

# Regenerate result tables + CIs from fold evidence
python scripts/summarize_person_benchmark.py --results-dir results/research/EXP-BENCH-PERSON

# Run the notebook pipeline (raw EDFs → metrics)
jupyter nbconvert --to notebook --execute notebooks/04_student_99k_complete_training.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/05_evaluation_and_benchmarking.ipynb --inplace

# Tests
python -m pytest tests/ -v

# Dashboard
streamlit run app/streamlit_app.py
```

## Team Contact

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

---

*Last updated: September 17, 2026*
*Project: NeuroSleep — VIT Bhopal University*
