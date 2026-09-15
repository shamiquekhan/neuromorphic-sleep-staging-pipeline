# Documentation Index — NeuroSleep

> The canonical project pipeline is the notebook series
> `notebooks/01` → `06`. These documents describe and evidence it.

## Documentation Structure

```
docs/
├── index.md             # This file
├── results.md           # SINGLE authoritative results document (numbers)
├── methodology.md       # Experimental protocol & research approach
├── adaptation.md        # Three-regime adaptation protocol + contamination record
├── lora.md              # LoRA mathematics, targets, guarantees
├── architecture.md      # Architecture deep dive
├── dataset.md           # Dataset documentation
├── deployment.md        # Edge deployment guide
├── LIMITATIONS.md       # Per-class analysis & limitations
├── exhibition.md        # Exhibition/demo guide
├── team.md              # Team roles
└── archive/
    └── development_15_subject.md  # 15-subject dev benchmark (historical)
```

## Evidence Hierarchy (applies to every document)

1. **Primary:** 92-subject from-scratch benchmark (EXP-BENCH-92SUBJ) —
   `results.md`
2. **Quarantined:** adaptation study (EXP-ADAPT-*) — `adaptation.md`
3. **Archived:** 15-subject development benchmark (EXP-DEV-15SUBJ) —
   `archive/development_15_subject.md`

## Documentation by Audience

### For Researchers

| Document | Description |
|----------|-------------|
| `results.md` | Authoritative metrics with 95% CIs and evidence status |
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
| `results.md` | Authoritative result block for display |
| `team.md` | Team roles |

## Quick Reference

### Evidence Status

> **P0:** SC4ss1/SC4ss2 = same person's two nights. The 92-record
> cohort is **52 persons**. Legacy record-level folds leaked at person
> level (10/10 folds); person-level folds are now primary.

| Tier | Experiment | Status |
|------|-----------|--------|
| Primary | EXP-BENCH-PERSON (person-level CV, 52 persons) | **Complete** (seed 42): 85.47% ± 3.99% |
| Superseded | EXP-BENCH-92SUBJ (record-level) — 87.66% ± 2.22% | Record-level estimate only |
| Quarantined | EXP-ADAPT-* | Contaminated base + leaky folds |
| Archived | EXP-DEV-15SUBJ (93.0%) | Historical |

### Primary Result (Person-Level, seed 42)

```
Improved Student — from scratch, person-level folds (52 persons)
Accuracy  = 85.47% ± 3.99%  (95% CI [82.62, 88.33]%)
Kappa     = 0.705 ± 0.128   (95% CI [0.613, 0.797])
Macro F1  = 0.697 ± 0.082   (95% CI [0.639, 0.756])
Parameters = 99,477
```

> Person-level numbers above are the honest primary (single seed; seeds
> 43/44 pending). Do **not** cite the archived 15-record result (93.0%)
> or the quarantined adaptation numbers as final.

### Run Commands

```bash
# Generate person-level folds (idempotent, seeded)
python scripts/build_person_groups.py
python scripts/generate_person_folds.py

# Primary benchmark (person-level; refuses leaky folds)
python scripts/run_100_subject_benchmark.py --seed 42 --device cuda \
    --folds-manifest data/manifests/person_folds_52subj.json \
    --output-dir results/benchmark_person_level

# Regenerate result tables + CIs from fold evidence
python scripts/summarize_benchmark.py --results-dir results/benchmark_person_level

# Verify protocol integrity (record + person level)
python scripts/verify_protocol.py

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

*Last updated: September 2026*
*Project: NeuroSleep — VIT Bhopal University*
