# NeuroSleep — Project Guide

## Overview

NeuroSleep is a compact edge-oriented sleep-stage scoring pipeline that
classifies 30-second polysomnography epochs into five AASM stages
(Wake, N1, N2, N3, REM) from a 300-second, 4-channel context. The
**canonical pipeline is the notebook series** (`notebooks/01` → `06`),
which takes the project from raw EDF recordings to final metrics with
per-epoch training logs.

## Evidence Hierarchy (read this before citing any number)

> **P0 note:** SC records are two nights per person — the 92-record
> cohort is **52 persons**. Legacy record-level folds leaked at person
> level (10/10 folds); numbers from them are record-level estimates.

1. **Primary:** person-level benchmark (EXP-BENCH-PERSON, 52-person
   10-fold CV, seeds 42/43/44) — **87.30% ± 0.33% accuracy,
   κ 0.738 ± 0.010, macro-F1 0.724 ± 0.005** →
   `docs/RESULTS.md`, `results/research/EXP-BENCH-PERSON/`
2. **Deployed standalone run:** notebooks 01→05, supervised CE,
   exhibition 70/15/15 split — **90.57% accuracy,
   κ 0.808, macro-F1 0.749** → `results/standalone_99k/`,
   `artifacts/standalone_99k/student_99477_best.pt`,
   `results/final/notebook_pipeline_result.csv`
4. **Quarantined:** adaptation study (Frozen / LoRA / Full-FT) —
   contaminated base checkpoint + record-level folds; internal
   comparison only → `docs/adaptation.md`

All benchmark numbers regenerate from raw fold evidence:
`python scripts/summarize_person_benchmark.py --results-dir results/research/EXP-BENCH-PERSON`.

## Primary Model

| Property | Value |
|----------|-------|
| Model | Improved Student (from scratch, supervised CE) |
| Parameters | 99,477 |
| Accuracy (person-level, 3 seeds × 10 folds) | 87.30% ± 0.33% |
| Cohen's κ | 0.738 ± 0.010 |
| Macro F1 | 0.724 ± 0.005 |
| Accuracy (standalone exhibition split, seed 42) | 90.57% (κ 0.808) |
| CPU latency | 6.2 ms/batch (measured) |
| Checkpoint | `artifacts/standalone_99k/student_99477_best.pt` |
| Dataset | Sleep-EDF Expanded — 92 records / 52 persons |
| Config | `configs/benchmark_person_level.yaml` |

## The Notebook Pipeline (canonical exhibition path)

| Notebook | Role | Key output |
|----------|------|-----------|
| 01 data import & dataset collection | manifest, pairing audit, subject split | `data/manifests/exhibition_15subj_v1.json` |
| 02 data preprocessing | filter → epoch → QC → normalize → cache | `data/cache/*.npz` + `cache_index.csv` |
| 03 exploratory data analysis | class balance, QC burden, spectra, transitions | diagnostics (in-notebook) |
| 04 student 99k complete training | supervised CE student (from scratch), per-epoch logs | `artifacts/standalone_99k/student_99477_best.pt` |
| 05 evaluation & benchmarking | held-out test metrics, confusion matrix, latency | `results/standalone_99k/`, `results/final/notebook_pipeline_result.csv` |
| 06 LoRA adaptation (extension) | adapter machinery demo | research extension only |

## Key Files

| File | Description |
|------|-------------|
| `notebooks/01–05_*.ipynb` | **The complete pipeline** (run in order) |
| `docs/RESULTS.md` | **Single authoritative results document** |
| `configs/experiments/person_level_cv.yaml` | Primary benchmark config (EXP-BENCH-PERSON) |
| `data/manifests/person_folds_52subj.json` | Person-level folds (52 persons, 10 folds) |
| `data/manifests/exhibition_15subj_v1.json` | Exhibition 70/15/15 split |
| `scripts/generate_person_folds.py` | Generates person-level folds |
| `scripts/summarize_person_benchmark.py` | Regenerates result tables + CIs from evidence |
| `scripts/verify_protocol.py` | Leakage (record + person level) + config-consistency gate |
| `scripts/protocol_fingerprint.py` | Pre-run consistency verification |
| `docs/lora.md` | LoRA mathematics, targets, verification guarantees |
| `docs/adaptation.md` | Three-regime protocol + contamination record (quarantined) |
| `app/streamlit_app.py` | Dashboard |
| `ROADMAP.md` | Experiment status + remaining work |

## Quick Start

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run the pipeline (raw EDFs in data/raw/sleep_edf/)
jupyter nbconvert --to notebook --execute notebooks/01_data_import_and_dataset_collection.ipynb --inplace
# ...continue through 05

# Dashboard
streamlit run app/streamlit_app.py

# Regenerate primary benchmark result tables from fold evidence
python scripts/summarize_person_benchmark.py --results-dir results/research/EXP-BENCH-PERSON

# Verify protocol integrity
python scripts/verify_protocol.py

# Tests
python -m pytest tests/ -v
```

## Repository Structure

```
notebooks/              The pipeline (01→06)
src/sleep_staging/      Core package (models, adaptation, data, training, evaluation)
app/                    Streamlit dashboard
scripts/                CLI tools
tests/                  Test suite (92 tests)
artifacts/              Deployed checkpoint (standalone_99k)
results/                Evaluation evidence (see evidence hierarchy)
configs/                Canonical experiment definitions
data/                   Manifests (tracked) + cache/raw (local only)
docs/                   Documentation (RESULTS.md is the numbers source of truth)
huggingface/            Hub model card + demo space assets
deployment/             Docker deployment
```

## Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

## License

CC-BY-4.0 (see `LICENSE`). Dataset terms are governed by PhysioNet/
Sleep-EDF upstream licenses and citation requirements (see
`docs/DATA_SOURCES_AND_LICENSES.md`); the dataset itself is not
redistributed under this repository's license.
