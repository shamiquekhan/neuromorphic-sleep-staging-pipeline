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

1. **Final submission run (EXP-FULL-AUG30, project freeze Sept 2026):**
   notebooks 01→05 on the complete corpus, `FINAL PROTOCOL AUDIT PASSED`
   — **90.48% accuracy, κ 0.8283, macro-F1 0.7899** on 16 held-out test
   subjects (197 recordings / 100 subjects, person-level 70/15/15,
   seed 42) → `results/final/final_metrics.json`,
   `artifacts/final/EXP-FULL-AUG30_seed42.pt`; fit diagnosis (train/
   val/test gap +5.12 pp) → `results/final/fit_diagnosis.json`
2. **Historical research benchmark:** person-level benchmark
   (EXP-BENCH-PERSON, 52-person 10-fold CV, seeds 42/43/44) —
   **87.30% ± 0.33% accuracy, κ 0.738 ± 0.010, macro-F1 0.724 ± 0.005**
   → `docs/RESULTS.md`, `results/research/EXP-BENCH-PERSON/`
3. **Deployed standalone run:** notebooks 01→05, supervised CE,
   exhibition 70/15/15 split — **90.57% accuracy,
   κ 0.808, macro-F1 0.749** → `results/standalone_99k/`,
   `artifacts/standalone_99k/student_99477_best.pt`
4. **Quarantined:** adaptation study (Frozen / LoRA / Full-FT) —
   contaminated base checkpoint + record-level folds; internal
   comparison only → `docs/adaptation.md`

All benchmark numbers regenerate from raw fold evidence:
`python scripts/summarize_person_benchmark.py --results-dir results/research/EXP-BENCH-PERSON`.

## Final Submission Result — EXP-FULL-AUG30

| Metric | Value |
|--------|-------|
| Accuracy | 90.48% |
| Cohen's κ | 0.8283 |
| Macro F1 | 0.7899 |
| Weighted F1 | 0.9089 |
| Best validation | Macro F1 0.7645 @ epoch 12 (early-stopped 17/30) |
| CPU latency | 8.94 ms per 5-minute window (measured) |
| Checkpoint | `artifacts/final/EXP-FULL-AUG30_seed42.pt` → promoted to `artifacts/student_improved_best.pt` |

**Fit diagnosis (train/val/test gap):** train 92.50% · val 87.37% ·
test 90.48% (identical stride-10 protocol) — train−val **+5.12 pp**,
train−test **+2.01 pp**; weakest class N1 is weak on train too
(F1 0.627) → mild, controlled generalization gap (label ambiguity, not
memorization) → `results/final/fit_diagnosis.json` (Notebook 05 §14).

## Primary Model

| Property | Value |
|----------|-------|
| Model | Improved Student (from scratch, supervised CE) |
| Parameters | 99,477 |
| Accuracy (person-level, 3 seeds × 10 folds) | 87.30% ± 0.33% |
| Cohen's κ | 0.738 ± 0.010 |
| Macro F1 | 0.724 ± 0.005 |
| Accuracy (standalone exhibition split, seed 42) | 90.57% (κ 0.808) |
| Accuracy (final submission, EXP-FULL-AUG30, 16 held-out subjects) | 90.48% (κ 0.8283, macro-F1 0.7899) |
| CPU latency | 6.2 ms/batch (measured, standalone); 8.94 ms per 5-min window (EXP-FULL-AUG30) |
| Checkpoint | `artifacts/standalone_99k/student_99477_best.pt`; submission: `artifacts/final/EXP-FULL-AUG30_seed42.pt` (promoted: `artifacts/student_improved_best.pt`) |
| Dataset | Sleep-EDF Expanded — 92 records / 52 persons (research tier); complete corpus 197 records / 100 subjects (final freeze) |
| Config | `configs/benchmark_person_level.yaml` |

## The Notebook Pipeline (canonical exhibition path)

| Notebook | Role | Key output |
|----------|------|-----------|
| 01 data import & dataset collection | full-corpus manifest, SHA-1 verification, subject split | `data/manifests/sleep_edf_full.csv` + `dataset_audit.json` |
| 02 data preprocessing | filter → epoch → QC → normalize → cache (mmap layout) | `data/cache_full/*_epochs.npy` + `cache_index.csv` |
| 03 exploratory data analysis | class balance, QC burden, spectra, transitions | diagnostics (in-notebook) |
| 04 student 99k complete training | supervised CE student (from scratch), augmentation, early stopping | `artifacts/final/EXP-FULL-AUG30_seed42.pt` |
| 05 evaluation & benchmarking | held-out test metrics, fit diagnosis, audit gates, checkpoint promotion | `results/final/final_metrics.json`, `results/final/fit_diagnosis.json`, `artifacts/student_improved_best.pt` |
| 06 LoRA adaptation (extension) | adapter machinery demo | research extension only |

## Key Files

| File | Description |
|------|-------------|
| `notebooks/01–05_*.ipynb` | **The complete pipeline** (run in order) |
| `docs/RESULTS.md` | **Single authoritative results document** |
| `results/final/fit_diagnosis.json` | Train/val/test gap diagnosis (NB05 §14) |
| `configs/experiments/person_level_cv.yaml` | Historical benchmark config (EXP-BENCH-PERSON) |
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

# Regenerate historical benchmark result tables from fold evidence
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
artifacts/              Checkpoints (submission EXP-FULL-AUG30 + standalone_99k)
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
