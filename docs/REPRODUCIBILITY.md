# NeuroSleep — Reproducibility Guide
## Exact Commands, Environment, and Artifacts for Full Reproduction

---

## Two Documented Reproduction Paths

| Path | Scope | Entry point |
|------|-------|-------------|
| **Final protocol (project freeze, Sept 2026)** | Complete Sleep-EDF Expanded corpus — 197 recordings / 100 subjects, notebooks 01→05 | This file, next section |
| **Person-level research benchmark (EXP-BENCH-PERSON)** | 92-record / 52-person cohort, 10-fold CV × 3 seeds | [Person-Level Benchmark Reproduction](#person-level-benchmark-reproduction-exp-bench-person) |

The final protocol is the authoritative freeze result: `results/final/final_metrics.json`
(κ 0.8283 / accuracy 90.48% / macro-F1 0.7899 on 16 held-out test subjects).

---

## Environment Specification

### Dependencies
```bash
pip install -r requirements-lock.txt
```
> Note: there is no `environment.yml` in the repository; `requirements-lock.txt` is the
> pinned dependency list.

### Key Versions (executed environment)
| Package | Version |
|---------|---------|
| Python | 3.13.9 (Anaconda) |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| NumPy | 1.26.4 |
| Pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| MNE | 1.12.1 |
| nbconvert / nbclient | 7.16.6 / 0.10.2 |

> NumPy 1.x note: the notebooks use `np.trapz` (not `np.trapezoid`), matching
> NumPy 1.26. NumPy ≥ 2.0 would require renaming it back.

### Hardware (used for the freeze run)
- GPU: NVIDIA GTX 1650 (4 GB VRAM) — training runs on CUDA; NB05 latency model runs on CPU
- RAM: 16 GB — **sufficient only with the memory-mapped cache layout** (see Troubleshooting)
- Disk: ~30 GB free (8.2 GB raw EDFs + 20 GB processed cache + artifacts)

---

## Deterministic Execution

### Seed Control
The final protocol uses seed 42 throughout. Notebook 04 sets:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

Augmentation is additionally reproducible per window via a dedicated generator:
`np.random.default_rng([SEED, window_index])` — independent of DataLoader shuffling.

> **Honest caveat:** the notebooks do **not** set
> `torch.backends.cudnn.deterministic = True` / `benchmark = False`. Re-running on the
> same GPU + PyTorch build reproduces the protocol and metric values as reported, but
> bit-exact weight reproduction is not guaranteed across GPU models or CUDA versions.

---

## Full-Corpus Protocol Reproduction (Final Freeze — Notebooks 01→05)

### Prerequisites
- Internet access to PhysioNet (NB01 falls back to the AWS Open Data mirror
  `physionet-open.s3.amazonaws.com` automatically; every file is SHA-1-verified against the
  official MNE record tables either way)
- ~30 GB free disk
- Run notebooks in order, fresh kernel each

### Notebook 01: Full-Corpus Data Import
```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_data_import_and_dataset_collection.ipynb
```
**Runtime:** ~1 h download (network-dependent; ~8.2 GB across 394 EDF files)  
**Outputs:**
- `data/raw/sleep_edf_full/sleep-cassette/` (306 EDFs) + `sleep-telemetry/` (88 EDFs), all SHA-1-verified
- `data/manifests/sleep_edf_full.csv` (197 rows), `sleep_edf.csv` (canonical copy),
  `subject_splits_full.csv`, `dataset_audit.json`

**Must print:** `PASS: 197 recordings / 100 subjects / 0 split overlaps`  
**Split (seed 42):** train 135 / val 30 / test 32 recordings; per-cohort 70/15/15 subject-level

### Notebook 02: Full Preprocessing
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_preprocessing.ipynb
```
**Runtime:** ~2.5 h (filter → 30-s epoch → AASM harmonize → z-score → QC → cache)  
**Outputs:** `data/cache_full/` — 197 recordings, **457,652 epochs** (4 × 3000 @ 100 Hz),
`cache_index.csv`, `preprocessing_summary.json` (`notch_applied: false` — 50 Hz = Nyquist)

### Notebook 03: Exploratory Data Analysis
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_exploratory_data_analysis.ipynb
```
**Runtime:** minutes. Outputs: EDA figures/statistics (no benchmark artifacts).

### Notebook 04: Final Training
```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/04_student_99k_complete_training.ipynb
```
**Runtime:** ~3 h total on GTX 1650 (~1.5 h training loop + cache/dataset cells);
early stopping typically ends the loop before the 30-epoch budget  
**Config:** batch 8, train stride 5 / eval stride 10 (gap-aware, no tail padding),
train-only augmentation, AdamW 3e-4 / wd 1e-4, clip 1.0, 10% warmup + cosine  
**Outputs:**
- `artifacts/final/EXP-FULL-AUG30_seed42.pt` (best validation macro-F1 checkpoint, full provenance payload)
- `results/final/training_history_full_dataset.csv`, `test_metrics_full_dataset.json`,
  `per_class_metrics_full_dataset.csv`, `confusion_matrix_full_dataset.csv`,
  `experiment_summary_full_dataset.json`

### Notebook 05: Evaluation & Protocol Audit
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation_and_benchmarking.ipynb
```
**Runtime:** ~10 min  
**Outputs:**
- Canonical: `results/final/final_metrics.json`, `final_result.csv`, `predictions.csv`,
  `confusion_matrix.csv` (+ `confusion_matrix.png`), `per_class_metrics.csv`,
  `fit_diagnosis.json` (+ `*_full_dataset` variants)
- Verified checkpoint promoted to `artifacts/student_improved_best.pt` (byte-checked copy)

**Must print:** `FINAL PROTOCOL AUDIT PASSED`

### Expected Canonical Metrics (from `results/final/final_metrics.json`)
| Metric | Expected value |
|--------|----------------|
| Accuracy | 0.9048 |
| Cohen's κ | 0.8283 |
| Macro F1 | 0.7899 |
| Weighted F1 | 0.9089 |
| Macro geometric mean | 0.7911 |
| CPU latency | ~9 ms per 5-minute window (hardware-dependent) |

Exact numbers can vary slightly with GPU model / library builds (see determinism caveat);
the audit gates, dataset scope (197/100), and protocol parameters must match exactly.

---

## Post-Run Verification Checklist

- [ ] `pytest tests/` passes (92 tests)
- [ ] `python tests/test_documentation_metrics.py` → PASS (docs match canonical artifacts)
- [ ] NB01 output contains `PASS: 197 recordings / 100 subjects / 0 split overlaps`
- [ ] `data/manifests/dataset_audit.json` has `sha_verified: true`, `split_seed: 42`
- [ ] NB05 output contains `FINAL PROTOCOL AUDIT PASSED`
- [ ] `artifacts/student_improved_best.pt` exists and equals `artifacts/final/EXP-FULL-AUG30_seed42.pt`
- [ ] `results/final/final_metrics.json` matches the table in `docs/RESULTS.md`
- [ ] `results/final/fit_diagnosis.json` reports train/val/test accuracy 0.9250 / 0.8737 / 0.9048

### Verify the Promoted Checkpoint
```bash
python - <<'EOF'
import torch
a = torch.load("artifacts/student_improved_best.pt", map_location="cpu", weights_only=False)
b = torch.load("artifacts/final/EXP-FULL-AUG30_seed42.pt", map_location="cpu", weights_only=False)
assert all(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]) for k in a["model_state_dict"])
print("promoted checkpoint == final checkpoint")
print("epoch:", a["epoch"], "| params:", a["parameter_count"], "| val kappa:", round(a["metrics"]["kappa"], 4))
print("provenance: git", a["git_commit"][:12], "| torch", a["torch_version"], "| cache_layout:", a["dataset"]["cache_layout"])
EOF
```

### Manifest Hash
NB05 prints the full-manifest SHA-256 (first 16 hex of the freeze run: `45ddd5774d7a7370`)
and stores it in `results/final/final_metrics.json` → `manifest_sha256`. Re-verify with:
```bash
sha256sum data/manifests/sleep_edf_full.csv
```

---

## Person-Level Benchmark Reproduction (EXP-BENCH-PERSON)

### 1. Generate Person-Level Folds (if not present)
```bash
python scripts/generate_person_folds.py
```
**Output:** `data/manifests/person_folds_52subj.json`

### 2. Verify Protocol
```bash
python scripts/verify_protocol.py
```
**Expected:** PASS

### 3. Run Training (per fold, per seed)
```bash
# Single fold, single seed
python scripts/run_person_level_benchmark.py --fold 0 --seed 42

# All folds, seed 42
for fold in {0..9}; do
    python scripts/run_person_level_benchmark.py --fold $fold --seed 42
done

# Multi-seed (seeds 42, 43, 44)
for seed in 42 43 44; do
    for fold in {0..9}; do
        python scripts/run_person_level_benchmark.py --fold $fold --seed $seed
    done
done
```

### 4. Summarize Results
```bash
python scripts/summarize_person_benchmark.py --results-dir results/research/EXP-BENCH-PERSON
```
**Outputs:** `results/research/EXP-BENCH-PERSON/summary_with_ci.json`, markdown table

**Expected (30 folds × 3 seeds):** accuracy 87.30% ± 0.33%, κ 0.738 ± 0.010,
macro-F1 0.724 ± 0.005

---

## Artifact Verification

### Checkpoint Self-Provenance
The final-protocol checkpoint embeds its own provenance — inspect with:
```bash
python - <<'EOF'
import torch
ck = torch.load("artifacts/final/EXP-FULL-AUG30_seed42.pt", map_location="cpu", weights_only=False)
for k in ("epoch", "seed", "batch_size", "max_epochs", "early_stopping_patience",
          "augmentation_enabled", "parameter_count", "git_commit", "torch_version"):
    print(f"{k}: {ck[k]}")
print("dataset:", ck["dataset"])
print("optimizer:", ck["optimizer"])
EOF
```
Expected: `epoch` = best-validation epoch (12), `batch_size` = 8, `max_epochs` = 30,
`early_stopping_patience` = 5,
`augmentation_enabled` = True, `parameter_count` = 99477,
`dataset.cache_layout` = "per-recording mmap epochs .npy + _meta.npz sidecars",
`dataset.n_recordings` = 197, `dataset.n_subjects` = 100.

---

## Testing

### Run Test Suite
```bash
pytest tests/ -v
```

### Key Tests
| Test File | Purpose |
|-----------|---------|
| `tests/test_model_contract.py` | Model input/output contract |
| `tests/test_sequence_dataset.py` | Sequence dataset correctness (gap-aware windows) |
| `tests/test_checkpoint.py` | Checkpoint load/save |
| `tests/test_seed.py` | Deterministic execution |
| `tests/test_documentation_metrics.py` | Docs quote canonical artifacts verbatim |

---

## Expected Outputs

### Final Protocol Run (authoritative freeze)
- Training: best val macro-F1 0.7645 @ epoch 12, early-stopped at epoch 17/30
- Held-out test (16 subjects, 72,200 labels): accuracy 90.48%, κ 0.8283,
  macro-F1 0.7899, weighted-F1 0.9089

### Notebook 05 Test Metrics (other tiers, for cross-checks)
**EXP-STANDALONE-99K (deployed checkpoint):** accuracy 90.57%, κ 0.8080, macro-F1 0.7490

**EXP-BENCH-PERSON (historical benchmark):** accuracy 87.30% ± 0.33%, κ 0.738 ± 0.010,
macro-F1 0.724 ± 0.005

---

## CI/CD Reproduction

### GitHub Actions
The `.github/workflows/tests.yml` runs:
1. `pytest tests/`
2. `python scripts/verify_protocol.py`

---

## Troubleshooting

### RAM exhaustion during training ("kernel killed", swap thrashing)
The final protocol trains from a **memory-mapped cache**
(`data/cache_full/*_epochs.npy` + `*_meta.npz`). If `data/cache_full/` still contains
per-recording `.npz` files instead, re-run Notebook 02 (or Notebook 04's conversion cell,
which normalizes the layout idempotently and byte-verifies every file before deleting the
original). Do **not** dense-load all training recordings — the full corpus does not fit
in 16 GB of RAM as dense arrays.

### PhysioNet download extremely slow
NB01 already tries the AWS Open Data mirror first
(`https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/`) and falls back to
physionet.org. All files are SHA-1-verified from the official MNE record tables either
way. If both sources fail, check network egress; re-running NB01 resumes (verified files
are skipped).

### CUDA Out of Memory (GPU)
Reduce `BATCH_SIZE` in Notebook 04's configuration cell (8 → 4) and record the change in
the experiment summary. The checkpoint payload will reflect the actual batch size, and
NB05's audit gate `batch_size == 8` will then fail by design — a deviation must be
documented, not silently audited.

### Non-Deterministic Results
Seeds are fixed (see Deterministic Execution). For bit-exact reproduction also set
`torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`
in Notebook 04 — note this changes the executed code relative to the freeze run.

### NumPy 2.x incompatibility
Notebook 03 uses `np.trapz` (NumPy 1.x name). Under NumPy ≥ 2.0 rename to
`np.trapezoid`, or pin `numpy<2`.

### Missing Raw Data
Do not download manually — Notebook 01 builds the 394-file list from the official MNE
record tables, downloads (mirror-first), and SHA-1-verifies everything into
`data/raw/sleep_edf_full/`. The legacy `data/raw/sleep_edf/` night-1 subset is retained
for provenance only and is not used by the final protocol.

---

## Contact
For reproduction issues, check:
1. `docs/EXPERIMENTS.md` — Experiment definitions (incl. EXP-FULL-AUG30)
2. `docs/RESULTS.md` — Canonical metrics
3. `docs/guide.md` — Final full-data training protocol guide
4. `scripts/verify_protocol.py` — Protocol integrity
5. `docs/adaptation.md` — Known limitations
