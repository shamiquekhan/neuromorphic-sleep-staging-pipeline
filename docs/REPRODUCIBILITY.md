# NeuroSleep — Reproducibility Guide
## Exact Commands, Environment, and Artifacts for Full Reproduction

---

## Environment Specification

### Conda Environment
```bash
conda env create -f environment.yml
conda activate neurosleep
```

### Pip Requirements (Locked)
```bash
pip install -r requirements-lock.txt
```

### Key Versions
| Package | Version |
|---------|---------|
| Python | 3.11.16 |
| PyTorch | 2.6.0 |
| CUDA | 12.4 (if GPU available) |
| NumPy | 1.26.4 |
| Pandas | 2.2.1 |
| scikit-learn | 1.4.2 |
| MNE | 1.7.1 |

### Hardware
- GPU: NVIDIA GTX 1650 or better (4GB VRAM minimum)
- RAM: 16 GB minimum
- Disk: ~10 GB for cache and artifacts

---

## Deterministic Execution

### Seed Control
All experiments use seed 42 (plus 43, 44 for multi-seed benchmarks).

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Full Pipeline Reproduction (Notebooks 01→05)

### Prerequisites
- Raw Sleep-EDF Expanded data in `data/raw/sleep_edf/`
- Run notebooks in order with fresh kernel

### Notebook 01: Data Import & Dataset Collection
```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_import_and_dataset_collection.ipynb
```
**Outputs:** `data/manifests/sleep_edf_expanded.json`, `data/manifests/exhibition_15subj_v1.json`

### Notebook 02: Preprocessing
```bash
jupyter nbconvert --to notebook --execute notebooks/02_data_preprocessing.ipynb
```
**Outputs:** Cached epochs in `data/cache/sleep_edf/` (232,219 epochs), `data/manifests/sleep_edf.csv`

### Notebook 03: Exploratory Data Analysis
```bash
jupyter nbconvert --to notebook --execute notebooks/03_exploratory_data_analysis.ipynb
```
**Outputs:** EDA figures and statistics

### Notebook 04: Model Architecture & Training
```bash
jupyter nbconvert --to notebook --execute notebooks/04_student_99k_complete_training.ipynb
```
**Runtime:** ~18 minutes on GTX 1650  
**Outputs:**
- `artifacts/standalone_99k/student_99477_best.pt`
- Per-epoch training logs embedded in notebook

### Notebook 05: Evaluation & Benchmarking
```bash
jupyter nbconvert --to notebook --execute notebooks/05_evaluation_and_benchmarking.ipynb
```
**Outputs:** Results in `results/standalone_99k/` and `results/research/EXP-BENCH-PERSON/`

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

---

## Artifact Verification

### Checkpoint Provenance
Every canonical checkpoint includes a `provenance.json`:

```json
{
  "experiment_id": "EXP-BENCH-PERSON",
  "git_commit": "fea0b6594b3d759ac53a34fb444c46ac92bae7d4",
  "manifest_sha256": "0365b29cc9226afefa0b1fce8fb697df5462fdbb570d874163b5a83c77475ecb",
  "checkpoint_sha256": "...",
  "environment": {...},
  "seeds": [42, 43, 44]
}
```

### Verify Checkpoint Integrity
```bash
python -c "
import hashlib, json
with open('results/research/EXP-BENCH-PERSON/provenance.json') as f:
    p = json.load(f)
h = hashlib.sha256()
with open('artifacts/standalone_99k/student_99477_best.pt', 'rb') as f:
    for chunk in iter(lambda: f.read(65536), b''):
        h.update(chunk)
print('student_99477_best.pt sha256:', h.hexdigest())
"
```

---

## Protocol Fingerprint

Before running multi-seed benchmarks, verify protocol consistency:

```bash
# Save seed-42 reference
python scripts/protocol_fingerprint.py --seed 42 --mode full_finetune --save-reference

# Verify seeds 43, 44 match
python scripts/protocol_fingerprint.py --seed 43 --mode full_finetune
python scripts/protocol_fingerprint.py --seed 44 --mode full_finetune
```

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
| `tests/test_sequence_dataset.py` | Sequence dataset correctness |
| `tests/test_checkpoint.py` | Checkpoint load/save |
| `tests/test_seed.py` | Deterministic execution |

---

## Expected Outputs

### Notebook 04 Training Logs
**Student 99k (20 epochs, seed 42):**
- Peak val κ: 0.8274 at epoch 18
- Final val κ: 0.8219 at epoch 20

### Notebook 05 Test Metrics
**EXP-STANDALONE-99K (deployed checkpoint):**
- Accuracy: 90.57%
- κ: 0.8080
- Macro F1: 0.7490

**EXP-BENCH-PERSON (primary):**
- Accuracy: 87.30% ± 0.33%
- κ: 0.738 ± 0.010
- Macro F1: 0.724 ± 0.005

---

## CI/CD Reproduction

### GitHub Actions
The `.github/workflows/tests.yml` runs:
1. `pytest tests/`
2. `python scripts/verify_protocol.py`

### Manual Verification Checklist
- [ ] `pytest tests/` passes (92 tests)
- [ ] `python scripts/verify_protocol.py` → PASS
- [ ] Notebook 04 executes without error
- [ ] Notebook 05 produces expected metrics
- [ ] Checkpoint provenance matches `provenance.json`
- [ ] `scripts/summarize_person_benchmark.py` matches `docs/RESULTS.md`

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
training:
  batch_size: 16  # instead of 32
```

### Non-Deterministic Results
Ensure:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- Same PyTorch/CUDA versions

### Missing Raw Data
Download Sleep-EDF Expanded from PhysioNet and place in `data/raw/sleep_edf/`

---

## Contact
For reproduction issues, check:
1. `docs/EXPERIMENTS.md` — Experiment definitions
2. `docs/RESULTS.md` — Canonical metrics
3. `scripts/verify_protocol.py` — Protocol integrity
4. `docs/adaptation.md` — Known limitations