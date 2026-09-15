# Dataset — Sleep-EDF Expanded

## Overview

| Property | Value |
|----------|-------|
| Source | PhysioNet Sleep-EDF Expanded (SC cassette study) |
| Type | Polysomnography (PSG) recordings |
| Records downloaded | 100 |
| Records excluded (wake-only) | 8 |
| **Eligible cohort** | **92 records — 52 persons** |
| Persons with both nights | 40 (SC4ss1 + SC4ss2) |
| Persons with one night | 12 (lost cassettes / wake-only exclusion) |
| Epoch length | 30 seconds |
| Sampling rate | 100 Hz |
| Target classes | 5 (AASM standard) |

> **Records ≠ persons (P0).** PhysioNet's SC naming is `SC4ssNE0` where
> *ss is the subject number and N is the night* (sleep-edfx README).
> `SC4051` and `SC4052` are two nights of the **same person** —
> confirmed by `SC-subjects.xls` (age/sex constant per ss) and the EDF
> headers (identical patient fields, adjacent recording dates). The
> "92-subject" phrasing used in earlier documentation is wrong: the
> cohort is 92 records from 52 persons. Splits must be made at the
> person level (`person_folds_52subj.json`,
> `scripts/generate_person_folds.py`).

---

## Data Source

**PhysioNet Sleep-EDF Expanded Database**
- URL: https://physionet.org/content/sleep-edfx/1.0.0/
- License: Open Access
- Citation: Goldberger et al., 2000

The Sleep-EDF dataset contains polysomnography recordings from healthy subjects. Each recording includes:
- EEG signals (multiple channels)
- EOG (electrooculography)
- EMG (electromyography)
- Hypnogram annotations (expert-scored sleep stages)

---

## Channels Used

| Channel | MNE Name | Purpose |
|---------|----------|---------|
| EEG 1 | EEG Fpz-Cz | Brain electrical activity (frontal) |
| EEG 2 | EEG Pz-Oz | Brain electrical activity (parietal-occipital) |
| EOG | EOG horizontal | Eye movement detection |
| EMG | EMG submental | Muscle activity |

### Why These Channels?

**EEG Fpz-Cz:**
- Primary channel for sleep stage classification
- Captures delta waves (N3), sleep spindles (N2), alpha rhythm (Wake)

**EEG Pz-Oz:**
- Complementary posterior EEG
- Helps distinguish N2 from N3 stages

**EOG:**
- Critical for REM detection (rapid eye movements)
- Helps identify Wake state (voluntary eye movements)

**EMG:**
- Muscle tone changes across sleep stages
- Low tone in REM, higher in Wake
- Helps distinguish Wake from N1

---

## Sleep Stages (AASM Standard)

| Code | Stage | Description | Typical EEG Features |
|------|-------|-------------|---------------------|
| 0 | Wake (W) | Awake state | Alpha rhythm (8-13 Hz), beta activity |
| 1 | N1 | Light sleep transition | Theta waves (4-8 Hz), vertex sharp waves |
| 2 | N2 | Stable light/intermediate sleep | Sleep spindles, K-complexes |
| 3 | N3 | Deep/slow-wave sleep | Delta waves (0.5-4 Hz), high amplitude |
| 4 | REM | Rapid eye movement sleep | Mixed frequency, low amplitude, rapid eye movements |

### Class Distribution

The dataset exhibits extreme class imbalance (untrimmed full-night recordings):

```
Wake:   68.8% of epochs (28,219) — majority class
N1:      3.4% of epochs (1,388)  — minority class
N2:     16.4% of epochs (6,718)
N3:      5.0% of epochs (2,070)
REM:     6.4% of epochs (2,642)
```

> **Why Wake is 68.8%:** These are untrimmed full-night recordings that include pre-sleep and post-sleep wake periods. Standard practice in the literature trims to ±30 min around sleep onset/offset, which would rebalance to ~10-15% Wake. This pipeline uses untrimmed recordings for now.

**Impact on training:** The pipeline uses:
- N1/REM class weighting (2x) to address minority classes
- Macro F1 and Cohen's κ as primary metrics (accuracy-insensitive to imbalance)

---

## Data Preprocessing Pipeline

### Raw Signal → Clean Epochs

```
Raw PSG (100 Hz, 4 channels)
    │
    ▼
┌─────────────────────────────────┐
│ 1. Bandpass Filter              │
│    - 4th order Butterworth      │
│    - 0.5 Hz ≤ f ≤ 35 Hz        │
│    - Removes: baseline drift,   │
│      high-frequency noise       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 2. Notch Filter                 │
│    - IIR notch at 50 Hz         │
│    - Q factor: 30               │
│    - Removes: power-line noise  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 3. Z-Score Normalization        │
│    - Per-channel, per-epoch     │
│    - x_norm = (x - μ) / σ      │
│    - Ensures zero mean, unit    │
│      variance                   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 4. Artifact Quality Control     │
│    - Clipping detection         │
│      (|x| > 8.0 σ)             │
│    - Flatline detection         │
│      (std < 0.05)               │
│    - NaN/Inf detection          │
│    - Flag rate: ~2%             │
└─────────────┬───────────────────┘
              │
              ▼
    Clean Epochs (30s × 4ch × 100Hz)
    Shape: [n_epochs, 4, 3000]
```

---

## Dataset Splits

### Person-Level Splitting (current protocol)

To prevent data leakage, splits are performed at the **person level**:
both nights of a person always travel together into the same role
(train / validation / test). The person-level protocol
(`person_folds_52subj.json`) uses:

- 10 test folds over the 47 pool persons (4–5 persons per fold)
- 5 fixed validation persons (~10% of persons)
- folds stratified by **age decade** — the SC cohort spans 25–101 yr
  and fold difficulty must be balanced

| Role | Persons | Records | Purpose |
|------|---------|---------|---------|
| Train (per fold) | ~42 | ~74 | Model training |
| Validation (fixed) | 5 | 8 | Checkpoint selection, early stopping |
| Test (per fold) | 4–5 | 6–10 | Final evaluation |

**Key principle:** no person appears in multiple roles, and no record's
same-person mate is ever in a different role. This is enforced by
`scripts/verify_protocol.py` and by the runner's refusal guard.

### Historical: Record-Level Splitting (superseded)

The earlier manifest (record-level, removed) treated
each record (night) as an independent "subject". This leaked at person
level in 10/10 folds: a test record's same-person mate was almost
always in the train pool. Numbers from those folds are record-level
estimates and are labeled as such in [`results.md`](results.md).

---

## Data Contract

The preprocessing pipeline outputs the following artifacts:

### Manifest File
```csv
subject_id,night,psg,hypnogram,split
SC4001,0,data/raw/sleep_edf/SC4001E0-PSG.edf,data/raw/sleep_edf/SC4001EC-Hypnogram.edf,train
SC4002,0,data/raw/sleep_edf/SC4002E0-PSG.edf,data/raw/sleep_edf/SC4002EC-Hypnogram.edf,train
```

### Cache Files
```python
# Per subject-night .npz file
{
    "epochs": np.ndarray,      # [n_epochs, 4, 3000] float32
    "labels": np.ndarray,      # [n_epochs] int64, values 0-4
    "qc_flag": np.ndarray,     # [n_epochs] bool
    "fs": float,               # 100.0
    "subject_id": str,         # "SC4001"
    "night": int,              # 0
    "split": str               # "train" | "val" | "test"
}
```

### Cache Index
```csv
subject_id,night,split,n_epochs,n_flagged,cache_path
SC4001,0,train,2649,50,data/cache/SC4001_night0.npz
SC4002,0,train,2829,56,data/cache/SC4002_night0.npz
```

---

## Data Governance

### Integrity Checks

The pipeline performs these validation steps:

1. **PSG-Hypnogram pairing:** Each PSG file must have a matching hypnogram
2. **File existence:** All referenced files must exist on disk
3. **No duplicate subject-nights:** Each (subject_id, night) pair is unique
4. **Channel availability:** All 4 required channels must be present
5. **Label validation:** All labels must be in {0, 1, 2, 3, 4}
6. **No NaN/Inf:** Processed signals must be finite

### Reproducibility

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Random seed | 42 | Reproducible subject splits |
| Filter order | 4 | Deterministic filtering |
| Normalization | Per-epoch z-score | Consistent across runs |

---

## Dataset Limitations

1. **Two nights per person only:** 40 of 52 persons have both nights; night-to-night variability beyond two nights is unrepresented.
2. **Healthy subjects only:** No pathological sleep patterns (apnea, narcolepsy, etc.).
3. **Wide age range (25–101 yr):** fold stratification mitigates but does not remove demographic imbalance.
4. **Class imbalance:** N1 and REM are underrepresented.
5. **Annotation granularity:** 30-second epochs may miss brief events.
6. **RK scoring conventions:** hypnograms use Rechtschaffen-Kales (1968), not AASM — labels are mapped to 5 classes.

---

## Citation

```bibtex
@article{goldberger2000physiobank,
  title={PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals},
  author={Goldberger, Ary L and others},
  journal={Circulation},
  volume={101},
  number={23},
  pages={e215--e220},
  year={2000}
}
```

---

*Last updated: September 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
