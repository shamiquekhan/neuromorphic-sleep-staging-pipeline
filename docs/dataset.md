# Dataset — Sleep-EDF Expanded

## Overview

| Property | Value |
|----------|-------|
| Source | PhysioNet Sleep-EDF Expanded |
| Type | Polysomnography (PSG) recordings |
| Subjects | 4 (development subset) |
| Recordings | 4 nights |
| Total epochs | ~11,128 (full cohort) |
| Epoch length | 30 seconds |
| Sampling rate | 100 Hz |
| Target classes | 5 (AASM standard) |

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

The dataset exhibits significant class imbalance:

```
Wake:   ~10-15% of epochs
N1:     ~5-10% of epochs
N2:     ~45-55% of epochs (majority class)
N3:     ~15-25% of epochs
REM:    ~15-20% of epochs
```

**Impact on training:** The pipeline uses:
- Log-inverse frequency class weights
- Focal loss for hard example mining
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

### Subject-Level Splitting

To prevent data leakage, splits are performed at the **subject level**, not at the epoch level.

| Split | Subjects | Recordings | Purpose |
|-------|----------|------------|---------|
| Train | 70% | ~55 subjects | Model training |
| Validation | 15% | ~12 subjects | Hyperparameter tuning, checkpoint selection |
| Test | 15% | ~12 subjects | Final evaluation |

**Key principle:** No subject appears in multiple splits. This ensures the model generalizes to unseen subjects.

### Split Assignment

```python
from sklearn.model_selection import train_test_split

subjects = sorted(manifest["subject_id"].unique())
train_subj, temp_subj = train_test_split(subjects, test_size=0.30, random_state=42)
val_subj, test_subj = train_test_split(temp_subj, test_size=0.50, random_state=42)
```

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

1. **Small development subset:** The current pipeline uses 4 subjects. Full deployment requires 78 subjects.
2. **Healthy subjects only:** No pathological sleep patterns (apnea, narcolepsy, etc.)
3. **Single night per subject:** Limited intra-subject variability
4. **Class imbalance:** N1 and REM are underrepresented
5. **Annotation granularity:** 30-second epochs may miss brief events

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

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
