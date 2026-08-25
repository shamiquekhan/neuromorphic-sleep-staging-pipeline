# Data Sources and Licenses

## NeuroSleep Data Sources

This document describes all datasets used in the NeuroSleep project, their access requirements, and licensing terms.

---

### 1. Sleep-EDF Expanded (Primary Dataset)

| Property | Value |
|----------|-------|
| **Provider** | PhysioNet / MIT Laboratory for Computational Physiology |
| **URL** | https://www.physionet.org/content/sleep-edfx/1.0.0/ |
| **DOI** | 10.13026/C2X676 |
| **Access** | Open Access (no registration required) |
| **License** | Open Data Commons Attribution License v1.0 |
| **Subjects** | 197 whole-night recordings |
| **Channels** | EEG (Fpz-Cz, Pz-Oz), EOG (horizontal), EMG (submental) |
| **Sampling Rate** | 100 Hz (EEG/EOG), 1 Hz (EMG envelope) |
| **Scoring** | Rechtschaffen & Kales (R&K) manual scoring |
| **Epoch Duration** | 30 seconds |

#### Subset Used

Our primary experiments use **153 healthy controls** (SC4xxxx subjects) from the Sleep Cassette study, recorded 1987-1991.

**Note:** Subject SC4013 was lost due to a failing cassette (per PhysioNet documentation).

#### Citation

> B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).

---

### 2. SHHS (Planned External Dataset)

| Property | Value |
|----------|-------|
| **Provider** | National Sleep Research Resource (NSRR) |
| **URL** | https://sleepdata.org/datasets/shhs/ |
| **Access** | **Data Use Agreement required** — must register at sleepdata.org |
| **License** | NSRR Data Use Agreement |
| **Subjects** | SHHS Visit 1: 5,793 subjects; Visit 2: 2,651 subjects |
| **Channels** | Variable montage (must verify from documentation) |
| **Scoring** | AASM-compatible staging |
| **Format** | EDF (PSG) + XML (staging annotations) |

#### Access Requirements

1. Register at https://sleepdata.org
2. Complete Data Use Agreement
3. Download files via NSRR portal or AWS CLI
4. **Do not redistribute raw data**

#### Usage Plan

SHHS will be used for:
1. **N1 enrichment** — controlled injection of SHHS N1 samples into Sleep-EDF training
2. **Cross-dataset pretraining** — train on SHHS + Sleep-EDF, evaluate on Sleep-EDF
3. **Domain adaptation** — LoRA adaptation from SHHS-pretrained base to Sleep-EDF target

#### Citation

> Quan SF, Howard BV, Iber C, et al. The Sleep Heart Health Study: a study of obstructive sleep apnea and cardiovascular disease in a community sample. Am J Respir Crit Care Med. 1997;155(3):1070-1077.

---

### 3. Data Processing Provenance

All processed data includes provenance tracking:

```
dataset: sleep_edf_expanded
version: 1.0.0
preprocessing:
  sampling_rate: 100 Hz
  epoch_seconds: 30
  bandpass: [0.5, 35.0] Hz
  notch: 50 Hz
  normalization: z-score
scoring_mapping: R&K → AASM canonical
  W → Wake (0)
  1 → N1 (1)
  2 → N2 (2)
  3,4 → N3 (3)
  R → REM (4)
  M,? → excluded
```

---

### 4. Redistribution Rules

| Dataset | Raw Data | Processed Cache | Model Checkpoints |
|---------|----------|-----------------|-------------------|
| Sleep-EDF | ✅ Allowed (ODC-BY) | ✅ Allowed | ✅ Allowed |
| SHHS | ❌ Requires NSRR permission | ⚠️ Check DUA | ✅ Allowed |
| Our Models | N/A | ✅ Allowed | ✅ Allowed |

---

### 5. File Organization

```
data/
├── raw/
│   └── sleep_edf/           # Raw EDF files (from PhysioNet)
│       ├── SC4001E0-PSG.edf  # 16 PSG files downloaded
│       ├── SC4002E0-PSG.edf
│       ├── ...
│       └── *-Hypnogram.edf   # Hypnogram annotations
├── cache/
│   ├── sleep_edf/           # Processed NPZ files (15 subjects)
│   │   ├── SC4001_night0.npz
│   │   ├── SC4002_night0.npz
│   │   ├── ...
│   │   └── checksums.json
│   └── shhs/                # Processed NPZ files (SHHS, future)
├── manifests/
│   ├── sleep_edf.csv        # Original 4-subject manifest
│   └── sleep_edf_expanded.json  # 15-subject manifest
```

#### Current State

- **16 PSG files** downloaded from PhysioNet (Sleep-EDF Expanded)
- **15 subjects** successfully preprocessed and cached as NPZ
- **1 subject** (SC4021) failed preprocessing (channel mismatch)
- **41,037 total epochs** across all cached subjects
- **N1 distribution:** 1,388 epochs (3.4%) — 4.4x increase from original 318
