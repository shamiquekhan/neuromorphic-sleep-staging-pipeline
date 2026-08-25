# Methodology — Neuromorphic Sleep Stage Scoring

## Research Approach

This project follows a systematic engineering methodology for developing an automated sleep-stage classification system. The approach emphasizes reproducibility, edge-deployment readiness, and exhibition-quality documentation.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    METHODOLOGY PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Data Acquisition & Governance                        │
│  ├── Sleep-EDF download & validation                           │
│  ├── PSG/Hypnogram pairing                                     │
│  ├── Subject-level splitting                                   │
│  └── Manifest generation                                       │
│                                                                 │
│  Phase 2: Signal Preprocessing                                 │
│  ├── Bandpass filtering (0.5–35 Hz)                            │
│  ├── Notch filtering (50 Hz)                                   │
│  ├── Z-score normalization                                     │
│  ├── Artifact quality control                                  │
│  └── Epoch caching                                             │
│                                                                 │
│  Phase 3: Exploratory Data Analysis                            │
│  ├── Class distribution analysis                               │
│  ├── Signal characteristic visualization                       │
│  ├── Temporal pattern analysis                                 │
│  └── Quality control validation                                │
│                                                                 │
│  Phase 4: Model Development                                    │
│  ├── Architecture design (Improved Student)                    │
│  ├── Teacher training (Improved Teacher)                       │
│  ├── Knowledge distillation                                    │
│  └── Checkpoint selection                                      │
│                                                                 │
│  Phase 5: LoRA Adaptation                                      │
│  ├── Rank sweep (r=2, r=4, r=8)                               │
│  ├── 4-fold held-out-subject CV                                │
│  ├── Multi-seed confirmation                                   │
│  ├── Latency & merge verification                              │
│  └── Parameter-efficiency analysis                             │
│                                                                 │
│  Phase 6: Evaluation & Deployment                              │
│  ├── Test set evaluation                                       │
│  ├── Official result documentation                             │
│  ├── Edge deployment preparation                               │
│  └── Exhibition demonstration                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Acquisition & Governance

### Objectives
- Obtain Sleep-EDF PSG recordings
- Validate data integrity
- Create reproducible subject-level splits
- Establish data contract for downstream processing

### Methods

**Data Source:** PhysioNet Sleep-EDF Expanded database
- 78 healthy subjects
- 2 nights per subject (most subjects)
- 100 Hz sampling rate
- 4 channels: EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG

**Subject-Level Splitting:**
```python
# Prevents data leakage from same subject in multiple splits
train_subj, temp_subj = train_test_split(subjects, test_size=0.30, random_state=42)
val_subj, test_subj = train_test_split(temp_subj, test_size=0.50, random_state=42)
```

**Quality Assurance:**
- PSG-Hypnogram file pairing validation
- Duplicate subject-night detection
- Channel availability verification

### Deliverables
- `data/manifests/sleep_edf.csv` — Subject/night manifest
- `data/manifests/subject_splits.csv` — Train/val/test splits

---

## Phase 2: Signal Preprocessing

### Objectives
- Reduce noise and artifacts
- Standardize signal characteristics
- Create model-ready epoch representations

### Methods

**Bandpass Filtering:**
- 4th-order Butterworth filter
- Cutoff: 0.5 Hz (high-pass) to 35 Hz (low-pass)
- Rationale: Removes baseline drift and high-frequency noise while preserving sleep-relevant frequencies

**Notch Filtering:**
- IIR notch filter at 50 Hz
- Quality factor: 30
- Rationale: Eliminates power-line interference

**Z-Score Normalization:**
```python
x_normalized = (x - mean) / std
```
- Per-channel, per-epoch
- Ensures zero mean and unit variance
- Makes model invariant to absolute signal amplitude

**Artifact Quality Control:**
| Check | Threshold | Action |
|-------|-----------|--------|
| Clipping | |x| > 8.0σ | Flag epoch |
| Flatline | std < 0.05 | Flag epoch |
| NaN/Inf | Non-finite values | Flag epoch |

### Rationale
Preprocessing is treated as a first-class engineering step, not an optional cleanup. The contract is fixed and reproducible.

### Deliverables
- `data/cache/*.npz` — Preprocessed epoch arrays
- `data/cache/cache_index.csv` — Cache metadata

---

## Phase 3: Exploratory Data Analysis

### Objectives
- Verify dataset characteristics
- Identify class imbalance
- Validate preprocessing quality
- Understand temporal patterns

### Analyses Performed

**Class Distribution:**
- Quantified imbalance across 5 sleep stages
- N2 dominates (~50% of epochs)
- N1 and REM are minority classes

**Signal Characteristics:**
- Visualized raw vs. preprocessed waveforms
- Verified frequency content after filtering
- Confirmed normalization effectiveness

**Temporal Patterns:**
- Analyzed sleep-stage transition frequencies
- Verified contiguous epoch ordering
- Confirmed no data leakage across recordings

### Key Findings
- Class imbalance requires imbalance-aware training (focal loss, class weights)
- Artifact flag rate ~2% (acceptable)
- Strong temporal dependencies justify sequence modeling

---

## Phase 4: Model Development

### Architecture Design Principles

1. **Multi-scale signal processing:** Capture patterns at different temporal scales
2. **Computational efficiency:** Use depthwise-separable convolutions for edge deployment
3. **Frequency awareness:** Include parametric Gabor filters for spectral patterns
4. **Temporal modeling:** Use GRU to capture stage transitions

### Improved Student Architecture

| Component | Description | Parameters |
|-----------|-------------|------------|
| Multi-Resolution Stem | Parallel short/long receptive fields | ~1,000 |
| Depthwise-Separable CNN | Efficient feature extraction | ~2,000 |
| Parametric Gabor FEB | Learnable frequency filters | ~1,300 |
| 2-Layer GRU | Temporal sequence modeling | ~16,500 |
| Classification Head | 5-class softmax | 320 |
| **Total** | | **99,477** |

### Training Strategy

**Knowledge Distillation:**
- Train a larger "teacher" model first
- Distill knowledge into the smaller "student"
- Student learns from both hard labels and softened teacher predictions

**Loss Function:**
```
L_total = α_ce × L_CE + α_kl × L_KL + α_feat × L_Feature
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Cross-entropy | 1.0 | Hard-label supervision |
| KL divergence | 1.0 | Teacher-student alignment |
| Feature MSE | 0.5 | Intermediate representation alignment |

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | AdamW | Adaptive learning rate with weight decay |
| Learning rate | 3e-4 | Standard for small models |
| Batch size | 16 | Balance between speed and stability |
| Sequence length | 10 | 5 minutes of context (300s) |
| Training epochs | 20 | Sufficient for convergence |

### Rationale
The architecture is intentionally compact for edge deployment. The 99,477 parameter count is below the project's 100K target while maintaining competitive accuracy (87.5% across 15 subjects).

### Deliverables
- `artifacts/student_improved_best.pt` — Final trained model
- `artifacts/teacher_improved_best.pt` — Teacher model (training only)

---

## Phase 5: Evaluation & Deployment

### Evaluation Protocol

**Test Set:** Held-out subjects never seen during training

**Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 87.5% ± 3.2% | Overall correct classifications |
| Cohen's κ | 0.763 ± 0.043 | Agreement beyond chance |
| Macro F1 | 0.721 ± 0.050 | Balanced performance across classes |

**Per-Class Performance:**

| Stage | F1 Score | Notes |
|-------|----------|-------|
| Wake | 0.9693 | Excellent — strong EEG/EMG signatures |
| N1 | 0.2006 | Challenging — brief, transitional stage |
| N2 | 0.8162 | Good — distinct spindle/K-complex features |
| N3 | 0.7849 | Good — clear delta wave patterns |
| REM | 0.3586 | Moderate — overlap with Wake/N1 |

### Official Result

The project's single authoritative result is:

```
Improved Student
87.34% Test Accuracy
Cohen's κ = 0.7551
99,477 Parameters
8.5 ms/batch CPU latency
```

### Deployment Preparation

1. Export to ONNX format
2. Apply INT8 post-training quantization
3. Validate on target hardware
4. Implement real-time inference pipeline

---

## Reproducibility

| Aspect | Implementation |
|--------|---------------|
| Random seeds | Python, NumPy, PyTorch all seeded to 42 |
| Deterministic operations | CUDA deterministic mode enabled |
| Configuration | YAML files for all parameters |
| Version control | Git-tracked code and manifests |
| Checkpointing | Best model saved by validation κ |

---

## Ethical Considerations

1. **Not a medical device:** This is a research prototype, not a clinical diagnostic system
2. **Limited validation:** Only validated on Sleep-EDF (healthy subjects)
3. **Expert oversight required:** Should supplement, not replace, expert scoring
4. **Data privacy:** Uses publicly available, de-identified data
5. **Responsible claims:** Avoid overstating clinical applicability

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
