# N1 Diagnosis Report

## 1. Problem Definition

The NeuroSleep Improved Student model achieves 87.34% test accuracy and Cohen's κ=0.7551 on 4-fold held-out-subject evaluation. However, N1 (Stage 1 sleep) performance is severely degraded:

| Model            | N1 F1  | N1 Recall | Accuracy | κ     |
|------------------|--------|-----------|----------|-------|
| Frozen Base      | 0.000  | 0.000     | 0.873    | 0.755 |
| LoRA r=2         | 0.000  | 0.000     | 0.899    | 0.790 |
| LoRA r=4         | 0.000  | 0.000     | 0.899    | 0.790 |
| LoRA r=8         | 0.000  | 0.000     | 0.907    | 0.809 |
| Full Fine-Tuning | 0.25±0.09 | 0.17±0.07 | 0.930 | 0.859 |
| **Baseline (this study)** | **0.179±0.058** | **0.210±0.076** | **0.846** | **0.723** |

The investigation traces N1 information loss through the complete pipeline.

## 2. Current Baseline

Training configuration for this diagnosis:
- Architecture: Improved Student (99,477 params)
- Context: 10 × 30s epochs (300s), target = last epoch (index 9)
- Stride: 5 (50% overlap)
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-2)
- Scheduler: CosineAnnealingLR (T_max=12)
- Epochs: up to 12, early stopping on validation macro F1 (patience=4)
- Validation: temporal last-20% split within training subjects
- Evaluation: 4-fold held-out-subject CV, stride=1 for test

## 3. Dataset Audit

### 3.1 Epoch-Level Distribution

| Subject | Total Epochs | N1 Epochs | N1 %  |
|---------|-------------|-----------|-------|
| SC4001  | 2,650       | 58        | 2.2%  |
| SC4002  | 2,829       | 59        | 2.1%  |
| SC4011  | 2,802       | 109       | 3.9%  |
| SC4012  | 2,848       | 92        | 3.2%  |
| **Total** | **11,129** | **318**   | **2.9%** |

N1 is the rarest class. Wake dominates at 67.9%.

### 3.2 Class Distribution (All Subjects)

| Class | Count | Percentage |
|-------|-------|------------|
| Wake  | 7,562 | 67.9%      |
| N1    | 318   | 2.9%       |
| N2    | 1,845 | 16.6%      |
| N3    | 718   | 6.5%       |
| REM   | 686   | 6.2%       |

### 3.3 QC Flag Rate by Class

| Class | Total | Flagged | Rate  |
|-------|-------|---------|-------|
| Wake  | 7,562 | 7,230   | 95.6% |
| N1    | 318   | 232     | 73.0% |
| N2    | 1,845 | 1,632   | 88.5% |
| N3    | 718   | 677     | 94.3% |
| REM   | 686   | 575     | 83.8% |

**Key finding:** QC flags are stored but NOT used to filter training data. All epochs are used regardless of QC status. N1 has the LOWEST QC flag rate (73.0%), meaning N1 epochs are actually cleaner than other classes on average.

## 4. Annotation & Label Mapping

The AASM mapping is correct:

```python
AASM_MAP = {
    "Sleep stage W": 0,  # Wake
    "Sleep stage 1": 1,  # N1
    "Sleep stage 2": 2,  # N2
    "Sleep stage 3": 3,  # N3 (merged with stage 4)
    "Sleep stage 4": 3,  # N3
    "Sleep stage R": 4,  # REM
}
```

All 4 subjects contain N1 labels (integer 1) in their cached arrays. No label mapping issues detected.

## 5. PSG/Hypnogram Alignment

MNE's `events_from_annotations` with `chunk_duration=30` creates events aligned to annotation boundaries. The pipeline uses `raw.set_annotations(mne.read_annotations(hyp_path))` which correctly synchronizes PSG and hypnogram timestamps. No offset issues detected.

## 6. Sequence Target Audit

| Fold | Test Subj | Train Seqs | N1-as-Target | N1-in-Window | Test Seqs | N1-Test-Target |
|------|-----------|------------|--------------|--------------|-----------|----------------|
| 1    | SC4001    | 1,691      | 54 (3.2%)    | 204 (12.1%)  | 529       | 13 (2.5%)      |
| 2    | SC4002    | 1,656      | 53 (3.2%)    | 194 (11.7%)  | 564       | 14 (2.5%)      |
| 3    | SC4011    | 1,661      | 42 (2.5%)    | 182 (11.0%)  | 559       | 25 (4.5%)      |
| 4    | SC4012    | 1,652      | 52 (3.1%)    | 170 (10.3%)  | 568       | 15 (2.6%)      |

**Key finding:** N1 appears as the training target in only ~3% of sequences, but appears somewhere in the 10-epoch window in ~11% of sequences. The model sees N1 as context much more often than as a target. The supervision signal for N1 is extremely稀疏.

## 7. Confusion Analysis (Frozen Base Model)

```
             Predicted
True    Wake   N1    N2    N3   REM
Wake    750     0     0     0     0
N1       33     0     1     0     0
N2       92     1   101     0     0
N3        0     0    28    39     0
REM      65     0     1     0     0
```

**N1 is confused with Wake 95.5% of the time (33/34 samples).** The model never predicts N1 (0/34 correct).

## 8. Probability Analysis (Frozen Base, True N1 Epochs)

| Metric | Value |
|--------|-------|
| P(N1) mean | 0.076 |
| P(N1) max | 0.182 |
| P(N1) > 0.1 | 9/34 (26.5%) |
| P(Wake) mean | 0.804 |
| P(N2) mean | 0.083 |
| N1 as argmax | 0/34 (0.0%) |

The model assigns Wake probability of ~80% to true N1 epochs. Even the most "N1-like" epoch only receives 18% probability for N1.

## 9. Confusion Analysis (Full Fine-Tuning)

```
True N1 → Wake:  28.4%
True N1 → N1:    16.4%  (recall)
True N1 → N2:    17.9%
True N1 → N3:     4.5%
True N1 → REM:   32.8%
```

Full FT improves N1 recall from 0% to 16.4%, but N1 predictions are scattered across all classes rather than concentrated on N1. This suggests the model has learned some N1 features but lacks discriminative power.

## 10. LoRA Confusion Analysis

```
True N1 → Wake:  44.8%
True N1 → N1:     0.0%
True N1 → N2:    13.4%
True N1 → REM:   41.8%
```

LoRA shifts N1 from Wake to REM (41.8%), but still achieves 0% N1 recall. The head-only adapter cannot learn the discriminative features needed for N1.

## 11. N1-Weighted Experiments

| Config           | Accuracy | κ     | N1 F1 | N1 Recall | Notes |
|------------------|----------|-------|-------|-----------|-------|
| n1w=1.0 os=1.0   | 0.846    | 0.723 | 0.179 | 0.210     | Baseline |
| n1w=1.5 os=1.0   | 0.543    | 0.407 | 0.189 | 0.219     | Accuracy destroyed |
| n1w=2.0 os=1.0   | 0.436    | 0.280 | 0.136 | 0.219     | Accuracy destroyed |
| n1w=2.0 os=2.0   | 0.187    | 0.114 | 0.179 | 0.276     | Accuracy destroyed |
| n1w=1.0 os=2.0   | 0.331    | 0.220 | 0.146 | 0.247     | Accuracy destroyed |
| focal n1w=2.0    | 0.192    | 0.105 | 0.185 | 0.278     | Accuracy destroyed |

**Key finding:** All weighting/oversampling/focal approaches improve N1 recall marginally (from 0.21 to ~0.28) but destroy overall accuracy (from 0.85 to 0.19-0.54). The model begins predicting N1 for many Wake samples, and since Wake is 68% of the data, accuracy collapses.

## 12. Root Cause Analysis

The N1 weakness has **multiple contributing factors**, not a single root cause:

### Factor 1: Extreme Class Imbalance (Primary)
- N1 = 2.9% of all epochs (318/11,129)
- N1 = ~3% of training sequence targets
- The model can achieve 87%+ accuracy by never predicting N1

### Factor 2: Physiological Overlap with Wake (Structural)
- N1 and Wake share similar EEG characteristics (low-amplitude, mixed-frequency)
- The frozen model assigns P(Wake)=0.80 to true N1 epochs
- Even full FT only achieves 16% N1 recall, with N1 scattered across all classes
- This is a known limitation in sleep staging research

### Factor 3: Sparse Supervision Signal
- Only the last epoch (index 9) is supervised during LoRA/training
- N1 appears in ~3% of training targets
- The gradient signal for N1 is extremely稀疏

### Factor 4: LoRA Capacity Insufficient
- LoRA adapts only the head layer (552 params, 0.55%)
- N1 discrimination requires deeper feature changes
- LoRA shifts N1→REM (41.8%) but cannot learn N1-specific features

### What is NOT the cause:
- **Label mapping**: Correct (verified)
- **Hypnogram alignment**: Correct (MNE handles this)
- **QC filtering**: Not used for training (all epochs included)
- **Sequence indexing**: Correct (target = last epoch)
- **Preprocessing**: Same as training pipeline

## 13. Corrective Action Summary

### What works:
1. **Full fine-tuning** achieves N1 F1=0.25 by updating all 99,477 parameters
2. **Baseline training** with proper class weighting achieves N1 F1=0.179

### What does NOT work:
1. N1-weighted loss (destroys accuracy)
2. N1 oversampling (destroys accuracy)
3. Focal loss (destroys accuracy)
4. LoRA adaptation (insufficient capacity)

### The honest assessment:
N1 F1 of ~0.18-0.25 is near the practical ceiling for this 4-subject dataset with this architecture. The limitation is structural (class imbalance + physiological overlap), not a training bug.

## 14. Before/After Results

| Metric | Frozen | Full FT | Baseline (this study) |
|--------|--------|---------|----------------------|
| Accuracy | 0.873 | 0.930 | 0.846 |
| κ | 0.755 | 0.859 | 0.723 |
| N1 F1 | 0.000 | 0.250 | 0.179 |
| N1 Recall | 0.000 | 0.170 | 0.210 |

## 15. Remaining Limitations

1. **N1 remains the weakest class** across all configurations
2. **4 subjects** provide insufficient N1 diversity
3. **Physiological overlap** between N1 and Wake is inherent to the problem
4. **Weighting tradeoff**: improving N1 recall hurts overall accuracy

## 16. Final Recommendation

For the exhibition, present the N1 limitation honestly:

> "N1 (Stage 1 sleep) is the most challenging class due to its physiological similarity to Wake and its representation as only 2.9% of the dataset. The model achieves zero N1 recall in frozen and LoRA configurations, and approximately 17-25% N1 F1 with full fine-tuning. This is a known limitation of small-dataset sleep staging that motivates future work with larger subject populations."

The project's scientific value lies in diagnosing and documenting this limitation, not in hiding it.
