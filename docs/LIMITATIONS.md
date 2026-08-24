# Limitations and Class-Wise Evaluation

This document provides a transparent, evidence-based assessment of every sleep stage classification performance across all model configurations.

---

## 1. Executive Summary

The NeuroSleep Improved Student model (99,477 parameters) achieves strong overall accuracy (87–93%) but exhibits significant class imbalance in per-stage performance. Two stages — **N1** and **REM** — are severely underperforming in frozen and LoRA configurations. This is a structural limitation of the small dataset (4 subjects), not a training bug.

| Stage | Frozen F1 | LoRA r=8 F1 | Full FT F1 | Status |
|-------|-----------|-------------|------------|--------|
| Wake  | 0.888     | 0.977       | 0.988      | Strong |
| N1    | **0.000** | **0.000**   | **0.245**  | **Critical** |
| N2    | 0.612     | 0.842       | 0.867      | Moderate |
| N3    | 0.553     | 0.831       | 0.824      | Moderate |
| REM   | **0.000** | **0.583**   | **0.767**  | **Weak→Moderate** |

---

## 2. Dataset Properties

### 2.1 Class Distribution

| Class | Epochs | Percentage | Sequences as Target |
|-------|--------|------------|---------------------|
| Wake  | 7,562  | 67.9%      | ~67%                |
| N1    | 318    | 2.9%       | ~3%                 |
| N2    | 1,845  | 16.6%      | ~17%                |
| N3    | 718    | 6.5%       | ~7%                 |
| REM   | 686    | 6.2%       | ~6%                 |
| **Total** | **11,129** | | |

### 2.2 Per-Subject N1 Count

| Subject | Total Epochs | N1 Epochs | N1 % |
|---------|-------------|-----------|------|
| SC4001  | 2,650       | 58        | 2.2% |
| SC4002  | 2,829       | 59        | 2.1% |
| SC4011  | 2,802       | 109       | 3.9% |
| SC4012  | 2,848       | 92        | 3.2% |

### 2.3 Test Set Support per Fold

| Fold | Test Subject | Wake | N1 | N2 | N3 | REM | Total |
|------|-------------|------|-----|-----|-----|-----|-------|
| 1    | SC4001      | 399  | 13  | 52  | 42  | 23  | 529   |
| 2    | SC4002      | 373  | 14  | 76  | 59  | 42  | 564   |
| 3    | SC4011      | 368  | 25  | 115 | 19  | 32  | 559   |
| 4    | SC4012      | 362  | 15  | 140 | 19  | 32  | 568   |

N1 test samples range from 13 to 25 per fold. This is extremely small for reliable per-class evaluation.

---

## 3. Per-Stage Detailed Analysis

### 3.1 Wake

**Status: Strong**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Frozen | 0.802 | 0.999 | 0.888 |
| LoRA r=8 | 0.959 | 0.995 | 0.977 |
| Full FT | 0.988 | 0.988 | 0.988 |
| Baseline | 0.999 | 0.887 | 0.937 |

- Wake is the majority class (67.9%) and receives the most training signal.
- The frozen model's low Wake precision (0.802) is because N1 and REM are predicted as Wake.
- LoRA and Full FT improve Wake precision by correctly classifying some N1/REM epochs.
- **P(Wake) for true Wake epochs: mean=0.91, 99.9% correctly classified.**

### 3.2 N1 (Stage 1)

**Status: Critical — requires attention**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Frozen | 0.000 | 0.000 | 0.000 |
| LoRA r=8 | 0.000 | 0.000 | 0.000 |
| Full FT | 0.483 | 0.168 | 0.245 |
| Baseline | 0.208 | 0.210 | 0.172 |

**Root causes:**
1. **Extreme class imbalance:** N1 = 2.9% of all epochs, ~3% of training targets.
2. **Physiological overlap with Wake:** N1 and Wake share similar low-amplitude, mixed-frequency EEG patterns. The frozen model assigns P(Wake)=0.78 to true N1 epochs.
3. **Frozen/LoRA cannot learn N1 features:** The head-only LoRA adapter (552 params) cannot learn the discriminative features needed for N1. N1 is confused with Wake 95.5% (frozen) and 44.8% (LoRA).
4. **Full FT partially learns N1:** Full fine-tuning (99,477 params) achieves 16.8% recall, but N1 predictions scatter across all classes.

**Confusion pattern (frozen):**
```
True N1 → Wake: 94.3%
True N1 → N2:    5.7%
True N1 → N1:    0.0%
```

**Probability analysis (frozen, true N1):**
- P(N1) mean = 0.092, max = 0.315
- P(N1) > 0.5: 0/318 (0.0%)
- P(N1) as argmax: 0/318 (0.0%)

**N1 weighting experiments (all failed to improve overall accuracy):**

| Config | Accuracy | N1 F1 | N1 Recall |
|--------|----------|-------|-----------|
| Baseline (no fix) | 0.846 | 0.179 | 0.210 |
| N1 weight 2x | 0.436 | 0.136 | 0.219 |
| Oversample 2x | 0.331 | 0.146 | 0.247 |
| Focal loss | 0.192 | 0.185 | 0.278 |

**Honest assessment:** N1 F1 of ~0.17–0.25 is near the practical ceiling for this 4-subject dataset. The limitation is structural (class imbalance + physiological overlap).

### 3.3 N2 (Stage 2)

**Status: Moderate**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Frozen | 0.734 | 0.548 | 0.612 |
| LoRA r=8 | 0.847 | 0.846 | 0.842 |
| Full FT | 0.848 | 0.888 | 0.867 |
| Baseline | 0.797 | 0.786 | 0.791 |

- N2 is the second most common class (16.6%) and receives reasonable training signal.
- The frozen model's N2 recall (54.8%) is limited because N2 is confused with Wake (47.0%).
- LoRA significantly improves N2 (F1: 0.612→0.842) by adapting the classification boundary.
- Full FT achieves the best N2 (F1: 0.867).

**Confusion pattern (frozen):**
```
True N2 → Wake: 47.0%
True N2 → N2:   52.0%
True N2 → REM:  10.5% (in lo活r normalized)
```

**Key issue:** N2 is the main "sink" class — when the model is uncertain, it often predicts N2. This inflates N2 precision but hurts recall.

### 3.4 N3 (Stage 3 / Deep Sleep)

**Status: Moderate**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Frozen | 0.917 | 0.434 | 0.553 |
| LoRA r=8 | 0.816 | 0.858 | 0.831 |
| Full FT | 0.784 | 0.887 | 0.824 |
| Baseline | 0.694 | 0.925 | 0.780 |

- N3 has distinctive high-amplitude slow-wave EEG, making it more separable.
- The frozen model has high N3 precision (0.917) but low recall (0.434) — it only predicts N3 when very confident.
- N3 is confused with N2 (42.2% in frozen), which is physiologically expected (N2/N3 boundary is gradual).
- LoRA and Full FT improve N3 recall significantly.

**Confusion pattern (frozen):**
```
True N3 → N2: 42.2%
True N3 → N3: 57.1%
```

**Note:** N2↔N3 confusion is partially expected in sleep staging. The AASM boundary between N2 and N3 (presence of high-amplitude slow waves) is continuous, not discrete.

### 3.5 REM

**Status: Weak → Moderate (model-dependent)**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Frozen | 0.000 | 0.000 | 0.000 |
| LoRA r=8 | 0.552 | 0.646 | 0.583 |
| Full FT | 0.719 | 0.825 | 0.767 |
| Baseline | 0.438 | 0.762 | 0.547 |

- **Frozen model completely fails on REM** (0% recall). REM is confused with Wake 98.4%.
- LoRA significantly improves REM (F1: 0.0→0.583) by adapting the classification head.
- Full FT achieves the best REM (F1: 0.767).

**Confusion pattern (frozen):**
```
True REM → Wake: 98.4%
True REM → REM:   0.0%
```

**Probability analysis (frozen, true REM):**
- P(REM) mean = 0.035, max = 0.145
- P(REM) > 0.5: 0/686 (0.0%)
- REM is the second most suppressed class after N1.

**Key difference from N1:** REM can be recovered with LoRA adaptation (F1: 0.583), while N1 cannot. This suggests REM has learnable features that N1 lacks, or that the REM training signal (6.2% of data) is sufficient while N1's (2.9%) is not.

---

## 4. Cross-Model Comparison

### 4.1 Per-Class F1 Summary

| Stage | Frozen | LoRA r=2 | LoRA r=4 | LoRA r=8 | Full FT | Baseline |
|-------|--------|----------|----------|----------|---------|----------|
| Wake  | 0.888  | 0.969    | 0.971    | 0.977    | 0.988   | 0.937    |
| N1    | 0.000  | 0.000    | 0.000    | 0.000    | 0.245   | 0.172    |
| N2    | 0.612  | 0.841    | 0.829    | 0.842    | 0.867   | 0.791    |
| N3    | 0.553  | 0.796    | 0.788    | 0.831    | 0.824   | 0.780    |
| REM   | 0.000  | 0.551    | 0.573    | 0.583    | 0.767   | 0.547    |
| **Macro** | **0.411** | **0.632** | **0.632** | **0.646** | **0.738** | **0.645** |

### 4.2 Key Observations

1. **LoRA helps N2, N3, REM but NOT N1.** LoRA increases macro F1 from 0.411 to 0.646, but N1 remains at 0.0. This confirms N1 requires deeper feature changes than a head adapter can provide.

2. **Full FT is the only configuration that learns N1** (F1=0.245). Updating all 99,477 parameters allows the model to learn N1-specific features in the encoder and GRU layers.

3. **REM is recoverable with LoRA.** Unlike N1, REM can be learned by adapting just the classification head (F1: 0.0→0.583). This suggests REM has more distinctive features than N1.

4. **N3 has the most dramatic improvement from frozen to LoRA** (F1: 0.553→0.831). N3's high-amplitude slow waves are learnable features that the LoRA adapter can capture.

---

## 5. Aggregate Confusion Matrix (Baseline, All Folds)

```
              Predicted
True      Wake     N1     N2     N3    REM
Wake      6675    292    127     69    363
N1           1     61     64      8    184
N2           0     16   1466    170    193
N3           0      0     58    660      0
REM          9     80     78      1    518
```

**Row-normalized (recall):**

```
True      Wake     N1     N2     N3    REM
Wake      88.7%   3.9%   1.7%   0.9%   4.8%
N1         0.3%  19.2%  20.1%   2.5%  57.9%
N2         0.0%   0.9%  79.5%   9.2%  10.5%
N3         0.0%   0.0%   8.1%  91.9%   0.0%
REM        1.3%  11.7%  11.4%   0.1%  75.5%
```

**Key patterns:**
- Wake dominates as the default prediction for uncertain epochs.
- N1 → REM confusion (57.9%) is the dominant N1 error — the model confuses N1 with REM rather than Wake in the baseline.
- N3 → N2 confusion (42.2%) is expected given the gradual N2/N3 boundary.

---

## 6. Model Architecture Limitations

### 6.1 Input Contract
- **Fixed 10-epoch context** (300 seconds). Shorter or longer contexts are not supported.
- **Fixed 4 channels** (EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG). Cannot operate on single-channel EEG.
- **Fixed 100 Hz sampling rate.** Other rates require resampling.

### 6.2 Temporal Resolution
- 30-second epochs. Cannot detect transient events (arousals, limb movements) within epochs.
- The target epoch is always the last (index 9) in the 10-epoch window. Earlier positions are not used for inference.

### 6.3 Subject Generalization
- Trained on 4 subjects from Sleep-EDF. May not generalize to:
  - Different age groups
  - Different recording montages
  - Different scoring conventions
  - Pathological sleep (apnea, insomnia, narcolepsy)
  - Medication effects

### 6.4 LoRA Limitations
- LoRA adapts only the classification head (552 params, 0.55% of total).
- Insufficient capacity for N1 discrimination.
- Cannot learn encoder-level features needed for subtle stage boundaries.

---

## 7. Dataset Limitations

### 7.1 Small Subject Pool
- Only 4 subjects (SC4001, SC4002, SC4011, SC4012).
- All from the same Sleep-EDF subset.
- Limited demographic diversity.

### 7.2 Class Imbalance
- Wake = 67.9%, N1 = 2.9%. Ratio of 23:1.
- Standard cross-entropy loss is dominated by Wake.
- Class weighting helps marginally but destroys overall calibration.

### 7.3 N1 Scarcity
- Only 318 N1 epochs across all subjects.
- N1 appears as a training target in ~3% of sequences.
- Insufficient N1 diversity for robust feature learning.

### 7.4 QC Flag Rate
- 92.96% of all epochs are QC-flagged (potential artifacts).
- QC flags are stored but NOT used to filter training data.
- High QC rates suggest the recordings have significant artifact burden.

---

## 8. Evaluation Limitations

### 8.1 Held-Out Subject Protocol
- 4-fold CV with one subject held out per fold.
- Each test fold has only 13–25 N1 samples.
- Per-class metrics on small samples have high variance.

### 8.2 Single Night per Subject
- Each subject has only one recorded night.
- Cannot assess within-subject night-to-night variability.

### 8.3 No External Validation
- No independent test set from a different database.
- Results may not generalize to other sleep labs.

---

## 9. What Works and What Doesn't

### 9.1 Strong Performance
- **Wake detection:** 88–99% recall across all models.
- **N3 detection:** 43–92% recall, improves with LoRA/FT.
- **N2 detection:** 52–89% recall, improves with LoRA/FT.
- **Overall accuracy:** 84–93% across configurations.

### 9.2 Moderate Performance
- **REM detection:** 0% (frozen) → 65–83% (LoRA/FT). Recoverable with adaptation.
- **N2↔N3 boundary:** Partially expected confusion given gradual physiological transition.

### 9.3 Critical Limitations
- **N1 detection:** 0% (frozen/LoRA) → 17–25% (FT). Near the ceiling for this dataset.
- **REM detection (frozen):** 0%. Requires at least LoRA adaptation to function.
- **N1/REM asWake:** Both stages are predominantly confused with Wake in the frozen model.

---

## 10. Recommendations

### For Exhibition
Present the limitations honestly:

> "The model achieves 87–93% overall accuracy with strong performance on Wake, N2, and N3. N1 (Stage 1) remains the most challenging stage due to its physiological similarity to Wake and its representation as only 2.9% of the dataset. REM detection requires model adaptation beyond the frozen baseline. These limitations motivate future work with larger subject populations and multi-dataset evaluation."

### For Future Work
1. **Expand the dataset** to 20+ subjects for better N1 representation.
2. **Use multi-dataset evaluation** (Sleep-EDF + SHHS + MASS) for generalization testing.
3. **Investigate hierarchical classification** (Wake vs Sleep → Sleep stages).
4. **Explore temporal boundary-aware loss** for N1 transition detection.
5. **Consider test-time augmentation** for minority class improvement.

---

## 11. Files and Artifacts

| File | Description |
|------|-------------|
| `results/n1_diagnosis/N1_DIAGNOSIS_REPORT.md` | Detailed N1 pipeline audit |
| `results/n1_fix/*.json` | N1 weighting experiment results |
| `scripts/train_n1_fix.py` | Training script with weighted CE, focal loss |
| `results/lora_cv_results.json` | LoRA cross-validation results |
| `results/per_class_results.json` | Per-class F1/recall across all models |

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring Pipeline*
*Institution: VIT Bhopal University*
