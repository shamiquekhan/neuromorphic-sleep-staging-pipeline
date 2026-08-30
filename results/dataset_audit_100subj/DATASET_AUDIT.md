# Dataset Audit Report — 100-Subject Sleep-EDF Expanded

**Audit Date:** 2026-08-28  
**Dataset:** Sleep-EDF Expanded v1.0.0 (PhysioNet)  
**Script:** `scripts/audit_100_subjects.py`

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total NPZ files | 100 |
| Valid subjects | 100 (0 load errors) |
| Wake-only subjects | 8 (excluded) |
| **Included subjects** | **92** |
| Total epochs (all) | 232,219 |
| **Total epochs (included)** | **228,532** |
| NaN / Inf issues | None |
| Shape issues | None (all 4ch × 3000 samples) |

---

## Global Class Distribution (92 Included Subjects)

| Class | Epochs | Percentage |
|-------|-------:|-----------:|
| Wake  | 157,566 | 68.95% |
| N1    |  10,580 |  4.63% |
| N2    |  37,561 | 16.44% |
| N3    |   7,943 |  3.48% |
| REM   |  14,882 |  6.51% |

**Key observation:** Severe class imbalance. N1 (4.63%) and N3 (3.48%) are the minority classes. Wake dominates at ~69%.

---

## Wake-Only Subjects (Excluded)

| Subject | Epochs | Duration | Wake% | Likely Cause |
|---------|-------:|---------:|------:|-------------|
| SC4082 | 462 | 3.85h | 100% | Corrupted/misaligned hypnogram |
| SC4111 | 864 | 7.20h | 100% | Corrupted/misaligned hypnogram |
| SC4142 | 648 | 5.40h | 100% | Corrupted/misaligned hypnogram |
| SC4162 | 517 | 4.31h | 100% | Corrupted/misaligned hypnogram |
| SC4172 | 41 | 0.34h | 100% | Truncated recording |
| SC4192 | 107 | 0.89h | 100% | Truncated recording |
| SC4232 | 331 | 2.76h | 100% | Corrupted/misaligned hypnogram |
| SC4301 | 717 | 5.97h | 100% | Corrupted/misaligned hypnogram |

**Action:** Excluded from training and CV. Not deleted from disk.

---

## N1 Distribution Analysis

| Metric | Value |
|--------|-------|
| Median N1 epochs/subject | 92 |
| Mean N1 epochs/subject | 105.8 |
| Median N1 % | 3.59% |
| Min N1 epochs | 0 |
| Max N1 epochs | 470 |

### N1 Categories

| Category | Count | Threshold |
|----------|------:|-----------|
| Low N1 | 24 | < 2% |
| Normal N1 | 71 | 2–10% |
| High N1 | 5 | > 10% |

**Top N1 subjects:**
- SC4732: 470 N1 epochs (18.7%)
- SC4661: 375 N1 epochs (14.1%)
- SC4622: 347 N1 epochs (12.2%)
- SC4721: 259 N1 epochs (11.1%)
- SC4621: 276 N1 epochs (10.6%)

---

## Subjects Missing Stages

| Category | Count | Subjects |
|----------|------:|----------|
| All 5 stages present | 77 | (majority) |
| Missing N3 only | 13 | SC4202, SC4321, SC4641, SC4712, SC4721, SC4722, SC4731, SC4732, SC4741, SC4742, SC4751, SC4762, SC4642 |
| Missing REM only | 1 | SC4221 |
| Missing N2+N3+REM | 1 | SC4522 |
| Wake-only (excluded) | 8 | SC4082, SC4111, SC4142, SC4162, SC4172, SC4192, SC4232, SC4301 |

**Note:** Subjects missing only N3 are still valid for training on Wake/N1/N2/REM stages. Only wake-only subjects are excluded.

---

## Quality Checks

| Check | Status |
|-------|--------|
| All labels in {0,1,2,3,4} | PASS (after excluding wake-only) |
| No NaN values | PASS |
| No Inf values | PASS |
| Shape = (N, 4, 3000) | PASS for all 100 |
| Sampling rate = 100 Hz | PASS |

---

## 10-Fold CV Class Balance by Fold

The canonical folds have been regenerated to exclude the 8 wake-only subjects.
Each fold's test set contains ~9–10 subjects from the 92 included subjects.

See: `class_distribution_by_fold.csv` and `fold_class_balance.png`

---

## Output Files

```
results/dataset_audit_100subj/
├── DATASET_AUDIT.md                      # This report
├── dataset_summary.json                  # Machine-readable summary
├── subject_statistics.csv                # Per-subject raw stats
├── per_subject_class_distribution.csv    # Per-subject class breakdown
├── class_distribution.csv                # Global class totals
├── class_distribution_by_fold.csv        # Per-fold class breakdown
├── qc_statistics.csv                     # Quality check flags
├── wake_only_subjects.csv                # Excluded subjects detail
├── subject_quality_manifest.csv          # Final inclusion/exclusion manifest
├── n1_distribution.png                   # N1 epoch histogram
├── class_distribution_overview.png       # Global class bar chart
└── fold_class_balance.png               # Fold class heatmap
```

---

## Recommended Next Steps

1. **Regenerate canonical folds** for 92 included subjects (excluding wake-only)
2. **Train 100-subject benchmark** using `configs/full_100_subject.yaml`
3. **Run 3-seed confirmation** (seeds 42, 43, 44)
4. **Compare 15 vs 100 subject results** — the key experiment
5. **Track Macro F1** as primary metric (not accuracy)
