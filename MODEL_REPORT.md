# NeuroSleep - Model Report

## Executive Summary

This report presents the final **Improved Student** model for five-stage
sleep classification (Wake, N1, N2, N3, REM) from polysomnography (PSG)
signals.

The final benchmark uses the **Sleep-EDF Expanded database**, with **100
subjects downloaded and 92 eligible subjects retained** after excluding
8 wake-only subjects. Evaluation uses **10-fold subject-level
cross-validation with 3 random seeds (30 evaluation folds per method)**.

The final full-fine-tuned model achieves:

-   **87.7% ± 2.7% accuracy**
-   **Cohen's κ = 0.763 ± 0.043**
-   **Macro F1 = 0.730 ± 0.037**
-   **Weighted F1 = 0.890 ± 0.021**
-   **MGm = 0.797 ± 0.040**
-   **99,477 trainable parameters**

The model is designed for compact, edge-oriented sleep staging while
retaining temporal context over 300 seconds of PSG data.

------------------------------------------------------------------------

## 1. Model Architecture

**Model:** Improved Student\
**Parameters:** 99,477 (\~100K)\
**Input:** `[B, T=10, C=4, S=3000]`\
**Output:** `[B, T=10, 5]`

### Signal Inputs

  Channel   Signal   Purpose
  --------- -------- -----------------------------------
  Fpz-Cz    EEG      Frontal-central brain activity
  Pz-Oz     EEG      Parietal-occipital brain activity
  EOG       EOG      Eye movement detection
  EMG       EMG      Muscle tone / REM detection

### Processing Pipeline

``` text
PSG Input (Fpz-Cz, Pz-Oz, EOG, EMG)
        ↓
Multi-Resolution Stem
  ├── Short kernel (25) — fast transients
  └── Long kernel (200) — slow oscillations
        ↓
Depthwise-Separable CNN
  ├── Block 0: DW Conv + PW Conv + BN
  └── Block 1: DW Conv + PW Conv + BN
        ↓
Adaptive Average Pooling
        ↓
Parametric Gabor Feature Extraction
  ├── 8 learnable filters
  ├── Learnable frequency: 0.5–30 Hz
  └── Learnable bandwidth
        ↓
Feature Fusion
        ↓
2-Layer GRU (hidden=64)
        ↓
Linear(64 → 5)
        ↓
Wake / N1 / N2 / N3 / REM
```

### Parameter Breakdown

  Module                          Parameters   \% of Total
  ----------------------------- ------------ -------------
  Multi-Resolution Stem                7,232         7.27%
  Depthwise-Separable Encoder          2,304         2.32%
  Gabor Feature Extraction               144         0.14%
  2-Layer GRU                         89,856        90.33%
  Classification Head                    325         0.33%
  **Total**                       **99,477**      **100%**

The GRU accounts for most of the parameter budget because it provides
sequence-level temporal modeling across the 300-second context window.

------------------------------------------------------------------------

## 2. Final Benchmark

### 2.1 Evaluation Protocol

  Property                      Value
  ----------------------------- ---------------------------------------
  Dataset                       Sleep-EDF Expanded
  Subjects downloaded           100
  Wake-only subjects excluded   8
  Final eligible cohort         **92 subjects**
  Cross-validation              10-fold subject-level CV
  Random seeds                  3
  Evaluation folds per method   **30**
  Sequence length               10 epochs
  Epoch duration                30 seconds
  Context window                300 seconds
  Sampling rate                 100 Hz
  Channels                      4
  Classes                       5
  Final model                   Improved Student --- Full Fine-Tuning
  Parameters                    99,477

**Leakage control:** subject-level folds are used so that each subject
appears in exactly one test fold. All ten positions in each 10-epoch
sequence are supervised.

> **Terminology note:** The project uses the name "100-Subject
> Benchmark" because 100 subjects were downloaded. The actual eligible
> evaluation cohort contains **92 subjects**, after excluding 8
> wake-only records. Results should therefore be described precisely as
> a **92-subject evaluation cohort from a 100-subject downloaded
> dataset**.

### 2.2 Overall Metrics

  Metric               Full Fine-Tuning
  ----------------- -------------------
  **Accuracy**         **87.7% ± 2.7%**
  **Cohen's κ**       **0.763 ± 0.043**
  **Macro F1**        **0.730 ± 0.037**
  **Weighted F1**      **89.0% ± 2.1%**
  **MGm**             **0.797 ± 0.040**
  **Parameters**             **99,477**

The difference between weighted and macro F1 reflects the strong class
imbalance in Sleep-EDF: Wake dominates the epoch distribution, while N1
is comparatively rare.

------------------------------------------------------------------------

## 3. Per-Stage Performance

  Stage                       F1           Precision              Recall
  ---------- ------------------- ------------------- -------------------
  **Wake**     **0.964 ± 0.016**   **0.995 ± 0.003**   **0.936 ± 0.031**
  **N1**       **0.445 ± 0.061**       0.330 ± 0.065   **0.712 ± 0.070**
  **N2**       **0.768 ± 0.041**   **0.878 ± 0.043**       0.687 ± 0.063
  **N3**       **0.681 ± 0.114**       0.562 ± 0.135   **0.896 ± 0.075**
  **REM**      **0.771 ± 0.078**       0.772 ± 0.077       0.785 ± 0.118

### 3.1 N1 Remains the Main Bottleneck

N1 is the weakest class, with an F1 of **0.445 ± 0.061**. This is
consistent with the transitional nature of N1, which is difficult to
separate from Wake and N2 using short physiological segments.

The model nevertheless achieves **0.712 ± 0.070 recall** for N1,
indicating that the low F1 is driven substantially by false positives
and class ambiguity rather than failure to detect N1 altogether.

### 3.2 Model Strengths

-   **Wake F1 = 0.964** with precision of **0.995**, showing highly
    reliable Wake classification.
-   **REM F1 = 0.771**, with balanced precision and recall.
-   **N2 F1 = 0.768**, providing strong performance on the dominant
    sleep stage.
-   **N3 recall = 0.896**, indicating that most deep-sleep epochs are
    successfully detected.
-   **Macro F1 = 0.730**, demonstrating useful performance beyond the
    dominant Wake class.
-   The complete model remains below **100K parameters**, supporting
    edge-oriented deployment.

------------------------------------------------------------------------

## 4. Adaptation Study

The project also evaluates three adaptation strategies under the same
subject-level benchmark.

  ---------------------------------------------------------------------------
  Method               Trainable       Accuracy      Cohen's κ       Macro F1
                          Params                               
  --------------- -------------- -------------- -------------- --------------
  Frozen Base                  0   87.1% ± 3.6%  0.738 ± 0.077  0.673 ± 0.074

  LoRA CNN + Head          1,448   83.6% ± 3.7%  0.693 ± 0.057  0.674 ± 0.045

  **Full              **99,477**      **87.7% ±      **0.763 ±      **0.730 ±
  Fine-Tuning**                          2.7%**        0.043**        0.037**
  ---------------------------------------------------------------------------

### Key Finding

Full fine-tuning provides the strongest overall and stage-balanced
performance.

LoRA trains only **1,448 parameters (1.43% of the model)** ---
approximately **68.7× fewer trainable parameters** than full fine-tuning
--- while retaining approximately **95.4% of full-fine-tuning
accuracy**.

This makes LoRA a potentially useful parameter-efficient adaptation
strategy, although it does not match full fine-tuning on the final
benchmark.

------------------------------------------------------------------------

## 5. Training Configuration

  Parameter                 Value
  ------------------------- -------------------
  Optimizer                 AdamW
  Learning rate             3e-4
  Weight decay              1e-4
  Maximum epochs            20
  Early stopping patience   5
  Batch size                32
  Scheduler                 Cosine Annealing
  Gradient clipping         max_norm = 1.0
  Mixed precision           CUDA AMP
  Class weights             N1 = 2×, REM = 2×
  Supervision               All-position

The final benchmark uses the canonical subject-level splits and
aggregates results across three seeds.

------------------------------------------------------------------------

## 6. Dataset

### Sleep-EDF Expanded

-   **100 subjects downloaded**
-   **8 wake-only subjects excluded**
-   **92 subjects in the final eligible cohort**
-   Channels: Fpz-Cz, Pz-Oz, EOG, EMG
-   Sampling rate: 100 Hz
-   Epoch length: 30 seconds
-   Five AASM-aligned classes: Wake, N1, N2, N3, REM

### Approximate Class Distribution

  Stage     Approx. Share
  ------- ---------------
  Wake              \~68%
  N1               \~4.6%
  N2                \~17%
  N3                 \~6%
  REM                \~6%

The strong imbalance makes macro-level metrics particularly important
when evaluating sleep-stage classification.

------------------------------------------------------------------------

## 7. Reproducibility and Evaluation Integrity

The benchmark is designed around subject-level separation to prevent
subject leakage.

The repository provides:

-   Canonical 92-subject fold assignments
-   Multi-seed aggregation
-   Dataset manifest hashing
-   Checkpoint hashing
-   Configuration hashing
-   Git commit tracking
-   PyTorch/CUDA version tracking
-   GPU metadata
-   Complete training hyperparameters
-   Fold-level and per-class evaluation outputs

The authoritative adaptation results are stored under:

``` text
results/100_subject_adaptation/final/
```

Key files include:

``` text
aggregate_metrics.json
overall_comparison.csv
per_class_comparison.csv
FINAL_ADAPTATION_RESULTS.md
```

------------------------------------------------------------------------

## 8. Edge-Deployment Characteristics

The final model contains **99,477 parameters**, approximately **400 KB
when serialized**, making it substantially smaller than conventional
deep sleep-staging networks.

The architecture is designed around:

-   Depthwise-separable convolutions
-   Compact parametric spectral features
-   A small recurrent temporal module
-   300-second temporal context
-   Four-channel physiological input
-   CPU/edge-oriented inference

The model is therefore suitable as a research prototype for deployment
on resource-constrained systems such as wearables and embedded
sleep-monitoring devices.

Actual latency, memory consumption, and energy usage should be measured
on the target hardware before making production deployment claims.

------------------------------------------------------------------------

## 9. Limitations

1.  **N1 remains difficult.** Its F1 is substantially below the other
    stages.
2.  **Class imbalance is significant**, so accuracy alone is
    insufficient for evaluation.
3.  The final cohort contains **92 eligible subjects**, not 100; the
    other 8 downloaded records are wake-only and excluded.
4.  Sleep-EDF is a single dataset; cross-dataset generalization remains
    to be established.
5.  The benchmark is subject-level, but external validation on an
    independent cohort would provide stronger evidence of
    generalization.
6.  The model is compact, but deployment claims should be supported with
    measurements on the intended edge hardware.
7.  Performance should not be interpreted as clinical-grade sleep
    staging without independent clinical validation.

------------------------------------------------------------------------

## 10. Conclusion

NeuroSleep demonstrates that a compact **99,477-parameter** model can
perform five-stage sleep staging from four-channel PSG while maintaining
a 300-second temporal context.

Across a **92-subject eligible cohort**, evaluated using **10-fold
subject-level cross-validation across three seeds**, the final
full-fine-tuned model achieves:

> **87.7% ± 2.7% accuracy, Cohen's κ = 0.763 ± 0.043, and Macro F1 =
> 0.730 ± 0.037.**

The strongest performance is observed for Wake, N2, N3, and REM, while
N1 remains the principal challenge.

The adaptation study further shows that **full fine-tuning gives the
best overall result**, whereas LoRA offers a compelling
parameter-efficient alternative with only **1.43% of the model
parameters trainable**.

Overall, the results support NeuroSleep as a compact research
architecture for subject-level sleep-stage classification and a
promising foundation for future edge and wearable deployment studies.
