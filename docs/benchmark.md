# Benchmark: ImprovedStudent Lightweight Sleep-Staging Model

**Project:** `shamiquekhan/neuromorphic-sleep-staging-pipeline`\
**Benchmark date:** 2026-10-01\
**Model:** `ImprovedStudent`

## 1. Executive summary

`ImprovedStudent` is a compact multimodal sleep-staging model with
**99,477 trainable parameters**. It combines:

-   4-channel PSG input: Fpz-Cz EEG, Pz-Oz EEG, EOG, EMG
-   10 × 30-second epochs = **300 seconds of context**
-   multi-resolution temporal convolution
-   depthwise-separable convolution
-   8 learnable Gabor filters
-   feature fusion
-   a 2-layer GRU with hidden size 64
-   five AASM classes: Wake, N1, N2, N3, REM

The project's final holdout run reports:

  Metric          ImprovedStudent
  ------------- -----------------
  Parameters           **99,477**
  Accuracy             **90.50%**
  Macro-F1             **78.89%**
  Weighted-F1          **90.89%**
  Cohen's κ            **0.8284**

The repository also reports a 10-fold person-level benchmark of
approximately **87.7 ± 2.7% accuracy**, **73.0 ± 3.7% Macro-F1**, and
**κ = 0.763 ± 0.043**.

**Important:** published cross-paper results below are contextual, not
apples-to-apples. Dataset, modality, context length, split,
preprocessing and evaluation protocols differ.

------------------------------------------------------------------------

## 2. Core benchmark table

  ---------------------------------------------------------------------------------------------------------------------
  Model                   Modality            Parameters   Published/Project       Macro-F1              κ Comparison
                                                                    Accuracy                               status
  ----------------------- ------------ ----------------- ------------------- -------------- -------------- ------------
  **ImprovedStudent**     4-ch PSG             **99.5K**        **90.50%**\*   **78.89%**\*   **0.8284**\* Project
                                                                                                           holdout

  **ULW-SleepNet**        Multimodal           **13.3K**               86.9%          80.7%           0.82 Contextual
                          PSG                                                                              

  **GamSleepNet**         Single EEG          **30.86K**              87.86%            ---            --- Contextual

  **Micro SleepNet**      Single EEG           **48.2K**               82.8%          75.3%           0.76 Contextual

  **PicoSleepNet**        Single EEG /   **14.0--25.8K**               83.5%          75.2%            --- Contextual
                          SNN                                                                              

  **EfficientSleepNet**   Single EEG           **83.8K**               84.4%          78.1%         \~0.79 Contextual
  ---------------------------------------------------------------------------------------------------------------------

`*` The project holdout result is not directly comparable to the
published rows.

------------------------------------------------------------------------

## 3. Architecture

``` text
4-channel PSG
     │
     ▼
Multi-resolution temporal stem
     │
     ▼
Depthwise-separable CNN
     │
     ├──────────────► Learnable Gabor frequency branch
     │
     └─────────────────────────┐
                               ▼
                         Feature fusion
                               │
                         272-D features
                               │
                               ▼
                       2-layer GRU (64)
                               │
                               ▼
                           Classifier
                               │
                               ▼
                     W / N1 / N2 / N3 / REM
```

Approximate parameter allocation:

  Component               Parameters         Share
  --------------------- ------------ -------------
  Stem                         7,232        \~7.3%
  CNN encoder                  1,904        \~1.9%
  Gabor branch                   160        \~0.2%
  GRU                         89,856   **\~90.3%**
  Classification head            325        \~0.3%
  **Total**               **99,477**      **100%**

**Implication:** roughly 90% of the parameter budget is in the GRU. This
makes the temporal module the most important target for future
efficiency experiments.

------------------------------------------------------------------------

## 4. Data augmentation benchmark

Augmentation is applied **only to training tensors**, after
normalization.

The training pipeline uses:

-   amplitude scaling
-   Gaussian noise
-   temporal masking
-   channel dropout

Validation and test data remain unaugmented.

This is **on-the-fly stochastic augmentation**, not a fixed
multiplication of the dataset. The same training window can receive
different perturbations on different training passes.

Do **not** interpret the experiment identifier `AUG30` as "30 augmented
copies per sample."

------------------------------------------------------------------------

## 5. Final class-level performance

  Sleep stage            F1
  ------------- -----------
  Wake            **0.981**
  N1              **0.535**
  N2              **0.831**
  N3              **0.762**
  REM             **0.835**

The weakest class is **N1**. This is consistent with the broader
literature: N1 is difficult because its boundaries are less distinct and
it frequently occurs around transitions.

The benchmark should therefore report **N1 F1 separately**, rather than
relying only on overall accuracy.

------------------------------------------------------------------------

## 6. ULW-SleepNet

ULW-SleepNet is the most important published multimodal comparator.

Reported configuration:

-   13.3K parameters
-   7.89M FLOPs
-   multimodal PSG
-   Sleep-EDF-20 accuracy: 86.9%
-   Macro-F1: 80.7%
-   κ: 0.82

Parameter ratio:

``` text
99,477 / 13,300 ≈ 7.48×
```

Therefore `ImprovedStudent` uses roughly 7.5× as many parameters.

This does **not** establish that ULW-SleepNet is more efficient or that
ImprovedStudent is more accurate in a fair comparison. A controlled
reimplementation under the project's exact input, split and training
protocol is required.

Source: Wang et al., *ULW-SleepNet*, 2026.\
https://arxiv.org/abs/2602.23852

------------------------------------------------------------------------

## 7. GamSleepNet

GamSleepNet is particularly relevant because it combines **learnable
Gabor features with Mamba**.

Reported:

-   30.86K parameters
-   87.86% accuracy on Sleep-EDF
-   single-channel EEG

The architectural comparison is:

``` text
ImprovedStudent:
Gabor → GRU

GamSleepNet:
Gabor → Mamba
```

This creates a strong future experiment:

``` text
same feature extractor
        │
        ├── GRU
        │
        └── lightweight Mamba/SSM
```

with both constrained to approximately the same parameter budget.

Source: Wei et al., *Lightweight ML-Based Automatic Sleep Staging
Framework with Constrained CNN and Mamba for Small-Sample EEG Datasets*,
2026.\
https://arxiv.org/abs/2607.04934

------------------------------------------------------------------------

## 8. Micro SleepNet

Micro SleepNet is a useful edge-device comparator.

Reported:

-   \~48.2K parameters
-   \~48.95M FLOPs
-   Sleep-EDF-20 accuracy: 82.8%
-   Macro-F1: 75.3%
-   κ: 0.76
-   30-second single-channel EEG input

The paper reports approximately 2.8 ms inference per EEG epoch on a
Qualcomm Snapdragon 865 and approximately 100 KB memory footprint.

However, it processes a 30-second single-channel epoch, while
ImprovedStudent processes a 300-second four-channel context. Latency and
FLOPs therefore require a common benchmark harness before direct
comparison.

Source: Liu et al., *Micro SleepNet*, Frontiers in Neuroscience, 2023.\
https://pmc.ncbi.nlm.nih.gov/articles/PMC10416229/

------------------------------------------------------------------------

## 9. PicoSleepNet

PicoSleepNet is the key comparator for the project's neuromorphic
direction.

It uses:

``` text
single-channel EEG
      ↓
level-crossing sampling
      ↓
event/spike representation
      ↓
sparse recurrent SNN
      ↓
sleep stage
```

Reported:

-   14.0--25.8K parameters
-   681.4--842.0K operations
-   Sleep-EDF-20 accuracy: 83.5%
-   Sleep-EDF-20 Macro-F1: 75.2%

**Critical distinction:** ImprovedStudent is currently a conventional
ANN (`CNN + Gabor + GRU`), not an SNN. Therefore the current project
should not claim that the model itself demonstrates neuromorphic
computing.

A more defensible description is:

> A lightweight, edge-oriented sleep-staging architecture with a future
> neuromorphic deployment direction.

Source: Liu et al., *PicoSleepNet*, IEEE Journal of Biomedical and
Health Informatics, 2026.\
https://pubmed.ncbi.nlm.nih.gov/40853814/

------------------------------------------------------------------------

## 10. Parameter-efficiency context

Approximate model sizes:

``` text
ULW-SleepNet       13.3K
PicoSleepNet       14–25.8K
GamSleepNet        30.86K
Micro SleepNet     48.2K
EfficientSleepNet  83.8K
ImprovedStudent    99.5K
```

Therefore:

> ImprovedStudent is lightweight, but it is not the smallest model in
> the current comparison set.

Its justification is the combination of:

-   multimodal PSG
-   300-second context
-   multi-scale temporal features
-   explicit frequency-aware features
-   recurrent temporal modelling

rather than minimum parameter count.

------------------------------------------------------------------------

## 11. Required parameter-matched benchmark

The strongest controlled baseline is a plain CNN with approximately 100K
parameters.

All of the following must be identical:

-   subjects
-   train/validation/test split
-   input channels
-   sampling rate
-   context length
-   normalization
-   augmentation
-   class weighting
-   optimizer
-   learning rate
-   scheduler
-   stopping rule
-   evaluation code
-   random seeds

Only the architecture should change.

Recommended baseline:

``` text
4-channel PSG
      ↓
plain temporal CNN
      ↓
global pooling
      ↓
5-class classifier
```

Target:

``` text
≈99K parameters
```

------------------------------------------------------------------------

## 12. Required ablation study

### A0 --- Plain CNN

Tests the baseline representation.

### A1 --- Multi-resolution CNN

Tests whether multiple temporal scales improve performance.

### A2 --- Depthwise-separable CNN

Tests the efficiency/performance trade-off.

### A3 --- Add Gabor

``` text
A2 + Gabor
```

Tests whether explicit frequency-aware features improve staging.

### A4 --- Add GRU

``` text
A3 + GRU
```

This is the complete ImprovedStudent.

### A5 --- Replace GRU

``` text
A3 + lightweight Mamba/SSM
```

Tests modern state-space temporal modelling under the same budget.

### A6 --- EEG only

Measures the contribution of multimodal PSG.

### A7 --- EEG + EOG

Measures incremental multimodal benefit.

### A8 --- Full four-channel PSG

Complete configuration.

------------------------------------------------------------------------

## 13. Context-length benchmark

Run:

  Configuration       Context
  --------------- -----------
  C1                     30 s
  C2                     60 s
  C3                    120 s
  C4                    180 s
  C5                **300 s**

Measure:

-   Accuracy
-   Macro-F1
-   κ
-   N1 F1
-   parameters
-   FLOPs
-   latency
-   RAM

This determines whether the 300-second context actually produces a
meaningful gain.

------------------------------------------------------------------------

## 14. Channel ablation

Run:

``` text
1-channel:
Fpz-Cz

2-channel:
Fpz-Cz + Pz-Oz

3-channel:
EEG + EOG

4-channel:
EEG + EEG + EOG + EMG
```

The key question is:

> How much performance is gained by each additional physiological
> modality?

This is especially important because several published lightweight
models use only one EEG channel.

------------------------------------------------------------------------

## 15. Efficiency benchmark

Every model should report:

### Model complexity

-   trainable parameters
-   model file size
-   FLOPs/MACs

### Runtime

-   batch-1 CPU latency
-   batch-1 GPU latency
-   P50
-   P90
-   P95
-   P99
-   throughput

### Memory

-   peak RAM
-   inference memory

### Deployment

-   FP32
-   FP16
-   INT8 if supported
-   ONNX/TorchScript runtime if applicable

------------------------------------------------------------------------

## 16. Standard latency protocol

Use:

``` text
batch = 1
fixed input shape
100 warm-up iterations
1000 measured iterations
```

Report:

``` text
mean
median
P50
P90
P95
P99
standard deviation
```

Always record:

-   CPU/GPU model
-   RAM
-   OS
-   PyTorch version
-   CUDA version
-   thread count
-   precision
-   compiler/runtime

A latency number without hardware information is not reproducible.

------------------------------------------------------------------------

## 17. FLOPs protocol

Use one profiler for every model:

-   `fvcore`
-   `ptflops`
-   or `THOP`

Use exactly the same input shape and counting convention.

For the project:

``` text
[batch=1, channels=4, samples=30,000]
```

Do not directly compare FLOPs reported by unrelated papers unless their
counting methods and input shapes match.

------------------------------------------------------------------------

## 18. Statistical protocol

Minimum:

``` text
3 independent random seeds
```

Report:

``` text
mean ± standard deviation
```

Preferred:

``` text
5- or 10-fold subject-level cross-validation
```

Report:

``` text
mean ± SD
95% CI
```

Keep the final test set untouched while selecting architectures and
hyperparameters.

------------------------------------------------------------------------

## 19. Statistical significance

For paired model comparisons, use subject/fold-level paired tests, such
as:

-   paired bootstrap over subjects,
-   Wilcoxon signed-rank test over folds,
-   permutation test.

Do not treat individual epochs as independent subjects when testing
statistical significance.

------------------------------------------------------------------------

## 20. Recommended final benchmark table

  ------------------------------------------------------------------------------------------------------------------------
  Model                      Params    FLOPs       Accuracy       Macro-F1              κ         N1 F1       P95      RAM
                                                                                                          Latency 
  --------------------- ----------- -------- -------------- -------------- -------------- ------------- --------- --------
  Plain CNN                  \~100K      TBD            TBD            TBD            TBD           TBD       TBD      TBD

  Multi-res CNN                 TBD      TBD            TBD            TBD            TBD           TBD       TBD      TBD

  \+ Depthwise                  TBD      TBD            TBD            TBD            TBD           TBD       TBD      TBD

  \+ Gabor                      TBD      TBD            TBD            TBD            TBD           TBD       TBD      TBD

  **ImprovedStudent**     **99.5K**      TBD   **90.50%**\*   **78.89%**\*   **0.8284**\*   **53.5%**\*       TBD      TBD

  CNN + Mamba                \~100K      TBD            TBD            TBD            TBD           TBD       TBD      TBD

  SNN                        \~100K      TBD            TBD            TBD            TBD           TBD       TBD      TBD
  ------------------------------------------------------------------------------------------------------------------------

`*` Current project holdout; replace with controlled mean ± SD for
publication.

------------------------------------------------------------------------

## 21. Recommended plots

1.  **Accuracy vs parameters**
2.  **Macro-F1 vs parameters**
3.  **Accuracy vs FLOPs**
4.  **N1 F1 vs parameters**
5.  **Accuracy vs latency**
6.  **Pareto frontier: Macro-F1 vs FLOPs**
7.  **Ablation waterfall**
8.  **Context-length vs Macro-F1**
9.  **Channel-count vs Macro-F1**

Use log scale for parameter/FLOP axes where appropriate.

------------------------------------------------------------------------

## 22. Downstream sleep-architecture benchmark

Epoch-level classification is not the only useful metric.

Eventually evaluate:

-   total sleep time
-   sleep efficiency
-   sleep latency
-   REM latency
-   Wake After Sleep Onset
-   N1/N2/N3 duration
-   stage-bout duration
-   transition counts
-   hypnogram fragmentation

The pipeline should therefore be evaluated as:

``` text
PSG
 ↓
sleep-stage sequence
 ↓
hypnogram
 ↓
sleep-architecture metrics
```

not only:

``` text
PSG → accuracy
```

------------------------------------------------------------------------

## 23. Research claims: what is currently justified?

### Supported

> ImprovedStudent is a sub-100K-parameter multimodal sleep-staging
> model.

### Supported

> It combines multi-resolution temporal convolution, depthwise-separable
> processing, learnable Gabor features and GRU-based temporal modelling.

### Supported

> The final holdout experiment achieved approximately 90.5% accuracy and
> κ=0.8284.

### Reasonable with qualification

> The architecture is suitable for edge-oriented sleep-staging research.

### Not established yet

> It is state-of-the-art.

### Not established yet

> It is the most efficient sleep-staging model.

### Not established yet

> Gabor features cause the performance improvement.

### Not established yet

> The model is neuromorphic.

The last three require controlled experiments.

------------------------------------------------------------------------

## 24. Highest-priority experiments

If compute is limited, prioritize:

1.  **\~100K parameter-matched CNN**
2.  **Remove Gabor**
3.  **Remove GRU**
4.  **30 s vs 300 s context**
5.  **1-channel vs 4-channel**
6.  **3 seeds**
7.  **CPU latency + RAM**
8.  **GRU vs lightweight Mamba/SSM**
9.  **ANN vs SNN**

The objective is not simply to obtain a larger accuracy number.

The objective is to establish **why the architecture works and what it
costs**.

------------------------------------------------------------------------

## 25. Final scientific positioning

A defensible description of the current work is:

> **A compact multimodal sleep-staging model using multi-scale temporal
> convolution, learnable Gabor frequency features and recurrent temporal
> context under a \~100K parameter budget.**

The most important next contribution is a controlled benchmark showing
the incremental value of:

``` text
multi-resolution CNN
        +
depthwise separable convolution
        +
Gabor frequency features
        +
300-second context
        +
GRU temporal modelling
```

This is substantially stronger scientifically than claiming novelty from
parameter count alone.

------------------------------------------------------------------------

## 26. References

1.  Wang et al., *ULW-SleepNet: An Ultra-Lightweight Network for
    Multimodal Sleep Stage Scoring*, 2026.\
    https://arxiv.org/abs/2602.23852

2.  Wei et al., *Lightweight ML-Based Automatic Sleep Staging Framework
    with Constrained CNN and Mamba for Small-Sample EEG Datasets*,
    2026.\
    https://arxiv.org/abs/2607.04934

3.  Liu et al., *Micro SleepNet: efficient deep learning model for
    mobile terminal real-time sleep staging*, Frontiers in Neuroscience,
    2023.\
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10416229/

4.  Liu et al., *PicoSleepNet: An Ultra Lightweight Sleep Stage
    Classification by Spike Neural Network Using Single-Channel EEG
    Signal*, IEEE JBHI, 2026.\
    https://pubmed.ncbi.nlm.nih.gov/40853814/

5.  PhysioNet, *Sleep-EDF Database*.\
    https://physionet.org/content/sleep-edfx/

6.  Project repository.\
    https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline

------------------------------------------------------------------------

## 27. Benchmark status

  Component                          Status
  ---------------------------------- ---------------
  Final model parameter count        **Available**
  Final holdout metrics              **Available**
  Subject-level CV                   **Available**
  Published lightweight comparison   **Available**
  Parameter-matched CNN              **TODO**
  Gabor ablation                     **TODO**
  GRU ablation                       **TODO**
  Context-length ablation            **TODO**
  Channel ablation                   **TODO**
  3-seed benchmark                   **TODO**
  Standardized FLOPs                 **TODO**
  Hardware latency                   **TODO**
  RAM benchmark                      **TODO**
  ANN vs SNN                         **TODO**
