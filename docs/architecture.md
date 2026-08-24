# Architecture — Neuromorphic Sleep Stage Scoring

## System Overview

The system classifies 30-second sleep epochs into five AASM stages (Wake, N1, N2, N3, REM) from multi-channel polysomnography (PSG) signals. The architecture is designed for **edge deployment** with fewer than 100K trainable parameters.

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: 10 × 30s PSG Epochs                  │
│                    Shape: [B, 10, 4, 3000]                      │
│                    Channels: EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-RESOLUTION STEM (2 parallel branches)        │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ Short Receptive     │    │ Long Receptive      │            │
│  │ Field Branch        │    │ Field Branch        │            │
│  │ Kernel: 25          │    │ Kernel: 100         │            │
│  │ Stride: 6           │    │ Stride: 25          │            │
│  │ Output: 10 channels │    │ Output: 10 channels │            │
│  └──────────┬──────────┘    └──────────┬──────────┘            │
│             │                          │                        │
│             └──────────┬───────────────┘                        │
│                        ▼                                        │
│              Concatenate → [B×T, 20, L]                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           DEPTHWISE-SEPARABLE CONVOLUTION BLOCK                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Block 1: DWSep(20→32, k=5, s=2)                        │   │
│  │   Depthwise Conv (groups=20) → Pointwise 1×1 → BN → ReLU6│   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Block 2: DWSep(32→32, k=5, s=2)                        │   │
│  │   Depthwise Conv (groups=32) → Pointwise 1×1 → BN → ReLU6│   │
│  └─────────────────────────────────────────────────────────┘   │
│                        │                                        │
│                        ▼                                        │
│              AdaptiveAvgPool1d(1) → [B×T, 32]                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           PARAMETRIC GABOR FEATURE EXTRACTION BLOCK             │
│                                                                 │
│  Input: raw 4-channel signal [B×T, 4, 3000]                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 8 learnable Gabor filters                               │   │
│  │ - Center frequencies: 0.5–30 Hz (normalized by fs)      │   │
│  │ - Bandwidths: learnable                                 │   │
│  │ - Kernel size: 51 samples                               │   │
│  │                                                         │   │
│  │ Each filter: g(t) = exp(-t²/2σ²) × cos(2πf₀t)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                        │                                        │
│                        ▼                                        │
│  Conv1d per channel → AdaptiveAvgPool1d(1)                     │
│  → Linear(4×8 → 32) → ReLU6                                   │
│  → Output: [B×T, 32]                                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE FUSION                                      │
│                                                                 │
│  CNN features:    [B×T, 32]                                     │
│  Gabor features:  [B×T, 32]                                     │
│                        │                                        │
│                        ▼                                        │
│  Concatenate → [B×T, 64]                                        │
│  Reshape → [B, T, 64]                                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              2-LAYER GRU TEMPORAL MODELING                       │
│                                                                 │
│  Input: [B, 10, 64] (10 epochs × 64 features)                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ GRU Layer 1: 64 → 64 hidden units                       │   │
│  │ GRU Layer 2: 64 → 64 hidden units                       │   │
│  │                                                         │   │
│  │ Models temporal dependencies across 300s context         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                        │                                        │
│                        ▼                                        │
│  Output: [B, 10, 64]                                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              CLASSIFICATION HEAD                                 │
│                                                                 │
│  Linear(64 → 5)                                                 │
│                        │                                        │
│                        ▼                                        │
│  Output: [B, 10, 5] logits                                     │
│  Softmax → per-epoch class probabilities                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Multi-Resolution Stem

**Purpose:** Capture both short and long temporal patterns in the physiological signal.

| Property | Short Branch | Long Branch |
|----------|-------------|-------------|
| Kernel size | 25 samples (250ms) | 100 samples (1s) |
| Stride | 6 | 25 |
| Output channels | 10 | 10 |
| Total output | 20 channels | |

**Why multi-resolution:**
- Sleep-related waveform patterns occur at different timescales
- Short kernels capture local morphology (spindles, K-complexes)
- Long kernels capture slower oscillatory structure (delta waves)
- Combining them improves representation diversity

### 2. Depthwise-Separable Convolution

**Purpose:** Reduce computational cost while maintaining feature extraction capability.

**Standard Convolution:**
```
Parameters = C_in × C_out × K
```

**Depthwise-Separable Convolution:**
```
Parameters = (C_in × K) + (C_in × C_out)
           = C_in × (K + C_out)
```

**Example:**
- Standard: 20 × 32 × 5 = 3,200 parameters
- Depthwise-separable: (20 × 5) + (20 × 32) = 740 parameters
- **Reduction: 77%**

| Layer | Input | Output | Kernel | Stride | Parameters |
|-------|-------|--------|--------|--------|------------|
| DWSep 1 | 20 | 32 | 5 | 2 | 740 |
| DWSep 2 | 32 | 32 | 5 | 2 | 1,120 |
| **Total** | | | | | **1,860** |

### 3. Parametric Gabor Feature Extraction Block

**Purpose:** Learn frequency-localized patterns using learnable Gabor-like filters.

**Gabor Filter Definition:**
```
g(t) = exp(-t² / (2σ²)) × cos(2π × f₀ × t)
```

Where:
- `f₀` = center frequency (learnable, initialized 0.5–30 Hz)
- `σ` = bandwidth (learnable)
- `t` = time index

**Why Gabor:**
- Sleep stages are characterized by different oscillatory content
- Gabor filters provide joint time-frequency localization
- Compact parameterization (8 filters × 2 params = 16 learnable params)

| Property | Value |
|----------|-------|
| Number of filters | 8 |
| Kernel size | 51 samples |
| Frequency range | 0.5–30 Hz |
| Output per channel | 8 features |
| Total output | 4 × 8 = 32 features |

### 4. GRU Temporal Modeling

**Purpose:** Model dependencies across consecutive sleep epochs.

**Why GRU over LSTM:**
- Fewer parameters (2 gates vs 3 gates)
- Comparable performance on sequential sleep data
- Better suited for edge deployment

| Property | Value |
|----------|-------|
| Layers | 2 |
| Hidden size | 64 |
| Input size | 64 |
| Parameters | ~16,500 |
| Context window | 10 epochs × 30s = 300s |

---

## Complete Parameter Count

| Component | Parameters |
|-----------|------------|
| Multi-Resolution Stem (DWSep) | 1,860 |
| Encoder DWSep layers | 1,860 |
| Gabor FEB (filters + projection) | 1,344 |
| CNN pooling + projection | ~2,000 |
| GRU (2 layers) | ~16,500 |
| Classification head | 320 |
| BatchNorm parameters | ~500 |
| **Total** | **~99,477** |

---

## Input/Output Specification

### Input
- **Shape:** `[batch, 10, 4, 3000]`
- **Description:** 10 consecutive 30-second epochs, 4 channels, 100 Hz sampling
- **Channels:** EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental
- **Preprocessing:** 0.5–35 Hz bandpass → 50 Hz notch → z-score normalization

### Output
- **Shape:** `[batch, 10, 5]`
- **Description:** Per-epoch logits for 5 sleep stages
- **Classes:** Wake (0), N1 (1), N2 (2), N3 (3), REM (4)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Sequence length | 10 |
| Sequence stride | 5 |
| Training epochs | 20 |
| Gradient clipping | 1.0 |
| Scheduler | Cosine with warmup |
| Warmup fraction | 10% |
| Checkpoint metric | Validation Cohen's κ |

---

## Knowledge Distillation

The student is trained using a teacher-guided objective:

```
L_total = α_ce × L_CE + α_kl × L_KL + α_feat × L_Feature
```

| Component | Weight | Description |
|-----------|--------|-------------|
| L_CE | 1.0 | Hard-label cross-entropy |
| L_KL | 1.0 | Softened teacher-student KL divergence |
| L_Feature | 0.5 | MSE feature alignment |
| Temperature | 4.0 | Softmax temperature for distillation |

**Note:** The teacher exists only during training. The final checkpoint is the student alone.

---

## Edge Deployment Considerations

| Property | Value | Deployment Impact |
|----------|-------|-------------------|
| Parameters | 99,477 | < 400 KB at FP32 |
| Input size | 30,000 samples | 120 KB per epoch |
| CPU latency | 8.5 ms/batch | Real-time capable |
| Operations | Depthwise-separable + GRU | MCU-friendly |

### Quantization Readiness
- ReLU6 activations clamp to [0, 6], aiding INT8 quantization
- Depthwise-separable convolutions are quantization-friendly
- GRU layers can be quantized with minimal accuracy loss

---

## Design Decisions

### Why not classify each epoch independently?
Sleep stages are temporally dependent. Classifying an isolated epoch (especially near transitions like W→N1 or N1→N2) is ambiguous. The 10-epoch GRU context provides 5 minutes of surrounding information.

### Why EEG + EOG + EMG?
- **EEG:** Primary signal for brain activity patterns (delta, theta, spindles)
- **EOG:** Eye movement patterns, critical for REM detection
- **EMG:** Muscle activity, helps distinguish Wake from sleep stages

### Why not a larger model?
The project targets edge/MCU deployment. A sub-100K parameter model:
- Fits on microcontrollers with limited RAM
- Enables on-device inference without cloud connectivity
- Demonstrates that compact models can achieve competitive accuracy (87.34%)

---

## LoRA Adaptation

The base model supports parameter-efficient adaptation via LoRA (Low-Rank Adaptation).

### LoRA Configuration

| Property | Value |
|----------|-------|
| Target module | `head` (Linear 64→5) |
| Rank | 8 |
| Alpha | 16 |
| Scaling | 2.0 |
| Dropout | 0.05 |
| Trainable parameters | 552 (0.55%) |

### How LoRA Works

```
Input x → frozen head.original → h_orig
                ↓
           x @ lora_A.T → x @ lora_A.T @ lora_B.T × scaling → Δh
                ↓
           h_orig + Δh → output
```

The base model weights are frozen. Only the low-rank matrices A (8×64) and B (5×8) are trained.

### 4-Fold CV Results

| Method | Trainable | κ | Macro F1 |
|---|---:|---:|---:|
| Frozen Base | 0 | 0.5001 ± 0.1329 | 0.4106 ± 0.0775 |
| Full Fine-Tuning | 99,477 | 0.8595 ± 0.0092 | 0.7379 ± 0.0390 |
| **LoRA r=8** | **552** | **0.8092 ± 0.0663** | **0.6464 ± 0.0603** |

LoRA r=8 achieves **94.2%** of full fine-tuning's κ with **0.55%** of the parameters.

---

## Architecture Variants (Historical)

The final architecture evolved through these stages:

| Variant | Parameters | Accuracy | Status |
|---------|-----------|----------|--------|
| Baseline Teacher | 303,789 | 79.12% | Historical |
| Improved Teacher | 193,197 | 79.28% | Historical |
| Baseline Student | 567,749 | 90.25% | Historical |
| **Improved Student** | **99,477** | **87.34%** | **Final** |

The Improved Student is the only model used for exhibition and deployment.

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
