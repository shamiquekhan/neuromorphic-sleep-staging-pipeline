---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: other
tags:
  - sleep-staging
  - polysomnography
  - eeg
  - eog
  - emg
  - biosignal
  - biosignals
  - signal-processing
  - time-series
  - healthcare
  - medical
  - edge-ai
  - tinyml
  - on-device
  - gru
  - cnn
  - depthwise-separable-convolution
  - gabor-filter
  - sleep-edf
  - physionet
  - lightweight
  - low-parameter
  - lora
  - peft
  - parameter-efficient-fine-tuning
  - adapter
  - fine-tuning
  - pytorch
  - sleep
  - classification
  - 5-class
  - aasm
  - epoch-classification
  - wearable
  - iot
  - microcontroller
  - arm
  - cortex-m
datasets:
  - siamakz/sleep_edf_expanded
  - physionet/sleep-edf
model-index:
  - name: NeuroSleep Improved Student
    results:
      - task:
          type: other
          name: Sleep Stage Classification
        dataset:
          type: sleep-edf-expanded
          name: Sleep-EDF Expanded (52 persons / 92 records, PhysioNet)
          config: default
          split: test
          revision: main
        metrics:
          - type: accuracy
            value: 0.8730
            name: Accuracy
            verified: false
          - type: cohen_kappa
            value: 0.738
            name: Cohen's Kappa
            verified: false
          - type: f1
            value: 0.724
            name: Macro F1
            verified: false
          - type: f1
            value: 0.880
            name: Weighted F1
            verified: false
widget:
  - src: https://huggingface.co/spaces/shamique/neurosleep-demo
    title: NeuroSleep Live Demo
    width: 600
    height: 400
---

# NeuroSleep — Light-Weight Sleep Stage Model

**99,477 parameters, 90.57% accuracy (κ=0.808, macro-F1=0.749, exhibition 15-subject holdout) — small enough for edge/wearable deployment, scoring Wake/N1/N2/N3/REM from 4-channel PSG.**

> **Evidence note:** the headline numbers are the **standalone notebook
> pipeline** results (seed 42, 92-record / 52-person eligible cohort
> from 100 downloaded Sleep-EDF Expanded records, 70/15/15 subject-level
> split, all-position evaluation, stride=5; supervised class-weighted
> cross-entropy — no distillation). Under the stricter person-level
> 10-fold causal protocol (EXP-BENCH-PERSON, seeds 42/43/44) the same
> architecture scores **87.30% ± 0.33% accuracy / κ 0.738 ± 0.010** —
> the honest person-generalization estimate.

> **Quick links:** [GitHub](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) · [Live Demo](https://huggingface.co/spaces/shamiquekhan/neurosleep-demo)

A compact PyTorch model for five-stage sleep-stage classification from polysomnography signals. Processes 300 seconds of context (10 × 30-second epochs) and classifies each epoch into Wake, N1, N2, N3, or REM. Designed for edge deployment on resource-constrained devices.

## Quick Start

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Download checkpoint
path = hf_hub_download(
    repo_id="shamique/Light-Weight-Neuromorphic-Sleep-Stage-Model",
    filename="student_full_finetuned.safetensors",
)

# Load model (see source repo for ImprovedStudent class definition)
# https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline
from sleep_staging.models.improved_student import ImprovedStudent

model = ImprovedStudent()
model.load_state_dict(load_file(path, device="cpu"))
model.eval()

# Run inference on preprocessed PSG data
# Input: [batch, 10, 4, 3000] — 10 epochs, 4 channels, 3000 samples @ 100Hz
x = torch.randn(1, 10, 4, 3000)  # replace with real data

with torch.inference_mode():
    logits = model(x)              # [1, 10, 5]
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)   # [1, 10]

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
for i in range(10):
    print(f"Epoch {i}: {STAGE_NAMES[preds[0, i].item()]} ({probs[0, i, preds[0, i]].item():.2%})")
```

## Architecture

```
PSG Input (Fpz-Cz, Pz-Oz, EOG, EMG)  [B, 10, 4, 3000]
      ↓
Multi-Resolution Stem (2 parallel Conv1d branches)
      ↓
Depthwise-Separable CNN (2 blocks)
      ↓
Parametric Gabor Feature Extraction (8 learnable filters)
      ↓
2-Layer GRU (hidden=64, 300s context)
      ↓
Linear(64→5) + Softmax
      ↓
Wake / N1 / N2 / N3 / REM
```

| Module | Parameters |
|--------|------------|
| Stem (S+L) | 7,232 |
| Encoder (2 blocks) | 2,304 |
| Gabor FEB | 144 |
| GRU | 89,856 |
| Head | 325 |
| **Total** | **99,477** |

## Input Format

- **Sampling rate:** 100 Hz
- **Channels:** Fpz-Cz, Pz-Oz, EOG, EMG
- **Epoch length:** 30 seconds (3000 samples)
- **Sequence length:** 10 epochs
- **Shape:** `[batch, 10, 4, 3000]`
- **Preprocessing:** 0.5–35 Hz bandpass → 50 Hz notch → z-score normalization

## Output Labels

| Index | Stage | Description |
|-------|-------|-------------|
| 0 | Wake | Awake state |
| 1 | N1 | Light sleep |
| 2 | N2 | Intermediate sleep |
| 3 | N3 | Deep sleep |
| 4 | REM | Rapid eye movement sleep |

## Evaluation

### Person-Level Primary Benchmark (EXP-BENCH-PERSON, 30 folds / 3 seeds)

| Metric | Value |
|--------|-------|
| Accuracy | 87.30% ± 0.33% (95% CI [85.80, 88.79]%) |
| Cohen's Kappa | 0.738 ± 0.010 (95% CI [0.691, 0.785]) |
| Macro F1 | 0.724 ± 0.005 (95% CI [0.693, 0.755]) |
| Weighted F1 | 0.880 ± 0.003 (95% CI [0.860, 0.900]) |

| Stage | F1 |
|-------|-----|
| Wake | 0.960 ± 0.026 |
| N1 | **0.445 ± 0.087** |
| N2 | 0.733 ± 0.163 |
| N3 | 0.700 ± 0.112 |
| REM | 0.782 ± 0.116 |

### Standalone Notebook Run (deployed checkpoint, 15-subject holdout, seed 42)

| Metric | Value |
|--------|-------|
| Accuracy | 90.57% |
| Cohen's Kappa | 0.8080 |
| Macro F1 | 0.7490 |
| Weighted F1 | 0.9115 |
| F1 (Wake / N1 / N2 / N3 / REM) | 0.978 / 0.477 / 0.823 / 0.705 / 0.762 |

> **Honest assessment:** N1 is the most challenging stage (F1≈0.45–0.48)
> due to its transitional nature and low prevalence (~4.6% of epochs).
> The stricter person-level protocol is the reference for
> generalization; the standalone numbers show the same architecture on
> the fixed exhibition split.

## Preprocessing

The model expects preprocessed data:

1. **Bandpass filter:** 0.5–35 Hz
2. **Notch filter:** 50 Hz
3. **Normalization:** z-score per channel
4. **Epoching:** 30-second windows at 100 Hz

See the [source repo](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) for the full preprocessing pipeline.

## Training Details

- **Dataset:** Sleep-EDF Expanded — 92-record eligible cohort (52
  persons) from 100 downloaded records (8 wake-only excluded), PhysioNet
- **Initialization:** from scratch (random init) — supervised
  class-weighted cross-entropy, no distillation
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Epochs:** 20 (cosine schedule with 10% warmup, best-κ checkpointing)
- **Class weights:** log-balanced from the training partition
- **Supervision:** All-position (every epoch in 10-epoch window)
- **Gradient clipping:** max_norm=1.0
- **Seeds:** 42 (standalone run); 42/43/44 (primary benchmark)

## LoRA Adaptation (Parameter-Efficient Fine-Tuning)

The model supports **LoRA (Low-Rank Adaptation)** for efficient fine-tuning on new datasets without updating all 99K parameters.

### LoRA Configuration

| Property | Value |
|----------|-------|
| Target modules | `enc.0.pw`, `enc.1.pw`, `head` |
| Rank | 8 |
| Alpha | 16 |
| Scaling | 2.0 |
| Trainable params | 1,448 (1.43% of total) |

### Apply LoRA

```python
from sleep_staging.adaptation.lora import LoRAConfig, apply_lora

lora_config = LoRAConfig(
    rank=8,
    alpha=16,
    target_modules=["enc.0.pw", "enc.1.pw", "head"],
    dropout=0.05,
)

model = ImprovedStudent()
model.load_state_dict(load_file(ckpt_path, device="cpu"))
model = apply_lora(model, lora_config)
# trainable params: 1,448 || all params: 99,477 || trainable%: 1.43%
```

### LoRA vs Full Fine-Tuning (quarantined internal comparison)

> **Evidence status:** the adaptation study's base checkpoint was
> trained on 15 subjects that overlap the evaluation folds
> (contaminated; ~+2.5pp inflation). These numbers are retained for
> like-for-like internal comparison only and must not be cited as
> generalization results. See `docs/adaptation.md` in the source repo.

| Method | Trainable Params | Accuracy | Macro F1 |
|--------|-----------------|----------|----------|
| Frozen Base | 0 | 87.1% | 0.673 |
| LoRA CNN+Head | 1,448 | 83.6% | 0.674 |
| **Full Fine-Tuning** | **99,477** | **87.7%** | **0.730** |

**Honest reading:** the tested LoRA CNN+Head configuration trains
**68.7× fewer parameters** than full fine-tuning, but in this
(contaminated) run produced lower accuracy and substantially lower
macro-F1 (0.674 vs 0.730). Whether low-rank adaptation — including
adapting the GRU, which holds 90.3% of the parameter budget — can close
the gap is an open research question being re-run on a leak-free base
checkpoint.

## Intended Use

- Research and educational sleep-stage classification
- Benchmarking and comparison with other sleep staging methods
- Edge deployment on resource-constrained devices (MCUs, wearables)
- Transfer learning via LoRA for new sleep datasets

## Limitations

- **Not clinically validated** — do not use for diagnosis or clinical decision-making
- N1 classification is challenging (F1≈0.45–0.48) due to brief, transitional light sleep
- Trained on Sleep-EDF Expanded (92-record / 52-person cohort); cross-dataset generalizability should be validated
- Requires 4-channel PSG (Fpz-Cz, Pz-Oz, EOG, EMG) — single-channel EEG not supported
- Class distribution is Wake-dominant (~68%) from untrimmed recordings
- The architecture is a conventional differentiable CNN–GRU designed for edge-deployment constraints, not a spiking/neuromorphic network

## Resources

| Resource | Link |
|----------|------|
| **Source Code** | [GitHub](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) |
| **Live Demo** | [Hugging Face Space](https://huggingface.co/spaces/shamique/neurosleep-demo) |
| **Reproduce** | Notebooks 01→05 in the source repo (run in order) |
| **Model Weights** | This page |

## Citation

```bibtex
@project{neurosleep_2026,
  title={NeuroSleep: Light-Weight Sleep Stage Scoring},
  author={Kaushik, P. and Vora, S. and Bhatt, S. and Khan, S. and Lone, A.J.},
  year={2026},
  institution={VIT Bhopal University}
}
```

## Download Counting

Hugging Face counts downloads per unique file. For this model, the primary tracked file is `student_full_finetuned.safetensors`. Each HTTP request (GET or HEAD) to this file counts as one download. Clone operations that download all files are counted once per file.

For granular download analytics (unique users, CI/CD filtering), see [Publisher Analytics](https://huggingface.co/docs/hub/en/publisher-analytics).
