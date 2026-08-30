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
          name: Sleep-EDF Expanded (92 subjects, PhysioNet)
          config: default
          split: test
          revision: main
        metrics:
          - type: accuracy
            value: 0.877
            name: Accuracy
            verified: false
          - type: cohen_kappa
            value: 0.763
            name: Cohen's Kappa
            verified: false
          - type: f1
            value: 0.730
            name: Macro F1
            verified: false
          - type: f1
            value: 0.890
            name: Weighted F1
            verified: false
widget:
  - src: https://huggingface.co/spaces/shamique/neurosleep-demo
    title: NeuroSleep Live Demo
    width: 600
    height: 400
---

# NeuroSleep — Light-Weight Sleep Stage Model

**99,477 parameters, 87.7% accuracy (κ=0.763) — small enough for edge/wearable deployment, scoring Wake/N1/N2/N3/REM from 4-channel PSG.**

> **Quick links:** [GitHub](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) · [Live Demo](https://huggingface.co/spaces/shamiquekhan/neurosleep-demo) · [Kaggle](https://www.kaggle.com/shamiquekhan/neurosleep-final)

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

## Evaluation (92 subjects, 10-fold subject-level CV, 3 seeds)

| Metric | Value |
|--------|-------|
| Accuracy | 87.7% ± 2.7% |
| Cohen's Kappa | 0.763 ± 0.043 |
| Macro F1 | 0.730 ± 0.037 |
| Weighted F1 | 89.0% ± 2.1% |

### Per-Class Performance (Full Fine-Tuning)

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.964 ± 0.016 | 0.995 | 0.936 |
| N1 | **0.445 ± 0.061** | 0.330 | 0.712 |
| N2 | 0.768 ± 0.041 | 0.878 | 0.687 |
| N3 | 0.681 ± 0.114 | 0.562 | 0.896 |
| REM | 0.771 ± 0.078 | 0.772 | 0.785 |

> **Honest assessment:** Overall accuracy (87.7%) is strong with balanced performance across all five stages. N1 is the most challenging stage (F1=0.445) due to its transitional nature and low prevalence (~4.6% of epochs).

## Preprocessing

The model expects preprocessed data:

1. **Bandpass filter:** 0.5–35 Hz
2. **Notch filter:** 50 Hz
3. **Normalization:** z-score per channel
4. **Epoching:** 30-second windows at 100 Hz

See the [source repo](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) for the full preprocessing pipeline.

## Training Details

- **Dataset:** Sleep-EDF Expanded (92 subjects, PhysioNet)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Epochs:** 20 (early stopping patience=5)
- **Class weights:** N1=2x, REM=2x
- **Supervision:** All-position (every epoch in 10-epoch window)
- **Gradient clipping:** max_norm=1.0
- **Mixed precision:** True (CUDA)
- **Seeds:** 42, 43, 44 (30 folds per method)

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

### LoRA vs Full Fine-Tuning (100-Subject Benchmark)

| Method | Trainable Params | Accuracy | Macro F1 |
|--------|-----------------|----------|----------|
| Frozen Base | 0 | 87.1% | 0.673 |
| LoRA CNN+Head | 1,448 | 83.6% | 0.674 |
| **Full Fine-Tuning** | **99,477** | **87.7%** | **0.730** |

LoRA CNN+Head uses **68.7× fewer trainable parameters** while retaining **95.4% of full FT accuracy**.

## Intended Use

- Research and educational sleep-stage classification
- Benchmarking and comparison with other sleep staging methods
- Edge deployment on resource-constrained devices (MCUs, wearables)
- Transfer learning via LoRA for new sleep datasets

## Limitations

- **Not clinically validated** — do not use for diagnosis or clinical decision-making
- N1 classification is challenging (F1=0.445) due to brief, transitional light sleep
- Trained on Sleep-EDF Expanded (92 subjects); generalizability should be validated
- Requires 4-channel PSG (Fpz-Cz, Pz-Oz, EOG, EMG) — single-channel EEG not supported
- Class distribution is Wake-dominant (~68%) from untrimmed recordings

## Resources

| Resource | Link |
|----------|------|
| **Source Code** | [GitHub](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline) |
| **Live Demo** | [Hugging Face Space](https://huggingface.co/spaces/shamique/neurosleep-demo) |
| **Reproduce** | [Kaggle Notebook](https://www.kaggle.com/shamiquekhan/neurosleep-final) |
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
