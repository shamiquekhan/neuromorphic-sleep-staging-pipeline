---
license: cc-by-4.0
library_name: pytorch
tags:
  - sleep-staging
  - polysomnography
  - eeg
  - eog
  - emg
  - sleep-edf
  - biomedical
  - neuromorphic
---

# NeuroSleep — Improved Student

## Model Summary

NeuroSleep is a compact PyTorch model for five-stage sleep-stage classification from four-channel polysomnography (PSG) signals. The model processes 300 seconds of context (10 x 30-second epochs) and classifies each epoch into Wake, N1, N2, N3, or REM.

## Architecture

- **Name:** Improved Student
- **Parameters:** 99,477
- **Input:** `[batch, 10, 4, 3000]` (10 epochs, 4 channels, 3000 samples each)
- **Output:** `[batch, 10, 5]` (5 sleep stage probabilities per epoch)
- **Context:** 300 seconds (10 x 30s epochs)

```
PSG Input (Fpz-Cz, Pz-Oz, EOG, EMG)
      ↓
Multi-Resolution Stem
      ↓
Depthwise-Separable CNN
      ↓
Parametric Gabor Feature Extraction
      ↓
2-Layer GRU (hidden=64)
      ↓
5-Class Softmax
```

## Input Format

- **Sampling rate:** 100 Hz
- **Channels:** Fpz-Cz, Pz-Oz, EOG, EMG
- **Epoch length:** 30 seconds (3000 samples)
- **Sequence length:** 10 epochs
- **Shape:** `[batch, 10, 4, 3000]`

## Output Labels

| Index | Stage | Description |
|-------|-------|-------------|
| 0 | Wake | Awake state |
| 1 | N1 | Light sleep |
| 2 | N2 | Intermediate sleep |
| 3 | N3 | Deep sleep |
| 4 | REM | Rapid eye movement sleep |

## Training

- **Dataset:** Sleep-EDF Expanded (15 subjects, PhysioNet)
- **Method:** Full fine-tuning
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-2)
- **Epochs:** 15
- **Class weights:** N1=2x, REM=2x
- **Supervision:** All-position (every epoch in sequence)
- **Cross-validation:** 4-fold subject-level CV

## Evaluation

- **Accuracy:** 93.0% ± 1.0%
- **Cohen's Kappa:** 0.861 ± 0.027
- **Macro F1:** 0.794 ± 0.036
- **Weighted F1:** 0.935 ± 0.007

### Per-Class F1

| Stage | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Wake | 0.983 ± 0.011 | 1.000 | 0.967 |
| N1 | 0.682 ± 0.090 | 1.000 | 0.552 |
| N2 | 0.912 ± 0.044 | 1.000 | 0.848 |
| N3 | 0.958 ± 0.016 | 1.000 | 0.912 |
| REM | 0.966 ± 0.017 | 1.000 | 0.930 |

## Intended Use

Research and educational sleep-stage classification. This model is designed for:

- Automated sleep staging in research settings
- Educational demonstrations of deep learning for biomedical signals
- Benchmarking and comparison with other sleep staging methods

## Limitations

- This model is **not clinically validated** and should not be used for diagnosis or clinical decision-making
- N1 classification remains challenging (F1=0.682) due to the brief and transitional nature of light sleep
- Performance may vary across different PSG设备 and recording protocols
- Trained on Sleep-EDF Expanded; generalizability to other datasets should be validated

## Ethical Considerations

- This model is for research use only
- Should not be used as a substitute for expert sleep technologist scoring
- Clinical deployment requires rigorous validation and regulatory approval

## Reproducibility

```python
from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
path = hf_hub_download(
    repo_id="shamiquekhan/neuromorphic-sleep-staging",
    filename="student_full_finetuned.pt"
)

# Load model
model = torch.load(path, weights_only=False)
```

## Source

- **GitHub:** [neuromorphic-sleep-staging-pipeline](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline)
- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/shamiquekhan/neurosleep-demo)

## Citation

```bibtex
@project{neurosleep_2026,
  title={NeuroSleep: Neuromorphic Sleep Stage Scoring},
  author={Kaushik, P. and Vora, S. and Bhatt, S. and Khan, S. and Lone, A.J.},
  year={2026},
  institution={VIT Bhopal University}
}
```
