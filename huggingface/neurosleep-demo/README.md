---
title: NeuroSleep Sleep Stage Scoring
emoji: "\U0001F4A4"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# NeuroSleep Demo

Interactive demo for five-stage sleep classification from polysomnography signals.

## Model

- **Architecture:** Improved Student (99,477 parameters)
- **Input:** 10 x 4 x 3000 (10 epochs, 4 channels, 3000 samples)
- **Output:** Wake, N1, N2, N3, REM
- **Accuracy:** 90.57% (κ 0.808) on the 15-subject exhibition holdout
  (deployed standalone checkpoint); the stricter person-level benchmark
  reports 87.30% ± 0.33% — see the source repository's `docs/RESULTS.md`

## Links

- **GitHub:** [neuromorphic-sleep-staging-pipeline](https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline)
- **Model:** [Hugging Face Model Hub](https://huggingface.co/shamiquekhan/neuromorphic-sleep-staging)
