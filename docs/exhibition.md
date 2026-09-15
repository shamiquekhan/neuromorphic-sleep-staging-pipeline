# Exhibition Guide — VIT Bhopal University

## Exhibition Overview

| Property | Value |
|----------|-------|
| Project | Neuromorphic Sleep Stage Scoring |
| Venue | VIT Bhopal University |
| Duration | 3–5 minutes |
| Format | Live demonstration + poster |
| Final Model | Improved Student (99,477 parameters) |

---

## 3–5 Minute Demo Script

### Minute 1: Problem Statement (60 seconds)

**Script:**
> "Sleep staging is clinically important but requires expert analysis of multi-channel polysomnography recordings. Manual scoring takes 2–4 hours per study and requires specialized training. We built an automated system to classify five sleep stages from physiological signals."

**Show:**
- Raw PSG waveform (EEG, EOG, EMG)
- Five sleep stages (Wake, N1, N2, N3, REM)
- Expert scoring timeline

**Key points:**
- Manual scoring is time-consuming
- Expert training is required
- Automation can assist clinicians

---

### Minute 2: Signal Processing (60 seconds)

**Script:**
> "Our pipeline processes raw PSG signals through bandpass filtering, notch filtering, and normalization. We apply artifact quality control and segment the signals into 30-second epochs."

**Show:**
- Before/after waveform comparison
- Filtering effects visualization
- Epoch segmentation

**Key points:**
- 0.5–35 Hz bandpass removes noise
- 50 Hz notch eliminates power-line interference
- Z-score normalization standardizes signals
- ~2% artifact flag rate

---

### Minute 3: Model Architecture (60 seconds)

**Script:**
> "The model uses a multi-resolution front end to capture patterns at different temporal scales. Depthwise-separable convolutions reduce computation while maintaining feature extraction. A compact Gabor block learns frequency-localized patterns. A two-layer GRU models temporal dependencies across five minutes of context."

**Show:**
- Architecture diagram
- Multi-resolution stem visualization
- Feature extraction flow
- GRU temporal modeling

**Key points:**
- Multi-scale feature capture
- Efficient depthwise-separable convolutions
- Learnable frequency filters
- 300-second temporal context

---

### Minute 4: Live Inference (60 seconds)

**Script:**
> "Let's run the model on a sample sequence. The model processes ten consecutive epochs and predicts the sleep stage for each."

**Show:**
- Load sample PSG sequence
- Run inference
- Display predictions with confidence
- Show hypnogram-like timeline

**Demo commands:**
```bash
# Run inference
python scripts/infer.py \
    # open notebooks/05_evaluation_and_benchmarking.ipynb (outputs embedded)
    # or run the dashboard:
    streamlit run app/streamlit_app.py
```

**Expected output:**
```
Predicted sleep stages:
Epoch  1: Wake   (confidence: 95.2%)
Epoch  2: N1     (confidence: 78.3%)
Epoch  3: N2     (confidence: 89.1%)
...
```

---

### Minute 5: Results & Impact (60 seconds)

**Script:**
> "Our historical exhibition model achieves **88.64% accuracy** with Cohen's kappa of **0.7725** on 15 held-out subjects from Sleep-EDF Expanded under the legacy all-position protocol. The model has only **99,477 parameters** — small enough for edge deployment on microcontrollers. 

> **Important:** This historical result (88.64%) uses the legacy all-position evaluation protocol which scores overlapping sequence windows. The primary research benchmark (person-level 10-fold CV, causal unique-epoch protocol) yields 87.30% accuracy. The two protocols are not directly comparable."

**Show:**
- Historical exhibition result block
- Parameter count comparison
- Latency measurement
- Edge deployment potential
- Protocol distinction note

**Historical Exhibition Result (EXP-EXHIBITION-15SUBJ):**
```
88.64%
HISTORICAL EXHIBITION ACCURACY (legacy protocol)

0.7725
COHEN'S κ

0.7186
MACRO F1

99,477
PARAMETERS

8.86 ms/batch
CPU LATENCY
```

**Primary Research Benchmark (EXP-BENCH-PERSON):**
```
87.30% ± 0.33%
PERSON-LEVEL ACCURACY (causal protocol)

0.738 ± 0.010
COHEN'S κ

0.724 ± 0.005
MACRO F1

99,477
PARAMETERS
```

---

## Exhibition Poster Layout

### Section A: Title

**Neuromorphic Sleep Stage Scoring**

*A Compact Deep Learning Pipeline for Five-Stage Sleep Classification from PSG Signals*

---

### Section B: Problem

- Sleep staging requires expert analysis
- Manual scoring is time-consuming
- Automation can assist clinicians

**Visual:** Raw PSG waveform with stage annotations

---

### Section C: Dataset

```
Sleep-EDF Expanded
EEG + EOG + EMG
92-record eligible cohort (52 persons)
Person-level 10-fold CV (primary)
Subject-level 70/15/15 split (exhibition)
```

---

### Section D: Signal Processing

```
Raw signal
    ↓
0.5–35 Hz bandpass
    ↓
50 Hz notch
    ↓
z-score
    ↓
artifact QC
    ↓
30 s epochs
```

---

### Section E: Architecture

```
Multi-Resolution Stem
    ↓
Depthwise-Separable CNN
    ↓
Parametric Gabor FEB
    ↓
2-Layer GRU
    ↓
5-Class Output
```

---

### Section F: Results

**Large typography — Historical Exhibition Result (EXP-EXHIBITION-15SUBJ):**

```
88.64%
HISTORICAL EXHIBITION ACCURACY
(Legacy All-Position Protocol)

0.7725
COHEN'S κ

0.7186
MACRO F1

99,477
PARAMETERS
```

**Small text note:** "Primary research benchmark (EXP-BENCH-PERSON): 87.30% ± 0.33% accuracy, 0.738 ± 0.010 κ, causal unique-epoch protocol."

---

### Section G: LoRA Adaptation (Quarantined)

> **Note:** The LoRA adaptation study used a contaminated base checkpoint and record-level folds with person-level leakage. Results are retained in `docs/adaptation.md` for internal comparison only and must not be presented as valid person-generalization estimates.

---

### Section H: Deployment

```
Compact model
    ↓
Low memory footprint
    ↓
Fast CPU inference
    ↓
Edge / MCU deployment
```

---

### Section I: Team

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

---

## Live Demo Dashboard

### Streamlit Layout

```
┌─────────────────────────────────────────────────────┐
│ NEUROMORPHIC SLEEP STAGE SCORING                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  PSG Waveform                                       │
│  ─────────────────────────────────────────────────  │
│                                                     │
│  Predicted Stage: N2                                │
│  Confidence: 89.1%                                  │
│                                                     │
│  5-Class Probabilities                              │
│  W   ███████████                                    │
│  N1  ██                                             │
│  N2  █████████████████                              │
│  N3  ███                                            │
│  REM ██                                             │
│                                                     │
│  Model: Improved Student                            │
│  Parameters: 99,477                                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Demo Buttons

- [Load Sample] — Load a sample PSG sequence
- [Run Inference] — Run model prediction
- [Show Signal] — Display multi-channel waveforms
- [Show Probabilities] — Display class probabilities
- [Show Hypnogram] — Display predicted timeline

---

## Q&A Preparation

### Technical Questions

**Q: Why use a 300-second context window?**
A: Sleep stages are temporally dependent. 10 consecutive 30-second epochs provide 5 minutes of context, allowing the GRU to model stage transitions.

**Q: Why not classify each epoch independently?**
A: Isolated epochs can be ambiguous, especially around transitions (W→N1, N1→N2). Sequence modeling provides contextual information.

**Q: Why use EEG, EOG, and EMG together?**
A: Each channel provides complementary information: EEG for brain activity, EOG for eye movements, EMG for muscle activity.

**Q: Why depthwise-separable convolutions?**
A: They reduce computational cost by ~77% compared to standard convolutions while maintaining feature extraction capability.

**Q: Why is accuracy not enough?**
A: Class imbalance means a model can have high accuracy while performing poorly on minority stages. Cohen's kappa and F1 are more informative.

**Q: What's the difference between 88.64% and 87.30%?**
A: The 88.64% is the historical exhibition result under the legacy all-position protocol (scoring overlapping windows). The 87.30% is the primary research benchmark under the causal unique-epoch protocol (one prediction per unique epoch, person-level splits). The stricter protocol yields a more honest person-generalization estimate.

### Deployment Questions

**Q: Can this run on a microcontroller?**
A: Yes. The model has 99,477 parameters (~400 KB at FP32) and 8.5 ms latency, suitable for ARM Cortex-M7 class devices.

**Q: What is the memory footprint?**
A: ~400 KB for weights, plus ~120 KB for input buffer. Total < 1 MB, fitting on most microcontrollers.

**Q: Can it run in real-time?**
A: Yes. At 8.5 ms per batch with 30-second epochs, the model has significant headroom for real-time inference.

### Limitation Questions

**Q: What are the main limitations?**
A: N1 and REM classification are challenging due to class imbalance and subtle signal differences. The model is trained on healthy subjects only.

**Q: Is this clinically validated?**
A: No. This is a research prototype. Clinical validation would require larger, more diverse datasets and regulatory approval.

---

## Checklist

### Before Exhibition

- [ ] Live demo working on exhibition machine
- [ ] Streamlit dashboard tested
- [ ] Sample data files available
- [ ] Poster printed and mounted
- [ ] Backup slides prepared
- [ ] Team roles assigned for Q&A

### During Exhibition

- [ ] Start with problem statement
- [ ] Show signal processing pipeline
- [ ] Demonstrate live inference
- [ ] Present historical exhibition result (88.64%, legacy protocol)
- [ ] Clarify primary benchmark is 87.30% (causal protocol)
- [ ] Answer questions confidently
- [ ] Acknowledge limitations honestly

### After Exhibition

- [ ] Collect feedback
- [ ] Document any issues
- [ ] Update project based on feedback
- [ ] Share results with team

---

*Last updated: September 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
