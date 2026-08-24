# Documentation Index — Neuromorphic Sleep Stage Scoring

## Documentation Structure

```
docs/
├── architecture.md      # Detailed model architecture
├── dataset.md           # Dataset documentation
├── deployment.md        # Edge deployment guide
├── exhibition.md        # VIT Bhopal exhibition guide
├── methodology.md       # Research methodology
├── results.md           # Official final results
├── team.md              # Team roles & contributions
└── index.md             # This file
```

---

## Documentation by Audience

### For Researchers

| Document | Description |
|----------|-------------|
| `architecture.md` | Model design decisions and parameter analysis |
| `methodology.md` | Research approach and experimental design |
| `dataset.md` | Data sources, preprocessing, and contracts |
| `results.md` | Official metrics and interpretation |

### For Engineers

| Document | Description |
|----------|-------------|
| `architecture.md` | Input/output specifications and components |
| `deployment.md` | Export, quantization, and edge deployment |
| `dataset.md` | Data formats and cache structure |
| `methodology.md` | Training configuration and reproducibility |

### For Exhibition

| Document | Description |
|----------|-------------|
| `exhibition.md` | Demo script, poster layout, Q&A |
| `results.md` | Official result block for display |
| `team.md` | Team roles for poster/presentation |
| `architecture.md` | Visual architecture diagrams |

---

## Quick Reference

### Official Result (Frozen Base)

```
Improved Student
87.34% Test Accuracy
Cohen's κ = 0.7551
99,477 Parameters
8.5 ms/batch CPU latency
```

### LoRA Adaptation Result

```
LoRA r=8
90.66% ± 3.59% Held-Out-Subject CV Test Accuracy
Cohen's κ = 0.8092 ± 0.0663
552 Trainable Parameters (0.55%)
```

### Model Checkpoint

```
artifacts/student_improved_best.pt
```

### Run Commands

```bash
# Verify model
python scripts/verify_model.py

# LoRA cross-validation
python scripts/run_lora_cv.py

# Train LoRA adapter
python scripts/train_lora_adapter.py --rank 8 --alpha 16 --output artifacts/lora/head_r8

# Evaluate LoRA adapter
python scripts/evaluate_lora_adapter.py --adapter artifacts/lora/head_r8

# Dashboard
streamlit run app/streamlit_app.py

# Tests
python -m pytest tests/ -v
```

---

## Team Contact

| Member | Role |
|--------|------|
| Param Kaushik | Dataset & Data Governance |
| Suha Vora | Signal Preprocessing |
| Shailendra Bhatt | Exploratory Data Analysis |
| Shamique Khan | Model Development & Training |
| Aasir Jaffer Lone | Evaluation & Performance |

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
