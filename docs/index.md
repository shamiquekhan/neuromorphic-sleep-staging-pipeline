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

### Official Result

```
Improved Student
87.34% Test Accuracy
Cohen's κ = 0.7551
99,477 Parameters
8.5 ms/batch CPU latency
```

### Model Checkpoint

```
artifacts/student_improved_best.pt
```

### Run Commands

```bash
# Prepare dataset
python scripts/prepare_dataset.py

# Train model
python scripts/train.py --config configs/training.yaml

# Evaluate
python scripts/evaluate.py --checkpoint artifacts/student_improved_best.pt

# Inference
python scripts/infer.py --checkpoint artifacts/student_improved_best.pt --input demo/sample_inputs/sample_epoch.npz

# Demo
streamlit run demo/demo.py

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
