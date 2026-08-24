# Team Roles & Contributions — Neuromorphic Sleep Stage Scoring

## Team Overview

| Member | Role | Primary Responsibility |
|--------|------|----------------------|
| Param Kaushik | Dataset & Data Governance Lead | Data acquisition, manifest, integrity |
| Suha Vora | Signal Preprocessing Lead | Filtering, normalization, QC |
| Shailendra Bhatt | Exploratory Data Analysis Lead | Class balance, signal patterns |
| Shamique Khan | Model Development & Training Lead | Architecture, training, distillation |
| Aasir Jaffer Lone | Evaluation & Performance Lead | Metrics, confusion matrix, exhibition |

---

## Detailed Responsibilities

### Param Kaushik — Dataset & Data Governance

**Notebook:** `01_data_import_and_dataset_collection.ipynb`

**Responsibilities:**
- Sleep-EDF dataset acquisition from PhysioNet
- PSG/Hypnogram file pairing validation
- Subject-level train/val/test splitting
- Manifest generation and validation
- Data contract for downstream notebooks
- Dataset integrity audit

**Deliverables:**
- `data/manifests/sleep_edf.csv` — Subject/night manifest
- `data/manifests/subject_splits.csv` — Split assignments
- Dataset provenance documentation

**Handoff to:** Suha Vora (preprocessing)

---

### Suha Vora — Signal Preprocessing

**Notebook:** `02_data_preprocessing.ipynb`

**Responsibilities:**
- Bandpass filtering (0.5–35 Hz)
- Notch filtering (50 Hz)
- Z-score normalization
- Artifact quality control
- Epoch construction (30s)
- Cache generation

**Deliverables:**
- `data/cache/*.npz` — Preprocessed epoch arrays
- `data/cache/cache_index.csv` — Cache metadata
- Preprocessing configuration

**Handoff to:** Shailendra Bhatt (EDA)

---

### Shailendra Bhatt — Exploratory Data Analysis

**Notebook:** `03_exploratory_data_analysis.ipynb`

**Responsibilities:**
- Class distribution analysis
- Signal characteristic visualization
- Temporal pattern analysis
- Quality control validation
- Feature importance insights

**Deliverables:**
- Class balance report
- Signal visualization plots
- Temporal dependency analysis
- EDA summary documentation

**Handoff to:** Shamique Khan (model development)

---

### Shamique Khan — Model Development & Training

**Notebook:** `04_model_architecture_and_training.ipynb`

**Responsibilities:**
- Improved Student architecture design
- Teacher model training
- Knowledge distillation implementation
- Training loop optimization
- Checkpoint management
- Architecture documentation

**Deliverables:**
- `artifacts/student_improved_best.pt` — Final model
- `artifacts/teacher_improved_best.pt` — Teacher model
- Architecture specification
- Training configuration

**Handoff to:** Aasir Jaffer Lone (evaluation)

---

### Aasir Jaffer Lone — Evaluation & Performance

**Notebook:** `05_evaluation_and_benchmarking.ipynb`

**Responsibilities:**
- Test set evaluation
- Official result documentation
- Confusion matrix analysis
- Per-class F1 computation
- Exhibition result formatting
- Performance visualization

**Deliverables:**
- `results/final_result.csv` — Official metrics
- Confusion matrix plots
- Per-class F1 charts
- Exhibition result block

**Handoff to:** Project team (exhibition)

---

## Shared Responsibilities

All team members share responsibility for:

1. **Reproducibility:** Ensuring all results can be regenerated
2. **Code quality:** Following project coding standards
3. **Documentation:** Maintaining clear notebook narratives
4. **Exhibition preparation:** Supporting the final demonstration
5. **Version control:** Proper Git commit practices

---

## Notebook Handoff Flow

```
01 Dataset (Param)
    │
    ▼
02 Preprocessing (Suha)
    │
    ▼
03 EDA (Shailendra)
    │
    ▼
04 Model Training (Shamique)
    │
    ▼
05 Evaluation (Aasir)
    │
    ▼
Exhibition Demo (All)
```

---

## Data Contract

The data contract between notebooks ensures consistency:

| Contract | Producer | Consumer | Artifact |
|----------|----------|----------|----------|
| Manifest | Notebook 01 | Notebook 02 | `data/manifests/sleep_edf.csv` |
| Cache | Notebook 02 | Notebooks 03, 04 | `data/cache/*.npz` |
| Split | Notebook 01 | All notebooks | `split` column in manifest |
| Model | Notebook 04 | Notebook 05 | `artifacts/student_improved_best.pt` |

---

## Contribution Matrix

| Task | Param | Suha | Shailendra | Shamique | Aasir |
|------|-------|------|------------|----------|-------|
| Dataset acquisition | **Lead** | Support | - | - | - |
| Data splitting | **Lead** | - | - | - | - |
| Manifest generation | **Lead** | - | - | - | - |
| Bandpass filtering | - | **Lead** | - | - | - |
| Notch filtering | - | **Lead** | - | - | - |
| Z-score normalization | - | **Lead** | - | - | - |
| Artifact QC | - | **Lead** | Support | - | - |
| Class analysis | - | - | **Lead** | Support | - |
| Signal visualization | - | Support | **Lead** | - | - |
| Temporal analysis | - | - | **Lead** | - | - |
| Architecture design | - | - | - | **Lead** | Support |
| Model training | - | - | - | **Lead** | - |
| Knowledge distillation | - | - | - | **Lead** | - |
| Test evaluation | - | - | - | Support | **Lead** |
| Official results | - | - | - | - | **Lead** |
| Exhibition demo | Support | Support | Support | Support | **Lead** |

---

## Communication Protocol

### Weekly Sync
- **When:** Every [day] at [time]
- **Agenda:** Progress updates, blockers, handoff planning
- **Duration:** 30 minutes

### Handoff Process
1. Producer completes their notebook section
2. Producer documents any assumptions or requirements
3. Consumer reviews and validates outputs
4. Both agree on artifact contract
5. Consumer begins their section

### Issue Resolution
1. Identify the issue
2. Document in project issue tracker
3. Assign to relevant team member
4. Resolve and verify
5. Update documentation

---

*Last updated: August 2026*
*Project: Neuromorphic Sleep Stage Scoring — VIT Bhopal University*
