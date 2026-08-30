# Team Contributions

This document describes individual contributions to the NeuroSleep project.

## Team Members

| Member | Role | Contributions |
|--------|------|---------------|
| Param Kaushik | Dataset & Data Governance, Streamlit Dashboard | Dataset acquisition, PhysioNet data pipeline, data versioning, Streamlit web application development |
| Suha Vora | Signal Preprocessing | MNE preprocessing pipeline, signal filtering, quality control |
| Shailendra Bhatt | Exploratory Data Analysis | EDA notebooks, class distribution analysis, visualization |
| Shamique Khan | Model Development & Training | ImprovedStudent architecture, LoRA adaptation, training loops, code consolidation |
| Aasir Jaffer Lone | Evaluation & Performance | Metrics implementation, cross-validation design, performance analysis |

## Git History Note

All commits were consolidated by Shamique Khan for repository hygiene. Individual contributions were made through collaborative workflows (pair programming, code reviews, notebook sharing) rather than direct git commits. The contributions listed above reflect the primary responsibility areas for each team member.

## Repository Structure

The repository was designed to clearly separate concerns:
- `src/sleep_staging/` — Core package (model, inference, data, training, evaluation)
- `scripts/` — CLI entry points for training, evaluation, and diagnostics
- `tests/` — Pytest test suite
- `notebooks/` — Analysis and EDA notebooks
- `app/` — Streamlit dashboard
- `configs/` — YAML configuration files
- `docs/` — Documentation (architecture, limitations, results)
