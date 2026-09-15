# Quarantine Directory (Artifacts)

This directory contains model checkpoints that are **quarantined** and must not be used for deployment or primary evaluation.

## Contents

### `student_full_finetuned_generic.pt`
- **Original name:** `artifacts/final/student_full_finetuned.pt`
- **Why quarantined:** 
  - Filename does not encode experiment ID, seed, or fold
  - No provenance metadata (git commit, manifest hash, etc.)
  - Ambiguous "final" in path suggests it's the canonical checkpoint but it's not versioned
- **Replacement:** `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/student_best.pt` (with full provenance)

---

## Active Checkpoints (Use These Instead)

| Experiment | Checkpoint | Provenance |
|------------|------------|------------|
| **EXP-EXHIBITION-15SUBJ** | `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/student_best.pt` | `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/provenance.json` |
| **EXP-EXHIBITION-15SUBJ** | `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/teacher_improved_best.pt` | `artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/provenance.json` |

All active checkpoints include:
- `experiment_id`
- `git_commit` (SHA)
- `manifest_sha256`
- `checkpoint_sha256`
- `seed`
- `training_config`
- `architecture_config`
- `created_at` timestamp