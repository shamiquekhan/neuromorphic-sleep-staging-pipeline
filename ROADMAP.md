# NeuroSleep — Roadmap

Living document. Everything not yet finished, ordered by the evidence
hierarchy: **CODE → CONFIG → EXPERIMENT → RAW RESULTS → AGGREGATION →
RESULT TABLES → MODEL_REPORT → README → HF MODEL CARD** (results must
always be regenerated from the protocol, never hand-edited).

Status legend: ☐ not started · ◐ partially done · ✔ done · ⛔ blocked

---

## 0. Experiment Matrix (Canonical IDs)

| ID | Experiment | Status | Evidence |
|----|-----------|--------|----------|
| EXP-DEV-15SUBJ | 15-subject development benchmark | ✔ archived (`docs/archive/development_15_subject.md`) | `results/final/`, 93.0% — historical only |
| EXP-BENCH-92SUBJ | 92-subject from-scratch benchmark, 10-fold subject-level CV, 3 seeds | ◐ seed 42 complete; **seeds 43/44 never run** | `results/benchmark_92_subject/` (seed 42: 87.7% ± 2.1%) |
| EXP-ADAPT-FROZEN / -LORA-R8-CNNHEAD / -FULLFT | Adaptation regimes from one base checkpoint | ⛔ **quarantined — contaminated base** (see §2) | `results/100_subject_adaptation/` |
| EXP-LORA-R{2,4,16}-CNNHEAD | LoRA rank ablation | ◐ r=8 exists (contaminated base); **r=2/4/16 never run** | `results/ablations/lora_rank/` |
| EXP-LORA-TARGET-ABLATION | Target-module matrix (head / CNN / Gabor / GRU combos) | ⛔ blocked on LoRA-GRU code + leak-free base | — |
| EXP-CROSS-DATASET | Cross-dataset validation (e.g., SHHS) | ☐ planned | — |

---

## 1. Completed Repairs (September 2026 audit)

- ✔ Split experimental narratives: 15-subject dev benchmark archived
  (`docs/archive/development_15_subject.md`), `final.yaml` →
  `configs/development_15_subject.yaml` with contamination warning
- ✔ Canonical config hierarchy: `benchmark_92_subject.yaml` (primary),
  `adaptation_92_subject.yaml`, `lora_rank_ablation.yaml`,
  `lora_target_ablation.yaml` — all with experiment IDs; ambiguous
  `benchmark_canonical.yaml` / `full_100_subject.yaml` removed
- ✔ Canonical cohort phrasing adopted: *"92-subject eligible cohort from
  100 downloaded Sleep-EDF Expanded records (8 wake-only excluded)"*
- ✔ `docs/lora.md` — math, target modules, frozen/trainable split, adapter
  save/load/merge, verification guarantees, honest result reading
- ✔ `docs/adaptation.md` — three-regime protocol, subject-disjointness
  requirement, contamination quantification
- ✔ `scripts/verify_protocol.py` — machine-detectable leakage +
  config-consistency checks (legacy checkpoint correctly FAILS)

---

## 2. CRITICAL — Fix the Contaminated Adaptation Benchmark ⛔

The base checkpoint `artifacts/final/student_full_finetuned.pt` was trained
on 15 subjects; **12 appear in the 92-subject eval test folds, 3 in eval
validation**. Every Frozen/LoRA/Full-FT number from
`results/100_subject_adaptation/` is inflated (frozen baseline by ~+2.5pp)
and quarantined.

- [ ] Train a **leak-free base checkpoint** on the 9 fixed evaluation
      validation subjects (or another disjoint set); record its training
      subject list (e.g., `artifacts/base_leakfree/train_subjects.json`)
- [ ] Re-validate: `python scripts/verify_protocol.py --base-subjects artifacts/base_leakfree/train_subjects.json`
- [ ] Re-run Frozen (10 folds; deterministic — 1 seed only, it is wrong to
      report 30 fold-evaluations for it)
- [ ] Re-run LoRA primary config × seeds 42/43/44
- [ ] Re-run Full-FT × seeds 42/43/44
- [ ] Re-run rank ablation on the leak-free base
- [ ] `aggregate_adaptation_results.py` → `statistical_analysis.py` →
      regenerate `docs/results.md` + README + HF card from the aggregates

Estimated compute: ~8–12 h on the GTX 1650 (frozen is fast; LoRA ≈ 1.6 h
per seed; full-FT ≈ 2.5 h per seed; rank ablation ×3 ranks ≈ 5 h).

---

## 3. Remaining Work — This Repair Effort ◐

- [ ] Complete LoRA rank ablation aggregation + write-up into
      `docs/results.md` and `docs/lora.md` (r=2/4/16 vs r=8 on the same —
      quarantined — base; label contamination clearly)
- [ ] Rewrite README.md as high-level summary only (no result tables
      duplicated ad hoc; link `docs/results.md`; fix "neuromorphic" claims)
- [ ] Rewrite MODEL_REPORT.md as the publication-style consolidated report
      (three-regime definitions, per-class tables, confusion matrices,
      parameter-efficiency table, honest LoRA claim)
- [ ] Rewrite GUIDE.md (currently still shows 93.0% as "Final Model")
- [ ] Rewrite `docs/results.md` as the single authoritative numbers file,
      with 95% CIs and quarantine notes
- [ ] Rewrite `docs/methodology.md` (currently a third, stale narrative:
      distillation teacher, 87.5%, 4-subject baseline)
- [ ] Rewrite `docs/index.md` (currently references non-existent scripts
      and stale results)
- [ ] Fix "neuromorphic" positioning repo-wide: current architecture is a
      conventional differentiable CNN–GRU; reposition as *compact
      edge-oriented sleep staging*, keep SNN conversion as future work
- [ ] Sweep remaining stale-number locations:
      `deployment/README.md`, `deployment/config/inference.yaml`,
      `huggingface/*/README.md`, `kaggle/neurosleep_final.ipynb`,
      `app/pages/04_Model_Information.py` fallback metrics, notebooks
- [ ] Final consistency grep (93.0 / 0.861 / 0.794 outside
      `docs/archive/`, `results/final/`, `results/LORA_RESULTS.md`)
- [ ] Full `pytest` run + `verify_protocol.py` in CI gate

---

## 4. Short-Term Science (Blocked on GPU Hours)

- [ ] **EXP-BENCH-92SUBJ seeds 43/44** — the primary result is currently a
      single-seed estimate (87.7% ± 2.1%, 10 folds). ~7 h GPU. This is the
      highest-priority *clean* experiment in the repo.
- [ ] LoRA-GRU support in `src/sleep_staging/adaptation/lora.py`
      (wrap `weight_ih_l*/weight_hh_l*` or apply low-rank update to GRU
      input/hidden maps) + tests — unblocks LoRA-D/E target experiments
- [ ] Confidence intervals on all primary metrics (bootstrap over folds)
- [ ] Per-method confusion-matrix figures for docs (`frozen`, `lora`,
      `full_finetune`) + Wake↔N1 / N1↔N2 / N2↔N3 / REM↔Wake discussion
- [ ] Experiment-ID stamped result dirs (`EXP-LORA-R8-CNNHEAD/...`) via
      `train_adaptation.py --experiment-id`

## 5. Medium-Term Science

- [ ] LoRA **target-module ablation** (LoRA-A…E matrix,
      `configs/lora_target_ablation.yaml`) on the leak-free base — the key
      open question: *does adapting the GRU (90.3% of parameters) close the
      LoRA-vs-full-FT gap?*
- [ ] LoRA **alpha ablation** (α = r, 2r, 4r) and **dropout ablation**
      (0 / 0.05 / 0.10)
- [ ] Early-stopping-patience / checkpoint-selection sensitivity check
      (current runs select on val macro-F1)
- [ ] Merge-procedure utility (`merge_adapter` function) + latency
      benchmark merged vs unmerged vs base
- [ ] Protocol-fingerprint enforcement in `train_adaptation.py`
      (refuse to start when `verify_protocol.py` fails)

## 6. Long-Term Research Directions

- [ ] **EXP-CROSS-DATASET** — train on Sleep-EDF, evaluate on SHHS
      (`src/sleep_staging/data/shhs.py` exists); report frozen vs LoRA vs
      full-FT transfer gap — this is where parameter-efficient adaptation
      has its strongest use case
- [ ] Single-channel EEG variant (Fpz-Cz only) for wearable deployment
- [ ] ONNX export + INT8 quantization validation on Cortex-M class target
- [ ] Neuromorphic/SNN conversion study (spike-encoded GRU or LIF layer
      with surrogate gradients) — only then revisit the "neuromorphic"
      naming; keep positioning as *edge-oriented* until then
- [ ] N1-focused work: the primary benchmark's N1 F1 (0.45) is the honest
      bottleneck; consider transitional-stage-specific loss or
      sequence-level smoothing

---

## 7. Regeneration Order (Never Hand-Edit Numbers)

```
code/fix → configs/*.yaml → verify_protocol.py → experiments (scripts/)
        → results/**/raw folds → aggregate_*.py → statistical_analysis.py
        → docs/results.md → MODEL_REPORT.md → README.md → hf_model_card.md
```

*Last updated: September 2026.*
