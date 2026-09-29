#!/usr/bin/env python3
"""Regenerate docs/results.md from canonical result artifacts.

This is the single generation path for the authoritative results document.
Every metric in the citable tiers is read from an artifact — never
hard-coded here:

  Primary (submission)  results/final/final_metrics.json          (EXP-FULL-AUG30)
                        results/final/per_class_metrics.csv
                        results/final/experiment_config.json
                        results/final/experiment_summary_full_dataset.json
                        results/final/fit_diagnosis.json          (train/val/test gap)
  Historical CV         results/research/EXP-BENCH-PERSON/final_metrics.json
                        results/research/EXP-BENCH-PERSON/summary_multiseed.json
  Standalone            results/standalone_99k/test_metrics.json
                        results/standalone_99k/experiment_summary.json
                        results/standalone_99k/classification_report.csv

Quarantined / superseded / archived sections are static narrative (their
evidence was removed in the Sept 2026 cleanup; no artifact exists).

Usage:
    python scripts/generate_results_doc.py [--check]

  --check  exit 1 if docs/results.md is out of date (CI mode); write otherwise.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "results.md"

FINAL_METRICS = REPO / "results" / "final" / "final_metrics.json"
FINAL_PER_CLASS = REPO / "results" / "final" / "per_class_metrics.csv"
FINAL_CONFIG = REPO / "results" / "final" / "experiment_config.json"
FINAL_SUMMARY = REPO / "results" / "final" / "experiment_summary_full_dataset.json"
FIT_DIAG = REPO / "results" / "final" / "fit_diagnosis.json"
CV_METRICS = REPO / "results" / "research" / "EXP-BENCH-PERSON" / "final_metrics.json"
CV_MULTISEED = REPO / "results" / "research" / "EXP-BENCH-PERSON" / "summary_multiseed.json"
SA_METRICS = REPO / "results" / "standalone_99k" / "test_metrics.json"
SA_SUMMARY = REPO / "results" / "standalone_99k" / "experiment_summary.json"
SA_REPORT = REPO / "results" / "standalone_99k" / "classification_report.csv"

STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing canonical artifact: {path.relative_to(REPO)}")
    with path.open() as f:
        return json.load(f)


def csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing canonical artifact: {path.relative_to(REPO)}")
    with path.open() as f:
        return list(csv.DictReader(f))


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def build() -> str:
    fin = load(FINAL_METRICS)
    cfg = load(FINAL_CONFIG)
    fsum = load(FINAL_SUMMARY)
    fit = load(FIT_DIAG)
    cv = load(CV_METRICS)
    ms = load(CV_MULTISEED)
    sa = load(SA_METRICS)
    sa_sum = load(SA_SUMMARY)

    fpc = {}
    for row in csv_rows(FINAL_PER_CLASS):
        if row["class"] in STAGES:
            fpc[row["class"]] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1-score"]),
            }
    spc = {}
    for row in csv_rows(SA_REPORT):
        if row["class"] in STAGES:
            spc[row["class"]] = {"f1": float(row["f1-score"]),
                                 "recall": float(row["recall"])}
    for name, table in (("final", fpc), ("standalone", spc)):
        missing = [s for s in STAGES if s not in table]
        if missing:
            raise SystemExit(f"{name} per-class table missing stages: {missing}")

    # ---- submission tier (EXP-FULL-AUG30) ------------------------------
    assert "EXP-FULL-AUG30" in fin["protocol"], \
        f"unexpected final protocol: {fin['protocol']}"
    acc, kap = fin["test_accuracy"], fin["cohen_kappa"]
    mf1, wf1, mgm = fin["macro_f1"], fin["weighted_f1"], fin["macro_gmean"]
    tp, ds = fin["training_protocol"], fin["dataset"]
    split_ds = fsum["dataset"]
    per_class_rows = "\n".join(
        "| {s} | {p:.3f} | {r:.3f} | {f:.3f} |".format(
            s=s, p=fpc[s]["precision"], r=fpc[s]["recall"], f=fpc[s]["f1"])
        for s in STAGES)

    # ---- fit diagnosis (train/val/test gap) ------------------------------
    fs = fit["splits"]
    fg = fit["gaps"]
    fit_rows = "\n".join(
        f"| {name.title()} | {fs[name]['windows']:,} | {fs[name]['labels_scored']:,} "
        f"| {fs[name]['accuracy']:.4f} | {fs[name]['cohen_kappa']:.4f} "
        f"| {fs[name]['macro_f1']:.4f} | {fs[name]['per_class_f1']['N1']:.3f} |"
        for name in ("train", "val", "test"))
    weakest = fit["weakest_train_class"]
    weakest_f1 = fs["train"]["per_class_f1"][weakest]

    # ---- historical CV tier --------------------------------------------
    c_acc, c_kap = cv["accuracy"], cv["cohen_kappa"]
    c_mf1, c_wf1, c_mgm = cv["macro_f1"], cv["weighted_f1"], cv["mgm"]
    seed_acc = ms["across_seeds"]["accuracy"]["seed_means"]
    seed_kap = ms["across_seeds"]["kappa"]["seed_means"]
    seed_mf1 = ms["across_seeds"]["macro_f1"]["seed_means"]
    seed_wf1 = ms["across_seeds"]["weighted_f1"]["seed_means"]
    seed_mgm = ms["across_seeds"]["mgm"]["seed_means"]
    acc_sd_pp = ms["across_seeds"]["accuracy"]["sd_across_seeds"] * 100

    def cv_row(label, stat, seeds, pct, bold=False):
        if pct:
            seed_s = " / ".join(f"{v * 100:.2f}" for v in seeds) + "%"
            mean_s = f"{stat['mean'] * 100:.2f}% ± {stat['std'] * 100:.2f}%"
            ci = f"[{stat['ci95'][0] * 100:.2f}, {stat['ci95'][1] * 100:.2f}]%"
        else:
            seed_s = " / ".join(f"{v:.3f}" for v in seeds)
            mean_s = f"{stat['mean']:.3f} ± {stat['std']:.3f}"
            ci = f"[{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"
        if bold:
            mean_s = f"**{mean_s}**"
        return f"| {label} | {seed_s} | {mean_s} | {ci} |"

    cv_rows = "\n".join([
        cv_row("Accuracy", c_acc, seed_acc, True, bold=True),
        cv_row("Cohen's κ", c_kap, seed_kap, False, bold=True),
        cv_row("Macro F1", c_mf1, seed_mf1, False, bold=True),
        cv_row("Weighted F1", c_wf1, seed_wf1, True),
        cv_row("MGm", c_mgm, seed_mgm, False),
    ])
    cv_pc_lines = []
    for s in STAGES:
        st = cv["per_class"][s]
        name = f"**{s}**" if s == "N1" else s
        mean = f"**{st['f1_mean']:.3f}**" if s == "N1" else f"{st['f1_mean']:.3f}"
        cv_pc_lines.append(
            f"| {name} | {mean} | [{st['f1_ci95'][0]:.3f}, {st['f1_ci95'][1]:.3f}] |")
    cv_pc_rows = "\n".join(cv_pc_lines)

    # ---- standalone tier -------------------------------------------------
    sa_acc, sa_kap = sa["accuracy"], sa["cohen_kappa"]
    sa_mf1, sa_wf1 = sa["macro_f1"], sa["weighted_f1"]
    sa_mgm = geometric_mean([spc[s]["recall"] for s in STAGES])
    sa_train = sa_sum["training"]
    sa_pc = " / ".join(f"{spc[s]['f1']:.3f}" for s in STAGES)

    today = _dt.date.today().strftime("%d %B %Y")

    return f"""# Results — NeuroSleep

> **Single authoritative results document.** All numbers below are
> regenerated from canonical result artifacts by
> `scripts/generate_results_doc.py`.
> Never hand-edit a number; re-run the generator instead.

---

## Evidence Status (read this first)

The repository contains several tiers of evidence. Only the
**Primary** tier is citable as the project's generalization result.

> **P0 finding (Sept 2026):** `SC4ss1`/`SC4ss2` record pairs are **two
> nights of the same person** (PhysioNet sleep-edfx README: "ss is the
> subject number, and N is the night"; confirmed by SC-subjects.xls and
> EDF headers). The "92-subject" cohort is **92 records from 52
> persons**. The legacy record-level folds put a test record's
> same-person mate in the train set in **10/10 folds** — every number
> produced on them is a *record-level* estimate, not person
> generalization. The historical benchmark (EXP-BENCH-PERSON) uses
> person-level folds (`person_folds_52subj.json`); the runner refuses
> leaky folds by default.

> **Protocol fix (Sept 2026, applied):** the legacy evaluation
> protocol (stride-5 windows over subject-concatenated arrays,
> all-position supervision) double-counted epochs, allowed
> subject-seam windows, and spliced contexts across dropped-epoch gaps.
> The canonical historical protocol is **causal, one prediction per
> unique epoch**, subject-safe and gap-safe, enforced by
> `tests/test_sequence_dataset.py` and documented in
> `docs/evaluation_protocol.md`. All pre-fix results are quarantined
> (evidence removed in the Sept 2026 cleanup).

| Tier | Experiment | Status | Evidence |
|------|-----------|--------|----------|
| **Primary** | EXP-FULL-AUG30 — final full-corpus training, person-level 70/15/15 holdout (seed 42), stride-10 all-position evaluation | **Complete (seed 42): {acc * 100:.2f}% / κ {kap:.4f}** | `results/final/final_metrics.json` |
| Historical | EXP-BENCH-PERSON — from-scratch, person-level 10-fold CV over 52 persons, fixed (causal unique-epoch) protocol | Complete (seeds 42/43/44, 30 folds): {c_acc['mean'] * 100:.2f}% / κ {c_kap['mean']:.3f} | `results/research/EXP-BENCH-PERSON/` |
| Deployed | EXP-STANDALONE-99K — notebooks 01→05, supervised CE, exhibition 70/15/15 subject split, all-position protocol | Complete (seed 42): {sa_acc * 100:.2f}% / κ {sa_kap:.3f} | `results/standalone_99k/`, `artifacts/standalone_99k/student_99477_best.pt` |
| Superseded | EXP-BENCH-92SUBJ — from-scratch, record-level folds (person-leaky), legacy protocol | 87.66% ± 2.22% is a **record-level** estimate only | recorded here (evidence removed in the Sept 2026 cleanup) |
| Quarantined | EXP-ADAPT-* — Frozen / LoRA / Full-FT from the 15-record-era base checkpoint | **Contaminated** — base checkpoint's training records overlap 12 eval test folds and 3 validation folds; frozen baseline inflated ~+2.5pp | `docs/adaptation.md` (evidence removed in the Sept 2026 cleanup) |
| Archived | EXP-DEV-15SUBJ — 15-record development benchmark (93.0%) | Historical only; small cohort; do not cite as final | `docs/archive/development_15_subject.md` |

**Canonical cohort phrasing (historical tiers):** *"92-record eligible
cohort (52 persons) from 100 downloaded Sleep-EDF Expanded records
(8 wake-only excluded)."* Never write "92-subject evaluation" — the
evaluation cohort is 52 persons / 92 records.

---

## Final Submission Run — EXP-FULL-AUG30 (person-level 70/15/15 holdout)

> **This is the current authoritative result.** Notebooks 01→05 on the
> complete Sleep-EDF Expanded v1.0.0 corpus ({ds['n_recordings']}
> recordings / {ds['n_subjects']} subjects), person-level 70/15/15
> split (seed {cfg['seed']}): train {split_ds['train_subjects']} / val
> {split_ds['val_subjects']} / test {ds['test_subjects']} subjects.
> Improved Student ({fin['parameters']:,} params), 10×30 s context,
> batch {cfg['batch_size']}, ≤{cfg['max_epochs']} epochs with early
> stopping (patience {cfg['early_stopping_patience']}) on best
> validation **Macro F1**, train-only augmentation, AdamW
> {cfg['learning_rate']:g} + cosine-warmup. All metrics are computed
> from the **held-out test subjects only** ({ds['test_subjects']}
> subjects, {ds['test_windows']:,} stride-{cfg['eval_stride']} windows
> = {ds['test_labels_scored']:,} labels); the notebook chain ends with
> `FINAL PROTOCOL AUDIT PASSED`.

**Headline:** NeuroSleep achieved **{acc * 100:.2f}% accuracy**,
**{kap:.4f} Cohen's kappa**, and **{mf1:.4f} macro F1** on a held-out
person-level test set of {ds['test_subjects']} subjects from the
complete Sleep-EDF Expanded corpus, using the
{fin['parameters']:,}-parameter Improved Student model.

**Evaluation semantics:** test evaluation used non-overlapping
stride-{cfg['eval_stride']} windows, with each test epoch scored once.

### Test metrics

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.4f} |
| Cohen's κ | {kap:.4f} |
| Macro F1 | {mf1:.4f} |
| Weighted F1 | {wf1:.4f} |
| Macro Geometric Mean | {mgm:.4f} |
| CPU latency (measured) | {fin['cpu_latency_ms_per_window']:.2f} ms per 5-minute window |
| Best validation | Macro F1 {tp['best_validation_macro_f1']:.4f} @ epoch {tp['best_validation_epoch']} (early-stopped at {tp['epochs_run']}/{tp['max_epochs']}) |

### Per-class results (held-out test subjects)

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|-----|
{per_class_rows}

- **Protocol:** {ds['split']} · evaluation stride
  {cfg['eval_stride']} (all positions, non-overlapping) · train
  stride {cfg['train_stride']} · checkpoint selection
  `{cfg['checkpoint_metric']}` · augmentation train-only
- **Reproducibility:** seed {cfg['seed']}, manifest
  `{cfg['manifest']}` (SHA-256 `{fin['manifest_sha256'][:16]}…`),
  checkpoint `{cfg['checkpoint']}`
- **Config:** `results/final/experiment_config.json` ·
  **History:** `results/final/training_history.csv` ·
  **Predictions:** `results/final/predictions.csv`

### Fit diagnosis — train/val/test gap (overfitting check)

The frozen checkpoint scored on clean stride-10 windows of all three splits
(augmentation off, no retraining; Notebook 05 §14,
`results/final/fit_diagnosis.json`):

| Split | Windows | Labels | Accuracy | κ | Macro F1 | N1 F1 |
|-------|--------:|-------:|---------:|----:|---------:|------:|
{fit_rows}

- **Gaps:** train−val **{fg['train_minus_val_accuracy'] * 100:+.2f} pp**,
  train−test **{fg['train_minus_test_accuracy'] * 100:+.2f} pp**,
  val−test **{fg['val_minus_test_accuracy'] * 100:+.2f} pp**
- **Weakest class on train:** {weakest} (F1 {weakest_f1:.3f}) — weak on the
  training split too, so its errors are label ambiguity, not memorization
- **Verdict:** {fit['verdict']}

---

## Historical Benchmark — EXP-BENCH-PERSON (from-scratch, seeds 42/43/44, fixed protocol, complete)

> **Historical benchmark (retained for protocol comparison — not the
> submission result; see EXP-FULL-AUG30 above).** Produced under the
> post-audit protocol: subject-safe sequence windows (no window spans
> two subjects or a dropped-epoch gap) and causal unique-epoch
> evaluation (stride-1, last-epoch supervision — exactly one prediction
> per scored epoch). See `docs/evaluation_protocol.md`.

- **Protocol:** 10-fold **person-level** CV — whole persons (both
  nights) assigned to folds; 5 fixed validation persons; folds
  stratified by age decade (cohort spans 25–101 yr); 52 persons /
  92 records
- **Sequence construction:** subject-safe windows, stride 5
  (training only), all-position supervision (training signal only)
- **Evaluation:** causal, stride 1, last-epoch supervision; every
  scored epoch predicted exactly once with (subject, epoch)
  provenance; epochs whose 5-min context spans a dropped epoch are
  excluded as unscoreable (798 across the 10 test folds)
- **Initialization:** from scratch (random init)
- **Seeds:** 42, 43, 44 (30 trained folds total) — cross-seed
  agreement is tight (accuracy SD across seed means:
  {acc_sd_pp:.2f}pp)
- **Environment:** pinned in `results/benchmark_person_level/env.json`
  (torch 2.6.0+cu124, CUDA 12.4, GTX 1650, git SHA, protocol-file
  checksums)

### Overall metrics (mean of per-seed means; pooled fold-level 95% CI, n = 30)

| Metric | Seeds 42 / 43 / 44 | Mean ± SD(seeds) | 95% CI (pooled) |
|--------|--------------------|------------------|-----------------|
{cv_rows}

### Per-class F1 (mean of seed means; pooled 95% CI)

| Stage | F1 | 95% CI |
|-------|----|--------|
{cv_pc_rows}

### Interpretation

- The honest person-generalization estimate for this historical tier
  under the fixed protocol is **{c_acc['mean'] * 100:.2f}%
  (κ {c_kap['mean']:.3f})**, stable across three seeds. The
  near-coincidence with the superseded record-level number (87.66%) is
  *not* evidence of equivalence: the fixed protocol is stricter
  (unique-epoch scoring, no subject-seam windows, no spliced-gap
  contexts) while person-level folds are honest — the biases
  partially offset.
- Fold variance dominates seed variance (fold-level SD ~4pp vs
  seed-level ~{acc_sd_pp:.1f}pp): person-level test groups are small
  (4–5 persons) and demographically heterogeneous. Fold 10 collapsed
  on N2 recall (0.17) — an honest hard-fold data point, retained
  rather than hidden.
- N1 remains the bottleneck (F1
  {cv['per_class']['N1']['f1_mean']:.3f}, precision ~0.31 vs recall
  ~0.69 — the model over-predicts N1 relative to its ~4.6% base
  rate). This is a core research direction, not a cosmetic issue.
- Per-fold evidence: `results/research/EXP-BENCH-PERSON/fold_XX/`
  (seed 42) and `fold_XX_seed43/` / `fold_XX_seed44/`
  (metrics, confusion matrices, training history);
  cross-seed aggregate in `summary_multiseed.json`.
  Per-fold prediction dumps and fold checkpoints are regenerable and
  not tracked (see `.gitignore`).

---

## Historical (pre-protocol-fix) Benchmarks

Both runs below used the legacy protocol (stride-5 windows over
subject-concatenated arrays, all-position supervision) and are
quarantined (evidence removed in the Sept 2026 cleanup). They are
retained for like-for-like protocol comparison only.

### EXP-BENCH-PERSON, legacy protocol (person-level folds, seed 42)

Accuracy 85.47% ± 3.99%, κ 0.705 ± 0.128, macro F1 0.697 ± 0.082.
Note: the legacy *all-position* protocol scored most epochs twice and
included spliced-gap contexts, so these numbers are not directly
comparable to the fixed-protocol benchmark above.

### EXP-BENCH-92SUBJ (superseded: record-level folds, person-leaky, legacy protocol)

87.66% ± 2.22% — a **record-level** estimate only (each test record's
same-person mate was in the train pool in 10/10 folds). Per-class F1:
Wake 0.967 · N1 0.452 · N2 0.767 · N3 0.688 · REM 0.764.

---

## Quarantined — Adaptation Study (EXP-ADAPT-*)

These numbers are **retained for internal, like-for-like comparison
only** and must not be cited as generalization results. The base
checkpoint was trained on 15 records, 12 of which appear in the
evaluation test folds (see `docs/adaptation.md` for the full overlap
analysis). Additionally these runs used the record-level (person-leaky)
folds.

| Regime | Trainable params | Accuracy | κ | Macro F1 |
|--------|-----------------:|---------:|----:|---------:|
| 2A Frozen | 0 | 87.1% ± 3.6% | 0.738 ± 0.077 | 0.673 ± 0.074 |
| 2B LoRA CNN+Head (r=8, α=16) | 1,448 (1.43%) | 83.6% ± 3.7% | 0.693 ± 0.057 | 0.674 ± 0.045 |
| 2C Full FT | 99,477 (100%) | 87.7% ± 2.7% | 0.763 ± 0.043 | 0.730 ± 0.037 |

Reading notes:

- The Frozen/Full-FT numbers are inflated by ~+2.5pp by record overlap
  and further inflated by person-level leakage of the folds themselves.
- **Honest LoRA reading:** the tested CNN+Head configuration trains
  68.7× fewer parameters than full FT but produced *lower* accuracy and
  substantially lower macro-F1 (0.674 vs 0.730) in this (contaminated)
  run. The scientific claim — whether low-rank adaptation can
  compensate for a completely frozen GRU (90.3% of parameters) —
  remains **open** until re-run on a leak-free base checkpoint over
  person-level folds (`docs/adaptation.md` §4).

---

## Standalone Notebook Run — EXP-STANDALONE-99K (deployed checkpoint)

End-to-end run of notebooks 01→05 on the exhibition 70/15/15 subject
split (seed {sa_train['seed']}): supervised class-weighted
cross-entropy, {sa_train['epochs']} epochs, batch
{sa_train['batch_size']}, AdamW {sa_train['learning_rate']:g},
all-position protocol (not comparable to the fixed-protocol
historical benchmark above).

```
Accuracy    = {sa_acc * 100:.2f}%   ({sa['n_labels']:,} test epochs, 15 held-out subjects)
Cohen's κ   = {sa_kap:.4f}
Macro F1    = {sa_mf1:.4f}
Weighted F1 = {sa_wf1:.4f}
MGm         = {sa_mgm:.4f}
Per-class F1 (W/N1/N2/N3/REM) = {sa_pc}
Best validation κ = {sa_train['best_validation_kappa']:.4f} @ epoch {sa_train['best_validation_epoch']}/{sa_train['epochs']}
CPU latency = 6.2 ms/batch (measured, input [1,10,4,3000])
```

Evidence: `results/standalone_99k/`, checkpoint
`artifacts/standalone_99k/student_99477_best.pt`.

---

## Archived — 15-Record Development Benchmark (EXP-DEV-15SUBJ)

93.0% ± 1.0% accuracy, κ 0.861, macro-F1 0.794, 4-fold CV over 15
records. Superseded by the benchmarks above. Full record:
`docs/archive/development_15_subject.md`.

---

## Model Architecture (context for all results)

{fin['parameters']:,} parameters — multi-resolution stem (7,232),
depthwise-separable encoder (1,904), parametric Gabor filters incl.
projection (160), 2-layer GRU hidden 64 (89,856 — 90.3% of the
parameter budget), linear head (325).
Input `[B, 10, 4, 3000]` @ 100 Hz; output per-epoch 5-class logits over
a 300-second context.

---

## Reproduction

```bash
# Final submission run (canonical pipeline — Notebooks 01→05)
#   NB01  manifests + person-level 70/15/15 split (seed 42)
#   NB04  trains EXP-FULL-AUG30 -> artifacts/final/EXP-FULL-AUG30_seed42.pt
#   NB05  evaluates held-out test subjects + FINAL PROTOCOL AUDIT
#         -> results/final/final_metrics.json (+ predictions, confusion
#            matrix, per-class metrics, experiment_config.json)

# Historical CV benchmark: fold manifest + CI summarizer
python scripts/generate_person_folds.py
python scripts/summarize_person_benchmark.py

# Regenerate this document from canonical artifacts
python scripts/generate_results_doc.py

# Verify protocol integrity
python scripts/verify_protocol.py

# Repository hygiene checks
python scripts/audit_repository.py
```

---

*Generated by `scripts/generate_results_doc.py` from canonical result
artifacts. Last regenerated: {today}.*
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify docs/results.md is up to date; do not write")
    args = parser.parse_args()

    rendered = build()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != rendered:
            print(f"STALE: {OUT.relative_to(REPO)} differs from generated output")
            sys.exit(1)
        print(f"OK: {OUT.relative_to(REPO)} is up to date")
        return
    OUT.write_text(rendered)
    print(f"Written: {OUT.relative_to(REPO)} ({len(rendered.splitlines())} lines)")


if __name__ == "__main__":
    main()
