"""Summarize fold-level person-level benchmark evidence with 95% confidence intervals.

Reads results/research/EXP-BENCH-PERSON/fold_summary.csv (one row per fold,
from-scratch person-level benchmark) and emits mean, sample std, and a
t-based 95% CI per metric. The unit of analysis is the fold (person
group), not the epoch.

Usage:
    python scripts/summarize_person_benchmark.py [--results-dir results/research/EXP-BENCH-PERSON]

Writes results/research/EXP-BENCH-PERSON/summary_with_ci.json and prints a
markdown table suitable for direct inclusion in docs/results.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO / "results" / "research" / "EXP-BENCH-PERSON"

# t(0.975, df) for two-sided 95% CI; keys are degrees of freedom (n - 1).
T975 = {9: 2.262, 29: 2.045}

METRICS = ["accuracy", "kappa", "macro_f1", "weighted_f1", "mgm"]
STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def fold_per_class_stats(results_dir: Path, n_folds: int) -> dict:
    """Per-class F1/precision/recall mean±std from fold_XX/metrics.json."""
    stats: dict = {}
    for stage in STAGES:
        agg = {k: [] for k in ("f1", "precision", "recall")}
        for i in range(1, n_folds + 1):
            mpath = results_dir / f"fold_{i:02d}" / "metrics.json"
            with mpath.open() as f:
                pc = json.load(f)["per_class"][stage]
            for k in agg:
                agg[k].append(float(pc[k]))
        stats[stage] = {}
        for k, vals in agg.items():
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            stats[stage][f"{k}_mean"] = mean
            stats[stage][f"{k}_std"] = std
    return stats


def write_dashboard_metrics(results_dir: Path, out: dict) -> None:
    """Regenerate results/final/final_metrics.json (dashboard data source).

    The previous copy of this file held the quarantined adaptation-study
    Full-FT numbers mislabeled as from-scratch; it is regenerated here
    from clean fold evidence only.
    """
    is_person = "person" in results_dir.name
    multiseed_path = results_dir / "summary_multiseed.json"
    if is_person and multiseed_path.exists():
        with multiseed_path.open() as f:
            ms = json.load(f)
        seeds = ms["seeds"]
        evaluation = (f"10-fold person-level CV over 52 persons, seeds "
                      f"{seeds}, causal unique-epoch protocol "
                      "(EXP-BENCH-PERSON)")
        n_seeds = len(seeds)
        acc_src = ms["across_seeds"]["accuracy"]
        kap_src = ms["across_seeds"]["kappa"]
        f1_src = ms["across_seeds"]["macro_f1"]
        wf1_src = ms["across_seeds"]["weighted_f1"]
        mgm_src = ms["across_seeds"]["mgm"]
        get = lambda s: {"mean": s["mean_of_seed_means"],
                         "std": s["sd_across_seeds"],
                         "ci95": s["pooled_fold_level"]["ci95"]}
        per_class = {
            stage: {
                "f1_mean": v["mean_of_seed_means"],
                "f1_std": v["pooled_fold_level"]["sd"],
                "f1_ci95": v["pooled_fold_level"]["ci95"],
            }
            for stage, v in ms["per_class_f1"].items()
        }
    else:
        seeds = None
        evaluation = ("10-fold person-level CV over 52 persons, seed 42 only "
                      "(seeds 43/44 pending; EXP-BENCH-PERSON, causal "
                      "unique-epoch protocol)") if is_person else \
                     ("10-fold record-level CV, seed 42 only "
                      "(seeds 43/44 pending; EXP-BENCH-92SUBJ — superseded)")
        n_seeds = 1
        acc_src = out["metrics"]["accuracy"]
        kap_src = out["metrics"]["kappa"]
        f1_src = out["metrics"]["macro_f1"]
        wf1_src = out["metrics"]["weighted_f1"]
        mgm_src = out["metrics"]["mgm"]
        get = lambda s: {"mean": s["mean"], "std": s["std"], "ci95": s["ci95"]}
        per_class = fold_per_class_stats(results_dir, out["n_folds"])
    dash = {
        "model": "Improved Student (from scratch)",
        "parameters": 99477,
        "dataset": "Sleep-EDF Expanded",
        "cohort": "92-record eligible cohort from 100 downloaded records "
                  "(8 wake-only excluded); 52 persons",
        "n_subjects": 92,
        "split_level": "person" if is_person else "record",
        "protocol": "causal_unique_epoch",
        "n_folds": out["n_folds"],
        "n_seeds": n_seeds,
        "seeds": seeds,
        "evaluation": evaluation,
        "accuracy": get(acc_src),
        "cohen_kappa": get(kap_src),
        "macro_f1": get(f1_src),
        "weighted_f1": get(wf1_src),
        "mgm": get(mgm_src),
        "per_class": per_class,
    }
    dest = REPO / "results" / "final" / "final_metrics.json"
    dest.write_text(json.dumps(dash, indent=2) + "\n")
    print(f"Written: {dest.relative_to(REPO)} (dashboard data source, clean evidence)")


def t_crit(n: int) -> float:
    df = n - 1
    if df not in T975:
        raise ValueError(f"No t-critical value tabulated for df={df} (n={n}).")
    return T975[df]


def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0, "ci95": [mean, mean]}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    half = t_crit(n) * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def write_fold_summary(results_dir: Path, n_folds: int) -> Path:
    """Build fold_summary.csv from fold_XX/metrics.json if absent."""
    dest = results_dir / "fold_summary.csv"
    if dest.exists():
        return dest
    rows = []
    for i in range(1, n_folds + 1):
        m = json.load(open(results_dir / f"fold_{i:02d}" / "metrics.json"))
        folds_manifest = json.load(open(
            REPO / "data" / "manifests" / "person_folds_52subj.json"))
        row = {
            "fold": i,
            "n_test_subjects": len(folds_manifest["folds"][f"fold_{i}"]["test"]),
            "test_subjects": ";".join(folds_manifest["folds"][f"fold_{i}"]["test"]),
            "accuracy": m["accuracy"], "kappa": m["kappa"],
            "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
            "mgm": m["mgm"],
            "n_unique_epochs": m.get("n_unique_epochs", ""),
            "protocol": m.get("protocol", ""),
        }
        for stage in STAGES:
            row[f"{stage}_f1"] = m["per_class"][stage]["f1"]
        rows.append(row)
    import csv
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {dest.relative_to(REPO)}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    if not args.results_dir.is_absolute():
        args.results_dir = REPO / args.results_dir

    summary_path = write_fold_summary(args.results_dir, 10)
    with summary_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No fold rows found in {summary_path}")

    out: dict = {"source": str(summary_path.relative_to(REPO)), "n_folds": len(rows),
                 "metrics": {}, "per_class_f1": {}}

    for m in METRICS:
        out["metrics"][m] = summarize([float(r[m]) for r in rows])
    for s in STAGES:
        out["per_class_f1"][s] = summarize([float(r[f"{s}_f1"]) for r in rows])

    dest = args.results_dir / "summary_with_ci.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    write_dashboard_metrics(args.results_dir, out)

    def fmt(m: str, pct: bool) -> str:
        s = out["metrics"][m]
        lo, hi = s["ci95"]
        scale = 100.0 if pct else 1.0
        suffix = "%" if pct else ""
        return (f"{s['mean'] * scale:.3f} ± {s['std'] * scale:.3f}{suffix}",
                f"[{lo * scale:.3f}, {hi * scale:.3f}]{suffix}")

    print(f"# {args.results_dir.name} (n={len(rows)} folds)\n")
    print("| Metric | Mean ± SD | 95% CI |")
    print("|--------|-----------|--------|")
    for label, m, pct in [("Accuracy", "accuracy", True), ("Cohen's κ", "kappa", False),
                          ("Macro F1", "macro_f1", False), ("Weighted F1", "weighted_f1", True),
                          ("MGm", "mgm", False)]:
        mean_s, ci_s = fmt(m, pct)
        print(f"| {label} | {mean_s} | {ci_s} |")
    print("\n| Stage | F1 mean ± SD | 95% CI |")
    print("|-------|--------------|--------|")
    for s in STAGES:
        c = out["per_class_f1"][s]
        lo, hi = c["ci95"]
        print(f"| {s} | {c['mean']:.3f} ± {c['std']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    print(f"\nWritten: {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
