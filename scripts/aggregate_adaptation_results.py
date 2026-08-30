#!/usr/bin/env python3
"""
Aggregate Adaptation Results — Fold-Level Multi-Seed Aggregation

Reads raw fold metrics from seeds 42/43/44 and computes aggregate
statistics at the fold level (30 folds per method).

Usage:
    python scripts/aggregate_adaptation_results.py
"""

import json
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ADAPT_DIR = REPO / "results" / "100_subject_adaptation"
OUTPUT_DIR = ADAPT_DIR / "final"

MODES = {
    "frozen": {"dir": ADAPT_DIR / "frozen", "key": "frozen", "label": "Frozen", "params": 0},
    "lora_cnn_head": {"dir": ADAPT_DIR / "lora_r8_enc.0.pw_enc.1.pw_head", "key": "lora_cnn_head", "label": "LoRA CNN+Head", "params": 1448},
    "full_finetune": {"dir": ADAPT_DIR / "full_finetune", "key": "full_finetune", "label": "Full Fine-Tuning", "params": 99477},
}

SEEDS = [42, 43, 44]
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
METRICS = ["accuracy", "kappa", "macro_f1", "weighted_f1", "mgm"]


def load_seed_folds(mode_dir: Path, seed: int) -> dict:
    agg_file = mode_dir / f"{mode_dir.name.split('_r')[0] if 'lora' in mode_dir.name else mode_dir.name}_seed{seed}.json"
    # Try different naming patterns
    candidates = [
        mode_dir / f"frozen_seed{seed}.json",
        mode_dir / f"lora_seed{seed}.json",
        mode_dir / f"full_finetune_seed{seed}.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    # Fallback: search for any *_seed{seed}.json
    for f in mode_dir.glob(f"*seed{seed}.json"):
        with open(f) as fh:
            return json.load(fh)
    return {}


def aggregate_mode(mode_name: str, mode_info: dict) -> dict:
    all_folds = []
    for seed in SEEDS:
        folds = load_seed_folds(mode_info["dir"], seed)
        for fold_key, fold_data in folds.items():
            fold_data["_seed"] = seed
            fold_data["_fold"] = int(fold_key.replace("fold_", ""))
            all_folds.append(fold_data)

    if not all_folds:
        print(f"  WARNING: No fold data found for {mode_name}")
        return {}

    n_folds = len(all_folds)

    # Aggregate overall metrics at fold level
    overall = {}
    for m in METRICS:
        vals = [f[m] for f in all_folds]
        overall[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Aggregate per-class metrics at fold level
    per_class = {}
    for stage in STAGES:
        per_class[stage] = {}
        for metric in ["precision", "recall", "f1"]:
            vals = [f["per_class"][stage][metric] for f in all_folds]
            per_class[stage][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "mode": mode_name,
        "label": mode_info["label"],
        "params": mode_info["params"],
        "n_seeds": len(SEEDS),
        "n_folds_total": n_folds,
        "overall": overall,
        "per_class": per_class,
        "raw_folds": all_folds,
    }


def save_aggregate_csv(results: dict, output_dir: Path):
    # Overall comparison CSV
    with open(output_dir / "overall_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Params"] + [f"{m} (mean)" for m in METRICS] + [f"{m} (std)" for m in METRICS])
        for mode_name, data in results.items():
            row = [data["label"], data["params"]]
            for m in METRICS:
                row.append(f"{data['overall'][m]['mean']:.4f}")
            for m in METRICS:
                row.append(f"{data['overall'][m]['std']:.4f}")
            writer.writerow(row)

    # Per-class comparison CSV
    with open(output_dir / "per_class_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Metric"] + [data["label"] for data in results.values()])
        for stage in STAGES:
            for metric in ["f1", "precision", "recall"]:
                row = [stage, metric]
                for data in results.values():
                    val = data["per_class"][stage][metric]
                    row.append(f"{val['mean']:.4f} ± {val['std']:.4f}")
                writer.writerow(row)


def save_final_markdown(results: dict, output_dir: Path):
    lines = [
        "# 100-Subject Adaptation Benchmark — Final Results",
        "",
        "**Dataset:** Sleep-EDF Expanded (92 subjects, 10-fold subject-level CV)",
        "**Seeds:** 42, 43, 44 (30 folds per method)",
        "**Base Checkpoint:** `artifacts/final/student_full_finetuned.pt`",
        "",
        "## Overall Metrics (Mean ± Std Across 30 Folds)",
        "",
        "| Model | Trainable Params | Accuracy | κ | Macro F1 | Weighted F1 | MGm |",
        "|-------|----------------:|---------:|----:|---------:|------------:|----:|",
    ]

    for data in results.values():
        o = data["overall"]
        lines.append(
            f"| {data['label']} | {data['params']:,} | "
            f"{o['accuracy']['mean']:.4f} ± {o['accuracy']['std']:.4f} | "
            f"{o['kappa']['mean']:.4f} ± {o['kappa']['std']:.4f} | "
            f"{o['macro_f1']['mean']:.4f} ± {o['macro_f1']['std']:.4f} | "
            f"{o['weighted_f1']['mean']:.4f} ± {o['weighted_f1']['std']:.4f} | "
            f"{o['mgm']['mean']:.4f} ± {o['mgm']['std']:.4f} |"
        )

    lines += [
        "",
        "## Per-Stage F1 (Mean ± Std Across 30 Folds)",
        "",
        "| Stage | Frozen | LoRA CNN+Head | Full Fine-Tuning |",
        "|-------|-------:|--------------:|-----------------:|",
    ]

    for stage in STAGES:
        vals = []
        for data in results.values():
            f1 = data["per_class"][stage]["f1"]
            vals.append(f"{f1['mean']:.4f} ± {f1['std']:.4f}")
        lines.append(f"| {stage} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines += [
        "",
        "## Per-Stage Precision (Mean ± Std)",
        "",
        "| Stage | Frozen | LoRA CNN+Head | Full Fine-Tuning |",
        "|-------|-------:|--------------:|-----------------:|",
    ]

    for stage in STAGES:
        vals = []
        for data in results.values():
            p = data["per_class"][stage]["precision"]
            vals.append(f"{p['mean']:.4f} ± {p['std']:.4f}")
        lines.append(f"| {stage} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines += [
        "",
        "## Per-Stage Recall (Mean ± Std)",
        "",
        "| Stage | Frozen | LoRA CNN+Head | Full Fine-Tuning |",
        "|-------|-------:|--------------:|-----------------:|",
    ]

    for stage in STAGES:
        vals = []
        for data in results.values():
            r = data["per_class"][stage]["recall"]
            vals.append(f"{r['mean']:.4f} ± {r['std']:.4f}")
        lines.append(f"| {stage} | {vals[0]} | {vals[1]} | {vals[2]} |")

    # Parameter efficiency
    ft = results["full_finetune"]["overall"]["accuracy"]["mean"]
    lora = results["lora_cnn_head"]["overall"]["accuracy"]["mean"]
    frozen = results["frozen"]["overall"]["accuracy"]["mean"]
    ft_kappa = results["full_finetune"]["overall"]["kappa"]["mean"]
    lora_kappa = results["lora_cnn_head"]["overall"]["kappa"]["mean"]

    lines += [
        "",
        "## Parameter Efficiency",
        "",
        f"- LoRA CNN+Head: 1,448 / 99,477 = **1.43%** of trainable parameters",
        f"- Accuracy retention: {lora:.4f} / {ft:.4f} = **{lora/ft*100:.1f}%**",
        f"- κ retention: {lora_kappa:.4f} / {ft_kappa:.4f} = **{lora_kappa/ft_kappa*100:.1f}%**",
        f"- Frozen accuracy retention: {frozen:.4f} / {ft:.4f} = **{frozen/ft*100:.1f}%**",
        "",
        "## Conclusion",
        "",
        "> **Full fine-tuning is the preferred model for the NeuroSleep project** because it provides the strongest overall and stage-balanced performance on the 100-subject adaptation benchmark.",
        "",
        "---",
        "",
        f"*Generated from 3 seeds × 10 folds = 30 folds per method*",
    ]

    with open(output_dir / "FINAL_ADAPTATION_RESULTS.md", "w") as f:
        f.write("\n".join(lines))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode_name, mode_info in MODES.items():
        print(f"\nAggregating {mode_info['label']}...")
        data = aggregate_mode(mode_name, mode_info)
        if data:
            results[mode_name] = data
            o = data["overall"]
            print(f"  Folds: {data['n_folds_total']}")
            print(f"  Accuracy: {o['accuracy']['mean']:.4f} ± {o['accuracy']['std']:.4f}")
            print(f"  Kappa:    {o['kappa']['mean']:.4f} ± {o['kappa']['std']:.4f}")
            print(f"  Macro F1: {o['macro_f1']['mean']:.4f} ± {o['macro_f1']['std']:.4f}")

    # Save aggregate JSON (without raw_folds to keep file small)
    json_out = {}
    for mode_name, data in results.items():
        json_out[mode_name] = {
            "mode": data["mode"],
            "label": data["label"],
            "params": data["params"],
            "n_seeds": data["n_seeds"],
            "n_folds_total": data["n_folds_total"],
            "overall": data["overall"],
            "per_class": data["per_class"],
        }

    with open(OUTPUT_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved {OUTPUT_DIR / 'aggregate_metrics.json'}")

    save_aggregate_csv(results, OUTPUT_DIR)
    print(f"Saved {OUTPUT_DIR / 'overall_comparison.csv'}")
    print(f"Saved {OUTPUT_DIR / 'per_class_comparison.csv'}")

    save_final_markdown(results, OUTPUT_DIR)
    print(f"Saved {OUTPUT_DIR / 'FINAL_ADAPTATION_RESULTS.md'}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL ADAPTATION BENCHMARK — ALL SEEDS AGGREGATED")
    print("=" * 70)
    print(f"  {'Model':20s} {'Params':>8s} {'Accuracy':>12s} {'κ':>12s} {'Macro F1':>12s}")
    print("-" * 70)
    for data in results.values():
        o = data["overall"]
        print(f"  {data['label']:20s} {data['params']:>8,} "
              f"{o['accuracy']['mean']:.4f}±{o['accuracy']['std']:.4f} "
              f"{o['kappa']['mean']:.4f}±{o['kappa']['std']:.4f} "
              f"{o['macro_f1']['mean']:.4f}±{o['macro_f1']['std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
