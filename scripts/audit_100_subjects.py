#!/usr/bin/env python3
"""
Comprehensive dataset audit for 100-subject Sleep-EDF Expanded cache.

Generates:
  - subject_statistics.csv
  - class_distribution.csv
  - qc_statistics.csv
  - wake_only_subjects.csv
  - subject_quality_manifest.csv
  - dataset_summary.json
  - class_distribution_by_fold.csv
  - n1_distribution.png
  - class_distribution_overview.png
  - fold_class_balance.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "sleep_edf"
MANIFEST_PATH = ROOT / "data" / "manifests" / "sleep_edf_expanded.json"
FOLDS_PATH = ROOT / "data" / "manifests" / "canonical_subject_folds.json"
OUT_DIR = ROOT / "results" / "dataset_audit_100subj"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# label names
LABEL_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}
EXPECTED_LABELS = {0, 1, 2, 3, 4}
N_CLASSES = 5
SAMPLING_RATE = 100  # Hz
EPOCH_SECONDS = 30


# ── load canonical folds ─────────────────────────────────────────────────
def load_folds():
    with open(FOLDS_PATH) as f:
        folds_data = json.load(f)
    return folds_data["folds"]


# ── audit one subject ────────────────────────────────────────────────────
def audit_subject(npz_path):
    """Return dict with per-subject statistics, or None on error."""
    result = {
        "subject_id": npz_path.stem.replace("_night0", ""),
        "file": npz_path.name,
        "file_size_mb": npz_path.stat().st_size / (1024 * 1024),
        "status": "ok",
        "error": "",
    }

    try:
        data = np.load(npz_path)
    except Exception as e:
        result["status"] = "load_error"
        result["error"] = str(e)
        return result

    # check required keys (support both naming conventions)
    if {"X", "y"}.issubset(data.keys()):
        X, y = data["X"], data["y"]
    elif {"epochs", "labels"}.issubset(data.keys()):
        X, y = data["epochs"], data["labels"]
    else:
        result["status"] = "missing_keys"
        result["error"] = f"Keys: {list(data.keys())}"
        return result

    # ── shape checks ──────────────────────────────────────────────────────
    if X.ndim != 3:
        result["status"] = "bad_shape_X"
        result["error"] = f"X.ndim={X.ndim}, expected 3"
        return result

    n_epochs, n_channels, n_samples = X.shape
    result["n_epochs"] = n_epochs
    result["n_channels"] = n_channels
    result["n_samples"] = n_samples
    result["duration_hours"] = round(n_epochs * EPOCH_SECONDS / 3600, 2)

    # ── label checks ──────────────────────────────────────────────────────
    unique_labels = set(np.unique(y).astype(int))
    result["unique_labels"] = sorted(unique_labels)
    result["unexpected_labels"] = sorted(unique_labels - EXPECTED_LABELS)

    if not unique_labels.issubset(EXPECTED_LABELS):
        result["status"] = "bad_labels"
        result["error"] = f"Unexpected labels: {unique_labels - EXPECTED_LABELS}"
        return result

    # ── class distribution ────────────────────────────────────────────────
    counts = {}
    for label_id in range(N_CLASSES):
        counts[f"count_{LABEL_NAMES[label_id]}"] = int(np.sum(y == label_id))
        pct = np.sum(y == label_id) / len(y) * 100 if len(y) > 0 else 0
        counts[f"pct_{LABEL_NAMES[label_id]}"] = round(pct, 2)

    result.update(counts)

    # ── N1 detail ─────────────────────────────────────────────────────────
    n1_mask = y == 1
    n1_runs = 0
    if np.any(n1_mask):
        diff = np.diff(n1_mask.astype(int))
        n1_runs = int(np.sum(diff == 1) + n1_mask[0])

    result["n1_runs"] = n1_runs

    # ── NaN / Inf check ──────────────────────────────────────────────────
    result["has_nan"] = bool(np.isnan(X).any())
    result["has_inf"] = bool(np.isinf(X).any())

    if result["has_nan"] or result["has_inf"]:
        result["status"] = "bad_values"
        result["error"] = f"NaN={result['has_nan']}, Inf={result['has_inf']}"
        return result

    # ── value range ──────────────────────────────────────────────────────
    result["x_min"] = float(np.min(X))
    result["x_max"] = float(np.max(X))
    result["x_mean"] = float(np.mean(X))
    result["x_std"] = float(np.std(X))

    # ── wake-only flag ───────────────────────────────────────────────────
    non_wake_classes = unique_labels - {0}
    result["wake_only"] = len(non_wake_classes) == 0

    # ── has all 5 stages ─────────────────────────────────────────────────
    result["has_all_5_stages"] = len(unique_labels) >= 5

    # ── missing stages ──────────────────────────────────────────────────
    missing = EXPECTED_LABELS - unique_labels
    result["missing_stages"] = sorted(missing)
    result["missing_stage_names"] = sorted([LABEL_NAMES[l] for l in missing])

    return result


# ── main audit ───────────────────────────────────────────────────────────
def main():
    npz_files = sorted(CACHE_DIR.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files in {CACHE_DIR}")

    # ── 1. per-subject audit ──────────────────────────────────────────────
    records = []
    for i, npz_path in enumerate(npz_files):
        if (i + 1) % 10 == 0:
            print(f"  Auditing subject {i+1}/{len(npz_files)}...")
        rec = audit_subject(npz_path)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "subject_statistics.csv", index=False)
    print(f"\nSubject statistics saved to {OUT_DIR / 'subject_statistics.csv'}")

    # ── 2. summary counts ────────────────────────────────────────────────
    total = len(df)
    ok = (df["status"] == "ok").sum()
    errors = total - ok
    wake_only = df["wake_only"].sum() if "wake_only" in df.columns else 0
    has_all5 = df["has_all_5_stages"].sum() if "has_all_5_stages" in df.columns else 0
    nan_count = df["has_nan"].sum() if "has_nan" in df.columns else 0
    inf_count = df["has_inf"].sum() if "has_inf" in df.columns else 0

    print(f"\n{'='*60}")
    print(f"DATASET AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total NPZ files:     {total}")
    print(f"Valid (status=ok):   {ok}")
    print(f"Errors:              {errors}")
    print(f"Wake-only subjects:  {wake_only}")
    print(f"Has all 5 stages:    {has_all5}")
    print(f"Has NaN:             {nan_count}")
    print(f"Has Inf:             {inf_count}")

    # ── 3. class distribution across all subjects ────────────────────────
    total_epochs = 0
    class_totals = {i: 0 for i in range(N_CLASSES)}

    for rec in records:
        if rec["status"] != "ok":
            continue
        total_epochs += rec["n_epochs"]
        for label_id in range(N_CLASSES):
            class_totals[label_id] += rec[f"count_{LABEL_NAMES[label_id]}"]

    print(f"\n{'='*60}")
    print(f"GLOBAL CLASS DISTRIBUTION (all valid subjects)")
    print(f"{'='*60}")
    print(f"Total epochs: {total_epochs:,}")
    for label_id in range(N_CLASSES):
        name = LABEL_NAMES[label_id]
        count = class_totals[label_id]
        pct = count / total_epochs * 100 if total_epochs > 0 else 0
        print(f"  {name:5s}: {count:>10,}  ({pct:5.1f}%)")

    # save class distribution CSV
    class_rows = []
    for label_id in range(N_CLASSES):
        name = LABEL_NAMES[label_id]
        count = class_totals[label_id]
        pct = count / total_epochs * 100 if total_epochs > 0 else 0
        class_rows.append({
            "class": name,
            "label_id": label_id,
            "total_epochs": count,
            "percentage": round(pct, 2),
        })
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "class_distribution.csv", index=False)

    # ── 4. QC statistics ─────────────────────────────────────────────────
    qc_rows = []
    for rec in records:
        qc_rows.append({
            "subject_id": rec["subject_id"],
            "status": rec["status"],
            "error": rec.get("error", ""),
            "has_nan": rec.get("has_nan", False),
            "has_inf": rec.get("has_inf", False),
            "unexpected_labels": str(rec.get("unexpected_labels", [])),
            "wake_only": rec.get("wake_only", False),
            "has_all_5_stages": rec.get("has_all_5_stages", False),
            "missing_stages": str(rec.get("missing_stage_names", [])),
        })
    pd.DataFrame(qc_rows).to_csv(OUT_DIR / "qc_statistics.csv", index=False)

    # ── 5. wake-only subjects ────────────────────────────────────────────
    wake_df = df[df["wake_only"] == True].copy()
    if len(wake_df) > 0:
        wake_df.to_csv(OUT_DIR / "wake_only_subjects.csv", index=False)
        print(f"\n{'='*60}")
        print(f"WAKE-ONLY SUBJECTS ({len(wake_df)})")
        print(f"{'='*60}")
        for _, row in wake_df.iterrows():
            print(f"  {row['subject_id']}: {row['n_epochs']} epochs, "
                  f"{row.get('duration_hours', 0)}h, "
                  f"Wake={row.get('pct_Wake', 0)}%")
    else:
        print("\nNo wake-only subjects found.")

    # ── 6. subject quality manifest ──────────────────────────────────────
    quality_rows = []
    for rec in records:
        if rec["status"] != "ok":
            status = "excluded"
            reason = rec["error"]
        elif rec.get("wake_only", False):
            status = "excluded"
            reason = "wake_only_recording"
        else:
            status = "included"
            reason = "ok"

        quality_rows.append({
            "subject_id": rec["subject_id"],
            "status": status,
            "reason": reason,
            "included_in_training": status == "included",
            "included_in_cv": status == "included",
        })

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(OUT_DIR / "subject_quality_manifest.csv", index=False)

    included_count = quality_df["included_in_training"].sum()
    excluded_count = len(quality_df) - included_count
    print(f"\nQuality manifest: {included_count} included, {excluded_count} excluded")

    # ── 7. N1 distribution ───────────────────────────────────────────────
    valid_df = df[df["status"] == "ok"].copy()
    n1_epochs = valid_df["count_N1"].values
    n1_pcts = valid_df["pct_N1"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(n1_epochs, bins=30, edgecolor="black", alpha=0.7, color="#2196F3")
    axes[0].set_xlabel("N1 Epochs per Subject")
    axes[0].set_ylabel("Number of Subjects")
    axes[0].set_title("N1 Epoch Distribution (100 Subjects)")
    axes[0].axvline(np.median(n1_epochs), color="red", linestyle="--",
                     label=f"Median={np.median(n1_epochs):.0f}")
    axes[0].legend()

    axes[1].hist(n1_pcts, bins=30, edgecolor="black", alpha=0.7, color="#4CAF50")
    axes[1].set_xlabel("N1 Percentage per Subject (%)")
    axes[1].set_ylabel("Number of Subjects")
    axes[1].set_title("N1 Percentage Distribution (100 Subjects)")
    axes[1].axvline(np.median(n1_pcts), color="red", linestyle="--",
                     label=f"Median={np.median(n1_pcts):.1f}%")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "n1_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"N1 distribution plot saved to {OUT_DIR / 'n1_distribution.png'}")

    # N1 categories
    low_n1 = valid_df[valid_df["pct_N1"] < 2.0]
    normal_n1 = valid_df[(valid_df["pct_N1"] >= 2.0) & (valid_df["pct_N1"] <= 10.0)]
    high_n1 = valid_df[valid_df["pct_N1"] > 10.0]

    print(f"\nN1 categories:")
    print(f"  Low N1 (<2%):     {len(low_n1)} subjects")
    print(f"  Normal (2-10%):   {len(normal_n1)} subjects")
    print(f"  High N1 (>10%):   {len(high_n1)} subjects")

    # ── 8. class distribution overview bar chart ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    class_names = [LABEL_NAMES[i] for i in range(N_CLASSES)]
    class_counts = [class_totals[i] for i in range(N_CLASSES)]
    colors = ["#FF9800", "#F44336", "#2196F3", "#9C27B0", "#4CAF50"]

    bars = ax.bar(class_names, class_counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, class_counts):
        pct = count / total_epochs * 100 if total_epochs > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_epochs*0.005,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Number of Epochs")
    ax.set_title(f"Global Class Distribution — {valid_df.shape[0]} Valid Subjects, "
                 f"{total_epochs:,} Total Epochs")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 9. fold class balance ────────────────────────────────────────────
    folds = load_folds()
    fold_rows = []
    for fold_idx, fold_key in enumerate(sorted(folds.keys())):
        fold = folds[fold_key]
        test_subjects = fold["test"]
        fold_class_counts = {i: 0 for i in range(N_CLASSES)}
        fold_total = 0

        for subj_id in test_subjects:
            match = valid_df[valid_df["subject_id"] == subj_id]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            for label_id in range(N_CLASSES):
                fold_class_counts[label_id] += row[f"count_{LABEL_NAMES[label_id]}"]
                fold_total += row[f"count_{LABEL_NAMES[label_id]}"]

        for label_id in range(N_CLASSES):
            name = LABEL_NAMES[label_id]
            count = fold_class_counts[label_id]
            pct = count / fold_total * 100 if fold_total > 0 else 0
            fold_rows.append({
                "fold": fold_idx + 1,
                "class": name,
                "count": count,
                "percentage": round(pct, 2),
                "n_test_subjects": len(test_subjects),
            })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "class_distribution_by_fold.csv", index=False)

    # fold balance heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = fold_df.pivot_table(index="fold", columns="class", values="percentage")
    pivot = pivot[["Wake", "N1", "N2", "N3", "REM"]]

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Wake", "N1", "N2", "N3", "REM"])
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(10)])
    ax.set_title("Test Set Class Distribution by Fold (%)")

    for i in range(10):
        for j in range(5):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if val > 30 else "black")

    plt.colorbar(im, label="Percentage")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fold_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 10. dataset summary JSON ─────────────────────────────────────────
    included_subjects = quality_df[quality_df["included_in_training"]]["subject_id"].tolist()
    excluded_subjects = quality_df[~quality_df["included_in_training"]]["subject_id"].tolist()

    # recompute with only included subjects
    included_total_epochs = 0
    included_class_totals = {i: 0 for i in range(N_CLASSES)}

    for rec in records:
        if rec["status"] != "ok" or rec.get("wake_only", False):
            continue
        included_total_epochs += rec["n_epochs"]
        for label_id in range(N_CLASSES):
            included_class_totals[label_id] += rec[f"count_{LABEL_NAMES[label_id]}"]

    summary = {
        "audit_date": pd.Timestamp.now().isoformat(),
        "dataset": "Sleep-EDF Expanded",
        "dataset_version": "1.0.0",
        "subjects_total": total,
        "subjects_valid": int(ok),
        "subjects_excluded": int(errors + wake_only),
        "subjects_excluded_reasons": {
            "load_errors": int(errors),
            "wake_only": int(wake_only),
        },
        "included_subjects": len(included_subjects),
        "excluded_subjects": excluded_subjects,
        "total_epochs_all": total_epochs,
        "total_epochs_included": included_total_epochs,
        "class_distribution_all": {
            LABEL_NAMES[i]: {
                "count": class_totals[i],
                "pct": round(class_totals[i] / total_epochs * 100, 2) if total_epochs > 0 else 0,
            }
            for i in range(N_CLASSES)
        },
        "class_distribution_included": {
            LABEL_NAMES[i]: {
                "count": included_class_totals[i],
                "pct": round(included_class_totals[i] / included_total_epochs * 100, 2)
                if included_total_epochs > 0 else 0,
            }
            for i in range(N_CLASSES)
        },
        "n1_stats": {
            "median_epochs": float(np.median(n1_epochs)),
            "median_pct": float(np.median(n1_pcts)),
            "mean_epochs": float(np.mean(n1_epochs)),
            "mean_pct": float(np.mean(n1_pcts)),
            "min_epochs": int(np.min(n1_epochs)),
            "max_epochs": int(np.max(n1_epochs)),
            "low_n1_subjects": len(low_n1),
            "normal_n1_subjects": len(normal_n1),
            "high_n1_subjects": len(high_n1),
        },
        "quality_checks": {
            "all_labels_valid": bool(df["unexpected_labels"].apply(lambda x: x == "[]").all())
            if "unexpected_labels" in df.columns else True,
            "no_nan": bool(nan_count == 0),
            "no_inf": bool(inf_count == 0),
            "all_shapes_valid": bool(
                (valid_df["n_channels"] == 4).all() and (valid_df["n_samples"] == 3000).all()
            ) if len(valid_df) > 0 else False,
        },
        "folds": {
            "n_folds": 10,
            "method": "subject_level",
            "validation_subjects_fixed": 9,
        },
        "parameters": {
            "sampling_rate_hz": SAMPLING_RATE,
            "epoch_seconds": EPOCH_SECONDS,
            "samples_per_epoch": 3000,
            "n_channels": 4,
            "channels": ["Fpz-Cz", "Pz-Oz", "EOG", "EMG"],
        },
    }

    with open(OUT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset summary saved to {OUT_DIR / 'dataset_summary.json'}")

    # ── 11. per-subject class distribution CSV ───────────────────────────
    if len(valid_df) > 0:
        class_dist_rows = []
        for _, row in valid_df.iterrows():
            for label_id in range(N_CLASSES):
                name = LABEL_NAMES[label_id]
                class_dist_rows.append({
                    "subject_id": row["subject_id"],
                    "class": name,
                    "count": row[f"count_{name}"],
                    "percentage": row[f"pct_{name}"],
                })
        pd.DataFrame(class_dist_rows).to_csv(
            OUT_DIR / "per_subject_class_distribution.csv", index=False
        )

    # ── 12. summary table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUBJECTS WITH ALL 5 STAGES: {has_all5}/{ok} valid subjects")
    print(f"{'='*60}")

    if has_all5 < ok:
        partial = valid_df[~valid_df["has_all_5_stages"]].copy()
        print(f"\nSubjects missing stages:")
        for _, row in partial.iterrows():
            missing = row["missing_stage_names"]
            print(f"  {row['subject_id']}: missing {missing}")

    print(f"\n{'='*60}")
    print(f"TOP 10 SUBJECTS BY N1 PERCENTAGE")
    print(f"{'='*60}")
    top_n1 = valid_df.nlargest(10, "pct_N1")[["subject_id", "count_N1", "pct_N1"]]
    for _, row in top_n1.iterrows():
        print(f"  {row['subject_id']}: {row['count_N1']} N1 epochs ({row['pct_N1']:.1f}%)")

    print(f"\n{'='*60}")
    print(f"TOP 10 SUBJECTS BY N3 PERCENTAGE")
    print(f"{'='*60}")
    top_n3 = valid_df.nlargest(10, "pct_N3")[["subject_id", "count_N3", "pct_N3"]]
    for _, row in top_n3.iterrows():
        print(f"  {row['subject_id']}: {row['count_N3']} N3 epochs ({row['pct_N3']:.1f}%)")

    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETE — all outputs in {OUT_DIR}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
