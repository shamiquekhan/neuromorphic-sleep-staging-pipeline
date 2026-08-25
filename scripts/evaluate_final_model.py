#!/usr/bin/env python3
"""Final Model Evaluation Script.

Generates the authoritative result package for the NeuroSleep project.
Evaluates the final Full Fine-Tuned model across all 4 canonical folds
with complete per-class metrics.

Usage:
    python scripts/evaluate_final_model.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.models.improved_student import ImprovedStudent
from sleep_staging.data.labels import CANONICAL_LIST, N_CLASSES
from sleep_staging.training.cross_dataset import (
    SequenceDataset, load_subjects, compute_class_weights, evaluate,
)

# Configuration
CACHE_DIR = REPO / "data" / "cache" / "sleep_edf"
CHECKPOINT = REPO / "artifacts" / "final" / "student_full_finetuned.pt"
FOLDS_PATH = REPO / "data" / "manifests" / "canonical_subject_folds.json"
OUTPUT_DIR = REPO / "results" / "final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 15


def load_folds():
    """Load canonical subject folds."""
    with open(FOLDS_PATH) as f:
        return json.load(f)["folds"]


def train_fold(fold_data, fold_num):
    """Train model on one fold and return best state."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    train_epochs, train_labels, _ = load_subjects(fold_data["train"], CACHE_DIR)
    val_epochs, val_labels, _ = load_subjects(fold_data["validation"], CACHE_DIR)

    train_ds = SequenceDataset(train_epochs, train_labels, seq_len=10, stride=5)
    val_ds = SequenceDataset(val_epochs, val_labels, seq_len=10, stride=5)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, pin_memory=True)

    # Build model
    model = ImprovedStudent().to(DEVICE)
    class_weights = compute_class_weights(train_labels, n1_weight=2.0, rem_weight=2.0).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # Training
    best_val_f1 = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, N_CLASSES), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state


@torch.no_grad()
def evaluate_fold(model, fold_data):
    """Evaluate model on test set with complete metrics."""
    model.eval()

    test_epochs, test_labels, _ = load_subjects(fold_data["test"], CACHE_DIR)
    test_ds = SequenceDataset(test_epochs, test_labels, seq_len=10, stride=5)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, pin_memory=True)

    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

        all_preds.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
        all_labels.append(y.cpu().numpy().reshape(-1))
        all_probs.append(probs.cpu().numpy().reshape(-1, N_CLASSES))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, labels=list(range(N_CLASSES)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # MGm (geometric mean of per-class recall)
    per_class_recall = []
    for i in range(N_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_recall.append((all_preds[mask] == i).sum() / mask.sum())
        else:
            per_class_recall.append(0)
    mgm = np.exp(np.mean(np.log(np.maximum(per_class_recall, 1e-10))))

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(CANONICAL_LIST):
        mask = all_labels == i
        if mask.sum() > 0:
            class_preds = all_preds[mask]
            binary_preds = (class_preds == i).astype(int)
            binary_labels = np.ones_like(binary_preds)

            per_class[name] = {
                "precision": float(precision_score(binary_labels, binary_preds, zero_division=0)),
                "recall": float(recall_score(binary_labels, binary_preds, zero_division=0)),
                "f1": float(f1_score(binary_labels, binary_preds, zero_division=0)),
                "support": int(mask.sum()),
                "mean_prob": float(all_probs[mask, i].mean()),
            }
        else:
            per_class[name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "mean_prob": 0.0,
            }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(N_CLASSES)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mgm": float(mgm),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
    }


def main():
    print("=" * 80)
    print("  NEUROSLEEP - FINAL MODEL EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load folds
    folds = load_folds()

    # Verify checkpoint exists
    if not CHECKPOINT.exists():
        print(f"\nCheckpoint not found at {CHECKPOINT}")
        print("Training final model from scratch...")
        # Train on fold 1 as representative
        fold_data = folds["fold_1"]
        state_dict = train_fold(fold_data, 1)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, CHECKPOINT)
        print(f"Saved to {CHECKPOINT}")

    # Evaluate all folds
    all_results = []
    fold_metrics = []

    for fold_num in range(1, 5):
        fold_key = f"fold_{fold_num}"
        fold_data = folds[fold_key]

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_num} - Test: {fold_data['test']}")
        print(f"{'='*60}")

        # Load checkpoint
        model = ImprovedStudent()
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model = model.to(DEVICE)

        # Train fresh model for this fold
        print(f"  Training fold {fold_num}...", end=" ", flush=True)
        t0 = time.time()
        state_dict = train_fold(fold_data, fold_num)
        elapsed = time.time() - t0
        print(f"Done ({elapsed:.1f}s)")

        # Load trained weights
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)

        # Evaluate
        print(f"  Evaluating...", end=" ", flush=True)
        metrics = evaluate_fold(model, fold_data)
        print("Done")

        # Store results
        all_results.append(metrics)
        fold_metrics.append({
            "fold": fold_num,
            "test_subject": fold_data["test"][0],
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mgm": metrics["mgm"],
        })

        print(f"  Acc={metrics['accuracy']:.3f} F1={metrics['macro_f1']:.3f} "
              f"Kappa={metrics['kappa']:.3f}")

        # Save fold predictions
        fold_df = pd.DataFrame({
            "fold": fold_num,
            "true": metrics["labels"],
            "pred": metrics["predictions"],
        })
        fold_df.to_csv(OUTPUT_DIR / f"predictions_fold{fold_num}.csv", index=False)

        # Save fold confusion matrix
        cm_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=CANONICAL_LIST,
            columns=CANONICAL_LIST,
        )
        cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_fold{fold_num}.csv")

        cm_norm_df = pd.DataFrame(
            metrics["confusion_matrix_normalized"],
            index=CANONICAL_LIST,
            columns=CANONICAL_LIST,
        )
        cm_norm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_normalized_fold{fold_num}.csv")

    # Aggregate per-class metrics across folds
    print("\n" + "=" * 80)
    print("  AGGREGATING RESULTS")
    print("=" * 80)

    # Compute mean and std for per-class metrics
    per_class_agg = {}
    for stage in CANONICAL_LIST:
        f1s = [r["per_class"][stage]["f1"] for r in all_results]
        precisions = [r["per_class"][stage]["precision"] for r in all_results]
        recalls = [r["per_class"][stage]["recall"] for r in all_results]
        supports = [r["per_class"][stage]["support"] for r in all_results]

        per_class_agg[stage] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "support_mean": float(np.mean(supports)),
        }

    # Compute overall metrics
    overall = {
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics])),
        "kappa_mean": float(np.mean([m["kappa"] for m in fold_metrics])),
        "kappa_std": float(np.std([m["kappa"] for m in fold_metrics])),
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "macro_f1_std": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "weighted_f1_mean": float(np.mean([m["weighted_f1"] for m in fold_metrics])),
        "weighted_f1_std": float(np.std([m["weighted_f1"] for m in fold_metrics])),
        "mgm_mean": float(np.mean([m["mgm"] for m in fold_metrics])),
        "mgm_std": float(np.std([m["mgm"] for m in fold_metrics])),
    }

    # Create final metrics JSON
    final_metrics = {
        "model": "Improved Student - Full Fine-Tuning",
        "parameters": 99477,
        "dataset": "Sleep-EDF Expanded",
        "n_subjects": 15,
        "n_folds": 4,
        "evaluation": "subject-level 4-fold cross-validation",
        "accuracy": {
            "mean": overall["accuracy_mean"],
            "std": overall["accuracy_std"],
        },
        "cohen_kappa": {
            "mean": overall["kappa_mean"],
            "std": overall["kappa_std"],
        },
        "macro_f1": {
            "mean": overall["macro_f1_mean"],
            "std": overall["macro_f1_std"],
        },
        "weighted_f1": {
            "mean": overall["weighted_f1_mean"],
            "std": overall["weighted_f1_std"],
        },
        "mgm": {
            "mean": overall["mgm_mean"],
            "std": overall["mgm_std"],
        },
        "per_class": per_class_agg,
        "fold_metrics": fold_metrics,
    }

    # Save final metrics
    with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save fold metrics CSV
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    # Save per-class metrics CSV
    per_class_rows = []
    for stage in CANONICAL_LIST:
        for fold_idx, r in enumerate(all_results, 1):
            per_class_rows.append({
                "fold": fold_idx,
                "stage": stage,
                "precision": r["per_class"][stage]["precision"],
                "recall": r["per_class"][stage]["recall"],
                "f1": r["per_class"][stage]["f1"],
                "support": r["per_class"][stage]["support"],
            })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

    # Save confusion matrices
    cm_sum = np.zeros((N_CLASSES, N_CLASSES))
    cm_norm_sum = np.zeros((N_CLASSES, N_CLASSES))
    for r in all_results:
        cm_sum += np.array(r["confusion_matrix"])
        cm_norm_sum += np.array(r["confusion_matrix_normalized"])
    cm_mean = cm_sum / len(all_results)
    cm_norm_mean = cm_norm_sum / len(all_results)

    cm_df = pd.DataFrame(cm_mean, index=CANONICAL_LIST, columns=CANONICAL_LIST)
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    cm_norm_df = pd.DataFrame(cm_norm_mean, index=CANONICAL_LIST, columns=CANONICAL_LIST)
    cm_norm_df.to_csv(OUTPUT_DIR / "confusion_matrix_normalized.csv")

    # Merge all predictions
    all_pred_dfs = []
    for fold_num in range(1, 5):
        pred_file = OUTPUT_DIR / f"predictions_fold{fold_num}.csv"
        if pred_file.exists():
            all_pred_dfs.append(pd.read_csv(pred_file))
    if all_pred_dfs:
        pd.concat(all_pred_dfs).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)

    print(f"\nModel: Improved Student - Full Fine-Tuning")
    print(f"Parameters: 99,477")
    print(f"Dataset: Sleep-EDF Expanded (15 subjects)")
    print(f"Evaluation: 4-fold subject-level cross-validation")

    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {overall['accuracy_mean']:>10.3f} {overall['accuracy_std']:>10.3f}")
    print(f"{'Cohen Kappa':<20} {overall['kappa_mean']:>10.3f} {overall['kappa_std']:>10.3f}")
    print(f"{'Macro F1':<20} {overall['macro_f1_mean']:>10.3f} {overall['macro_f1_std']:>10.3f}")
    print(f"{'Weighted F1':<20} {overall['weighted_f1_mean']:>10.3f} {overall['weighted_f1_std']:>10.3f}")
    print(f"{'MGm':<20} {overall['mgm_mean']:>10.3f} {overall['mgm_std']:>10.3f}")

    print(f"\nPer-Class F1 Scores:")
    print(f"{'Stage':<10} {'F1 Mean':>10} {'F1 Std':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 50)
    for stage in CANONICAL_LIST:
        agg = per_class_agg[stage]
        print(f"{stage:<10} {agg['f1_mean']:>10.3f} {agg['f1_std']:>10.3f} "
              f"{agg['precision_mean']:>10.3f} {agg['recall_mean']:>10.3f}")

    print(f"\nResults saved to {OUTPUT_DIR / 'final_metrics.json'}")
    print("Done!")


if __name__ == "__main__":
    main()
