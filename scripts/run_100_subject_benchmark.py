#!/usr/bin/env python3
"""
100-Subject Full Benchmark — 10-Fold Subject-Level Cross-Validation

Trains the Improved Student (99,477 params) on 92 included Sleep-EDF subjects
using 10-fold subject-level CV. Saves per-fold metrics, confusion matrices,
and predictions.

Usage:
    python scripts/run_100_subject_benchmark.py --seed 42 --device cuda
    python scripts/run_100_subject_benchmark.py --seed 42 --device cuda --fold 1
"""

import argparse
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
    precision_recall_fscore_support, confusion_matrix,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.models.improved_student import ImprovedStudent, count_parameters
from sleep_staging.data.loader import load_cached_subject
from sleep_staging.data.labels import CANONICAL_LIST, N_CLASSES
from sleep_staging.training.cross_dataset import (
    SequenceDataset, compute_class_weights,
)

# ── Config ───────────────────────────────────────────────────────────────
CACHE_DIR = REPO / "data" / "cache" / "sleep_edf"
FOLDS_PATH = REPO / "data" / "manifests" / "canonical_subject_folds_92subj.json"
OUTPUT_DIR = REPO / "results" / "full_100_subject"

MAX_EPOCHS = 20
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEQ_LEN = 10
SEQ_STRIDE = 5
N1_WEIGHT = 2.0
REM_WEIGHT = 2.0
PATIENCE = 5  # early stopping patience


def load_folds():
    with open(FOLDS_PATH) as f:
        return json.load(f)["folds"]


def load_subjects(subject_ids, cache_dir):
    all_epochs, all_labels, loaded = [], [], []
    for sid in subject_ids:
        try:
            data = load_cached_subject(sid, cache_dir)
            all_epochs.append(data["epochs"])
            all_labels.append(data["labels"])
            loaded.append(sid)
        except FileNotFoundError:
            print(f"  WARNING: Cache not found for {sid}, skipping")
    if not all_epochs:
        raise ValueError("No subjects loaded")
    return np.concatenate(all_epochs), np.concatenate(all_labels), loaded


def build_dataloaders(train_subjects, val_subjects, test_subjects, cache_dir):
    print(f"  Loading {len(train_subjects)} train subjects...")
    train_epochs, train_labels, _ = load_subjects(train_subjects, cache_dir)
    print(f"    Train epochs: {len(train_epochs):,}")

    print(f"  Loading {len(val_subjects)} val subjects...")
    val_epochs, val_labels, _ = load_subjects(val_subjects, cache_dir)
    print(f"    Val epochs: {len(val_epochs):,}")

    print(f"  Loading {len(test_subjects)} test subjects...")
    test_epochs, test_labels, _ = load_subjects(test_subjects, cache_dir)
    print(f"    Test epochs: {len(test_epochs):,}")

    train_ds = SequenceDataset(train_epochs, train_labels, SEQ_LEN, SEQ_STRIDE)
    val_ds = SequenceDataset(val_epochs, val_labels, SEQ_LEN, SEQ_STRIDE)
    test_ds = SequenceDataset(test_epochs, test_labels, SEQ_LEN, SEQ_STRIDE)

    print(f"  Train windows: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_labels


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits.view(-1, N_CLASSES), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits.view(-1, N_CLASSES), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits.view(-1, N_CLASSES), y.view(-1))
        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
        all_labels.append(y.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, labels=list(range(N_CLASSES)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(N_CLASSES)), zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(CANONICAL_LIST):
        per_class[name] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }

    # MGm (geometric mean of per-class recalls)
    recalls_vals = [per_class[name]["recall"] for name in CANONICAL_LIST]
    mgm = float(np.exp(np.mean(np.log(np.maximum(np.array(recalls_vals), 1e-8)))))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(N_CLASSES)))

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mgm": mgm,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(all_labels),
    }


def run_fold(fold_num, fold_data, seed, device, output_dir, out_suffix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)

    fold_dir = output_dir / f"fold_{fold_num:02d}{out_suffix}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_subjects = fold_data["train"]
    val_subjects = fold_data["validation"]
    test_subjects = fold_data["test"]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num} — Seed {seed}")
    print(f"  Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")
    print(f"  Test subjects: {test_subjects}")
    print(f"{'='*60}")

    # Build data
    train_loader, val_loader, test_loader, train_labels = build_dataloaders(
        train_subjects, val_subjects, test_subjects, CACHE_DIR,
    )

    # Build model
    model = ImprovedStudent().to(device)
    print(f"  Model parameters: {count_parameters(model):,}")

    # Loss
    class_weights = compute_class_weights(train_labels, N1_WEIGHT, REM_WEIGHT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    history = []

    print(f"\n  Training for {MAX_EPOCHS} epochs...")
    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_kappa": val_metrics["kappa"],
            "val_macro_f1": val_metrics["macro_f1"],
            "time_s": elapsed,
        })

        marker = ""
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *best*"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch+1:2d}/{MAX_EPOCHS} ({elapsed:.1f}s): "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_F1={val_metrics['macro_f1']:.3f}{marker}"
        )

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n  TEST RESULTS (Fold {fold_num}):")
    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Kappa:     {test_metrics['kappa']:.4f}")
    print(f"    Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"    MGm:       {test_metrics['mgm']:.4f}")
    for name in CANONICAL_LIST:
        pc = test_metrics["per_class"][name]
        print(f"    {name:5s}: P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")

    # Save predictions
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
            all_labels.append(y.numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pred_df = pd.DataFrame({
        "true_label": all_labels,
        "pred_label": all_preds,
        "true_name": [CANONICAL_LIST[int(l)] for l in all_labels],
        "pred_name": [CANONICAL_LIST[int(p)] for p in all_preds],
    })
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)

    # Save confusion matrix
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=[f"true_{n}" for n in CANONICAL_LIST],
        columns=[f"pred_{n}" for n in CANONICAL_LIST],
    )
    cm_df.to_csv(fold_dir / "confusion_matrix.csv")

    # Save metrics
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save history
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    # Save checkpoint
    if best_state:
        torch.save({
            "model_state_dict": best_state,
            "fold": fold_num,
            "seed": seed,
            "test_metrics": test_metrics,
        }, fold_dir / "best_model.pt")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="100-Subject Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold only")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix for per-fold output dirs, e.g. '_seed43' "
                             "to avoid overwriting seed 42 artifacts")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("  100-SUBJECT FULL BENCHMARK — 10-Fold Subject-Level CV")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence: {SEQ_LEN} epochs, stride {SEQ_STRIDE}")
    print(f"  Class weights: N1={N1_WEIGHT}x, REM={REM_WEIGHT}x")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = load_folds()

    # Check for existing results
    results_file = OUTPUT_DIR / f"benchmark_seed{args.seed}.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {results_file}")
    else:
        all_results = {}

    fold_range = [args.fold] if args.fold else range(1, 11)

    for fold_num in fold_range:
        fold_key = f"fold_{fold_num}"
        if fold_key not in folds:
            print(f"Fold {fold_num} not found in folds file")
            continue

        if str(fold_num) in all_results:
            print(f"\nFold {fold_num}: Already completed, skipping")
            continue

        fold_data = folds[fold_key]
        t0 = time.time()

        metrics = run_fold(fold_num, fold_data, args.seed, device, OUTPUT_DIR,
                           out_suffix=args.out_suffix)
        elapsed = time.time() - t0

        # Store results
        all_results[str(fold_num)] = {
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mgm": metrics["mgm"],
            "per_class": metrics["per_class"],
            "test_subjects": fold_data["test"],
            "n_test_subjects": len(fold_data["test"]),
            "time_s": elapsed,
        }

        # Save after each fold
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n  Fold {fold_num} completed in {elapsed:.1f}s")

    # Final summary
    if all_results:
        print("\n" + "=" * 70)
        print(f"  FINAL RESULTS — Seed {args.seed}")
        print("=" * 70)

        accs = [v["accuracy"] for v in all_results.values()]
        kappas = [v["kappa"] for v in all_results.values()]
        f1s = [v["macro_f1"] for v in all_results.values()]
        wf1s = [v["weighted_f1"] for v in all_results.values()]
        mgms = [v["mgm"] for v in all_results.values()]

        print(f"  Folds completed: {len(all_results)}/10")
        print(f"  Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Kappa:        {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        print(f"  Macro F1:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  Weighted F1:  {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")
        print(f"  MGm:          {np.mean(mgms):.4f} ± {np.std(mgms):.4f}")

        # Per-class averages
        print(f"\n  Per-class F1:")
        for name in CANONICAL_LIST:
            class_f1s = [v["per_class"][name]["f1"] for v in all_results.values()]
            print(f"    {name:5s}: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")

        # Save summary
        summary = {
            "seed": args.seed,
            "n_folds": len(all_results),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "weighted_f1_mean": float(np.mean(wf1s)),
            "weighted_f1_std": float(np.std(wf1s)),
            "mgm_mean": float(np.mean(mgms)),
            "mgm_std": float(np.std(mgms)),
            "per_class_f1": {
                name: {
                    "mean": float(np.mean([v["per_class"][name]["f1"] for v in all_results.values()])),
                    "std": float(np.std([v["per_class"][name]["f1"] for v in all_results.values()])),
                }
                for name in CANONICAL_LIST
            },
        }

        with open(OUTPUT_DIR / f"summary_seed{args.seed}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Results saved to {OUTPUT_DIR / f'summary_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
