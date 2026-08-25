#!/usr/bin/env python3
"""4-Fold Canonical Ablation - Simplified Version.

Runs experiments one at a time with proper intermediate saves.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.models.improved_student import ImprovedStudent
from sleep_staging.data.labels import CANONICAL_LIST, N_CLASSES
from sleep_staging.training.cross_dataset import (
    SequenceDataset, load_subjects, compute_class_weights,
    train_one_epoch, evaluate,
)
from sleep_staging.adaptation.lora import LoRAConfig, apply_lora

# Configuration
CACHE_DIR = "data/cache/sleep_edf"
CHECKPOINT = "artifacts/student_improved_best.pt"
MAX_EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = REPO / "results" / "canonical_ablation"

EXPERIMENTS = {
    "A0_frozen": {"method": "frozen", "targets": [], "lr": 3e-4},
    "A1_head": {"method": "lora", "targets": ["head"], "lr": 1e-3},
    "A3_cnn_head": {"method": "lora", "targets": ["enc.0.pw", "enc.1.pw", "head"], "lr": 1e-3},
    "A5_full_ft": {"method": "full_ft", "targets": [], "lr": 3e-4},
}


def load_folds():
    with open(REPO / "data" / "manifests" / "canonical_subject_folds.json") as f:
        return json.load(f)["folds"]


def build_model(method, targets):
    model = ImprovedStudent()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)

    if method == "lora" and targets:
        config = LoRAConfig(rank=8, alpha=16, target_modules=targets)
        model = apply_lora(model, config)
    elif method == "frozen":
        for p in model.parameters():
            p.requires_grad = False

    return model.to(DEVICE)


def run_fold(exp_name, config, fold_data, fold_num):
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    train_epochs, train_labels, _ = load_subjects(fold_data["train"], CACHE_DIR)
    val_epochs, val_labels, _ = load_subjects(fold_data["validation"], CACHE_DIR)
    test_epochs, test_labels, _ = load_subjects(fold_data["test"], CACHE_DIR)

    train_ds = SequenceDataset(train_epochs, train_labels, seq_len=10, stride=5)
    val_ds = SequenceDataset(val_epochs, val_labels, seq_len=10, stride=5)
    test_ds = SequenceDataset(test_epochs, test_labels, seq_len=10, stride=5)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, pin_memory=True)

    # Build model
    model = build_model(config["method"], config["targets"])

    # Loss and optimizer
    class_weights = compute_class_weights(train_labels, n1_weight=2.0, rem_weight=2.0).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if config["method"] != "frozen":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"], weight_decay=1e-2)
    else:
        optimizer = None

    # Training
    best_val_f1 = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        if config["method"] != "frozen":
            train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate on test
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return evaluate(model, test_loader, criterion, DEVICE)


def main():
    print("=" * 80)
    print("  4-FOLD CANONICAL ABLATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    folds = load_folds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing results
    results_file = OUTPUT_DIR / "4fold_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print("Loaded existing results")
    else:
        all_results = {exp: {} for exp in EXPERIMENTS}

    for fold_num in range(1, 5):
        fold_key = f"fold_{fold_num}"
        fold_data = folds[fold_key]

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_num} - Test: {fold_data['test']}")
        print(f"{'='*60}")

        for exp_name, config in EXPERIMENTS.items():
            # Skip if already completed
            if str(fold_num) in all_results.get(exp_name, {}):
                print(f"  {exp_name}: Already completed")
                continue

            print(f"  Running {exp_name}...", end=" ", flush=True)
            t0 = time.time()

            metrics = run_fold(exp_name, config, fold_data, fold_num)
            elapsed = time.time() - t0

            # Store results
            if exp_name not in all_results:
                all_results[exp_name] = {}
            all_results[exp_name][str(fold_num)] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "kappa": metrics["kappa"],
                "n1_f1": metrics["per_class"].get("N1", {}).get("f1", 0),
                "rem_f1": metrics["per_class"].get("REM", {}).get("f1", 0),
            }

            # Save after each fold
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"Done ({elapsed:.1f}s)")
            print(f"    Acc={metrics['accuracy']:.3f} F1={metrics['macro_f1']:.3f} "
                  f"N1={metrics['per_class'].get('N1', {}).get('f1', 0):.3f}")

    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS (4-FOLD)")
    print("=" * 80)

    for exp_name in EXPERIMENTS:
        if exp_name in all_results:
            accs = [v["accuracy"] for v in all_results[exp_name].values()]
            f1s = [v["macro_f1"] for v in all_results[exp_name].values()]
            n1s = [v["n1_f1"] for v in all_results[exp_name].values()]
            print(f"{exp_name}:")
            print(f"  Acc: {np.mean(accs):.3f}±{np.std(accs):.3f}")
            print(f"  F1:  {np.mean(f1s):.3f}±{np.std(f1s):.3f}")
            print(f"  N1:  {np.mean(n1s):.3f}±{np.std(n1s):.3f}")


if __name__ == "__main__":
    main()
