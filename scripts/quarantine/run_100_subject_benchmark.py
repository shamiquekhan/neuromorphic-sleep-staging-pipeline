#!/usr/bin/env python3
"""
Person-Level Benchmark — Subject-Safe Sequences + Causal Unique-Epoch Eval

Trains the Improved Student (99,477 params) on the 92-record / 52-person
Sleep-EDF cohort using 10-fold person-level CV.

Protocol (post-audit, supersedes the stride-5 / all-position protocol):
  * TRAIN: windows built per subject (never cross a subject boundary or
    an unlabeled-epoch gap), all positions supervised, stride 5.
  * VAL/TEST: stride-1 causal windows, supervising only the last epoch —
    exactly one prediction per scored epoch, with (subject, epoch)
    provenance (see src/sleep_staging/evaluation/protocol.py).

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.models.improved_student import ImprovedStudent, count_parameters
from sleep_staging.data.loader import load_cached_subject
from sleep_staging.data.sequence_dataset import (
    SubjectSequenceDataset, CausalEvalDataset, make_subject_list,
)
from sleep_staging.data.labels import CANONICAL_LIST, N_CLASSES
from sleep_staging.evaluation.protocol import evaluate_causal
from sleep_staging.training.cross_dataset import compute_class_weights
from sleep_staging.training.seed import seed_everything, worker_init_fn
from sklearn.metrics import confusion_matrix

# ── Config ───────────────────────────────────────────────────────────────
CACHE_DIR = REPO / "data" / "cache" / "sleep_edf"
FOLDS_PATH = REPO / "data" / "manifests" / "person_folds_52subj.json"
OUTPUT_DIR = REPO / "results" / "benchmark_92_subject"

MAX_EPOCHS = 20
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEQ_LEN = 10
SEQ_STRIDE = 5          # training windows only
N1_WEIGHT = 2.0
REM_WEIGHT = 2.0
PATIENCE = 5  # early stopping patience


def load_folds(folds_path=FOLDS_PATH):
    with open(folds_path) as f:
        return json.load(f)["folds"]


def verify_person_disjointness(folds, folds_path):
    """Fail fast if any person's records are split across train/test/val.

    SC4ss1/SC4ss2 are two nights of the same person (PhysioNet
    sleep-edfx naming; see data/manifests/person_groups.json). A fold
    whose train set contains the same-person mate of a test record
    measures record-level, not person-level, generalization.
    """
    def person_of(r):
        if not (r.startswith("SC") and len(r) == 6 and r[5] in "12"):
            raise ValueError(f"unexpected record id {r}")
        return r[:5]

    problems = []
    for name, fold in folds.items():
        tr = {person_of(r) for r in fold["train"]}
        te = {person_of(r) for r in fold["test"]}
        va = {person_of(r) for r in fold["validation"]}
        if tr & te or tr & va or te & va:
            problems.append(name)
    if problems:
        raise SystemExit(
            f"REFUSING TO RUN: person-level leakage in {folds_path}: folds "
            f"{problems} train on the same-person mate of evaluation "
            "records. Use person_folds_52subj.json (generate via "
            "scripts/generate_person_folds.py) or fix the manifest."
        )


def load_subjects(subject_ids, cache_dir):
    """Load per-subject caches WITHOUT concatenation.

    Unlike the legacy ``cross_dataset.load_subjects`` (which concatenated
    all subjects into one array and thereby allowed windows to span
    subject boundaries), this keeps each subject as the unit of
    sequence construction. Missing caches raise immediately instead of
    being silently skipped, so a fold can never quietly shrink.
    """
    subjects = []
    for sid in subject_ids:
        data = load_cached_subject(sid, cache_dir)  # raises FileNotFoundError
        subjects.append(data)
    if not subjects:
        raise ValueError("No subjects loaded")
    return subjects


def build_dataloaders(train_subjects, val_subjects, test_subjects, cache_dir):
    """Build train loader + causal val/test datasets (loaded once).

    Returns (train_loader, val_ds, test_ds, train_labels). The causal
    datasets double as DataLoaders' backing store — callers iterate
    them via DataLoader(val_ds, ...) or pass them directly to
    evaluate(); subjects are loaded exactly once (no double caching of
    ~3000x4x3000 float32 arrays per subject).
    """
    print(f"  Loading {len(train_subjects)} train subjects...")
    train_list = load_subjects(train_subjects, cache_dir)
    n_train_epochs = sum(len(s["labels"]) for s in train_list)
    print(f"    Train epochs: {n_train_epochs:,}")

    print(f"  Loading {len(val_subjects)} val subjects...")
    val_list = load_subjects(val_subjects, cache_dir)
    print(f"    Val epochs: {len(val_list):,} subjects, "
          f"{sum(len(s['labels']) for s in val_list):,} epochs")

    print(f"  Loading {len(test_subjects)} test subjects...")
    test_list = load_subjects(test_subjects, cache_dir)
    print(f"    Test epochs: {sum(len(s['labels']) for s in test_list):,} "
          f"across {len(test_list)} subjects")

    # Subject-safe training windows (P0.1) — all-position supervision is
    # a training signal only; reporting uses the causal protocol below.
    train_ds = SubjectSequenceDataset(train_list, SEQ_LEN, SEQ_STRIDE)

    # Causal unique-epoch evaluation (P0.6): stride-1, last-epoch target.
    val_ds = CausalEvalDataset(val_list, SEQ_LEN)
    test_ds = CausalEvalDataset(test_list, SEQ_LEN)

    for role, ds in (("val", val_ds), ("test", test_ds)):
        gap_subjs = ds.subject_ids_with_gaps()
        if gap_subjs:
            print(f"    WARNING [{role}]: {len(gap_subjs)} subjects on legacy "
                  "caches without orig_epoch_idx — temporal-gap protection "
                  "unavailable; regenerate caches (scripts/"
                  "preprocess_sleep_edf_expanded.py)")
        if ds.gap_excluded:
            n_ex = sum(ds.gap_excluded.values())
            print(f"    NOTE [{role}]: {n_ex} epochs excluded — causal "
                  f"context spans a dropped epoch in {len(ds.gap_excluded)} "
                  "subjects (unscoreable under contiguous-context protocol)")

    print(f"  Train windows: {len(train_ds):,} | "
          f"Val scored epochs: {len(val_ds):,} | "
          f"Test scored epochs: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    # Class weights from the true per-epoch train label distribution.
    train_labels = np.concatenate([s["labels"] for s in train_list])

    return train_loader, val_ds, test_ds, train_labels


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train over subject-safe windows with all-position supervision.

    Window boundaries never cross subjects (SubjectSequenceDataset), so
    all-position supervision here is a training signal only — no
    reporting happens on overlapping windows.
    """
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


def evaluate(model, dataset, criterion, device):
    """Causal unique-epoch evaluation (P0.6).

    ``dataset`` is a CausalEvalDataset; every scored epoch receives
    exactly one prediction from the 5 minutes of context ending at it.
    ``criterion`` is accepted for signature compatibility but unused —
    loss is not part of the causal reporting protocol.
    """
    return evaluate_causal(
        model, dataset, device, batch_size=BATCH_SIZE, num_workers=0,
    )


def run_fold(fold_num, fold_data, seed, device, output_dir, out_suffix=""):
    seed_everything(seed)  # python/numpy/torch/cuda/cudnn, all controlled

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

    # Build data (datasets loaded once; val/test are causal datasets)
    train_loader, val_ds, test_ds, train_labels = build_dataloaders(
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
        val_metrics = evaluate(model, val_ds, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
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
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_F1={val_metrics['macro_f1']:.3f}{marker}"
        )

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    test_metrics = evaluate(model, test_ds, criterion, device)

    print(f"\n  TEST RESULTS (Fold {fold_num}) — causal unique-epoch protocol:")
    print(f"    Scored epochs: {test_metrics['n_unique_epochs']:,} "
          f"({test_metrics['n_subjects']} subjects)")
    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Kappa:     {test_metrics['kappa']:.4f}")
    print(f"    Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"    MGm:       {test_metrics['mgm']:.4f}")
    for name in CANONICAL_LIST:
        pc = test_metrics["per_class"][name]
        print(f"    {name:5s}: P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")

    # Save predictions with (subject, epoch) provenance — one row per
    # unique scored epoch (P0.6).
    preds = test_metrics["predictions"]
    pred_df = pd.DataFrame({
        "subject_id": preds["subject_id"],
        "epoch_index": preds["epoch_index"],
        "true_label": preds["y_true"],
        "pred_label": preds["y_pred"],
        "true_name": [CANONICAL_LIST[int(l)] for l in preds["y_true"]],
        "pred_name": [CANONICAL_LIST[int(p)] for p in preds["y_pred"]],
    })
    for i, name in enumerate(CANONICAL_LIST):
        pred_df[f"prob_{name}"] = preds["probs"][:, i]
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)

    # Save confusion matrix
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=[f"true_{n}" for n in CANONICAL_LIST],
        columns=[f"pred_{n}" for n in CANONICAL_LIST],
    )
    cm_df.to_csv(fold_dir / "confusion_matrix.csv")

    # Save metrics (predictions are serialized separately above; the
    # numpy arrays inside would not round-trip through json).
    serializable_metrics = {
        k: v for k, v in test_metrics.items() if k != "predictions"
    }
    serializable_metrics["confusion_matrix"] = np.asarray(
        test_metrics["confusion_matrix"]
    ).tolist()
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Save history
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    # Save checkpoint
    if best_state:
        torch.save({
            "model_state_dict": best_state,
            "fold": fold_num,
            "seed": seed,
            "test_metrics": serializable_metrics,
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
    parser.add_argument("--folds-manifest", default=str(FOLDS_PATH),
                        help="Folds manifest. Default is the legacy "
                             "record-level folds (will FAIL the "
                             "person-disjointness guard — use "
                             "data/manifests/person_folds_52subj.json "
                             "for person-level evaluation)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Results directory (default results/benchmark_92_subject)")
    parser.add_argument("--allow-record-level", action="store_true",
                        help="Explicitly acknowledge running on record-level "
                             "(leaky) folds; output is labeled record-level")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    folds = load_folds(Path(args.folds_manifest))
    if not args.allow_record_level:
        verify_person_disjointness(folds, args.folds_manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_kind = ("record-level (leaky)" if args.allow_record_level
                     else "person-level")

    # Everything from here on is teed into
    # <output_dir>/run_logs/benchmark_seed<seed>.log — the console log
    # is preserved as run evidence next to the metrics it produced.
    from sleep_staging.utils.runlog import tee_run_log

    with tee_run_log(output_dir, f"benchmark_seed{args.seed}"):
        _run_benchmark(args, folds, device, output_dir, manifest_kind)


def _run_benchmark(args, folds, device, output_dir, manifest_kind):
    print("=" * 70)
    print("  FULL BENCHMARK — 10-Fold Cross-Validation")
    print(f"  Folds: {args.folds_manifest} ({manifest_kind})")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {MAX_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Train windows: {SEQ_LEN} epochs, stride {SEQ_STRIDE} (subject-safe)")
    print(f"  Eval protocol: causal, stride 1, last-epoch (unique epochs)")
    print(f"  Class weights: N1={N1_WEIGHT}x, REM={REM_WEIGHT}x")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Check for existing results
    results_file = output_dir / f"benchmark_seed{args.seed}.json"
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

        metrics = run_fold(fold_num, fold_data, args.seed, device, output_dir,
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
            "per_class_accuracy": metrics["per_class_accuracy"],
            "n_unique_epochs": metrics["n_unique_epochs"],
            "protocol": "causal_unique_epoch",
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
        print(f"\n  Per-class F1 (unique-epoch protocol):")
        for name in CANONICAL_LIST:
            class_f1s = [v["per_class"][name]["f1"] for v in all_results.values()]
            print(f"    {name:5s}: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")

        # Save summary
        summary = {
            "seed": args.seed,
            "n_folds": len(all_results),
            "folds_manifest": args.folds_manifest,
            "split_level": "record" if args.allow_record_level else "person",
            "protocol": "causal_unique_epoch",
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

        with open(output_dir / f"summary_seed{args.seed}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Results saved to {output_dir / f'summary_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
