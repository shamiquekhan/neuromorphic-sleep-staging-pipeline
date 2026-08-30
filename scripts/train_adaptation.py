#!/usr/bin/env python3
"""
100-Subject Adaptation Study — Frozen / LoRA / Full Fine-Tuning

All three modes start from the same base checkpoint and are evaluated on the
same 10 subject-level folds (canonical_subject_folds_92subj.json) used by the
full_100_subject benchmark.

Modes:
    frozen          Load base checkpoint, freeze everything, evaluate test folds only.
    lora            Load base checkpoint, freeze base, train LoRA adapters per fold.
    full_finetune   Load base checkpoint, train all parameters per fold.

Usage:
    python scripts/train_adaptation.py --mode frozen --fold 1 --device cuda
    python scripts/train_adaptation.py --mode lora --rank 8 --alpha 16 \
        --targets enc.0.pw,enc.1.pw,head --fold 1 --device cuda
    python scripts/train_adaptation.py --mode full_finetune --fold 1 --device cuda
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
from sklearn.metrics import confusion_matrix

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_100_subject_benchmark import (
    CACHE_DIR, FOLDS_PATH, MAX_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    GRAD_CLIP, SEQ_LEN, SEQ_STRIDE, N1_WEIGHT, REM_WEIGHT, PATIENCE,
    load_folds, load_subjects, build_dataloaders, train_one_epoch, evaluate,
)

from sleep_staging.models.improved_student import ImprovedStudent, count_parameters
from sleep_staging.adaptation.lora import (
    LoRAConfig, apply_lora, count_lora_parameters, get_lora_targets,
    assert_lora_targets,
)
from sleep_staging.data.labels import CANONICAL_LIST

OUTPUT_ROOT = REPO / "results" / "100_subject_adaptation"
DEFAULT_BASE = REPO / "artifacts" / "final" / "student_full_finetuned.pt"


def load_base(model: nn.Module, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    model.load_state_dict(state)


def mode_dir_name(mode: str, args) -> Path:
    if mode == "frozen":
        return OUTPUT_ROOT / "frozen"
    if mode == "full_finetune":
        return OUTPUT_ROOT / "full_finetune"
    tag = "_".join(args.targets.split(",")) if args.targets else "head"
    return OUTPUT_ROOT / f"lora_r{args.rank}_{tag}"


def run_fold(mode, fold_num, fold_data, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = mode_dir_name(mode, args) / f"fold_{fold_num:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_subjects = fold_data["train"]
    val_subjects = fold_data["validation"]
    test_subjects = fold_data["test"]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num} — {mode} — Seed {args.seed}")
    print(f"  Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, train_labels = build_dataloaders(
        train_subjects, val_subjects, test_subjects, CACHE_DIR,
    )

    model = ImprovedStudent()
    load_base(model, args.base_checkpoint)
    print(f"  Loaded base checkpoint: {args.base_checkpoint}")

    recorded = {"mode": mode, "seed": args.seed, "fold": fold_num}

    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        model = model.to(args.device)
        criterion = nn.CrossEntropyLoss(weight=torch.ones(5).to(args.device))
        test_metrics = evaluate(model, test_loader, criterion, args.device)
        recorded.update({
            "trainable_params": 0,
            "trainable_pct": 0.0,
            "total_params": sum(p.numel() for p in model.parameters()),
            "test_metrics": test_metrics,
        })
    else:
        if mode == "lora":
            lora_cfg = LoRAConfig(
                rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout,
                target_modules=args.targets.split(","),
            )
            model = apply_lora(model, lora_cfg)
            assert_lora_targets(model, lora_cfg.target_modules)
            print(f"  LoRA targets wrapped: {get_lora_targets(model)}")
            pc = count_lora_parameters(model)
            print(f"  LoRA trainable: {pc['trainable']:,} ({pc['trainable_pct']}%)")
            recorded.update({
                "rank": args.rank, "alpha": args.alpha,
                "lora_targets": get_lora_targets(model),
                "trainable_params": pc["trainable"],
                "trainable_pct": pc["trainable_pct"],
                "total_params": pc["total"],
            })
            params_to_train = [p for p in model.parameters() if p.requires_grad]
        else:
            for p in model.parameters():
                p.requires_grad = True
            recorded.update({
                "trainable_params": sum(p.numel() for p in model.parameters()),
                "trainable_pct": 100.0,
                "total_params": sum(p.numel() for p in model.parameters()),
            })
            params_to_train = model.parameters()

        model = model.to(args.device)

        class_weights = None
        # Compute weighted loss from train labels
        counts = np.bincount(train_labels, minlength=5).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = 1.0 / counts
        w = w / w.sum() * 5
        w[1] *= N1_WEIGHT
        w[4] *= REM_WEIGHT
        class_weights = torch.from_numpy(w).float().to(args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(params_to_train, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        scaler = torch.amp.GradScaler("cuda") if args.device.type == "cuda" else None

        history = []
        best_val_f1 = 0
        best_state = None
        patience = 0

        print(f"\n  Training for {MAX_EPOCHS} epochs...")
        for epoch in range(MAX_EPOCHS):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, args.device, scaler,
            )
            val_metrics = evaluate(model, val_loader, criterion, args.device)
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
                patience = 0
                marker = " *best*"
            else:
                patience += 1

            print(
                f"  Epoch {epoch+1:2d}/{MAX_EPOCHS} ({elapsed:.1f}s): "
                f"loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_F1={val_metrics['macro_f1']:.3f}{marker}"
            )
            if patience >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        test_metrics = evaluate(model, test_loader, criterion, args.device)
        recorded["test_metrics"] = test_metrics
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        if mode == "lora":
            from sleep_staging.adaptation.lora import save_adapter
            save_adapter(model, out_dir / "adapter")
        elif best_state:
            torch.save({"model_state_dict": best_state, **recorded}, out_dir / "best_model.pt")

    # Print test summary
    print(f"\n  TEST RESULTS ({mode}, Fold {fold_num}):")
    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Kappa:     {test_metrics['kappa']:.4f}")
    print(f"    Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"    MGm:       {test_metrics['mgm']:.4f}")
    for name in CANONICAL_LIST:
        pcv = test_metrics["per_class"][name]
        print(f"    {name:5s}: P={pcv['precision']:.3f} R={pcv['recall']:.3f} "
              f"F1={pcv['f1']:.3f} (n={pcv['support']})")

    # Save outputs
    cm = np.array(test_metrics["confusion_matrix"])
    pd.DataFrame(cm, index=[f"true_{n}" for n in CANONICAL_LIST],
                 columns=[f"pred_{n}" for n in CANONICAL_LIST]).to_csv(
        out_dir / "confusion_matrix.csv")

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(args.device, non_blocking=True)
            logits = model(x)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
            all_labels.append(y.numpy().reshape(-1))
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    pd.DataFrame({
        "true_label": all_labels, "pred_label": all_preds,
        "true_name": [CANONICAL_LIST[int(l)] for l in all_labels],
        "pred_name": [CANONICAL_LIST[int(p)] for p in all_preds],
    }).to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({**recorded, **test_metrics}, f, indent=2, default=str)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="100-Subject Adaptation Study")
    parser.add_argument("--mode", required=True,
                        choices=["frozen", "lora", "full_finetune"])
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--targets", default="head",
                        help="Comma-separated LoRA targets, e.g. enc.0.pw,enc.1.pw,head")
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.device = device

    print("=" * 70)
    print(f"  100-SUBJECT ADAPTATION STUDY — {args.mode}")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Base checkpoint: {args.base_checkpoint}")
    if args.mode == "lora":
        print(f"  LoRA: r={args.rank}, alpha={args.alpha}, dropout={args.lora_dropout}")
        print(f"  Targets: {args.targets}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    folds = load_folds()
    fold_range = [args.fold] if args.fold else range(1, 11)
    agg_file = mode_dir_name(args.mode, args) / f"{args.mode}_seed{args.seed}.json"
    agg = {}
    if agg_file.exists():
        agg = json.load(open(agg_file))

    for fold_num in fold_range:
        key = f"fold_{fold_num}"
        if key not in folds:
            print(f"Fold {fold_num} not found")
            continue
        if str(fold_num) in agg:
            print(f"\nFold {fold_num}: already completed, skipping")
            continue
        t0 = time.time()
        metrics = run_fold(args.mode, fold_num, folds[key], args)
        elapsed = time.time() - t0
        agg[str(fold_num)] = {
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mgm": metrics["mgm"],
            "per_class": metrics["per_class"],
            "test_subjects": folds[key]["test"],
            "time_s": elapsed,
        }
        with open(agg_file, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  Fold {fold_num} completed in {elapsed:.1f}s")

    if agg:
        vals = {k: [v[k] for v in agg.values()] for k in
                ["accuracy", "kappa", "macro_f1", "weighted_f1", "mgm"]}
        print("\n" + "=" * 70)
        print(f"  {args.mode.upper()} — SEED {args.seed} — FOLDS {len(agg)}/10")
        print("=" * 70)
        for k, v in vals.items():
            print(f"  {k:12s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        summary = {
            "mode": args.mode, "seed": args.seed, "n_folds": len(agg),
            "means": {k: float(np.mean(v)) for k, v in vals.items()},
            "stds": {k: float(np.std(v)) for k, v in vals.items()},
        }
        with open(agg_file.parent / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved summary to {agg_file.parent / 'summary.json'}")


if __name__ == "__main__":
    main()