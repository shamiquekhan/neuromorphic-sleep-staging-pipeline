#!/usr/bin/env python3
"""Train the leak-free base checkpoint for the adaptation study.

The base model is trained ONLY on the 5 fixed validation PERSONS (8
records, both nights each) of ``person_folds_52subj.json`` — persons
that never appear in any adaptation-evaluation test fold. Every fold
of the adaptation study then starts from this checkpoint:

    frozen       eval-only
    lora          train adapters (strict PEFT, BN frozen)
    full_finetune train everything

Provenance is stored INSIDE the checkpoint payload (training subjects,
protocol, env, git SHA) so the base can always be audited — the
pre-fix leak (15-record-era checkpoint with 12 records in eval test
folds) must be structurally impossible to reproduce silently.

Usage:
    python scripts/train_base_leakfree.py --device cuda
    python scripts/train_base_leakfree.py --device cuda --seed 42
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sleep_staging.models.improved_student import ImprovedStudent, count_parameters
from sleep_staging.data.sequence_dataset import SubjectSequenceDataset, CausalEvalDataset, make_subject_list
from sleep_staging.data.labels import CANONICAL_LIST, N_CLASSES
from sleep_staging.evaluation.protocol import evaluate_causal
from sleep_staging.training.cross_dataset import compute_class_weights
from sleep_staging.training.seed import seed_everything, worker_init_fn

CACHE_DIR = REPO / "data" / "cache" / "sleep_edf"
PERSON_FOLDS = REPO / "data" / "manifests" / "person_folds_52subj.json"
OUT_DIR = REPO / "artifacts" / "base_leakfree"
SEQ_LEN, SEQ_STRIDE = 10, 5
MAX_EPOCHS, BATCH_SIZE, PATIENCE = 20, 32, 5
LR, WEIGHT_DECAY, GRAD_CLIP = 3e-4, 1e-4, 1.0
N1_WEIGHT, REM_WEIGHT = 2.0, 2.0


def person_of(record: str) -> str:
    return f"P{record[2:5]}"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    # 1. Resolve training records: the 5 fixed validation PERSONS.
    with open(PERSON_FOLDS) as f:
        folds = json.load(f)["folds"]
    val_records = sorted(set(next(iter(folds.values()))["validation"]))
    val_persons = sorted({person_of(r) for r in val_records})
    print(f"Base training records ({len(val_records)}): {val_records}")
    print(f"Base training persons ({len(val_persons)}): {val_persons}")

    # 2. Safety: these records must appear in NO test fold (any fold).
    all_test = set()
    for fold in folds.values():
        all_test.update(fold["test"])
    overlap = set(val_records) & all_test
    if overlap:
        raise SystemExit(
            f"REFUSING: validation records {sorted(overlap)} appear in "
            "test folds — the person-level manifest is inconsistent"
        )

    seed_everything(args.seed)

    # 3. Build data (subject-safe windows, current gap-aware caches).
    subjects = make_subject_list(val_records, CACHE_DIR)
    train_ds = SubjectSequenceDataset(subjects, SEQ_LEN, SEQ_STRIDE)
    print(f"Train windows: {len(train_ds):,}")

    # Hold out one person (both nights) for checkpoint selection so the
    # base's early stopping is not selected on its own training data.
    holdout_person = val_persons[0]
    holdout_records = [r for r in val_records if person_of(r) == holdout_person]
    train_records = [r for r in val_records if person_of(r) != holdout_person]
    print(f"Checkpoint-selection holdout person: {holdout_person} "
          f"({holdout_records})")
    train_subjects = make_subject_list(train_records, CACHE_DIR)
    holdout_subjects = make_subject_list(holdout_records, CACHE_DIR)
    train_ds = SubjectSequenceDataset(train_subjects, SEQ_LEN, SEQ_STRIDE)
    holdout_ds = CausalEvalDataset(holdout_subjects, SEQ_LEN)
    print(f"Fit windows: {len(train_ds):,} | holdout scored epochs: "
          f"{len(holdout_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
        drop_last=True, worker_init_fn=worker_init_fn,
    )

    # 4. Train with the canonical benchmark hyperparameters.
    model = ImprovedStudent().to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    train_labels = np.concatenate([s["labels"] for s in train_subjects])
    class_weights = compute_class_weights(train_labels, N1_WEIGHT, REM_WEIGHT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_f1, best_state, patience = 0.0, None, 0
    history = []
    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = correct = total = 0
        for x, y in train_loader:
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
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total

        hm = evaluate_causal(model, holdout_ds, device=device, batch_size=BATCH_SIZE)
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "train_acc": train_acc,
            "holdout_accuracy": hm["accuracy"],
            "holdout_macro_f1": hm["macro_f1"],
            "time_s": elapsed,
        })
        marker = ""
        if hm["macro_f1"] > best_f1:
            best_f1 = hm["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            marker = " *best*"
        else:
            patience += 1
        print(f"Epoch {epoch + 1:2d}/{MAX_EPOCHS} ({elapsed:.1f}s): "
              f"loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"holdout_acc={hm['accuracy']:.3f} "
              f"holdout_F1={hm['macro_f1']:.3f}{marker}")
        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise SystemExit("no checkpoint selected — training failed")

    # 5. Persist with full provenance.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": best_state,
        "provenance": {
            "role": "leak-free base for adaptation study",
            "train_records": train_records,
            "train_persons": sorted({person_of(r) for r in train_records}),
            "holdout_person_for_selection": holdout_person,
            "n_train_records": len(train_records),
            "seed": args.seed,
            "protocol": "subject_safe_windows_train__causal_holdout",
            "hyperparameters": {
                "epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP, "n1_weight": N1_WEIGHT,
                "rem_weight": REM_WEIGHT, "patience": PATIENCE,
                "seq_len": SEQ_LEN, "train_stride": SEQ_STRIDE,
            },
            "best_holdout_macro_f1": best_f1,
            "git_commit": git_sha(),
            "torch_version": torch.__version__,
        },
    }
    ckpt_path = OUT_DIR / "student_full_finetuned.pt"
    torch.save(payload, ckpt_path)

    with open(OUT_DIR / "train_subjects.json", "w") as f:
        json.dump({
            "base_training_subject_ids": val_records,
            "base_training_person_ids": val_persons,
            "n_records": len(val_records),
            "seed": args.seed,
            "config": "train_base_leakfree.py (canonical benchmark "
                      "hyperparameters, subject-safe windows, "
                      "causal holdout selection)",
            "holdout_person_for_selection": holdout_person,
            "protocol": "subject_safe_windows_train__causal_holdout",
        }, f, indent=2)

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 6. Machine-verify: training records disjoint from all test folds.
    assert not (set(val_records) & all_test), "leak check"
    print(f"\nBase checkpoint saved: {ckpt_path}")
    print(f"Best holdout macro-F1: {best_f1:.4f}")
    print(f"Provenance embedded: {len(val_records)} records "
          f"({len(val_persons)} persons), 0 test-fold overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
