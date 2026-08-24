#!/usr/bin/env python
"""Train a LoRA adapter on top of the frozen Improved Student base model.

Usage:
    python scripts/train_lora_adapter.py --rank 4 --alpha 8 --output artifacts/lora/head_r4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sleep_staging.adaptation import LoRAConfig, apply_lora, count_lora_parameters, save_adapter
from sleep_staging.config import CHECKPOINT_PATH, CACHE_DIR, StudentConfig
from sleep_staging.data.loader import available_subjects, load_cached_subject, get_contiguous_sequence
from sleep_staging.evaluation import compute_all_metrics
from sleep_staging.models import ImprovedStudent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_sequences(
    subject_ids: list[str],
    seq_len: int = 10,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build epoch sequences and labels from cached data."""
    all_seqs, all_labels = [], []
    for sid in subject_ids:
        data = load_cached_subject(sid)
        epochs = data["epochs"]
        labels = data["labels"]
        n = len(epochs)
        for start in range(0, n - seq_len, stride):
            seq = epochs[start : start + seq_len]
            target = labels[start + seq_len - 1]  # label of the target epoch
            all_seqs.append(seq)
            all_labels.append(target)
    return np.array(all_seqs, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LoRA adapter")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", nargs="+", default=["head"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts" / "lora" / "head_r4")
    parser.add_argument("--train-subjects", nargs="+", default=None)
    parser.add_argument("--val-subjects", nargs="+", default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = StudentConfig()
    lora_config = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.targets,
    )

    # ── Load base model ────────────────────────────────────────────────
    log.info("Loading base model from %s", args.checkpoint)
    model = ImprovedStudent(config)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)

    # ── Apply LoRA ─────────────────────────────────────────────────────
    model = apply_lora(model, lora_config)
    params = count_lora_parameters(model)
    log.info(
        "LoRA applied: %d trainable / %d total (%.2f%%)",
        params["trainable"], params["total"], params["trainable_pct"],
    )

    # ── Prepare data ───────────────────────────────────────────────────
    subjects = available_subjects()
    log.info("Available subjects: %s", subjects)

    if args.train_subjects:
        train_ids = args.train_subjects
    else:
        train_ids = subjects[:3] if len(subjects) >= 3 else subjects[:1]

    if args.val_subjects:
        val_ids = args.val_subjects
    else:
        val_ids = subjects[-1:] if len(subjects) > 1 else subjects[:1]

    log.info("Train subjects: %s", train_ids)
    log.info("Val subjects: %s", val_ids)

    train_seqs, train_labels = build_sequences(train_ids, stride=5)
    val_seqs, val_labels = build_sequences(val_ids, stride=10)
    log.info("Train: %d sequences, Val: %d sequences", len(train_seqs), len(val_seqs))

    train_ds = TensorDataset(
        torch.from_numpy(train_seqs), torch.from_numpy(train_labels)
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_seqs), torch.from_numpy(val_labels)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Train ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)

    best_val_kappa = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for seq_batch, label_batch in train_loader:
            seq_batch, label_batch = seq_batch.to(device), label_batch.to(device)
            logits = model(seq_batch)
            # Use only the target epoch (last in sequence)
            loss = criterion(logits[:, -1, :], label_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.inference_mode():
            for seq_batch, label_batch in val_loader:
                seq_batch = seq_batch.to(device)
                logits = model(seq_batch)
                preds = logits[:, -1, :].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(label_batch.numpy())

        metrics = compute_all_metrics(np.array(all_true), np.array(all_preds))
        kappa = metrics["kappa"]
        acc = metrics["accuracy"]

        log.info(
            "Epoch %d/%d — loss: %.4f — val_acc: %.4f — val_kappa: %.4f",
            epoch, args.epochs, avg_loss, acc, kappa,
        )

        history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "val_acc": round(acc, 4),
            "val_kappa": round(kappa, 4),
            "val_macro_f1": round(metrics["macro_f1"], 4),
        })

        if kappa > best_val_kappa:
            best_val_kappa = kappa
            save_adapter(model, args.output)
            log.info("  → New best adapter saved (kappa=%.4f)", kappa)

    # ── Save training summary ──────────────────────────────────────────
    summary = {
        "lora_config": {
            "rank": lora_config.rank,
            "alpha": lora_config.alpha,
            "dropout": lora_config.dropout,
            "target_modules": lora_config.target_modules,
        },
        "train_subjects": train_ids,
        "val_subjects": val_ids,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_kappa": round(best_val_kappa, 4),
        "params": params,
        "history": history,
    }
    with open(args.output / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Training complete. Best val_kappa: %.4f", best_val_kappa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
