#!/usr/bin/env python
"""Train the Improved Student model with knowledge distillation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import SleepSequenceDataset
from src.models.improved_student import ImprovedStudent, count_parameters
from src.training.losses import DistillationObjective, compute_class_weights
from src.training.trainer import train_student_with_teacher


def main():
    parser = argparse.ArgumentParser(description="Train Improved Student")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--teacher-checkpoint", default="artifacts/teacher_improved_best.pt")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output", default="artifacts/student_improved_best.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_index = pd.read_csv(Path(args.cache_dir) / "cache_index.csv")
    train_ds = SleepSequenceDataset(cache_index, "train", seq_len=cfg["training"]["sequence"]["length"])
    val_ds = SleepSequenceDataset(cache_index, "val", seq_len=cfg["training"]["sequence"]["length"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_labels = []
    for _, row in cache_index.loc[cache_index["split"] == "train"].iterrows():
        train_labels.append(np.load(row["cache_path"])["labels"])
    train_labels = np.concatenate(train_labels)
    class_weights = compute_class_weights(torch.from_numpy(train_labels))

    student = ImprovedStudent(n_classes=5).to(device)
    print(f"Student parameters: {count_parameters(student):,}")

    teacher = ImprovedStudent(n_classes=5).to(device)
    teacher_path = Path(args.teacher_checkpoint)
    if teacher_path.exists():
        teacher_payload = torch.load(teacher_path, map_location=device, weights_only=False)
        if isinstance(teacher_payload, dict) and "model_state_dict" in teacher_payload:
            teacher.load_state_dict(teacher_payload["model_state_dict"])
        else:
            teacher.load_state_dict(teacher_payload)
        print(f"Loaded teacher from {teacher_path}")
    else:
        print(f"WARNING: Teacher checkpoint not found at {teacher_path}")

    objective = DistillationObjective(
        class_weights=class_weights,
        temperature=cfg["training"]["distillation"]["temperature"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_student_with_teacher(
        student, teacher, objective,
        train_loader, val_loader,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
        checkpoint_path=output_path,
        device=device,
    )

    history.to_csv("results/training_history.csv", index=False)
    print(f"\nTraining complete. Best checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
