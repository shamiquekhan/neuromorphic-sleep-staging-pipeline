"""Training loop with knowledge distillation."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score


def train_student_with_teacher(
    student, teacher, objective,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20, lr: float = 3e-4, weight_decay: float = 1e-4,
    grad_clip: float = 1.0, checkpoint_path: Path = None,
    device: torch.device = torch.device("cpu"),
):
    """Train student with teacher distillation, checkpointing best validation kappa."""
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    sched = LambdaLR(optimizer, lambda s: (
        s / (0.1 * total_steps) if s < 0.1 * total_steps
        else 0.5 * (1 + math.cos(math.pi * (s - 0.1 * total_steps) / (0.9 * total_steps)))
    ))

    best_kappa = -1.0
    history = []

    teacher.eval()

    for epoch in range(1, epochs + 1):
        student.train()
        running = 0.0
        batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                t_logits, t_feats = teacher(x, return_features=True)

            s_logits, s_feats = student(x, return_features=True)
            loss, parts = objective(s_logits, s_feats, t_logits, t_feats, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            sched.step()

            running += float(loss.detach())
            batches += 1

        val_kappa = _validation_kappa(student, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(batches, 1), "val_kappa": val_kappa, **parts}
        history.append(row)

        print(f"Epoch {epoch:02d}/{epochs} | loss={row['train_loss']:.4f} | val_kappa={val_kappa:.4f}")

        if val_kappa > best_kappa and checkpoint_path:
            best_kappa = val_kappa
            torch.save({
                "model_state_dict": student.state_dict(),
                "epoch": epoch,
                "val_kappa": val_kappa,
            }, checkpoint_path)

    return pd.DataFrame(history)


@torch.no_grad()
def _validation_kappa(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        logits = model(x.to(device))
        yp.extend(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
        yt.extend(y.numpy().reshape(-1))
    return float(cohen_kappa_score(np.array(yt), np.array(yp)))
