"""Cross-dataset training pipeline.

Supports training on:
1. Sleep-EDF only (current baseline)
2. Sleep-EDF Expanded (full 183 subjects)
3. Sleep-EDF + SHHS N1 enrichment
4. SHHS pretraining → Sleep-EDF LoRA adaptation

All experiments use the same Improved Student architecture.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..config import StudentConfig, CACHE_DIR, SLEEP_EDF_CACHE_DIR
from ..data.loader import load_cached_subject
from ..data.labels import CANONICAL_LIST, N_CLASSES

log = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Dataset for sliding-window sequences over multi-epoch PSG data.

    Each sample is a contiguous window of `seq_len` epochs.
    """

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 10,
        stride: int = 5,
        subject_id: str = "",
        dataset: str = "sleep_edf",
    ):
        self.epochs = torch.from_numpy(epochs).float()
        self.labels = torch.from_numpy(labels).long()
        self.seq_len = seq_len
        self.stride = stride
        self.subject_id = subject_id
        self.dataset = dataset

        # Precompute valid start positions
        n_epochs = len(labels)
        self.starts = list(range(0, n_epochs - seq_len + 1, stride))
        if not self.starts and n_epochs >= seq_len:
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len

        x = self.epochs[start:end]  # [seq_len, C, S]
        y = self.labels[start:end]  # [seq_len]

        return x, y


def load_subjects(
    subject_ids: list[str],
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and concatenate multiple subjects' data.

    Returns:
        (all_epochs, all_labels, subject_ids_loaded)
    """
    all_epochs = []
    all_labels = []
    loaded = []

    for sid in subject_ids:
        try:
            data = load_cached_subject(sid, cache_dir)
            all_epochs.append(data["epochs"])
            all_labels.append(data["labels"])
            loaded.append(sid)
        except FileNotFoundError:
            log.warning("Cache not found for %s, skipping", sid)

    if not all_epochs:
        raise ValueError("No subjects loaded")

    return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0), loaded


def create_train_val_splits(
    subject_ids: list[str],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split subjects into train/val sets."""
    rng = np.random.RandomState(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def build_dataloaders(
    train_subjects: list[str],
    val_subjects: list[str,
    ],
    cache_dir: str | Path | None = None,
    seq_len: int = 10,
    stride: int = 5,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from subject lists."""
    train_epochs, train_labels, _ = load_subjects(train_subjects, cache_dir)
    val_epochs, val_labels, _ = load_subjects(val_subjects, cache_dir)

    train_ds = SequenceDataset(train_epochs, train_labels, seq_len, stride, dataset="sleep_edf")
    val_ds = SequenceDataset(val_epochs, val_labels, seq_len, stride, dataset="sleep_edf")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


def compute_class_weights(labels: np.ndarray, n1_weight: float = 2.0, rem_weight: float = 2.0) -> torch.Tensor:
    """Compute inverse-frequency class weights with optional N1/REM boost."""
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * N_CLASSES

    # Apply minority-class boost
    weights[1] *= n1_weight  # N1
    weights[4] *= rem_weight  # REM

    return torch.from_numpy(weights).float()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # [B, T, C]
        loss = criterion(logits.view(-1, N_CLASSES), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, N_CLASSES), y.view(-1))

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

        all_preds.append(preds.cpu().numpy().reshape(-1))
        all_labels.append(y.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Per-class metrics
    from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, labels=list(range(N_CLASSES)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    per_class = {}
    for i, name in enumerate(CANONICAL_LIST):
        mask = all_labels == i
        if mask.sum() > 0:
            class_preds = all_preds[mask]
            # Binary: correct (1) vs incorrect (0) for this class
            binary_preds = (class_preds == i).astype(int)
            binary_labels = np.ones_like(binary_preds)
            per_class[name] = {
                "f1": float(f1_score(binary_labels, binary_preds, average="binary", zero_division=0)),
                "recall": float((class_preds == i).sum() / mask.sum()),
                "support": int(mask.sum()),
            }

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy,
        "kappa": kappa,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def run_experiment(
    train_subjects: list[str],
    val_subjects: list[str],
    cache_dir: str | Path | None = None,
    n1_weight: float = 2.0,
    rem_weight: float = 2.0,
    max_epochs: int = 10,
    lr: float = 3e-4,
    batch_size: int = 16,
    device: str = "auto",
    save_path: str | Path | None = None,
) -> dict:
    """Run a full training experiment.

    Returns:
        Dict with training history and final metrics.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        train_subjects, val_subjects, cache_dir,
        batch_size=batch_size,
    )

    # Build model
    config = StudentConfig()
    from ..models.improved_student import ImprovedStudent
    model = ImprovedStudent(config).to(device)

    # Loss with class weights
    train_epochs, train_labels, _ = load_subjects(train_subjects, cache_dir)
    class_weights = compute_class_weights(train_labels, n1_weight, rem_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    history = []
    best_macro_f1 = 0
    best_state = None

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
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

        log.info(
            "Epoch %d/%d (%.1fs): loss=%.4f acc=%.3f | val_loss=%.4f val_acc=%.3f val_F1=%.3f",
            epoch + 1, max_epochs, elapsed,
            train_loss, train_acc,
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"],
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    if save_path and best_state:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        log.info("Saved best model to %s (F1=%.3f)", save_path, best_macro_f1)

    return {
        "history": history,
        "best_macro_f1": best_macro_f1,
        "n_train_subjects": len(train_subjects),
        "n_val_subjects": len(val_subjects),
    }
