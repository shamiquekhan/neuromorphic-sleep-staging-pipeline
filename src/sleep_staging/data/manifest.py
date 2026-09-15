"""Dataset manifest utilities."""

import random
from pathlib import Path

import pandas as pd

from ..config import MANIFEST_PATH


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    """Load the sleep-edf manifest CSV.

    Returns:
        DataFrame with columns including ``subject_id``, ``split``,
        ``psg``, ``hypnogram``.
    """
    p = Path(path) if path else MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return pd.read_csv(p)


def get_subjects(df: pd.DataFrame, split: str | None = None) -> list[str]:
    """Return unique subject IDs, optionally filtered by split."""
    if split:
        df = df[df["split"] == split]
    return sorted(df["subject_id"].unique().tolist())


def build_subject_splits(
    subject_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Split subjects into train/val/test sets.

    Args:
        subject_ids: List of subject IDs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        Dict mapping subject_id → split name.
    """
    rng = random.Random(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {}
    for i, sid in enumerate(shuffled):
        if i < n_train:
            splits[sid] = "train"
        elif i < n_train + n_val:
            splits[sid] = "val"
        else:
            splits[sid] = "test"
    return splits
