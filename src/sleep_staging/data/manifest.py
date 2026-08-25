"""Dataset manifest utilities.

Supports both the original CSV manifest and the new JSON manifest format.
"""

import json
from pathlib import Path

import pandas as pd

from ..config import MANIFEST_PATH, EXPANDED_MANIFEST_PATH


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    """Load the sleep-edf manifest CSV.

    Returns:
        DataFrame with columns including ``subject_id``, ``split``, ``edf_path``,
        ``hypnogram_path``.
    """
    p = Path(path) if path else MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return pd.read_csv(p)


def load_expanded_manifest(path: str | Path | None = None) -> dict:
    """Load the expanded Sleep-EDF manifest (JSON format).

    Returns:
        Dict with dataset metadata and per-subject information.
    """
    p = Path(path) if path else EXPANDED_MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"Expanded manifest not found: {p}")
    with open(p) as f:
        return json.load(f)


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
    import random

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


def build_manifest_df(
    manifest_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a DataFrame from the expanded manifest with optional splits.

    Returns:
        DataFrame with columns: subject_id, n_epochs, n1_epochs, split, etc.
    """
    manifest = load_expanded_manifest(manifest_path)

    rows = []
    for subject in manifest["subjects"]:
        dist = subject["class_distribution"]
        rows.append({
            "subject_id": subject["subject_id"],
            "n_epochs": subject["n_epochs"],
            "n_channels": subject["n_channels"],
            "wake_epochs": dist.get("Wake", 0),
            "n1_epochs": dist.get("N1", 0),
            "n2_epochs": dist.get("N2", 0),
            "n3_epochs": dist.get("N3", 0),
            "rem_epochs": dist.get("REM", 0),
            "cache_path": subject["cache_path"],
        })

    df = pd.DataFrame(rows)

    # Add splits if provided
    if split_manifest_path:
        with open(split_manifest_path) as f:
            splits = json.load(f)
        df["split"] = df["subject_id"].map(splits)
    else:
        df["split"] = "train"  # Default

    return df
