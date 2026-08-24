"""Dataset manifest utilities."""

from pathlib import Path

import pandas as pd

from ..config import MANIFEST_PATH


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


def get_subjects(df: pd.DataFrame, split: str | None = None) -> list[str]:
    """Return unique subject IDs, optionally filtered by split."""
    if split:
        df = df[df["split"] == split]
    return sorted(df["subject_id"].unique().tolist())
