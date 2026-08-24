"""Load preprocessed epoch arrays from the cache."""

from pathlib import Path

import numpy as np

from ..config import CACHE_DIR


def load_cached_subject(
    subject_id: str,
    cache_dir: str | Path | None = None,
) -> dict:
    """Load a cached NPZ file for one subject.

    The cache files are named ``{subject_id}_night0.npz`` and contain:
        - ``epochs``: ``[n_epochs, n_channels, n_samples]``
        - ``labels``: ``[n_epochs]`` integer stage labels.

    Returns:
        Dict with keys ``epochs``, ``labels``, ``subject_id``.
    """
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    path = d / f"{subject_id}_night0.npz"
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    data = np.load(path)
    return {
        "epochs": data["epochs"],
        "labels": data["labels"],
        "subject_id": subject_id,
    }


def get_contiguous_sequence(
    epochs: np.ndarray,
    start: int,
    seq_len: int = 10,
) -> np.ndarray:
    """Extract a contiguous sequence of ``seq_len`` epochs.

    Args:
        epochs: ``[n_epochs, n_channels, n_samples]``.
        start: Starting epoch index.
        seq_len: Number of epochs in the context window.

    Returns:
        Array of shape ``[1, seq_len, n_channels, n_samples]``.
    """
    end = start + seq_len
    if end > epochs.shape[0]:
        raise ValueError(
            f"Cannot extract {seq_len} epochs starting at {start}: "
            f"only {epochs.shape[0]} epochs available."
        )
    return epochs[start:end][np.newaxis, ...].astype(np.float32)


def available_subjects(cache_dir: str | Path | None = None) -> list[str]:
    """List subject IDs present in the cache directory."""
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    return sorted(
        p.stem.replace("_night0", "")
        for p in d.glob("*_night0.npz")
    )
