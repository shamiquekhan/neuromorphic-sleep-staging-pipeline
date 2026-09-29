"""Load preprocessed epoch arrays from the cache."""

from pathlib import Path

import numpy as np

from ..config import CACHE_DIR


def load_cached_subject(
    subject_id: str,
    cache_dir: str | Path | None = None,
) -> dict:
    """Load one cached recording for one subject.

    Two cache layouts are supported:

    Canonical (Notebook 02, ``data/cache_full``) — ``{subject_id}_epochs.npy``
    memory-mapped array ``[n_epochs, n_channels, n_samples]`` plus a
    ``{subject_id}_meta.npz`` sidecar with ``labels``, ``onsets`` (sample
    offsets), ``qc_flag`` and ``fs``. ``orig_epoch_idx`` is derived from
    ``onsets / (fs * 30)``.

    Legacy (night-1 subset, ``data/cache``) — single ``{subject_id}_nightE0.npz``
    containing ``epochs``, ``labels`` and optionally ``orig_epoch_idx``
    (index of each cached epoch within the raw 30 s annotation grid;
    preprocessing drops unlabeled epochs so cached rows can be temporally
    discontiguous — see ``data/sequence_dataset.py``).

    Returns:
        Dict with keys ``epochs``, ``labels``, ``subject_id``, and
        ``orig_epoch_idx`` (None when unavailable).
    """
    d = Path(cache_dir) if cache_dir else CACHE_DIR

    epochs_path = d / f"{subject_id}_epochs.npy"
    meta_path = d / f"{subject_id}_meta.npz"
    if epochs_path.exists() and meta_path.exists():
        epochs = np.load(epochs_path, mmap_mode="r")
        meta = np.load(meta_path)
        orig_epoch_idx = None
        if "onsets" in meta and "fs" in meta:
            samples_per_epoch = float(meta["fs"]) * 30.0
            orig_epoch_idx = np.round(
                np.asarray(meta["onsets"]) / samples_per_epoch
            ).astype(np.int64)
        return {
            "epochs": epochs,
            "labels": np.asarray(meta["labels"]),
            "subject_id": subject_id,
            "orig_epoch_idx": orig_epoch_idx,
        }

    legacy_path = d / f"{subject_id}_nightE0.npz"
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {epochs_path} or {legacy_path}"
        )

    data = np.load(legacy_path)
    return {
        "epochs": data["epochs"],
        "labels": data["labels"],
        "subject_id": subject_id,
        "orig_epoch_idx": (
            data["orig_epoch_idx"] if "orig_epoch_idx" in data else None
        ),
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
    """List recording IDs present in the cache directory.

    Canonical layout (``{id}_meta.npz``) wins when present; otherwise falls
    back to legacy ``{id}_night*.npz`` files.
    """
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    canonical = sorted(
        p.stem[: -len("_meta")] for p in d.glob("*_meta.npz")
    )
    if canonical:
        return canonical
    return sorted(
        p.stem.split("_night")[0]
        for p in d.glob("*_night*.npz")
    )
