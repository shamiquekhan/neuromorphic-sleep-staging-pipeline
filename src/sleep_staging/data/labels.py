"""Canonical label mapping for cross-dataset sleep staging.

All datasets map to a single canonical 5-stage system:
    0 = Wake
    1 = N1 (Stage 1)
    2 = N2 (Stage 2)
    3 = N3 (Stage 3 + Stage 4)
    4 = REM

Dataset-specific adapters convert raw annotations to these canonical labels.
"""

from enum import IntEnum


class SleepStage(IntEnum):
    """Canonical sleep stage indices."""
    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4


CANONICAL_NAMES = {
    SleepStage.WAKE: "Wake",
    SleepStage.N1: "N1",
    SleepStage.N2: "N2",
    SleepStage.N3: "N3",
    SleepStage.REM: "REM",
}

CANONICAL_LIST = ["Wake", "N1", "N2", "N3", "REM"]

N_CLASSES = 5


# ── Sleep-EDF label mapping (Rechtschaffen & Kales → AASM canonical) ────

SLEEP_EDF_MAP = {
    "Sleep stage W": SleepStage.WAKE,
    "Sleep stage 1": SleepStage.N1,
    "Sleep stage 2": SleepStage.N2,
    "Sleep stage 3": SleepStage.N3,
    "Sleep stage 4": SleepStage.N3,   # R&K stage 4 → AASM N3
    "Sleep stage R": SleepStage.REM,
    "Sleep stage ?": None,            # Unknown → exclude
    "Movement time": None,            # Movement → exclude
}


# ── SHHS label mapping (NSRR XML staging → canonical) ──────────────────
# SHHS uses: W, 1, 2, 3, 4, R, M, ?
# Source: NSRR SHHS documentation

SHHS_MAP = {
    "W": SleepStage.WAKE,
    "1": SleepStage.N1,
    "2": SleepStage.N2,
    "3": SleepStage.N3,
    "4": SleepStage.N3,   # Stage 4 → N3
    "R": SleepStage.REM,
    "M": None,            # Movement → exclude
    "?": None,            # Unknown → exclude
}


def map_labels(raw_labels: list[str], dataset: str) -> list[int | None]:
    """Map raw annotation strings to canonical integer labels.

    Args:
        raw_labels: List of raw annotation strings from the dataset.
        dataset: Dataset identifier ("sleep_edf" or "shhs").

    Returns:
        List of canonical label integers (or None for excluded epochs).
    """
    if dataset == "sleep_edf":
        mapping = SLEEP_EDF_MAP
    elif dataset == "shhs":
        mapping = SHHS_MAP
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return [mapping.get(label) for label in raw_labels]


def get_valid_mask(labels: list[int | None]) -> list[bool]:
    """Return a boolean mask indicating which labels are valid (not None)."""
    return [label is not None for label in labels]


def filter_valid(raw_labels: list[str], dataset: str) -> tuple[list[int], list[bool]]:
    """Map and filter labels, returning only valid canonical labels and their mask.

    Returns:
        Tuple of (valid_labels, mask) where mask indicates original positions.
    """
    mapped = map_labels(raw_labels, dataset)
    mask = get_valid_mask(mapped)
    valid = [int(label) for label, m in zip(mapped, mask) if m]
    return valid, mask
