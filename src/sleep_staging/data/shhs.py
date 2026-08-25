"""SHHS (Sleep Heart Health Study) dataset adapter — skeleton.

SHHS is a large multicenter cohort study of sleep-disordered breathing.
NSRR provides PSG recordings in EDF format with XML staging annotations.

Dataset: https://sleepdata.org/datasets/shhs/
Documentation: https://sleepdata.org/datasets/shhs/pages/

NOTE: This is a skeleton. Actual SHHS access requires NSRR data use agreement.
"""

import logging
from pathlib import Path

import numpy as np

from .labels import SHHS_MAP, CANONICAL_LIST, N_CLASSES

log = logging.getLogger(__name__)

# ── SHHS constants ───────────────────────────────────────────────────────

SHHS_VISITS = ["shhs1", "shhs2"]

# SHHS sampling rates vary by visit and channel
SHHS_NATIVE_FS = 125  # Hz (typical for SHHS1)

# SHHS channel names (exact names may vary — verify from actual files)
SHHS_CHANNELS = {
    "EEG": ["EEG1", "EEG2"],          # Select best 2 EEG derivations
    "EOG": ["EOG-L", "EOG-R"],        # Horizontal EOG
    "EMG": ["EMG"],                     # Chin EMG
}

TARGET_FS = 100  # Resample to match Sleep-EDF


def shhs_download_info():
    """Return information about SHHS access requirements."""
    return {
        "url": "https://sleepdata.org/datasets/shhs/files",
        "access": "NSRR Data Use Agreement required",
        "format": "EDF (PSG) + XML (staging)",
        "visits": ["shhs1 (n=5793)", "shhs2 (n=2651)"],
        "channels": "Variable montage — verify from documentation",
        "scoring": "30-second epochs, AASM-like staging",
        "note": "Must register at sleepdata.org before downloading",
    }


def load_shhs_subject(subject_id: str, visit: str, raw_dir: Path):
    """Load one SHHS subject — skeleton implementation.

    SHHS files are typically:
        PSG: {subject_id}-PSG.edf
        Hyp: {subject_id}-Hypn.xml (XML staging)

    Returns:
        Tuple of (epochs, labels, fs) — same interface as Sleep-EDF adapter.

    Raises:
        NotImplementedError: Until actual SHHS files are available.
    """
    raise NotImplementedError(
        "SHHS loading requires:\n"
        "1. NSRR registration and data download\n"
        "2. Channel name verification from actual files\n"
        "3. XML staging annotation parsing\n"
        "4. Resampling from native rate to 100 Hz\n\n"
        "See shhs_download_info() for access requirements."
    )


def map_shhs_staging(xml_labels: list[str]) -> list[int | None]:
    """Map SHHS XML staging labels to canonical integers.

    SHHS uses single-letter labels: W, 1, 2, 3, 4, R, M, ?
    """
    from .labels import map_labels
    return map_labels(xml_labels, "shhs")
