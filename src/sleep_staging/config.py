"""Centralized configuration for the NeuroSleep package."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "standalone_99k" / "student_99477_best.pt"
RESULTS_PATH = PROJECT_ROOT / "results" / "final" / "final_metrics.json"


def _resolve_cache_dir() -> Path:
    """Resolve the default epoch cache directory.

    Order:
    1. ``data/cache_full`` — canonical Notebook 02 output (197 recordings,
       ``*_epochs.npy`` + ``*_meta.npz`` sidecars).
    2. ``data/cache`` — legacy night-1 cache (``*_nightE0.npz``).
    3. ``data/cache_demo`` — small committed subset so the deployed
       Streamlit Cloud app has data without the full cache.
    """
    full = PROJECT_ROOT / "data" / "cache_full"
    if (full / "cache_index.csv").exists():
        return full
    legacy = PROJECT_ROOT / "data" / "cache"
    if legacy.exists() and any(legacy.glob("*_night*.npz")):
        return legacy
    demo = PROJECT_ROOT / "data" / "cache_demo"
    if any(demo.glob("*_meta.npz")):
        return demo
    return full


CACHE_DIR = _resolve_cache_dir()
SLEEP_EDF_CACHE_DIR = CACHE_DIR
SHHS_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "shhs"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sleep_edf"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sleep_edf.csv"

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_LIST = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "Wake": "#FF6B6B",
    "N1": "#FFA07A",
    "N2": "#4ECDC4",
    "N3": "#2C73D2",
    "REM": "#9B59B6",
}


@dataclass(frozen=True)
class StudentConfig:
    """Configuration for the Improved Student model.

    Defaults describe the trained architecture exactly (99,477
    parameters; matches the deployed standalone_99k checkpoint).
    Changing any of these values changes the architecture and
    invalidates existing checkpoints — `ImprovedStudent` now builds
    every layer from this config, so the mismatch between documented
    and actual widths that existed pre-fix (hardcoded 8/16/272 vs
    config 10/32/32) can no longer occur silently.
    """

    n_channels: int = 4
    n_classes: int = 5
    sampling_rate: int = 100
    gru_hidden: int = 64
    gru_layers: int = 2
    stem_width: int = 8
    encoder_channels: tuple = (32, 32)
    gabor_n_filters: int = 8
    gabor_out_dim: int = 16
    seq_len: int = 10
    epoch_seconds: int = 30
    gabor_kernel_size: int = 51

    @property
    def samples_per_epoch(self) -> int:
        return self.epoch_seconds * self.sampling_rate

    @property
    def input_shape(self) -> tuple:
        """Expected single-sequence input shape [B, T, C, S]."""
        return (1, self.seq_len, self.n_channels, self.samples_per_epoch)

    @property
    def output_shape(self) -> tuple:
        """Expected output shape [B, T, n_classes]."""
        return (1, self.seq_len, self.n_classes)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing parameters matching the training pipeline."""

    bandpass_low: float = 0.5
    bandpass_high: float = 35.0
    bandpass_order: int = 4
    notch_freq: float = 50.0
    notch_quality: float = 30.0
    normalization: str = "z-score"
    sampling_rate: int = 100
    epoch_seconds: int = 30

    @property
    def samples_per_epoch(self) -> int:
        return self.epoch_seconds * self.sampling_rate
