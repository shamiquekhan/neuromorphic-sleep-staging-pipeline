"""Centralized configuration for the NeuroSleep package."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "student_improved_best.pt"
IMPROVED_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "full_model_trained" / "student_full_trained.pt"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SLEEP_EDF_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "sleep_edf"
SHHS_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "shhs"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sleep_edf"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sleep_edf.csv"
EXPANDED_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sleep_edf_expanded.json"

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
    """Configuration for the Improved Student model."""

    n_channels: int = 4
    n_classes: int = 5
    sampling_rate: int = 100
    gru_hidden: int = 64
    gru_layers: int = 2
    stem_width: int = 10
    encoder_channels: tuple = (32, 32)
    gabor_n_filters: int = 8
    gabor_out_dim: int = 32
    seq_len: int = 10
    epoch_seconds: int = 30

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
