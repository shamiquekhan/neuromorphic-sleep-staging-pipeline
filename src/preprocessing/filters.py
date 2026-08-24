"""Signal preprocessing: bandpass, notch, normalization, artifact QC."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos


def bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    """Butterworth bandpass filter in second-order sections."""
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def notch_sos(freq: float, fs: float, q: float = 30.0):
    """IIR notch filter in second-order sections."""
    b, a = iirnotch(freq, q, fs)
    return tf2sos(b, a)


def filter_signal(x: np.ndarray, fs: float,
                  bandpass: tuple = (0.5, 35.0), notch_hz: float = 50.0) -> np.ndarray:
    """Apply bandpass then notch filtering to a signal array."""
    x = sosfiltfilt(bandpass_sos(*bandpass, fs=fs), x, axis=-1)
    x = sosfiltfilt(notch_sos(notch_hz, fs=fs), x, axis=-1)
    return x


def normalize_epoch(ep: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization."""
    mu = ep.mean(axis=-1, keepdims=True)
    sig = ep.std(axis=-1, keepdims=True)
    sig = np.where(sig < 1e-8, 1e-8, sig)
    return (ep - mu) / sig


def qc_flag(ep: np.ndarray) -> bool:
    """Return True if an epoch is flagged as an artifact."""
    clipped = bool(np.any(np.abs(ep) > 8.0))
    flatline = bool(np.any(ep.std(axis=-1) < 0.05))
    nan_inf = bool(~np.isfinite(ep).all())
    return clipped or flatline or nan_inf
