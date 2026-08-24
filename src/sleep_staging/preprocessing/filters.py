"""Bandpass and notch filtering."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal: np.ndarray,
    low_hz: float = 0.5,
    high_hz: float = 35.0,
    fs: int = 100,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        signal: Input signal (1-D or 2-D with last axis = time).
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal of the same shape.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1).astype(signal.dtype)


def notch_filter(
    signal: np.ndarray,
    freq_hz: float = 50.0,
    quality: float = 30.0,
    fs: int = 100,
) -> np.ndarray:
    """Zero-phase notch filter at *freq_hz*."""
    b, a = iirnotch(freq_hz, quality, fs)
    return filtfilt(b, a, signal, axis=-1).astype(signal.dtype)
