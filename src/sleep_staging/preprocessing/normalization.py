"""Channel-wise normalization."""

import numpy as np


def zscore_normalize(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Per-channel z-score normalization.

    Args:
        signal: Input array (1-D or N-D).
        axis: Axis along which to compute mean/std.

    Returns:
        Normalized array with zero mean and unit variance per channel.
    """
    mu = np.mean(signal, axis=axis, keepdims=True)
    sigma = np.std(signal, axis=axis, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return ((signal - mu) / sigma).astype(np.float32)
