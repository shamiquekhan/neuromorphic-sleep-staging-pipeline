"""Signal quality control checks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class QCResult:
    """Quality-control result for a single channel."""

    channel_name: str
    passed: bool
    clipping_detected: bool = False
    flatline_detected: bool = False
    message: str = "PASS"


def check_channel_quality(
    signal: np.ndarray,
    channel_name: str = "",
    clipping_threshold: float = 8.0,
    flatline_std_threshold: float = 0.05,
) -> QCResult:
    """Run basic QC checks on a single-channel signal.

    Checks:
        1. Amplitude clipping (values exceeding threshold after z-score).
        2. Flatline (std below threshold).

    Args:
        signal: 1-D array for one channel.
        channel_name: Label for the channel.
        clipping_threshold: Max absolute z-score before flagging.
        flatline_std_threshold: Min std before flagging as flat.

    Returns:
        ``QCResult`` indicating pass/fail.
    """
    clipped = bool(np.max(np.abs(signal)) > clipping_threshold)
    flat = bool(np.std(signal) < flatline_std_threshold)

    passed = not (clipped or flat)
    parts = []
    if clipped:
        parts.append("clipping")
    if flat:
        parts.append("flatline")

    return QCResult(
        channel_name=channel_name,
        passed=passed,
        clipping_detected=clipped,
        flatline_detected=flat,
        message="PASS" if passed else "; ".join(parts),
    )


def check_epoch_quality(
    epoch: np.ndarray,
    channel_names: list[str] | None = None,
) -> list[QCResult]:
    """QC for a single epoch across all channels.

    Args:
        epoch: Array of shape ``[n_channels, n_samples]``.
        channel_names: Optional channel labels.

    Returns:
        List of ``QCResult``, one per channel.
    """
    n_ch = epoch.shape[0]
    if channel_names is None:
        channel_names = [f"CH{i}" for i in range(n_ch)]

    return [
        check_channel_quality(epoch[i], channel_names[i])
        for i in range(n_ch)
    ]
