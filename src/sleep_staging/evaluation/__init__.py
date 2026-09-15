"""Evaluation metrics and protocols for sleep-stage classification."""

from .metrics import compute_all_metrics
from .protocol import (
    CausalEvalDataset,
    collect_causal_predictions,
    evaluate_causal,
)

__all__ = [
    "compute_all_metrics",
    "CausalEvalDataset",
    "collect_causal_predictions",
    "evaluate_causal",
]
