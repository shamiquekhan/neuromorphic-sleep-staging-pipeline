"""Evaluation metrics for sleep-stage classification."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute the full set of official project metrics.

    Returns a dict with keys: accuracy, kappa, macro_f1, weighted_f1,
    mgm, report, confusion_matrix.
    """
    if stage_names is None:
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

    acc = float(np.mean(y_true == y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(stage_names))))
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    per_class_acc = recalls
    mgm = float(np.prod(np.clip(recalls, 1e-12, 1.0)) ** (1.0 / len(stage_names)))

    return {
        "accuracy": acc,
        "kappa": kappa,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mgm": mgm,
        "report": classification_report(
            y_true, y_pred, target_names=stage_names, zero_division=0,
        ),
        "confusion_matrix": cm,
        "per_class_accuracy": dict(zip(stage_names, per_class_acc.tolist())),
    }
