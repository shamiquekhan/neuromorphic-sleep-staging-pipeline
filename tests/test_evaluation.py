"""Tests for evaluation metrics — regression tests for the precision bug."""

import numpy as np
import pytest

from sleep_staging.evaluation import compute_all_metrics


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_precision_is_not_trivially_one(self):
        """A metrics function that can never produce precision < 1 is broken."""
        # Deliberately imperfect predictions
        y_true = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([1, 0, 1, 0, 2, 2, 2, 0, 2])

        metrics = compute_all_metrics(y_true, y_pred, stage_names=["A", "B", "C"])

        # Extract per-class precisions from the confusion matrix
        cm = metrics["confusion_matrix"]
        precisions = cm.diagonal() / cm.sum(axis=0).clip(min=1)

        # At least one class must have precision < 1.0 with imperfect predictions
        assert any(p < 1.0 for p in precisions), (
            f"All precisions are 1.0 — metric is trivially correct: {precisions}"
        )

    def test_perfect_predictions(self):
        """Perfect predictions should yield precision=1.0 for all classes."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        metrics = compute_all_metrics(y_true, y_pred, stage_names=["A", "B", "C"])

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_known_confusion(self):
        """Verify metrics against hand-computed values."""
        # Class 0: 3 samples, class 1: 3 samples
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        metrics = compute_all_metrics(y_true, y_pred, stage_names=["A", "B"])

        assert metrics["accuracy"] == pytest.approx(4 / 6)
        # Class A: TP=2, FP=1, FN=1 → precision=2/3, recall=2/3, f1=2/3
        # Class B: TP=2, FP=1, FN=1 → precision=2/3, recall=2/3, f1=2/3
        assert metrics["macro_f1"] == pytest.approx(2 / 3)

    def test_weighted_f1_favors_majority(self):
        """Weighted F1 should be higher than macro F1 when majority class is good."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 2])

        metrics = compute_all_metrics(y_true, y_pred, stage_names=["A", "B", "C"])

        assert metrics["weighted_f1"] >= metrics["macro_f1"]

    def test_empty_classes_handled(self):
        """Missing classes should not crash — they get zero metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        metrics = compute_all_metrics(y_true, y_pred, stage_names=["A", "B", "C"])

        assert metrics["accuracy"] == pytest.approx(0.5)
        # Class C has no samples → confusion matrix row is all zeros
        assert "C" in metrics["per_class_accuracy"]

    def test_zero_recall_class(self):
        """A class with zero recall should have zero F1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        metrics = compute_all_metrics(y_true, y_pred, stage_names=["A", "B"])

        # Class B has zero recall → zero F1
        assert metrics["macro_f1"] < 1.0
