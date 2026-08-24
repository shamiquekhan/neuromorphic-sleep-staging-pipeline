"""Tests for inference pipeline."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.models.improved_student import ImprovedStudent, count_parameters
from src.inference.predict import load_model, run_inference, format_prediction


@pytest.fixture
def saved_checkpoint(tmp_path):
    model = ImprovedStudent(n_classes=5)
    path = tmp_path / "test_checkpoint.pt"
    torch.save(model.state_dict(), path)
    return path


class TestInference:
    def test_load_model(self, saved_checkpoint):
        model = load_model(saved_checkpoint)
        assert isinstance(model, ImprovedStudent)
        assert not model.training

    def test_run_inference_shape(self, saved_checkpoint):
        model = load_model(saved_checkpoint)
        epochs = np.random.randn(10, 4, 3000).astype(np.float32)
        probs, preds = run_inference(model, epochs)
        assert probs.shape == (10, 5)
        assert preds.shape == (10,)

    def test_run_inference_batched(self, saved_checkpoint):
        model = load_model(saved_checkpoint)
        epochs = np.random.randn(2, 10, 4, 3000).astype(np.float32)
        probs, preds = run_inference(model, epochs)
        assert probs.shape == (10, 5)
        assert preds.shape == (10,)

    def test_probabilities_sum_to_one(self, saved_checkpoint):
        model = load_model(saved_checkpoint)
        epochs = np.random.randn(10, 4, 3000).astype(np.float32)
        probs, _ = run_inference(model, epochs)
        sums = probs.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_predictions_in_range(self, saved_checkpoint):
        model = load_model(saved_checkpoint)
        epochs = np.random.randn(10, 4, 3000).astype(np.float32)
        _, preds = run_inference(model, epochs)
        assert all(0 <= p < 5 for p in preds)

    def test_format_prediction(self):
        preds = np.array([2, 2, 0])
        probs = np.array([
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.05, 0.05, 0.8, 0.05, 0.05],
            [0.9, 0.05, 0.02, 0.02, 0.01],
        ])
        results = format_prediction(preds, probs)
        assert len(results) == 3
        assert results[0]["predicted"] == "N2"
        assert results[2]["predicted"] == "Wake"
