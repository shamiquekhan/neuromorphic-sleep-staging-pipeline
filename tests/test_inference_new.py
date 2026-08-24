"""Tests for the inference engine."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.config import CHECKPOINT_PATH, StudentConfig
from sleep_staging.inference import PredictionResult, SleepStagePredictor


@pytest.fixture(scope="module")
def predictor():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("Checkpoint not found")
    return SleepStagePredictor(device="cpu")


@pytest.fixture
def sample_sequence():
    config = StudentConfig()
    return np.random.randn(*config.input_shape).astype(np.float32)


class TestInputValidation:
    def test_valid_input(self, predictor, sample_sequence):
        predictor.validate_input(sample_sequence)

    def test_wrong_ndim(self, predictor):
        with pytest.raises(ValueError, match="ndim"):
            predictor.validate_input(np.random.randn(10, 4, 3000))

    def test_wrong_seq_len(self, predictor):
        with pytest.raises(ValueError, match="epochs"):
            predictor.validate_input(np.random.randn(1, 5, 4, 3000).astype(np.float32))

    def test_wrong_channels(self, predictor):
        with pytest.raises(ValueError, match="channels"):
            predictor.validate_input(np.random.randn(1, 10, 3, 3000).astype(np.float32))

    def test_wrong_samples(self, predictor):
        with pytest.raises(ValueError, match="samples"):
            predictor.validate_input(np.random.randn(1, 10, 4, 2000).astype(np.float32))


class TestPrediction:
    def test_predict_returns_result(self, predictor, sample_sequence):
        result = predictor.predict(sample_sequence, target_epoch=9)
        assert isinstance(result, PredictionResult)

    def test_result_fields(self, predictor, sample_sequence):
        result = predictor.predict(sample_sequence, target_epoch=9)
        assert result.stage in ("Wake", "N1", "N2", "N3", "REM")
        assert 0 <= result.stage_index <= 4
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 5
        assert result.latency_ms >= 0
        assert result.target_epoch == 9

    def test_probabilities_sum_to_one(self, predictor, sample_sequence):
        result = predictor.predict(sample_sequence, target_epoch=9)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_predict_sequence(self, predictor, sample_sequence):
        results = predictor.predict_sequence(sample_sequence)
        assert len(results) == 10
        for r in results:
            assert isinstance(r, PredictionResult)

    def test_determinism(self, predictor, sample_sequence):
        r1 = predictor.predict(sample_sequence, target_epoch=9)
        r2 = predictor.predict(sample_sequence, target_epoch=9)
        assert r1.stage == r2.stage
        assert abs(r1.confidence - r2.confidence) < 1e-5

    def test_metadata(self, predictor):
        assert "name" in predictor.model_info
        assert predictor.model_info["name"] == "Improved Student"
