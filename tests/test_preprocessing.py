"""Tests for signal preprocessing and model shapes."""

import numpy as np
import pytest
import torch

from src.preprocessing.filters import filter_signal, normalize_epoch, qc_flag, bandpass_sos, notch_sos
from src.models.improved_student import ImprovedStudent, count_parameters


# ── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_bandpass_sos_shape(self):
        sos = bandpass_sos(0.5, 35.0, fs=100)
        assert sos.ndim == 2

    def test_notch_sos_shape(self):
        sos = notch_sos(50.0, fs=100)
        assert sos.ndim == 2

    def test_filter_signal_preserves_shape(self):
        x = np.random.randn(4, 3000).astype(np.float32)
        y = filter_signal(x, fs=100)
        assert y.shape == x.shape

    def test_normalize_epoch_shape(self):
        ep = np.random.randn(4, 3000).astype(np.float32)
        normed = normalize_epoch(ep)
        assert normed.shape == ep.shape

    def test_normalize_epoch_statistics(self):
        ep = np.random.randn(4, 3000).astype(np.float32)
        normed = normalize_epoch(ep)
        for ch in range(4):
            assert abs(normed[ch].mean()) < 0.1
            assert abs(normed[ch].std() - 1.0) < 0.1

    def test_qc_flag_clean(self):
        ep = np.random.randn(4, 3000).astype(np.float32) * 0.5
        assert qc_flag(ep) is False

    def test_qc_flag_clipped(self):
        ep = np.random.randn(4, 3000).astype(np.float32) * 0.5
        ep[0, 100] = 10.0
        assert qc_flag(ep) is True

    def test_qc_flag_flatline(self):
        ep = np.random.randn(4, 3000).astype(np.float32) * 0.5
        ep[2, :] = 0.0
        assert qc_flag(ep) is True


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestModel:
    @pytest.fixture
    def model(self):
        return ImprovedStudent(n_classes=5)

    def test_output_shape(self, model):
        x = torch.randn(2, 10, 4, 3000)
        out = model(x)
        assert out.shape == (2, 10, 5)

    def test_output_features_shape(self, model):
        x = torch.randn(2, 10, 4, 3000)
        logits, features = model(x, return_features=True)
        assert logits.shape == (2, 10, 5)
        assert features.shape[0] == 2
        assert features.shape[1] == 10

    def test_parameter_count(self, model):
        n = count_parameters(model)
        assert n < 150_000, f"Model too large: {n} params"
        assert n > 50_000, f"Model too small: {n} params"

    def test_single_batch(self, model):
        x = torch.randn(1, 10, 4, 3000)
        out = model(x)
        assert out.shape == (1, 10, 5)

    def test_deterministic(self, model):
        model.eval()
        x = torch.randn(1, 10, 4, 3000)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)
