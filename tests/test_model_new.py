"""Tests for the Improved Student model."""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.config import StudentConfig
from sleep_staging.models import ImprovedStudent, count_parameters


@pytest.fixture
def config():
    return StudentConfig()


@pytest.fixture
def model(config):
    return ImprovedStudent(config)


class TestArchitecture:
    def test_parameter_count(self, model):
        n = count_parameters(model)
        assert n > 0
        assert n < 200_000

    def test_output_shape(self, model, config):
        x = torch.randn(*config.input_shape)
        with torch.inference_mode():
            logits = model(x)
        assert logits.shape == (1, config.seq_len, config.n_classes)

    def test_probability_normalization(self, model, config):
        x = torch.randn(*config.input_shape)
        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_determinism(self, model, config):
        x = torch.randn(*config.input_shape)
        model.eval()
        with torch.inference_mode():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_return_features(self, model, config):
        x = torch.randn(*config.input_shape)
        with torch.inference_mode():
            logits, features = model(x, return_features=True)
        assert logits.shape == (1, config.seq_len, config.n_classes)
        assert features.shape[0] == 1
        assert features.shape[1] == config.seq_len

    def test_batch_size_2(self, model, config):
        x = torch.randn(2, config.seq_len, config.n_channels, config.samples_per_epoch)
        with torch.inference_mode():
            logits = model(x)
        assert logits.shape == (2, config.seq_len, config.n_classes)
