"""Tests for checkpoint loading and model contract."""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.config import CHECKPOINT_PATH, StudentConfig
from sleep_staging.models import ImprovedStudent, count_parameters


@pytest.fixture(scope="module")
def checkpoint():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("Checkpoint not found")
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    # Checkpoint is a raw state_dict (no "model_state_dict" wrapper)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    return raw


class TestCheckpoint:
    def test_checkpoint_is_dict(self, checkpoint):
        assert isinstance(checkpoint, dict)

    def test_state_dict_loads(self, checkpoint):
        config = StudentConfig()
        model = ImprovedStudent(config)
        model.load_state_dict(checkpoint, strict=True)

    def test_parameter_count_matches(self, checkpoint):
        config = StudentConfig()
        model = ImprovedStudent(config)
        model.load_state_dict(checkpoint, strict=True)
        n = count_parameters(model)
        assert n == 99_477

    def test_forward_pass_with_checkpoint(self, checkpoint):
        config = StudentConfig()
        model = ImprovedStudent(config)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        x = torch.randn(*config.input_shape)
        with torch.inference_mode():
            logits = model(x)
        assert logits.shape == tuple(config.output_shape)
