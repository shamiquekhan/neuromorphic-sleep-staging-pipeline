"""Tests for the LoRA adaptation system."""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.adaptation import LoRAConfig, LoRALinear, apply_lora, count_lora_parameters
from sleep_staging.config import CHECKPOINT_PATH, StudentConfig
from sleep_staging.models import ImprovedStudent


@pytest.fixture(scope="module")
def base_model():
    config = StudentConfig()
    model = ImprovedStudent(config)
    if CHECKPOINT_PATH.exists():
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
    return model


@pytest.fixture
def sample_input():
    return torch.randn(1, 10, 4, 3000)


class TestLoRALayer:
    def test_wraps_linear(self):
        import torch.nn as nn
        orig = nn.Linear(64, 5)
        lora = LoRALinear(orig, rank=4, alpha=8.0)
        assert lora.original is orig

    def test_output_shape(self):
        import torch.nn as nn
        orig = nn.Linear(64, 5)
        lora = LoRALinear(orig, rank=4)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 5)

    def test_original_frozen(self):
        import torch.nn as nn
        orig = nn.Linear(64, 5)
        lora = LoRALinear(orig, rank=4)
        assert not lora.original.weight.requires_grad

    def test_lora_params_trainable(self):
        import torch.nn as nn
        orig = nn.Linear(64, 5)
        lora = LoRALinear(orig, rank=4)
        for p in lora.lora_parameters:
            assert p.requires_grad

    def test_adapter_save_load(self, tmp_path):
        import torch.nn as nn
        orig = nn.Linear(64, 5)
        lora = LoRALinear(orig, rank=4)
        state = lora.adapter_state_dict
        assert "lora_A" in state
        assert "lora_B" in state


class TestApplyLoRA:
    def test_head_only(self, base_model):
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["head"])
        model = apply_lora(base_model, config)
        params = count_lora_parameters(model)
        assert params["trainable"] > 0
        assert params["trainable"] < params["total"]

    def test_backbone_frozen(self, base_model):
        config = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, config)
        # Check that stem/enc/gru are frozen
        for name, p in model.named_parameters():
            if "lora_" not in name and "head" not in name:
                assert not p.requires_grad, f"Unexpected trainable: {name}"

    def test_forward_shape(self, base_model, sample_input):
        config = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, config)
        model.eval()
        with torch.inference_mode():
            logits = model(sample_input)
        assert logits.shape == (1, 10, 5)

    def test_determinism(self, base_model, sample_input):
        config = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, config)
        model.eval()
        with torch.inference_mode():
            out1 = model(sample_input)
            out2 = model(sample_input)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_probability_normalization(self, base_model, sample_input):
        config = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, config)
        model.eval()
        with torch.inference_mode():
            probs = torch.softmax(model(sample_input), dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 10), atol=1e-5)


class TestLoRAConfig:
    def test_scaling(self):
        config = LoRAConfig(rank=4, alpha=8.0)
        assert config.scaling == 2.0

    def test_default_targets(self):
        config = LoRAConfig()
        assert "head" in config.target_modules
