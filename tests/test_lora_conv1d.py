"""Tests for LoRA Conv1d support.

These tests verify that the LoRA implementation correctly handles
nn.Conv1d layers, which was a critical bug in the original implementation.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.models.improved_student import ImprovedStudent
from sleep_staging.adaptation.lora import (
    LoRAConfig,
    LoRALinear,
    LoRAConv1d,
    apply_lora,
    count_lora_parameters,
    get_lora_targets,
    assert_lora_targets,
)


class TestLoRAConv1d:
    """Tests for LoRAConv1d layer."""

    def test_wraps_conv1d(self):
        """LoRAConv1d should wrap an nn.Conv1d layer."""
        conv = nn.Conv1d(16, 32, kernel_size=1)
        lora = LoRAConv1d(conv, rank=8, alpha=16)
        assert isinstance(lora, LoRAConv1d)
        assert lora.original is conv

    def test_output_shape(self):
        """LoRAConv1d should preserve output shape."""
        conv = nn.Conv1d(16, 32, kernel_size=1)
        lora = LoRAConv1d(conv, rank=8, alpha=16)
        x = torch.randn(2, 16, 100)
        out = lora(x)
        assert out.shape == (2, 32, 100)

    def test_original_frozen(self):
        """Original Conv1d weights should be frozen."""
        conv = nn.Conv1d(16, 32, kernel_size=1)
        lora = LoRAConv1d(conv, rank=8, alpha=16)
        for param in lora.original.parameters():
            assert not param.requires_grad

    def test_lora_params_trainable(self):
        """LoRA parameters should be trainable."""
        conv = nn.Conv1d(16, 32, kernel_size=1)
        lora = LoRAConv1d(conv, rank=8, alpha=16)
        for param in lora.lora_parameters:
            assert param.requires_grad

    def test_adapter_save_load(self):
        """Adapter state should be saveable and loadable."""
        conv = nn.Conv1d(16, 32, kernel_size=1)
        lora1 = LoRAConv1d(conv, rank=8, alpha=16)
        lora2 = LoRAConv1d(conv, rank=8, alpha=16)

        # Save state from lora1
        state = lora1.adapter_state_dict

        # Load into lora2
        lora2.load_adapter(state)

        # Verify states match
        assert torch.allclose(lora1.lora_A, lora2.lora_A)
        assert torch.allclose(lora1.lora_B, lora2.lora_B)


class TestConv1dLoRAIntegration:
    """Integration tests for Conv1d LoRA with ImprovedStudent."""

    @pytest.fixture
    def model_with_checkpoint(self):
        """Load model with checkpoint."""
        model = ImprovedStudent()
        ckpt = torch.load(
            "artifacts/student_improved_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(ckpt)
        return model

    def test_cnn_targets_actually_wrapped(self, model_with_checkpoint):
        """CNN targets should actually be wrapped by LoRA (critical bug fix)."""
        model = model_with_checkpoint
        config = LoRAConfig(
            rank=8,
            alpha=16,
            target_modules=["enc.0.pw", "enc.1.pw", "head"],
        )
        model = apply_lora(model, config)

        targets = get_lora_targets(model)
        assert "enc.0.pw" in targets, f"enc.0.pw not in targets: {targets}"
        assert "enc.1.pw" in targets, f"enc.1.pw not in targets: {targets}"
        assert "head" in targets, f"head not in targets: {targets}"

    def test_cnn_targets_are_lora_conv1d(self, model_with_checkpoint):
        """CNN targets should be LoRAConv1d, not LoRALinear."""
        model = model_with_checkpoint
        config = LoRAConfig(
            rank=8,
            alpha=16,
            target_modules=["enc.0.pw", "enc.1.pw"],
        )
        model = apply_lora(model, config)

        for name, module in model.named_modules():
            if name in ["enc.0.pw", "enc.1.pw"]:
                assert isinstance(module, LoRAConv1d), (
                    f"{name} is {type(module).__name__}, expected LoRAConv1d"
                )

    def test_cnn_lora_has_more_params_than_head_only(self, model_with_checkpoint):
        """CNN+head LoRA should have MORE trainable params than head-only."""
        model_head = model_with_checkpoint
        config_head = LoRAConfig(rank=8, alpha=16, target_modules=["head"])
        model_head = apply_lora(model_head, config_head)
        params_head = count_lora_parameters(model_head)

        model_cnn = model_with_checkpoint
        config_cnn = LoRAConfig(
            rank=8, alpha=16, target_modules=["enc.0.pw", "enc.1.pw", "head"]
        )
        model_cnn = apply_lora(model_cnn, config_cnn)
        params_cnn = count_lora_parameters(model_cnn)

        assert params_cnn["trainable"] > params_head["trainable"], (
            f"CNN+head ({params_cnn['trainable']}) should have more params "
            f"than head-only ({params_head['trainable']})"
        )

    def test_forward_pass_with_cnn_lora(self, model_with_checkpoint):
        """Forward pass should work with CNN LoRA."""
        model = model_with_checkpoint
        config = LoRAConfig(
            rank=8,
            alpha=16,
            target_modules=["enc.0.pw", "enc.1.pw", "head"],
        )
        model = apply_lora(model, config)

        batch = torch.randn(2, 10, 4, 3000)
        logits = model(batch)
        assert logits.shape == (2, 10, 5)

    def test_gradients_flow_through_cnn_lora(self, model_with_checkpoint):
        """Gradients should flow through CNN LoRA parameters."""
        model = model_with_checkpoint
        config = LoRAConfig(
            rank=8,
            alpha=16,
            target_modules=["enc.0.pw", "enc.1.pw", "head"],
        )
        model = apply_lora(model, config)

        batch = torch.randn(2, 10, 4, 3000)
        logits = model(batch)
        logits.sum().backward()

        # Check that CNN LoRA parameters have gradients
        for name, param in model.named_parameters():
            if "enc.0.pw.lora" in name or "enc.1.pw.lora" in name:
                assert param.grad is not None, f"No gradient for {name}"

    def test_assert_lora_targets_passes(self, model_with_checkpoint):
        """assert_lora_targets should pass when targets are properly applied."""
        model = model_with_checkpoint
        config = LoRAConfig(
            rank=8,
            alpha=16,
            target_modules=["enc.0.pw", "enc.1.pw", "head"],
        )
        model = apply_lora(model, config)

        # Should not raise
        assert_lora_targets(model, ["enc.0.pw", "enc.1.pw", "head"])

    def test_assert_lora_targets_fails_when_missing(self, model_with_checkpoint):
        """assert_lora_targets should fail when targets are missing."""
        model = model_with_checkpoint
        config = LoRAConfig(rank=8, alpha=16, target_modules=["head"])
        model = apply_lora(model, config)

        # Should raise because enc.0.pw is not wrapped
        with pytest.raises(RuntimeError, match="not actually adapted"):
            assert_lora_targets(model, ["enc.0.pw", "enc.1.pw", "head"])
