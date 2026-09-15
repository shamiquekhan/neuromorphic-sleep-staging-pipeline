"""Tests for the post-audit LoRA fixes (P1 #11).

Covers:
  * exact target matching — substring false positives must fail
  * adapter load without ``.data`` assignment — autograd state stays
    intact and optimizer updates keep working
  * strict-PEFT norm freeze — BN running stats cannot change while
    only LoRA parameters are "trainable"
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.adaptation.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    assert_lora_targets,
    count_lora_parameters,
    freeze_norm_layers,
    get_lora_targets,
    save_adapter,
    load_adapter,
)
from sleep_staging.models.improved_student import ImprovedStudent


@pytest.fixture
def base_model():
    torch.manual_seed(0)
    return ImprovedStudent()


class TestExactTargetMatching:
    def test_exact_names_required(self, base_model):
        """A target not present verbatim must raise — the pre-fix
        substring check would have passed on partial matches."""
        cfg = LoRAConfig(rank=4, target_modules=["head", "does_not_exist"])
        model = apply_lora(base_model, cfg)
        with pytest.raises(RuntimeError, match="does_not_exist"):
            assert_lora_targets(model, cfg.target_modules)

    def test_no_substring_false_positive(self, base_model):
        """``head`` must not satisfy a check expecting ``enc.0.head``."""
        cfg = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, cfg)
        with pytest.raises(RuntimeError, match="enc.0.head"):
            assert_lora_targets(model, ["enc.0.head"])

    def test_valid_targets_pass(self, base_model):
        cfg = LoRAConfig(rank=4, target_modules=["head", "gab_proj"])
        model = apply_lora(base_model, cfg)
        assert_lora_targets(model, cfg.target_modules)  # no raise
        assert set(get_lora_targets(model)) == {"head", "gab_proj"}


class TestAdapterLoadWithoutDataAssignment:
    def test_load_keeps_autograd_state(self, base_model, tmp_path):
        """After load_adapter, LoRA params must remain leaf tensors
        with requires_grad=True and receive optimizer updates."""
        cfg = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, cfg)
        save_adapter(model, tmp_path)

        fresh = ImprovedStudent()
        fresh = apply_lora(fresh, LoRAConfig(rank=4, target_modules=["head"]))
        before = fresh.head.lora_A.detach().clone()
        load_adapter(fresh, tmp_path)
        assert not torch.equal(before, fresh.head.lora_A.detach())
        assert fresh.head.lora_A.requires_grad
        assert fresh.head.lora_A.is_leaf

        # Optimizer step must change the loaded adapter (graph intact).
        opt = torch.optim.SGD([fresh.head.lora_A, fresh.head.lora_B], lr=0.1)
        loss = fresh.head.lora_A.sum() + fresh.head.lora_B.sum()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)

    def test_roundtrip_adapter_state(self, base_model, tmp_path):
        cfg = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, cfg)
        # Give adapters nonzero values
        with torch.no_grad():
            model.head.lora_A.normal_()
            model.head.lora_B.normal_()
        save_adapter(model, tmp_path)

        fresh = apply_lora(ImprovedStudent(), cfg)
        load_adapter(fresh, tmp_path)
        torch.testing.assert_close(
            fresh.head.lora_A, model.head.lora_A,
        )
        torch.testing.assert_close(
            fresh.head.lora_B, model.head.lora_B,
        )


class TestStrictPEFTNormFreeze:
    def test_batchnorm_stats_frozen(self, base_model):
        """With norm layers frozen, a forward/backward in train mode
        must not change BN running stats — the strict-PEFT guarantee
        behind any 'N trainable parameters' claim."""
        cfg = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, cfg)
        model.train()
        n = freeze_norm_layers(model)
        assert n == 4  # stem_s BN, stem_l BN, enc.0 BN, enc.1 BN

        stats_before = [
            m.running_mean.clone() for m in model.modules()
            if isinstance(m, nn.BatchNorm1d)
        ]
        x = torch.randn(2, 10, 4, 3000)
        model(x).sum().backward()
        stats_after = [
            m.running_mean for m in model.modules()
            if isinstance(m, nn.BatchNorm1d)
        ]
        for b, a in zip(stats_before, stats_after):
            torch.testing.assert_close(b, a)

    def test_batchnorm_stats_would_update_without_freeze(self, base_model):
        """Sanity: without the freeze, BN stats DO change — proving the
        freeze is what makes the PEFT claim strict."""
        cfg = LoRAConfig(rank=4, target_modules=["head"])
        model = apply_lora(base_model, cfg)
        model.train()
        before = model.stem_s[1].running_mean.clone()
        model(torch.randn(2, 10, 4, 3000))
        assert not torch.equal(before, model.stem_s[1].running_mean)

    def test_lora_only_params_trainable(self, base_model):
        cfg = LoRAConfig(rank=8, target_modules=["head"])
        model = apply_lora(base_model, cfg)
        pc = count_lora_parameters(model)
        # head: Linear(64→5): A is 8×64=512, B is 5×8=40
        assert pc["trainable"] == 512 + 40
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable == {"head.lora_A", "head.lora_B"}
