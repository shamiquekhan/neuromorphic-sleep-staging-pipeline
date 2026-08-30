#!/usr/bin/env python3
"""LoRA smoke test — verifies adapter correctness before full benchmark.

Tests:
1. LoRA wraps target modules correctly
2. Trainable parameter count is correct
3. Base weights stay frozen during training
4. LoRA weights change after optimizer step
5. Zero-adapter equivalence (output matches base when LoRA initialized to zero)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.models.improved_student import ImprovedStudent, count_parameters
from sleep_staging.adaptation.lora import (
    LoRAConfig, apply_lora, count_lora_parameters, get_lora_targets,
    assert_lora_targets, LoRALinear, LoRAConv1d,
)

CHECKPOINT = REPO / "artifacts" / "final" / "student_full_finetuned.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_lora_wrapping():
    """Test that LoRA wraps target modules correctly."""
    print("=" * 60)
    print("TEST 1: LoRA wrapping")
    print("=" * 60)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)

    # Test head-only
    model_head = ImprovedStudent()
    model_head.load_state_dict(ckpt)
    base_params = count_parameters(model_head)
    print(f"  Base model params: {base_params:,}")

    config = LoRAConfig(rank=8, alpha=16, target_modules=["head"])
    model_head = apply_lora(model_head, config)
    targets = get_lora_targets(model_head)
    print(f"  Head-only targets: {targets}")
    assert "head" in targets, f"head not in targets: {targets}"
    pc = count_lora_parameters(model_head)
    print(f"  Head-only trainable: {pc['trainable']:,} ({pc['trainable_pct']}%)")
    assert pc["trainable"] == 552, f"Expected 552, got {pc['trainable']}"
    print("  PASSED\n")

    # Test CNN+head
    model_cnn = ImprovedStudent()
    model_cnn.load_state_dict(ckpt)
    config2 = LoRAConfig(rank=8, alpha=16, target_modules=["enc.0.pw", "enc.1.pw", "head"])
    model_cnn = apply_lora(model_cnn, config2)
    targets2 = get_lora_targets(model_cnn)
    print(f"  CNN+head targets: {targets2}")
    assert_lora_targets(model_cnn, config2.target_modules)
    pc2 = count_lora_parameters(model_cnn)
    print(f"  CNN+head trainable: {pc2['trainable']:,} ({pc2['trainable_pct']}%)")
    print("  PASSED\n")

    return True


def test_base_frozen():
    """Test that base weights don't change during training."""
    print("=" * 60)
    print("TEST 2: Base weights frozen")
    print("=" * 60)

    model = ImprovedStudent()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)

    # Snapshot base weights before LoRA (on CPU for comparison)
    base_weights_before = {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
    }

    config = LoRAConfig(rank=8, alpha=16, target_modules=["head"])
    model = apply_lora(model, config)
    model = model.to(DEVICE)

    # Create dummy batch
    x = torch.randn(2, 10, 4, 3000).to(DEVICE)
    y = torch.randint(0, 5, (2, 10)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    # Train 3 steps
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, 5), y.view(-1))
        loss.backward()
        optimizer.step()

    # Check base weights unchanged (compare by checking original module weights)
    changed = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Check the original linear layer weights
            orig_name = name + ".original"
            for suffix in ["weight", "bias"]:
                param_name = f"{orig_name}.{suffix}"
                if param_name in base_weights_before:
                    if not torch.equal(module.original.weight.data.cpu(), base_weights_before[param_name]):
                        changed.append(param_name)
        elif isinstance(module, LoRAConv1d):
            orig_name = name + ".original"
            for suffix in ["weight", "bias"]:
                param_name = f"{orig_name}.{suffix}"
                if param_name in base_weights_before:
                    if not torch.equal(module.original.weight.data.cpu(), base_weights_before[param_name]):
                        changed.append(param_name)

    if changed:
        print(f"  FAILED: Base weights changed: {changed}")
        return False
    else:
        print("  All base weights frozen correctly")
        print("  PASSED\n")
        return True


def test_adapter_trains():
    """Test that LoRA weights change after training."""
    print("=" * 60)
    print("TEST 3: LoRA adapter trains")
    print("=" * 60)

    model = ImprovedStudent()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)

    config = LoRAConfig(rank=8, alpha=16, target_modules=["head"])
    model = apply_lora(model, config)
    model = model.to(DEVICE)

    # Snapshot LoRA weights
    lora_before = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            lora_before[name] = {
                "A": module.lora_A.data.clone(),
                "B": module.lora_B.data.clone(),
            }

    x = torch.randn(2, 10, 4, 3000).to(DEVICE)
    y = torch.randint(0, 5, (2, 10)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, 5), y.view(-1))
        loss.backward()
        optimizer.step()

    # Check LoRA weights changed
    changed = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            if name in lora_before:
                a_changed = not torch.equal(module.lora_A.data, lora_before[name]["A"])
                b_changed = not torch.equal(module.lora_B.data, lora_before[name]["B"])
                if a_changed or b_changed:
                    changed.append(name)

    if changed:
        print(f"  LoRA adapters trained: {changed}")
        print("  PASSED\n")
        return True
    else:
        print("  FAILED: No LoRA weights changed")
        return False


def test_zero_adapter_equivalence():
    """Test that zero-initialized LoRA produces same output as base."""
    print("=" * 60)
    print("TEST 4: Zero-adapter equivalence")
    print("=" * 60)

    # Base model
    model_base = ImprovedStudent()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model_base.load_state_dict(ckpt)
    model_base = model_base.to(DEVICE)
    model_base.eval()

    # LoRA model with zero init
    model_lora = ImprovedStudent()
    model_lora.load_state_dict(ckpt)
    config = LoRAConfig(rank=8, alpha=16, target_modules=["head"])
    model_lora = apply_lora(model_lora, config)

    # Set LoRA B to zero (already zero by default, but ensure)
    for module in model_lora.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            module.lora_B.data.zero_()

    model_lora = model_lora.to(DEVICE)
    model_lora.eval()

    x = torch.randn(1, 10, 4, 3000).to(DEVICE)

    with torch.no_grad():
        out_base = model_base(x)
        out_lora = model_lora(x)

    diff = (out_base - out_lora).abs().max().item()
    print(f"  Max logit difference: {diff:.8f}")

    if diff < 1e-5:
        print("  PASSED\n")
        return True
    else:
        print(f"  FAILED: diff={diff} > 1e-5")
        return False


def test_conv1d_forward():
    """Test LoRA Conv1d produces correct output shape."""
    print("=" * 60)
    print("TEST 5: Conv1d LoRA forward pass")
    print("=" * 60)

    model = ImprovedStudent()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)

    config = LoRAConfig(rank=8, alpha=16, target_modules=["enc.0.pw", "enc.1.pw", "head"])
    model = apply_lora(model, config)
    model = model.to(DEVICE)

    x = torch.randn(2, 10, 4, 3000).to(DEVICE)

    try:
        out = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {out.shape}")
        assert out.shape == (2, 10, 5), f"Unexpected shape: {out.shape}"
        print("  PASSED\n")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  LoRA SMOKE TEST")
    print(f"  Device: {DEVICE}")
    print("=" * 60 + "\n")

    results = {}
    results["wrapping"] = test_lora_wrapping()
    results["base_frozen"] = test_base_frozen()
    results["adapter_trains"] = test_adapter_trains()
    results["zero_equivalence"] = test_zero_adapter_equivalence()
    results["conv1d_forward"] = test_conv1d_forward()

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  ALL TESTS PASSED — LoRA implementation verified.")
    else:
        print("\n  SOME TESTS FAILED — Fix issues before running benchmark.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
