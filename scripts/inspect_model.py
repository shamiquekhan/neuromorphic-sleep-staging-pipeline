#!/usr/bin/env python3
"""Model inspection script for NeuroSleep Improved Student.

Prints module name, class, input/output dimensions, parameter count,
and requires_grad for every module in the model.

Usage:
    python scripts/inspect_model.py
    python scripts/inspect_model.py --checkpoint artifacts/student_improved_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sleep_staging.models.improved_student import ImprovedStudent
from sleep_staging.adaptation.lora import (
    apply_lora, LoRAConfig, count_lora_parameters, get_lora_targets
)


def inspect_model(model: torch.nn.Module, title: str = "Model") -> None:
    """Print detailed model inspection."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

    total_params = 0
    trainable_params = 0

    print(f"{'Module Name':<35} {'Class':<25} {'Params':>10} {'Trainable':>10}")
    print("-" * 85)

    # Only count leaf parameters to avoid double counting
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Print module summary (only leaf modules with params)
    seen_modules = set()
    for name, module in model.named_modules():
        # Skip containers that just hold other modules
        if len(list(module.children())) > 0 and name != '':
            continue
        if name in seen_modules:
            continue
        seen_modules.add(name)

        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

        if params > 0:
            class_name = module.__class__.__name__
            print(f"{name:<35} {class_name:<25} {params:>10,} {trainable:>10,}")

    print("-" * 85)
    print(f"{'TOTAL':<35} {'':<25} {total_params:>10,} {trainable_params:>10,}")

    # Verify expected count
    expected = 99477
    if total_params == expected:
        print(f"\n✓ Parameter count verified: {total_params:,} == {expected:,}")
    else:
        print(f"\n✗ Parameter count mismatch: {total_params:,} != {expected:,}")

    return total_params, trainable_params


def inspect_lora_targets(model: torch.nn.Module) -> None:
    """Verify LoRA targets were actually applied."""
    targets = get_lora_targets(model)
    print(f"\nLoRA targets found: {targets}")

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            print(f"  {name}: {module.__class__.__name__}")
            if hasattr(module, 'rank'):
                print(f"    rank={module.rank}, alpha={module.alpha}, scaling={module.scaling}")


def forward_test(model: torch.nn.Module) -> None:
    """Test forward pass with random input."""
    model.eval()
    batch = torch.randn(2, 10, 4, 3000)  # [B, T, C, samples]
    with torch.no_grad():
        logits = model(batch)
    print(f"\nForward pass test:")
    print(f"  Input shape: {batch.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits dtype: {logits.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Inspect ImprovedStudent model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--lora-targets", nargs="+", default=None, help="Apply LoRA and inspect")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    args = parser.parse_args()

    # Load model
    model = ImprovedStudent()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    inspect_model(model, "ImprovedStudent - Base Model")
    forward_test(model)

    if args.lora_targets:
        config = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_rank * 2,
            target_modules=args.lora_targets,
        )
        model = apply_lora(model, config)
        inspect_model(model, f"ImprovedStudent - LoRA ({', '.join(args.lora_targets)})")
        inspect_lora_targets(model)
        forward_test(model)


if __name__ == "__main__":
    main()
