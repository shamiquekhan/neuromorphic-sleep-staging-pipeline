"""LoRA (Low-Rank Adaptation) for the Improved Student model.

Implements parameter-efficient adaptation by freezing the base model
and training low-rank adapter matrices on selected linear and conv layers.

Target modules for the Improved Student:
    - ``head``: nn.Linear(64, 5) — classification head
    - ``gab_proj``: nn.Linear(8, 16) — Gabor feature projection
    - ``enc.0.pw``: nn.Conv1d(16, 32, 1) — CNN projection block 0
    - ``enc.1.pw``: nn.Conv1d(32, 32, 1) — CNN projection block 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""

    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["head"])
    bias: str = "none"  # "none", "all", or "lora_only"

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with low-rank adapter matrices.

    The forward pass computes: ``output = original(x) + dropout(x) @ A^T @ B^T * scaling``

    Args:
        original: The frozen linear layer.
        rank: LoRA rank.
        alpha: LoRA alpha (scaling = alpha / rank).
        dropout: Dropout probability on the adapter input.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    @property
    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    @property
    def adapter_state_dict(self) -> dict:
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
        }

    def load_adapter(self, state: dict) -> None:
        self.lora_A.data = state["lora_A"]
        self.lora_B.data = state["lora_B"]


class LoRAConv1d(nn.Module):
    """Wraps an ``nn.Conv1d`` with low-rank adapter matrices.

    For a 1x1 pointwise convolution (kernel_size=1), this is mathematically
    equivalent to applying a low-rank linear transformation at each time position.

    The forward pass computes:
        ``output = original(x) + conv1d(dropout(x), lora_weight, bias=None) * scaling``

    where ``lora_weight`` is constructed from the low-rank factorization.

    Args:
        original: The frozen Conv1d layer.
        rank: LoRA rank.
        alpha: LoRA alpha (scaling = alpha / rank).
        dropout: Dropout probability on the adapter input.
    """

    def __init__(
        self,
        original: nn.Conv1d,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = original.in_channels
        out_channels = original.out_channels

        # For Conv1d, the weight shape is [out_channels, in_channels, kernel_size]
        # We apply LoRA to the first two dimensions: [out_channels, in_channels]
        self.lora_A = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)

        # For kernel_size=1, we can apply LoRA directly
        # For larger kernels, we apply LoRA to the pointwise component
        if self.original.kernel_size[0] == 1:
            # Direct LoRA: reshape x for matrix multiplication
            lora_out = self.lora_dropout(x)
            # lora_out: [B, C_in, T] -> [B, T, C_in]
            lora_out = lora_out.permute(0, 2, 1)
            # Apply low-rank: [B, T, C_in] @ [C_in, rank] @ [rank, C_out] = [B, T, C_out]
            lora_out = lora_out @ self.lora_A.T @ self.lora_B.T * self.scaling
            # [B, T, C_out] -> [B, C_out, T]
            lora_out = lora_out.permute(0, 2, 1)
        else:
            # For non-kernel_size=1 convolutions, apply LoRA as a separate 1x1 conv
            lora_out = self.lora_dropout(x)
            # Reshape weight: [out, in, k] -> [out, in*k] -> low-rank -> [out, in, k]
            weight_flat = self.lora_B @ self.lora_A  # [out, in]
            weight_lora = weight_flat.unsqueeze(2)  # [out, in, 1]
            lora_out = nn.functional.conv1d(
                lora_out, weight_lora,
                padding=self.original.kernel_size[0] // 2,
            ) * self.scaling

        return base_out + lora_out

    @property
    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    @property
    def adapter_state_dict(self) -> dict:
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
        }

    def load_adapter(self, state: dict) -> None:
        self.lora_A.data = state["lora_A"]
        self.lora_B.data = state["lora_B"]


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Apply LoRA adapters to targeted modules in-place.

    Returns the same model with selected ``nn.Linear`` and ``nn.Conv1d`` layers
    replaced by ``LoRALinear``/``LoRAConv1d`` wrappers. Base weights are frozen.

    Args:
        model: The base ImprovedStudent model.
        config: LoRA configuration.

    Returns:
        The modified model (same object, modified in-place).
    """
    target_names = set(config.target_modules)
    replaced = []

    for name, module in model.named_modules():
        if name in target_names:
            parent = _get_parent(model, name)
            attr = name.split(".")[-1]

            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, attr, lora_layer)
                replaced.append(name)
            elif isinstance(module, nn.Conv1d):
                lora_layer = LoRAConv1d(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, attr, lora_layer)
                replaced.append(name)
            else:
                log.warning(
                    "Target module %s is %s (not Linear or Conv1d), skipping",
                    name, type(module).__name__
                )

    if not replaced:
        log.warning("No target modules matched: %s", target_names)
    else:
        log.info("LoRA applied to: %s", replaced)

    # Freeze all non-LoRA parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            for param in module.lora_parameters:
                param.requires_grad = True

    return model


def _get_parent(model: nn.Module, name: str) -> nn.Module:
    """Get the parent module of a named parameter/module."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[part] if part.isdigit() else getattr(parent, part)
    return parent


def count_lora_parameters(model: nn.Module) -> dict:
    """Count trainable (LoRA) and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(trainable / total * 100, 4),
    }


def get_lora_targets(model: nn.Module) -> list[str]:
    """Get list of modules that have LoRA adapters."""
    return [
        name for name, module in model.named_modules()
        if isinstance(module, (LoRALinear, LoRAConv1d))
    ]


def assert_lora_targets(model: nn.Module, expected: list[str]) -> None:
    """Assert that all expected LoRA targets were actually applied.

    Args:
        model: The model with LoRA applied.
        expected: List of expected target module names.

    Raises:
        RuntimeError: If any expected targets are missing.
    """
    targets = get_lora_targets(model)
    missing = [
        target for target in expected
        if not any(target in name for name in targets)
    ]
    if missing:
        raise RuntimeError(
            f"LoRA targets not actually adapted: {missing}. "
            f"Found targets: {targets}"
        )


def save_adapter(model: nn.Module, path: str | Path) -> None:
    """Save LoRA adapter weights and config."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter_state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            adapter_state[name] = module.adapter_state_dict

    torch.save(adapter_state, path / "adapter_model.pt")

    # Find LoRA config from any adapter layer
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            meta = {
                "rank": module.rank,
                "alpha": module.alpha,
                "scaling": module.scaling,
                "target_modules": [
                    n for n, m in model.named_modules()
                    if isinstance(m, (LoRALinear, LoRAConv1d))
                ],
            }
            break
    else:
        meta = {}

    with open(path / "adapter_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Adapter saved to %s", path)


def load_adapter(model: nn.Module, path: str | Path) -> nn.Module:
    """Load LoRA adapter weights onto a model that already has LoRA layers."""
    path = Path(path)
    adapter_state = torch.load(path / "adapter_model.pt", map_location="cpu")

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)) and name in adapter_state:
            module.load_adapter(adapter_state[name])

    log.info("Adapter loaded from %s", path)
    return model
