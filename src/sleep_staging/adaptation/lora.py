"""LoRA (Low-Rank Adaptation) for the Improved Student model.

Implements parameter-efficient adaptation by freezing the base model
and training low-rank adapter matrices on selected linear layers.

Target modules for the Improved Student:
    - ``head``: nn.Linear(64, 5) — classification head
    - ``gab_proj``: nn.Linear(8, 16) — Gabor feature projection
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


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Apply LoRA adapters to targeted modules in-place.

    Returns the same model with selected ``nn.Linear`` layers replaced
    by ``LoRALinear`` wrappers. Base weights are frozen.

    Args:
        model: The base ImprovedStudent model.
        config: LoRA configuration.

    Returns:
        The modified model (same object, modified in-place).
    """
    target_names = set(config.target_modules)
    replaced = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in target_names:
            parent = _get_parent(model, name)
            attr = name.split(".")[-1]
            lora_layer = LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(parent, attr, lora_layer)
            replaced.append(name)

    if not replaced:
        log.warning("No target modules matched: %s", target_names)
    else:
        log.info("LoRA applied to: %s", replaced)

    # Freeze all non-LoRA parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
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


def save_adapter(model: nn.Module, path: str | Path) -> None:
    """Save LoRA adapter weights and config."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapter_state[name] = module.adapter_state_dict

    torch.save(adapter_state, path / "adapter_model.pt")

    # Find LoRA config from any adapter layer
    for module in model.modules():
        if isinstance(module, LoRALinear):
            meta = {
                "rank": module.rank,
                "alpha": module.alpha,
                "scaling": module.scaling,
                "target_modules": [
                    n for n, m in model.named_modules() if isinstance(m, LoRALinear)
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
        if isinstance(module, LoRALinear) and name in adapter_state:
            module.load_adapter(adapter_state[name])

    log.info("Adapter loaded from %s", path)
    return model
