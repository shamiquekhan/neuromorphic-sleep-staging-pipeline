"""Parameter-efficient adaptation (LoRA) for sleep-stage models."""

from .lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    freeze_norm_layers,
    load_adapter,
    save_adapter,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "count_lora_parameters",
    "freeze_norm_layers",
    "load_adapter",
    "save_adapter",
]
