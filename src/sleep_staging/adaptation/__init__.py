"""Parameter-efficient adaptation (LoRA) for sleep-stage models."""

from .lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    load_adapter,
    save_adapter,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "count_lora_parameters",
    "load_adapter",
    "save_adapter",
]
