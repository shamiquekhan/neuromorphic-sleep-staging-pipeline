"""Device utilities for NeuroSleep."""

import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed device information."""
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return info


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device with non-blocking transfer."""
    return tensor.to(device, non_blocking=True)
