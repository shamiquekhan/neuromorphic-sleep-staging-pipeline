"""Deterministic seeding for reproducible experiments (P1 #12).

The pre-fix benchmark seeded only ``torch.manual_seed`` and
``np.random.seed`` — leaving Python's ``random`` module, CUDA RNGs, and
cuDNN nondeterminism uncontrolled. ``seed_everything`` controls all of
them and is a no-op-compatible drop-in for benchmark/adaptation runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger(__name__)

__all__ = ["seed_everything", "worker_init_fn", "REPRO_ENV_VARS"]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and (optionally) force deterministic kernels.

    Controls: Python ``random``, NumPy, torch CPU, torch CUDA (all
    devices), and cuDNN benchmark/deterministic behavior. Also sets
    ``CUBLAS_WORKSPACE_CONFIG`` so cuBLAS runs deterministically where
    the platform supports it.

    Note: full determinism can change throughput on CUDA. For exact
    cross-run reproduction of *reported* numbers, keep
    ``deterministic=True``; set False only for exploratory runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Restore PyTorch defaults (autotune on, nondeterministic algos
        # allowed) so the choice is symmetric.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init: derive a per-worker seed from torch's
    base seed so shuffling augmentation order is reproducible."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
