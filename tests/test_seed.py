"""Tests for deterministic seeding (P1 #12)."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.training.seed import seed_everything, worker_init_fn


def test_seeds_all_rngs():
    seed_everything(123)
    a = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )
    seed_everything(123)
    b = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )
    assert a == b


def test_different_seeds_differ():
    seed_everything(1)
    a = torch.rand(3)
    seed_everything(2)
    b = torch.rand(3)
    assert not torch.allclose(a, b)


def test_cuda_seeded_when_available():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    seed_everything(7)
    x1 = torch.rand(4, device="cuda")
    seed_everything(7)
    x2 = torch.rand(4, device="cuda")
    torch.testing.assert_close(x1, x2)


def test_deterministic_flag_controls_cudnn():
    seed_everything(5, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    seed_everything(5, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    # restore for other tests
    seed_everything(5)


def test_worker_init_fn_seeds_per_worker():
    seed_everything(99)
    torch.manual_seed(99)
    worker_init_fn(0)
    a = np.random.rand()
    seed_everything(99)
    torch.manual_seed(99)
    worker_init_fn(3)
    b = np.random.rand()
    # Different workers must get different streams
    assert a != b
