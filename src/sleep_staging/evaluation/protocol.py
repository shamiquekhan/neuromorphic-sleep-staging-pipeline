"""Causal unique-epoch evaluation protocol (P0.6).

The legacy benchmark evaluated ``[B, T, 5]`` logits with all-position
supervision and ``stride=5`` windows, so a typical epoch appeared in two
overlapping windows and was scored twice. Metrics were therefore
computed over window-position pairs, not over scored epochs.

This module implements the protocol the audit recommends:

    Input:   10 consecutive 30 s epochs = 5 minutes of PSG context
    Output:  prediction for the LAST (current) epoch
    Stride:  1 epoch
    Result:  exactly one prediction per scored epoch

Every prediction carries ``(subject_id, epoch_index)`` provenance, so
result artifacts map 1:1 onto unique scored epochs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.sequence_dataset import CausalEvalDataset
from .metrics import compute_all_metrics

__all__ = ["evaluate_causal", "collect_causal_predictions"]


@torch.no_grad()
def collect_causal_predictions(
    model: torch.nn.Module,
    dataset: CausalEvalDataset,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Run the model over a CausalEvalDataset, one prediction per epoch.

    Returns dict with keys:
        subject_id : (N,) object array of subject ids
        epoch_index: (N,) int64, position in the subject's raw epoch grid
        y_true     : (N,) int64
        y_pred     : (N,) int64
        probs      : (N, n_classes) float32
    """
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda" or str(device) == "cuda"),
    )

    subj_batch: list[np.ndarray] = []
    ep_batch: list[np.ndarray] = []
    true_batch: list[np.ndarray] = []
    pred_batch: list[np.ndarray] = []
    prob_batch: list[np.ndarray] = []

    for x, y_last, subject_id, epoch_index in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)              # [B, T, C]
        last_logits = logits[:, -1, :]  # causal: score only epoch T
        probs = torch.softmax(last_logits, dim=-1)
        preds = last_logits.argmax(dim=-1)

        subj_batch.append(np.asarray(subject_id, dtype=object))
        ep_batch.append(epoch_index.numpy().astype(np.int64))
        true_batch.append(y_last.numpy().astype(np.int64))
        pred_batch.append(preds.cpu().numpy().astype(np.int64))
        prob_batch.append(probs.cpu().numpy().astype(np.float32))

    out = {
        "subject_id": np.concatenate(subj_batch),
        "epoch_index": np.concatenate(ep_batch),
        "y_true": np.concatenate(true_batch),
        "y_pred": np.concatenate(pred_batch),
        "probs": np.concatenate(prob_batch),
    }

    _assert_unique_epochs(out)
    return out


def evaluate_causal(
    model: torch.nn.Module,
    dataset: CausalEvalDataset,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """Unique-epoch causal evaluation with full metric suite.

    Guarantees (enforced by ``collect_causal_predictions``):
      * no subject appears twice with different identities,
      * every (subject_id, epoch_index) pair is unique.
    """
    preds = collect_causal_predictions(
        model, dataset, device, batch_size, num_workers,
    )

    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    metrics = compute_all_metrics(y_true, y_pred)
    metrics["n_unique_epochs"] = int(len(y_true))
    metrics["n_subjects"] = int(len(np.unique(preds["subject_id"])))
    metrics["protocol"] = "causal_unique_epoch"
    metrics["predictions"] = preds
    return metrics


def _assert_unique_epochs(preds: dict[str, np.ndarray]) -> None:
    n = len(preds["y_true"])
    keys = np.array(
        list(zip(preds["subject_id"], preds["epoch_index"])),
        dtype=object,
    )
    seen = set(map(tuple, keys))
    if len(seen) != n:
        dup = n - len(seen)
        raise RuntimeError(
            f"causal protocol violated: {dup} duplicate (subject, epoch) "
            "predictions — every scored epoch must receive exactly one "
            "prediction"
        )
