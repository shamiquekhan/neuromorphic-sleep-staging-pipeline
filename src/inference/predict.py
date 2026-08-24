"""Inference utilities for the final Improved Student model."""

from pathlib import Path

import numpy as np
import torch

from src.models.improved_student import ImprovedStudent


def load_model(checkpoint_path: Path, device: torch.device = torch.device("cpu")):
    """Load the final Improved Student from a checkpoint file.

    Handles both raw state_dict files and wrapped checkpoint dicts.
    """
    model = ImprovedStudent(n_classes=5).to(device)

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and any(isinstance(v, torch.Tensor) for v in payload.values()):
        state_dict = payload
    else:
        state_dict = payload

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_inference(model, epochs: np.ndarray, device: torch.device = torch.device("cpu")):
    """Run inference on a sequence of preprocessed epochs.

    Args:
        model: Loaded ImprovedStudent model.
        epochs: numpy array of shape [seq_len, n_channels, n_samples] or [1, seq_len, n_channels, n_samples].
        device: torch device.

    Returns:
        probs: [seq_len, n_classes] probabilities.
        preds: [seq_len] predicted class indices.
    """
    if epochs.ndim == 3:
        epochs = np.expand_dims(epochs, 0)

    x = torch.from_numpy(epochs).float().to(device)
    with torch.no_grad():
        logits = model(x)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    preds = probs.argmax(axis=-1)
    return probs, preds


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def format_prediction(preds, probs, stage_names=None):
    """Format predictions for display."""
    if stage_names is None:
        stage_names = STAGE_NAMES

    results = []
    for i, (p, prob) in enumerate(zip(preds, probs)):
        results.append({
            "epoch": i + 1,
            "predicted": stage_names[p],
            "confidence": float(prob[p]),
            **{f"prob_{stage_names[j]}": float(prob[j]) for j in range(len(stage_names))},
        })
    return results
