"""Inference engine — single source of truth for model loading and prediction."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from ..config import CHECKPOINT_PATH, STAGE_NAMES, StudentConfig
from ..models import ImprovedStudent, count_parameters
from ..adaptation import LoRAConfig, apply_lora, count_lora_parameters, load_adapter
from .result import PredictionResult

log = logging.getLogger(__name__)


class SleepStagePredictor:
    """Deterministic inference engine for the Improved Student model.

    Args:
        checkpoint_path: Path to ``.pt`` checkpoint.
        device: ``"cpu"`` or ``"cuda"``.
        config: Model configuration.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        config: StudentConfig | None = None,
        adapter_path: str | Path | None = None,
        lora_config: LoRAConfig | None = None,
    ):
        self.device = torch.device(device)
        self.config = config or StudentConfig()

        self.model = ImprovedStudent(self.config)
        self._load_checkpoint(
            checkpoint_path or CHECKPOINT_PATH,
        )

        self._adapter_name = None
        if adapter_path is not None:
            cfg = lora_config or LoRAConfig(target_modules=["head"])
            self.model = apply_lora(self.model, cfg)
            load_adapter(self.model, adapter_path)
            self._adapter_name = Path(adapter_path).name
            params = count_lora_parameters(self.model)
            log.info("LoRA adapter loaded: %s (%d trainable)", adapter_path, params["trainable"])

        self.model.to(self.device)
        self.model.eval()

        self._n_params = count_parameters(self.model)
        log.info(
            "Model loaded: %s (%s params) on %s",
            self.model_name, f"{self._n_params:,}", self.device,
        )

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle both raw state_dict and wrapped checkpoint dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self._metadata = {
                "epoch": checkpoint.get("epoch"),
                "val_kappa": checkpoint.get("val_kappa"),
                "seed": checkpoint.get("seed"),
                "seq_len": checkpoint.get("seq_len", self.config.seq_len),
            }
        else:
            # Raw state_dict (keys are parameter names directly)
            state_dict = checkpoint
            self._metadata = {}

        self.model.load_state_dict(state_dict, strict=True)
        log.info("Checkpoint loaded: %s", path)

    @property
    def model_name(self) -> str:
        return "Improved Student"

    @property
    def n_parameters(self) -> int:
        return self._n_params

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def model_info(self) -> dict:
        info = {
            "name": self.model_name,
            "parameters": self._n_params,
            "device": str(self.device),
            "input_shape": list(self.config.input_shape),
            "n_classes": self.config.n_classes,
            "seq_len": self.config.seq_len,
            "mode": "eval",
            "adapter": self._adapter_name,
            **self._metadata,
        }
        return info

    @staticmethod
    def validate_input(x: np.ndarray, config: StudentConfig | None = None) -> None:
        """Raise ``ValueError`` if *x* violates the model contract."""
        cfg = config or StudentConfig()
        if x.ndim != 4:
            raise ValueError(f"Expected 4-D tensor [B, T, C, S], got ndim={x.ndim}")
        if x.shape[1] != cfg.seq_len:
            raise ValueError(f"Expected {cfg.seq_len} epochs, got {x.shape[1]}")
        if x.shape[2] != cfg.n_channels:
            raise ValueError(f"Expected {cfg.n_channels} channels, got {x.shape[2]}")
        if x.shape[3] != cfg.samples_per_epoch:
            raise ValueError(
                f"Expected {cfg.samples_per_epoch} samples per epoch, got {x.shape[3]}"
            )

    def predict(
        self,
        sequence: np.ndarray,
        target_epoch: int = 9,
    ) -> PredictionResult:
        """Run inference and return a structured result for *target_epoch*."""
        self.validate_input(sequence, self.config)

        x = torch.from_numpy(sequence).float().to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(x)
        latency_ms = (time.perf_counter() - start) * 1000

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        epoch_probs = probs[target_epoch]
        pred_idx = int(epoch_probs.argmax())
        confidence = float(epoch_probs[pred_idx])

        prob_dict = {
            STAGE_NAMES[i]: float(epoch_probs[i])
            for i in range(self.config.n_classes)
        }

        return PredictionResult(
            stage=STAGE_NAMES[pred_idx],
            stage_index=pred_idx,
            confidence=confidence,
            probabilities=prob_dict,
            latency_ms=latency_ms,
            target_epoch=target_epoch,
            model_name=self.model_name,
        )

    def predict_sequence(self, sequence: np.ndarray) -> list[PredictionResult]:
        """Predict all 10 temporal positions."""
        self.validate_input(sequence, self.config)

        x = torch.from_numpy(sequence).float().to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(x)
        total_ms = (time.perf_counter() - start) * 1000

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        results = []
        for t_idx in range(self.config.seq_len):
            epoch_probs = probs[t_idx]
            pred_idx = int(epoch_probs.argmax())
            results.append(PredictionResult(
                stage=STAGE_NAMES[pred_idx],
                stage_index=pred_idx,
                confidence=float(epoch_probs[pred_idx]),
                probabilities={
                    STAGE_NAMES[i]: float(epoch_probs[i])
                    for i in range(self.config.n_classes)
                },
                latency_ms=total_ms / self.config.seq_len,
                target_epoch=t_idx,
                model_name=self.model_name,
            ))
        return results

    def benchmark(
        self, sequence: np.ndarray, warmup: int = 5, repeats: int = 20,
    ) -> dict:
        """Measure inference latency on this machine."""
        self.validate_input(sequence, self.config)
        x = torch.from_numpy(sequence).float().to(self.device)

        with torch.inference_mode():
            for _ in range(warmup):
                _ = self.model(x)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                _ = self.model(x)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        total = time.perf_counter() - start
        latency_ms = (total / repeats) * 1000

        return {
            "warmup": warmup,
            "repeats": repeats,
            "total_time_s": round(total, 4),
            "mean_latency_ms": round(latency_ms, 2),
            "device": str(self.device),
        }
