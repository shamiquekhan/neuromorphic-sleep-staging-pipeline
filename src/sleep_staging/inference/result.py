"""Structured prediction result."""

from dataclasses import dataclass, field


@dataclass
class PredictionResult:
    """Result from a single inference call."""

    stage: str
    stage_index: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    target_epoch: int
    model_name: str = "Improved Student"

    def __str__(self) -> str:
        return (
            f"{self.model_name} → {self.stage} "
            f"(confidence={self.confidence:.1%}, latency={self.latency_ms:.1f} ms)"
        )
