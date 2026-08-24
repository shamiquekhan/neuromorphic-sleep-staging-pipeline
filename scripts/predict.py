#!/usr/bin/env python
"""CLI inference tool for the Improved Student model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from sleep_staging.config import CHECKPOINT_PATH, STAGE_NAMES
from sleep_staging.inference import SleepStagePredictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NeuroSleep — run inference on cached epoch sequences",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH,
        help="Path to model checkpoint (default: artifacts/student_improved_best.pt)",
    )
    parser.add_argument(
        "--cache", type=Path, required=True,
        help="Path to cached .npz file",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0,
        help="Starting epoch index in the cache (default: 0)",
    )
    parser.add_argument(
        "--target", type=int, default=9,
        help="Target epoch within the 10-epoch window (0–9, default: 9)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.cache.exists():
        print(f"Error: cache file not found: {args.cache}", file=sys.stderr)
        return 1

    predictor = SleepStagePredictor(
        checkpoint_path=args.checkpoint, device=args.device,
    )

    data = np.load(args.cache)
    epochs = data["epochs"]

    start = args.start_epoch
    end = start + 10
    if end > epochs.shape[0]:
        print(
            f"Error: cannot extract 10 epochs starting at {start} "
            f"(only {epochs.shape[0]} available)", file=sys.stderr,
        )
        return 1

    sequence = epochs[start:end][np.newaxis, ...].astype(np.float32)
    result = predictor.predict(sequence, target_epoch=args.target)

    print("NeuroSleep Prediction")
    print("-" * 30)
    print(f"  Model:       {result.model_name}")
    print(f"  Checkpoint:  {args.checkpoint.name}")
    print(f"  Source:      {args.cache.name}")
    print(f"  Target epoch: {start + result.target_epoch}")
    print(f"  Prediction:  {result.stage}")
    print(f"  Confidence:  {result.confidence:.1%}")
    print(f"  Latency:     {result.latency_ms:.1f} ms")
    print()
    print("  Probabilities:")
    for stage, prob in result.probabilities.items():
        bar = "#" * int(prob * 30)
        print(f"    {stage:>4s}  {bar:<30s} {prob:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
