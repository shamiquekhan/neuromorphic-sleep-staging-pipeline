#!/usr/bin/env python
"""Benchmark inference latency of the Improved Student model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from sleep_staging.config import CHECKPOINT_PATH, StudentConfig
from sleep_staging.inference import SleepStagePredictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    predictor = SleepStagePredictor(
        checkpoint_path=args.checkpoint, device=args.device,
    )

    config = StudentConfig()
    seq = np.random.randn(*config.input_shape).astype(np.float32)

    result = predictor.benchmark(seq, warmup=args.warmup, repeats=args.repeats)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("NeuroSleep Inference Benchmark")
        print("-" * 40)
        print(f"  Model:           {predictor.model_name}")
        print(f"  Parameters:      {predictor.n_parameters:,}")
        print(f"  Device:          {result['device']}")
        print(f"  Warmup runs:     {result['warmup']}")
        print(f"  Timed runs:      {result['repeats']}")
        print(f"  Mean latency:    {result['mean_latency_ms']:.2f} ms")
        print(f"  Total time:      {result['total_time_s']:.4f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
