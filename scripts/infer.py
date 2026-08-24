#!/usr/bin/env python
"""Run inference on a sample sequence using the final checkpoint."""

import argparse
from pathlib import Path

import numpy as np

from src.inference.predict import load_model, run_inference, format_prediction


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser(description="Run sleep-stage inference")
    parser.add_argument("--checkpoint", default="artifacts/student_improved_best.pt")
    parser.add_argument("--input", required=True, help="Path to .npz sequence file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    model = load_model(Path(args.checkpoint), device)

    data = np.load(args.input)
    epochs = data["epochs"]

    if epochs.ndim == 3:
        epochs = np.expand_dims(epochs, 0)

    probs, preds = run_inference(model, epochs, device)
    results = format_prediction(preds, probs)

    print("\nPredicted sleep stages:")
    print("-" * 50)
    for r in results:
        print(f"  Epoch {r['epoch']:2d}: {r['predicted']:5s} (confidence: {r['confidence']:.2%})")

    print(f"\nHypnogram: {' | '.join(r['predicted'] for r in results)}")

    print("\nProbabilities per epoch:")
    for r in results:
        w = r["prob_Wake"]
        n1 = r["prob_N1"]
        n2 = r["prob_N2"]
        n3 = r["prob_N3"]
        rem = r["prob_REM"]
        print(f"  Epoch {r['epoch']:2d}: W={w:.3f} N1={n1:.3f} N2={n2:.3f} N3={n3:.3f} REM={rem:.3f}")


if __name__ == "__main__":
    main()
