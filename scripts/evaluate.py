#!/usr/bin/env python
"""Evaluate the final Improved Student checkpoint on the test set."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import SleepSequenceDataset
from src.models.improved_student import ImprovedStudent, count_parameters
from src.evaluation.metrics import compute_all_metrics, plot_confusion_matrix, plot_per_class_f1


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Improved Student")
    parser.add_argument("--checkpoint", default="artifacts/student_improved_best.pt")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.device("cuda").is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    cache_index = pd.read_csv(Path(args.cache_dir) / "cache_index.csv")
    test_ds = SleepSequenceDataset(cache_index, "test", seq_len=10, stride=10)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
    print(f"Test windows: {len(test_ds)}")

    model = ImprovedStudent(n_classes=5).to(device)
    payload = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    else:
        model.load_state_dict(payload)
    model.eval()

    print(f"Parameters: {count_parameters(model):,}")

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
            y_true.extend(y.numpy().reshape(-1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = compute_all_metrics(y_true, y_pred)

    print("\n" + "=" * 60)
    print("FINAL OFFICIAL RESULT — Improved Student")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    official_df = pd.DataFrame([
        ("Model", "Improved Student"),
        ("Test Accuracy", f"{metrics['test_accuracy']:.2%}"),
        ("Cohen's Kappa", f"{metrics['cohen_kappa']:.4f}"),
        ("Macro F1", f"{metrics['macro_f1']:.4f}"),
        ("Weighted F1", f"{metrics['weighted_f1']:.4f}"),
        ("Macro Geometric Mean", f"{metrics['macro_gmean']:.4f}"),
        ("Parameters", f"{count_parameters(model):,}"),
    ], columns=["Metric", "Final value"])
    official_df.to_csv(results_dir / "final_result.csv", index=False)

    per_class_df = pd.DataFrame({
        "Sleep stage": STAGE_NAMES,
        "F1": [metrics[f"f1_{name.lower()}"] for name in STAGE_NAMES],
    })
    per_class_df.to_csv(results_dir / "per_class_f1.csv", index=False)

    cm = plot_confusion_matrix(y_true, y_pred, save_path=results_dir / "confusion_matrix.png")
    plot_per_class_f1(metrics, save_path=results_dir / "per_class_f1.png")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
