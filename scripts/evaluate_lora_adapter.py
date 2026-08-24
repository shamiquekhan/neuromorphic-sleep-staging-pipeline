#!/usr/bin/env python
"""Evaluate a LoRA-adapted model on the held-out test set.

Usage:
    python scripts/evaluate_lora_adapter.py --adapter artifacts/lora/head_r4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from sleep_staging.adaptation import LoRAConfig, apply_lora, count_lora_parameters, load_adapter
from sleep_staging.config import CHECKPOINT_PATH, StudentConfig
from sleep_staging.data.loader import available_subjects, load_cached_subject
from sleep_staging.evaluation import compute_all_metrics
from sleep_staging.models import ImprovedStudent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--test-subjects", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = StudentConfig()

    # Load base model
    model = ImprovedStudent(config)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)

    # Load adapter config and apply LoRA
    with open(args.adapter / "adapter_config.json") as f:
        meta = json.load(f)
    lora_config = LoRAConfig(
        rank=meta["rank"],
        alpha=meta["alpha"],
        target_modules=meta["target_modules"],
    )
    model = apply_lora(model, lora_config)
    load_adapter(model, args.adapter)
    model.eval()

    log.info("Model: %d total, %d trainable", *count_lora_parameters(model).values()[:2])

    # Prepare test data
    subjects = available_subjects()
    test_ids = args.test_subjects or subjects[-1:]
    log.info("Test subjects: %s", test_ids)

    all_seqs, all_labels = [], []
    for sid in test_ids:
        data = load_cached_subject(sid)
        n = len(data["epochs"])
        for start in range(0, n - 10, 10):
            seq = data["epochs"][start : start + 10]
            label = data["labels"][start + 9]
            all_seqs.append(seq)
            all_labels.append(label)

    x = torch.from_numpy(np.array(all_seqs, dtype=np.float32))
    y_true = np.array(all_labels, dtype=np.int64)

    with torch.inference_mode():
        logits = model(x)
        y_pred = logits[:, -1, :].argmax(dim=-1).numpy()

    metrics = compute_all_metrics(y_true, y_pred)

    print("\n" + "=" * 50)
    print("  LORA ADAPTER EVALUATION")
    print("=" * 50)
    print(f"  Adapter:       {args.adapter.name}")
    print(f"  Rank:          {meta['rank']}")
    print(f"  Target modules: {meta['target_modules']}")
    print(f"  Test subjects: {test_ids}")
    print(f"  Test samples:  {len(y_true)}")
    print()
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Cohen's κ:     {metrics['kappa']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
    print(f"  MGm:           {metrics['mgm']:.4f}")
    print()
    print("  Per-class accuracy:")
    for stage, acc_val in metrics["per_class_accuracy"].items():
        print(f"    {stage:>4s}: {acc_val:.4f}")
    print("=" * 50)

    if args.output:
        result = {
            "adapter": str(args.adapter),
            "lora_config": meta,
            "test_subjects": test_ids,
            "n_samples": len(y_true),
            "metrics": {
                "accuracy": round(metrics["accuracy"], 4),
                "kappa": round(metrics["kappa"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "mgm": round(metrics["mgm"], 4),
                "per_class_accuracy": {
                    k: round(v, 4) for k, v in metrics["per_class_accuracy"].items()
                },
            },
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Results saved to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
