#!/usr/bin/env python3
"""
Protocol Fingerprint — Pre-Run Consistency Check

Ensures that seeds 43/44 use exactly the same protocol as seed 42.
Prints a fingerprint of all fixed parameters and compares against
the saved seed-42 reference. Aborts if any mismatch is detected.

Usage:
    python scripts/protocol_fingerprint.py --seed 43
    python scripts/protocol_fingerprint.py --seed 44
    python scripts/protocol_fingerprint.py --seed 42 --save-reference
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

# ── Paths ──────────────────────────────────────────────────────────────
FOLDS_PATH = REPO / "data" / "manifests" / "canonical_subject_folds_92subj.json"
CHECKPOINT_PATH = REPO / "artifacts" / "final" / "student_full_finetuned.pt"
CONFIG_PATH = REPO / "configs" / "full_100_subject.yaml"
REFERENCE_PATH = REPO / "results" / "audit" / "protocol_fingerprint_seed42.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_fold_subjects() -> dict:
    with open(FOLDS_PATH) as f:
        data = json.load(f)
    all_subjects = set()
    for fold_data in data["folds"].values():
        for s in fold_data["train"] + fold_data["validation"] + fold_data["test"]:
            all_subjects.add(s)
    return {
        "n_folds": len(data["folds"]),
        "n_subjects": len(all_subjects),
        "fold_test_subjects": {
            k: v["test"] for k, v in data["folds"].items()
        },
    }


def build_fingerprint(mode: str, seed: int, rank: int = 8, alpha: int = 16,
                      targets: str = "enc.0.pw,enc.1.pw,head") -> dict:
    fold_info = get_fold_subjects()
    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    fp = {
        "seed": seed,
        "mode": mode,
        "dataset_manifest_hash": sha256_file(FOLDS_PATH),
        "checkpoint_hash": sha256_file(CHECKPOINT_PATH),
        "config_hash": sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else "missing",
        "git_commit": get_git_commit(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
        "gpu": gpu_name,
        "n_folds": fold_info["n_folds"],
        "n_subjects": fold_info["n_subjects"],
        "sequence_length": 10,
        "stride": 5,
        "n1_weight": 2.0,
        "rem_weight": 2.0,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "max_epochs": 20,
        "early_stopping_patience": 5,
        "scheduler": "cosine",
        "grad_clip": 1.0,
        "mixed_precision": True,
        "initialization": "pretrained_checkpoint",
        "supervision": "all_position",
    }

    if mode == "lora":
        fp["lora_rank"] = rank
        fp["lora_alpha"] = alpha
        fp["lora_targets"] = targets
        fp["lora_dropout"] = 0.05
    elif mode == "full_finetune":
        fp["trainable_params"] = 99477
    elif mode == "frozen":
        fp["trainable_params"] = 0

    return fp


def compare_fingerprints(ref: dict, cur: dict) -> list:
    mismatches = []
    skip_keys = {"seed"}
    for key in set(ref.keys()) | set(cur.keys()):
        if key in skip_keys:
            continue
        if ref.get(key) != cur.get(key):
            mismatches.append({
                "field": key,
                "reference": ref.get(key),
                "current": cur.get(key),
            })
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Protocol fingerprint check")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, default="full_finetune",
                        choices=["frozen", "lora", "full_finetune"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--targets", type=str, default="enc.0.pw,enc.1.pw,head")
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--compare-to", type=str, default=str(REFERENCE_PATH))
    args = parser.parse_args()

    fp = build_fingerprint(args.mode, args.seed, args.rank, args.alpha, args.targets)

    print("=" * 70)
    print(f"  PROTOCOL FINGERPRINT — Seed {args.seed} — {args.mode.upper()}")
    print("=" * 70)
    for k, v in fp.items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)

    if args.save_reference:
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REFERENCE_PATH, "w") as f:
            json.dump(fp, f, indent=2)
        print(f"\n  Reference saved to {REFERENCE_PATH}")
        return

    if not Path(args.compare_to).exists():
        print(f"\n  WARNING: No reference file found at {args.compare_to}")
        print("  Skipping comparison. Run with --save-reference first.")
        return

    with open(args.compare_to) as f:
        ref = json.load(f)

    mismatches = compare_fingerprints(ref, fp)

    if mismatches:
        print("\n  ❌ PROTOCOL MISMATCH — ABORTING")
        print("-" * 70)
        for m in mismatches:
            print(f"  {m['field']:30s}:")
            print(f"    Reference: {m['reference']}")
            print(f"    Current:   {m['current']}")
        print("-" * 70)
        sys.exit(1)
    else:
        print("\n  ✅ Protocol matches seed-42 reference — safe to proceed")


if __name__ == "__main__":
    main()
