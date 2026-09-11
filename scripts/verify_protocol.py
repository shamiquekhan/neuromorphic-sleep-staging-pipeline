#!/usr/bin/env python3
"""
Verify Adaptation Protocol — Subject-Disjointness + Config Consistency
======================================================================

Checks, in order:
  1. Folds manifest integrity (92 subjects, 10 folds, disjoint test sets)
  2. Base-checkpoint training subjects vs evaluation folds:
       base_train ∩ eval_test       = ∅
       base_train ∩ eval_validation  = ∅
  3. Adaptation config hierarchy consistency (epochs/batch/lr/wd agree
     with the canonical benchmark config)

The known 15-subject-era checkpoint
(artifacts/final/student_full_finetuned.pt) FAILS check 2 — that is the
documented contamination (see docs/adaptation.md). This script exists so
that failure is machine-detectable, not just documented.

Usage:
    python scripts/verify_protocol.py                 # full verification
    python scripts/verify_protocol.py --quiet         # exit code only
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FOLDS_PATH = REPO / "data" / "manifests" / "canonical_subject_folds_92subj.json"
BENCH_CFG = REPO / "configs" / "benchmark_92_subject.yaml"
ADAPT_CFG = REPO / "configs" / "adaptation_92_subject.yaml"
LEGACY_CKPT = REPO / "artifacts" / "final" / "student_full_finetuned.pt"

# Training subjects of the legacy 15-subject-era checkpoint, recovered from
# the historical fold manifest (see docs/archive/development_15_subject.md).
LEGACY_BASE_TRAIN_SUBJECTS = {
    "SC4001", "SC4002", "SC4011", "SC4012", "SC4022",
    "SC4031", "SC4032", "SC4041", "SC4042", "SC4051",
    "SC4052", "SC4061", "SC4062", "SC4071", "SC4072",
}


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_folds_manifest() -> dict:
    """Check 1: folds manifest integrity."""
    with open(FOLDS_PATH) as f:
        manifest = json.load(f)

    folds = manifest["folds"]
    subjects = set(manifest["subjects"])
    problems = []

    if manifest.get("n_subjects") != 92:
        problems.append(f"n_subjects = {manifest.get('n_subjects')}, expected 92")
    if len(folds) != 10:
        problems.append(f"{len(folds)} folds, expected 10")
    if len(subjects) != 92:
        problems.append(f"subject list has {len(subjects)} unique subjects")

    seen_test = set()
    for name, fold in folds.items():
        test = set(fold["test"])
        overlap = test & seen_test
        if overlap:
            problems.append(f"{name}: test subjects appear in multiple folds: {sorted(overlap)}")
        seen_test |= test

    # Every subject must be in exactly one test fold or in validation only
    if problems:
        for p in problems:
            fail(f"folds manifest: {p}")
        return {}
    ok(f"folds manifest: 10 folds, 92 subjects, test sets disjoint "
       f"({len(seen_test)} subjects appear in some test fold)")
    return manifest


def check_subject_disjointness(manifest: dict, base_subjects: set | None = None) -> bool:
    """Check 2: base checkpoint training subjects vs evaluation folds."""
    folds = manifest["folds"]
    eval_test = set()
    eval_val = set()
    for fold in folds.values():
        eval_test.update(fold["test"])
        eval_val.update(fold["validation"])

    if base_subjects is None:
        label = "legacy base checkpoint (15-subject era)"
        train_subjects = LEGACY_BASE_TRAIN_SUBJECTS
    else:
        label = "provided --base-subjects file"
        train_subjects = base_subjects

    contaminated = train_subjects & (eval_test | eval_val)
    overlap_test = train_subjects & eval_test
    overlap_val = train_subjects & eval_val

    if contaminated:
        fail(
            f"{label} is CONTAMINATED: "
            f"{len(overlap_test)}/{len(train_subjects)} training subjects appear in eval TEST folds "
            f"({sorted(overlap_test)}), "
            f"{len(overlap_val)}/{len(train_subjects)} in eval VALIDATION folds ({sorted(overlap_val)})"
        )
        print("        -> any frozen/LoRA/full-FT result produced from this "
              "checkpoint is quarantined (see docs/adaptation.md)")
        return False

    ok(f"{label}: {len(train_subjects)} training subjects are disjoint "
       "from eval test+validation")
    return True


def check_configs() -> bool:
    """Check 3: config hierarchy consistency (canonical hyperparameters)."""
    try:
        import yaml
    except ImportError:
        fail("PyYAML not installed — cannot verify configs")
        return False

    problems = []
    for cfg_path in (BENCH_CFG, ADAPT_CFG):
        if not cfg_path.exists():
            problems.append(f"missing config: {cfg_path.name}")
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            problems.append(f"{cfg_path.name}: parse error: {e}")
            continue

        tr = cfg.get("training", {})
        expected = {"epochs": 20, "batch_size": 32,
                    "learning_rate": 3.0e-4, "weight_decay": 1.0e-4}
        for key, want in expected.items():
            got = tr.get(key)
            if got is None:
                continue
            if float(got) != float(want):
                problems.append(
                    f"{cfg_path.name}: training.{key} = {got}, canonical is {want}"
                )

    if ADAPT_CFG.exists():
        with open(ADAPT_CFG) as f:
            cfg = yaml.safe_load(f)
        forbidden = cfg.get("base_checkpoint", {}).get("forbidden", "")
        if forbidden and "student_full_finetuned.pt" in str(forbidden):
            pass  # forbidden path correctly declared
        else:
            problems.append("adaptation_92_subject.yaml: forbidden base checkpoint not declared")

    if problems:
        for p in problems:
            fail(f"config consistency: {p}")
        return False
    ok("config hierarchy consistent with canonical benchmark hyperparameters "
       "(epochs=20, batch=32, lr=3e-4, wd=1e-4)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify adaptation protocol")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--base-subjects", type=str, default=None,
                        help="Path to a JSON file/text list of subjects the base "
                             "checkpoint was trained on. When omitted, the legacy "
                             "15-subject-era training set is checked (expected to FAIL).")
    args = parser.parse_args()

    print("=" * 70)
    print("  PROTOCOL VERIFICATION")
    print("=" * 70)

    base_subjects = None
    if args.base_subjects:
        p = Path(args.base_subjects)
        if not p.exists():
            fail(f"base-subjects file not found: {p}")
            return 1
        if p.suffix == ".json":
            data = json.load(open(p))
            if isinstance(data, dict):
                for key in ("subjects", "train_subjects", "base_train_subjects"):
                    if key in data:
                        data = data[key]
                        break
            base_subjects = set(data)
        else:
            base_subjects = {ln.strip() for ln in p.read_text().splitlines()
                             if ln.strip() and not ln.startswith("#")}
        if not base_subjects:
            fail("base-subjects file is empty")
            return 1
        print(f"  checking {len(base_subjects)} provided base-training subjects")

    checks = []
    manifest = check_folds_manifest()
    if manifest:
        checks.append(check_subject_disjointness(manifest, base_subjects))
    else:
        checks.append(False)
    checks.append(check_configs())

    if LEGACY_CKPT.exists():
        print(f"\n  legacy checkpoint sha256: {sha256(LEGACY_CKPT)[:16]}… "
              f"({LEGACY_CKPT.name})")

    print("=" * 70)
    if all(c for c in checks if c is not None) and checks and all(checks):
        print("  RESULT: PASS (protocol safe to run)")
        return 0
    n_fail = sum(1 for c in checks if not c)
    print(f"  RESULT: FAIL ({n_fail} check group(s) failed)")
    print("  See docs/adaptation.md for remediation steps.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
