#!/usr/bin/env python3
"""
Regenerate 10-fold subject-level CV splits for 92 included subjects.
Excludes 8 wake-only subjects from canonical_subject_folds.json.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS_IN = ROOT / "data" / "manifests" / "canonical_subject_folds.json"
QUALITY_MANIFEST = ROOT / "results" / "dataset_audit_100subj" / "subject_quality_manifest.csv"
FOLDS_OUT = ROOT / "data" / "manifests" / "canonical_subject_folds_92subj.json"

EXCLUDED = {
    "SC4082", "SC4111", "SC4142", "SC4162",
    "SC4172", "SC4192", "SC4232", "SC4301",
}

FIXED_VALIDATION = [
    "SC4212", "SC4211", "SC4712", "SC4042",
    "SC4441", "SC4252", "SC4002", "SC4512", "SC4072",
]


def main():
    # load original folds
    with open(FOLDS_IN) as f:
        orig = json.load(f)

    # collect all subjects from original folds (excluding wake-only)
    all_subjects = set()
    for fold_key, fold in orig["folds"].items():
        for subj in fold["test"]:
            if subj not in EXCLUDED:
                all_subjects.add(subj)
        for subj in fold["validation"]:
            if subj not in EXCLUDED:
                all_subjects.add(subj)
        for subj in fold["train"]:
            if subj not in EXCLUDED:
                all_subjects.add(subj)

    all_subjects = sorted(all_subjects)
    print(f"Total included subjects: {len(all_subjects)}")

    # remove fixed validation subjects from pool
    train_pool = [s for s in all_subjects if s not in FIXED_VALIDATION]
    print(f"Train pool (excl. validation): {len(train_pool)}")

    # verify validation subjects are in included set
    for v in FIXED_VALIDATION:
        assert v in all_subjects, f"Validation subject {v} not in included set!"

    # shuffle train pool deterministically
    random.seed(42)
    random.shuffle(train_pool)

    # create 10 folds
    n_folds = 10
    fold_size = len(train_pool) // n_folds
    remainder = len(train_pool) % n_folds

    folds = {}
    idx = 0
    for fold_i in range(n_folds):
        # distribute remainder across first folds
        extra = 1 if fold_i < remainder else 0
        size = fold_size + extra
        test_subjects = sorted(train_pool[idx : idx + size])
        idx += size

        # train = all included minus test minus validation
        train_subjects = sorted(
            set(all_subjects) - set(test_subjects) - set(FIXED_VALIDATION)
        )

        folds[f"fold_{fold_i + 1}"] = {
            "test": test_subjects,
            "validation": FIXED_VALIDATION,
            "train": train_subjects,
        }

        print(f"  Fold {fold_i+1:2d}: test={len(test_subjects):2d}, "
              f"val={len(FIXED_VALIDATION)}, train={len(train_subjects)}")

    # build output
    output = {
        "version": "2.1.0",
        "description": "10-fold subject-level CV for 92 Sleep-EDF Expanded subjects "
                       "(8 wake-only excluded). Validation set fixed across folds.",
        "dataset": "sleep_edf_expanded",
        "n_subjects": len(all_subjects),
        "n_folds": n_folds,
        "split_method": "subject_level",
        "excluded_subjects": sorted(EXCLUDED),
        "subjects": all_subjects,
        "folds": folds,
    }

    with open(FOLDS_OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {FOLDS_OUT}")

    # verify no overlap
    for fold_key, fold in folds.items():
        test_set = set(fold["test"])
        val_set = set(fold["validation"])
        train_set = set(fold["train"])
        assert test_set.isdisjoint(val_set), f"{fold_key}: test/val overlap"
        assert test_set.isdisjoint(train_set), f"{fold_key}: test/train overlap"
        assert val_set.isdisjoint(train_set), f"{fold_key}: val/train overlap"
        assert test_set | val_set | train_set == set(all_subjects), \
            f"{fold_key}: missing subjects"

    print("All folds validated: no overlaps, all subjects covered.")


if __name__ == "__main__":
    main()
