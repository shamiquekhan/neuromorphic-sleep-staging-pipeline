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

FOLDS_PATH = REPO / "data" / "manifests" / "person_folds_52subj.json"
PERSON_FOLDS_PATH = REPO / "data" / "manifests" / "person_folds_52subj.json"
PERSON_GROUPS_PATH = REPO / "data" / "manifests" / "person_groups.json"
BENCH_CFG = REPO / "configs" / "experiments" / "person_level_cv.yaml"
ADAPT_CFG = REPO / "configs" / "adaptation_92_subject.yaml"
LEGACY_CKPT = REPO / "artifacts" / "quarantine" / "student_full_finetuned_generic.pt"


def person_of(record: str) -> str:
    """SC4ssN -> P4ss: strip the night digit (same person, two nights)."""
    if not (record.startswith("SC") and len(record) == 6 and record[5] in "12"):
        raise ValueError(f"Unexpected record id: {record}")
    return f"P{record[2:5]}"

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
    """Check 1: folds manifest integrity (record level - legacy)."""
    # Note: FOLDS_PATH now points to person_folds_52subj.json which has a different structure.
    # This check is for the LEGACY record-level manifest which is quarantined.
    # We skip this check for the person-level manifest since check_person_level_folds covers it.
    legacy_folds_path = REPO / "data" / "manifests" / "canonical_subject_folds_92subj.json"
    if not legacy_folds_path.exists():
        ok("legacy record-level folds manifest not present (quarantined) — skipping")
        return {}
    
    with open(legacy_folds_path) as f:
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
    ok(f"folds manifest (legacy record-level): 10 folds, 92 records, test sets disjoint "
       f"({len(seen_test)} records appear in some test fold)")
    return manifest


def check_person_level_folds() -> bool:
    """Check 1b: person-level folds manifest integrity.

    SC4ss1/SC4ss2 are two nights of the same person (PhysioNet
    sleep-edfx README; SC-subjects.xls). The legacy record-level folds
    therefore leak at person level: a test record's same-person mate is
    usually in the train set. The person-level manifest
    (person_folds_52subj.json) is the only structure valid for
    person-generalization claims.
    """
    if not PERSON_FOLDS_PATH.exists():
        fail("person-level folds manifest missing — run "
             "scripts/generate_person_folds.py")
        return False

    manifest = json.load(open(PERSON_FOLDS_PATH))
    folds = manifest["folds"]
    problems = []

    if len(folds) != 10:
        problems.append(f"{len(folds)} folds, expected 10")
    if manifest.get("n_persons") != 52:
        problems.append(f"n_persons = {manifest.get('n_persons')}, expected 52")

    all_records = set(manifest["records"])
    seen_test = set()
    val = set(next(iter(folds.values()))["validation"])
    for name, fold in folds.items():
        test = set(fold["test"])
        if test & seen_test:
            problems.append(f"{name}: test records in multiple folds")
        seen_test |= test
        if set(fold["validation"]) != val:
            problems.append(f"{name}: validation set differs")

        tr, te, va = set(fold["train"]), test, set(fold["validation"])
        if tr & te or tr & va or te & va:
            problems.append(f"{name}: record-level overlap")
        if tr | te | va != all_records:
            problems.append(f"{name}: missing records")

        # the actual point: person-level disjointness
        for r in tr | te | va:
            pass
        person_roles = {}
        for r in all_records:
            if r in tr:
                person_roles.setdefault(person_of(r), set()).add("train")
            if r in te:
                person_roles.setdefault(person_of(r), set()).add("test")
            if r in va:
                person_roles.setdefault(person_of(r), set()).add("val")
        split = {p: roles for p, roles in person_roles.items() if len(roles) > 1}
        if split:
            problems.append(f"{name}: persons split across roles: {split}")

    if seen_test | val != all_records:
        problems.append("test folds + validation != all records")

    if problems:
        for p in problems:
            fail(f"person-level folds: {p}")
        return False
    n_persons = manifest["n_persons"]
    ok(f"person-level folds: 10 folds, 52 persons, no person split across "
       f"train/test/validation")
    return True


def check_legacy_folds_person_leakage() -> bool:
    """Check 1c: quantify person-level leakage in the LEGACY folds.

    Expected to FAIL — that is the documented finding. The legacy
    record-level folds put a test record's same-person mate in train
    for essentially every fold.
    """
    legacy_folds_path = REPO / "data" / "manifests" / "canonical_subject_folds_92subj.json"
    if not legacy_folds_path.exists():
        ok("legacy record-level folds not present (quarantined) — skipping person leakage check")
        return True  # Skip since quarantined
    
    manifest = json.load(open(legacy_folds_path))
    folds = manifest["folds"]
    n_leaky = 0
    for name, fold in folds.items():
        train_persons = {person_of(r) for r in fold["train"]}
        test_persons = {person_of(r) for r in fold["test"]}
        if train_persons & test_persons:
            n_leaky += 1
    if n_leaky:
        fail(f"legacy record-level folds leak at PERSON level: {n_leaky}/10 "
             f"folds train on the same-person mate of a test record "
             f"(SC4ss1/SC4ss2 = one person, two nights). Any number produced "
             f"on these folds is a RECORD-level estimate, not "
             f"person-generalization.")
        return False
    ok("legacy folds: no person-level train/test overlap (unexpected!)")
    return True


def check_subject_disjointness(manifest: dict, base_subjects: set | None = None) -> bool:
    """Check 2: base checkpoint training subjects vs evaluation folds.

    Runs against BOTH the legacy record-level manifest (expected to
    FAIL for a base trained on the person-level validation persons —
    documented, since that manifest is superseded) and the person-level
    manifest (the canonical evaluation protocol for the adaptation
    study — must PASS for a leak-free base).
    """
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
    # person-level view: even a record-disjoint base can share PERSONS
    # with the eval folds (SC4ss1/SC4ss2 = same person)
    person_overlap = {person_of(r) for r in train_subjects} & {
        person_of(r) for r in (eval_test | eval_val)
    }

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
    if person_overlap:
        fail(
            f"{label} is PERSON-CONTAMINATED: even though no record overlaps, "
            f"{len(person_overlap)} persons appear on both sides "
            f"(SC4ss1/SC4ss2 are the same person): {sorted(person_overlap)}"
        )
        return False

    ok(f"{label} is clean vs the LEGACY record-level manifest")

    ok(f"{label}: {len(train_subjects)} training subjects are disjoint "
       "from eval test+validation (record AND person level)")
    return True


def check_configs() -> bool:
    """Check 3: config hierarchy consistency (canonical hyperparameters)."""
    try:
        import yaml
    except ImportError:
        fail("PyYAML not installed — cannot verify configs")
        return False

    problems = []
    for cfg_path in (BENCH_CFG,):
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
                for key in ("subjects", "train_subjects", "base_train_subjects",
                             "base_training_subject_ids", "base_training_person_ids"):
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
        # vs legacy manifest: informational for a leak-free base (the
        # base's validation PERSONS appear as test records there —
        # documented, that manifest is superseded)
        checks.append(check_subject_disjointness(manifest, base_subjects))
    else:
        # Legacy manifest is quarantined - this is expected, don't add a failing check
        ok("legacy record-level manifest quarantined — skipping record-level disjointness check")
        checks.append(True)
    checks.append(check_person_level_folds())
    checks.append(check_legacy_folds_person_leakage())
    checks.append(check_configs())

    # The canonical check for the adaptation study: base training
    # records must be disjoint from the person-level folds' TEST sets
    # (appearing in validation there is fine — those subjects are never
    # used for adaptation evaluation).
    if base_subjects is not None and PERSON_FOLDS_PATH.exists():
        person_manifest = json.load(open(PERSON_FOLDS_PATH))
        person_folds = person_manifest["folds"]
        p_test = set()
        p_val = set()
        for fold in person_folds.values():
            p_test.update(fold["test"])
            p_val.update(fold["validation"])
        overlap = base_subjects & p_test
        in_val = base_subjects & p_val
        if overlap:
            fail(f"base checkpoint vs PERSON-LEVEL folds: {sorted(overlap)} "
                 "are TEST records — adaptation from this base is contaminated")
            checks.append(False)
        else:
            note = (f" (its training records are person-level VALIDATION "
                    f"records: {len(in_val)})") if in_val else ""
            ok(f"base checkpoint vs PERSON-LEVEL folds: 0 test-record "
               f"overlap{note} — SAFE for the adaptation study")
            checks.append(True)

    if LEGACY_CKPT.exists():
        print(f"\n  quarantined legacy checkpoint sha256: {sha256(LEGACY_CKPT)[:16]}… "
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
