#!/usr/bin/env python3
"""Generate person-level 10-fold CV splits for the 92-record cohort.

Groups records by PERSON (SC4ss1/SC4ss2 = same person's two nights; see
scripts/build_person_groups.py) and assigns whole persons — never single
nights — to folds. This is the only split structure that measures
person-level generalization.

Design:
  * 52 persons total (from the 92-record eligible cohort)
  * 5 fixed validation persons (~10% of persons, both nights when
    available) — never in any train or test fold
  * remaining 47 persons -> 10 test folds, stratified by AGE DECADE so
    fold difficulty is balanced (the SC cohort spans 25-101 yr)
  * per fold: train = all persons except test fold's persons (and
    except validation persons); both nights of every train person are
    used, night 1 and night 2 alike

All randomness is seeded (42) and deterministic.

Usage:
    python scripts/generate_person_folds.py            # writes manifest
    python scripts/generate_person_folds.py --validate  # check disjointness
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PERSON_GROUPS_PATH = REPO / "data" / "manifests" / "person_groups.json"
FOLDS_PATH = REPO / "data" / "manifests" / "person_folds_52subj.json"
OUT_PATH = REPO / "data" / "manifests" / "person_folds_52subj.json"

# Age per person, transcribed from PhysioNet SC-subjects.xls
# (subject number ss -> age in years). ss is the two digits after 'SC4'.
AGE_BY_SS = {
    "00": 33, "01": 33, "02": 26, "03": 26, "04": 34, "05": 28,
    "06": 31, "07": 30, "08": 25, "09": 25, "10": 26, "11": 26,
    "12": 26, "13": 27, "14": 27, "15": 31, "16": 32, "17": 31,
    "18": 28, "19": 28, "20": 51, "21": 51, "22": 56, "23": 50,
    "24": 54, "25": 56, "26": 51, "27": 54, "28": 56, "29": 51,
    "30": 50, "31": 54, "32": 57, "33": 60, "34": 54, "35": 57,
    "36": 51, "37": 52, "38": 51, "40": 67, "41": 66, "42": 69,
    "43": 73, "44": 74, "45": 66, "46": 66, "47": 73, "48": 67,
    "49": 67, "50": 71, "51": 70, "52": 69, "53": 67, "54": 73,
    "55": 71, "56": 72, "57": 66, "58": 67, "59": 67, "60": 89,
    "61": 101, "62": 95, "63": 91, "64": 85, "65": 88, "66": 88,
    "67": 87, "70": 89, "71": 88, "72": 88, "73": 97, "74": 92,
    "75": 96, "76": 90, "77": 85, "80": 54, "81": 57, "82": 56,
}

SEED = 42
N_FOLDS = 10
N_VALIDATION_PERSONS = 5


def age_decade(age: int) -> str:
    return f"{age // 10 * 10}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Only run the disjointness validation on the "
                             "existing manifest")
    args = parser.parse_args()

    groups = json.load(open(PERSON_GROUPS_PATH))["record_to_person"]
    with open(FOLDS_PATH) as f:
        records = json.load(f)["subjects"]

    person_records = defaultdict(list)
    for r in records:
        person_records[groups[r]].append(r)
    persons = sorted(person_records)  # 52 persons

    # person -> age (via ss digits: P4ss -> ss)
    def person_age(p: str) -> int:
        ss = p[2:]
        if ss not in AGE_BY_SS:
            raise KeyError(f"no age for person {p}")
        return AGE_BY_SS[ss]

    if args.validate:
        manifest = json.load(open(OUT_PATH))
        problems = validate(manifest, person_records)
        if problems:
            for p in problems:
                print(f"  [FAIL] {p}")
            raise SystemExit(1)
        print("  [OK] person-level folds valid")
        return

    # ── 1. choose fixed validation persons: one per spread of ages ──
    # Stratified pick: sort persons by age, take evenly spaced picks.
    rng = random.Random(SEED)
    by_age = sorted(persons, key=lambda p: (person_age(p), p))
    step = len(by_age) / N_VALIDATION_PERSONS
    val_persons = set()
    for i in range(N_VALIDATION_PERSONS):
        candidate = by_age[round(i * step)]
        val_persons.add(candidate)
    # extend if rounding collided
    while len(val_persons) < N_VALIDATION_PERSONS:
        for p in by_age:
            if p not in val_persons:
                val_persons.add(p)
                break

    val_records = sorted(r for p in val_persons for r in person_records[p])
    pool = [p for p in persons if p not in val_persons]  # 47 persons

    # ── 2. stratified assignment of pool persons to 10 test folds ──
    # Group by age decade, shuffle, deal round-robin so every fold gets a
    # mix of decades.
    by_decade = defaultdict(list)
    for p in pool:
        by_decade[age_decade(person_age(p))].append(p)
    for lst in by_decade.values():
        rng.shuffle(lst)

    fold_persons = [[] for _ in range(N_FOLDS)]
    i = 0
    for decade in sorted(by_decade):
        for p in by_decade[decade]:
            fold_persons[i].append(p)
            i = (i + 1) % N_FOLDS

    # ── 3. build folds ──
    folds = {}
    for k, plist in enumerate(fold_persons, start=1):
        test_records = sorted(r for p in plist for r in person_records[p])
        train_persons = [p for p in pool if p not in set(plist)]
        train_records = sorted(
            r for p in train_persons for r in person_records[p]
        )
        folds[f"fold_{k}"] = {
            "test": test_records,
            "validation": val_records,
            "train": train_records,
            "test_persons": sorted(plist),
            "train_persons": sorted(train_persons),
            "validation_persons": sorted(val_persons),
        }

    # ── 4. validation before writing ──
    manifest = {
        "version": "1.0.0",
        "description": "Person-level 10-fold CV over 52 persons "
                       "(92 eligible records; SC4ss1/SC4ss2 are the same "
                       "person's two nights). Validation fixed to 5 persons. "
                       "Stratified by age decade. Supersedes "
                       "person_folds_52subj.json, which is "
                       "record-level and leaks at person level.",
        "dataset": "sleep_edf_expanded",
        "n_records": len(records),
        "n_persons": len(persons),
        "n_folds": N_FOLDS,
        "split_method": "person_level_stratified_by_age_decade",
        "seed": SEED,
        "records": records,
        "persons": persons,
        "folds": folds,
    }
    problems = validate(manifest, person_records)
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        raise SystemExit(1)

    OUT_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    # report
    print(f"persons: {len(persons)} (val {len(val_persons)}, pool {len(pool)})")
    for k, f in folds.items():
        ages = [person_age(p) for p in f["test_persons"]]
        print(f"  {k}: {len(f['test_persons'])} test persons "
              f"({len(f['test'])} records), ages {min(ages)}-{max(ages)}, "
              f"median {statistics.median(ages):.0f} | "
              f"train {len(f['train'])} records")
    print(f"written: {OUT_PATH.relative_to(REPO)}")


def validate(manifest: dict, person_records: dict) -> list[str]:
    problems = []
    folds = manifest["folds"]
    all_records = set(manifest["records"])

    # every record appears in exactly one test fold OR is validation
    seen_test = set()
    val = set(folds["fold_1"]["validation"])
    for name, f in folds.items():
        test = set(f["test"])
        if test & seen_test:
            problems.append(f"{name}: record in multiple test folds")
        seen_test |= test
        if set(f["validation"]) != val:
            problems.append(f"{name}: validation set differs")
        tr, te, va = set(f["train"]), test, set(f["validation"])
        if tr & te or tr & va or te & va:
            problems.append(f"{name}: record-level overlap")
        # person-level: no person's records may be split across roles
        role = {}
        for r in f["train"]:
            role.setdefault(r, "train")
        for r in f["test"]:
            if role.get(r, "test") != "test":
                problems.append(f"{name}: {r} in two roles")
        for p, recs in person_records.items():
            roles = set()
            for r in recs:
                if r in tr:
                    roles.add("train")
                if r in te:
                    roles.add("test")
                if r in va:
                    roles.add("val")
            if len(roles) > 1:
                problems.append(f"{name}: person {p} split across {roles}")
        # every record accounted for
        if tr | te | va != all_records:
            problems.append(f"{name}: missing records")

    if seen_test | val != all_records:
        problems.append("union of test folds + validation != all records")
    return problems


if __name__ == "__main__":
    main()
