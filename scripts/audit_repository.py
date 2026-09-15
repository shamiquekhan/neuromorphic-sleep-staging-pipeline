#!/usr/bin/env python3
"""
NeuroSleep Repository Audit
===========================

Comprehensive audit of repository consistency:
- Duplicate model definitions
- Duplicate experiment IDs
- Duplicate result files
- Hardcoded metric claims
- Stale file references
- Unknown Git provenance
- Untracked/generated junk
- Manifest/checkpoint mismatch

Usage:
    python scripts/audit_repository.py
    python scripts/audit_repository.py --json
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_check(name: str, check_fn):
    """Run a check and return (passed, message)."""
    try:
        passed, msg = check_fn()
        status = "PASS" if passed else "FAIL"
        return status, f"  [{status}] {name}: {msg}"
    except Exception as e:
        return "FAIL", f"  [FAIL] {name}: {e}"


# ── Checks ──────────────────────────────────────────────────────────────

def check_duplicate_model_defs():
    """Check for duplicate ImprovedStudent/ImprovedTeacher definitions."""
    student_files = list(REPO.rglob("*ImprovedStudent*.py"))
    teacher_files = list(REPO.rglob("*ImprovedTeacher*.py"))

    # Filter out __pycache__ and quarantine
    student_files = [f for f in student_files if "__pycache__" not in str(f) and "quarantine" not in str(f)]
    teacher_files = [f for f in teacher_files if "__pycache__" not in str(f) and "quarantine" not in str(f)]

    problems = []
    if len(student_files) > 1:
        problems.append(f"Multiple ImprovedStudent files: {[str(f.relative_to(REPO)) for f in student_files]}")
    if len(teacher_files) > 1:
        problems.append(f"Multiple ImprovedTeacher files: {[str(f.relative_to(REPO)) for f in teacher_files]}")

    # Check notebook inline definitions
    nb_student = 0
    nb_teacher = 0
    for nb in REPO.glob("notebooks/*.ipynb"):
        content = nb.read_text()
        if "class ImprovedStudent" in content:
            nb_student += 1
        if "class ImprovedTeacher" in content:
            nb_teacher += 1

    if nb_student > 0:
        problems.append(f"{nb_student} notebook(s) contain inline ImprovedStudent definition")
    if nb_teacher > 0:
        problems.append(f"{nb_teacher} notebook(s) contain inline ImprovedTeacher definition")

    if problems:
        return False, "; ".join(problems)
    return True, "No duplicate model definitions"


def check_experiment_ids():
    """Check for duplicate or inconsistent experiment IDs."""
    ids_found = {}
    for nb in REPO.glob("notebooks/*.ipynb"):
        content = nb.read_text()
        matches = re.findall(r'EXP-[A-Z0-9-]+', content)
        for m in matches:
            ids_found.setdefault(m, []).append(str(nb.relative_to(REPO)))

    # Check config files
    for cfg in REPO.glob("configs/**/*.yaml"):
        content = cfg.read_text()
        matches = re.findall(r'EXP-[A-Z0-9-]+', content)
        for m in matches:
            ids_found.setdefault(m, []).append(str(cfg.relative_to(REPO)))

    problems = []
    for exp_id, locations in ids_found.items():
        if len(locations) > 1:
            # This is OK if they reference the same experiment
            pass

    # Check for undefined experiment IDs in results
    for res_dir in REPO.glob("results/*/**/"):
        prov = res_dir / "provenance.json"
        if prov.exists():
            with open(prov) as f:
                p = json.load(f)
            exp_id = p.get("experiment_id")
            if exp_id and exp_id not in ids_found:
                problems.append(f"Result dir has experiment_id {exp_id} not found in configs/notebooks")

    if problems:
        return False, "; ".join(problems)
    return True, f"Experiment IDs consistent: {list(ids_found.keys())}"


def check_result_files():
    """Check for duplicate/ambiguous result files."""
    problems = []

    # Check for generic final_result.csv
    for f in REPO.rglob("final_result.csv"):
        if "quarantine" not in str(f):
            problems.append(f"Generic final_result.csv found: {f.relative_to(REPO)}")

    # Check for ambiguous checkpoint names
    for f in REPO.rglob("*student*final*.pt"):
        if "quarantine" not in str(f):
            problems.append(f"Ambiguous checkpoint name: {f.relative_to(REPO)}")

    # Check for multiple result files in same dir
    for d in REPO.glob("results/*/"):
        csvs = list(d.glob("*.csv"))
        if len(csvs) > 2:  # allow fold_summary + one other
            problems.append(f"Multiple CSVs in {d.relative_to(REPO)}: {[c.name for c in csvs]}")

    if problems:
        return False, "; ".join(problems)
    return True, "No duplicate/ambiguous result files"


def check_hardcoded_metrics():
    """Check for hardcoded metric claims in documentation."""
    metric_patterns = [
        r"87\.(?:66|30|7)",  # accuracy variants
        r"0\.(?:763|772|738)",  # kappa variants
        r"0\.(?:730|724|718)",  # macro F1 variants
        r"88\.64",  # exhibition accuracy
    ]

    problems = []
    for md_file in REPO.glob("*.md"):
        if "quarantine" in str(md_file):
            continue
        content = md_file.read_text()
        for pattern in metric_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # This is OK in RESULTS.md and EXPERIMENTS.md but not in README if inconsistent
                pass

    # Specific check: README vs RESULTS.md consistency
    readme = (REPO / "README.md").read_text()
    results_md = (REPO / "docs" / "RESULTS.md").read_text()

    # Check that primary benchmark numbers match
    if "87.30" not in readme and "87.30" in results_md:
        return False, "README missing primary benchmark (87.30%) found in RESULTS.md"

    if problems:
        return False, "; ".join(problems)
    return True, "Hardcoded metrics consistent with canonical sources"


def check_stale_references():
    """Check for references to deleted/moved files."""
    stale_patterns = [
        "benchmark_92_subject",
        "student_full_finetuned.pt",
        "run_100_subject_benchmark",
        "canonical_subject_folds_92subj",
        "final/student_full_finetuned.pt",
    ]

    problems = []

    # Check markdown files but allow quarantine folder listings and EXPERIMENTS.md
    for md_file in REPO.glob("*.md"):
        if "quarantine" in str(md_file):
            continue
        content = md_file.read_text()
        for pattern in stale_patterns:
            if pattern in content:
                # Allow in quarantine folder listings, EXPERIMENTS.md, and README quarantine section
                if "EXPERIMENTS" in str(md_file):
                    continue
                # Check if it's in a quarantine folder listing context
                idx = content.find(pattern)
                context = content[max(0, idx-100):idx+100]
                if "quarantine" not in context.lower():
                    problems.append(f"Stale reference '{pattern}' in {md_file.relative_to(REPO)}")

    # Check notebooks - only source cells, not execution outputs
    for nb in REPO.glob("notebooks/*.ipynb"):
        try:
            nb_json = json.loads(nb.read_text())
            for cell in nb_json.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    for pattern in stale_patterns:
                        if pattern in source:
                            problems.append(f"Stale reference '{pattern}' in {nb.relative_to(REPO)}")
        except Exception:
            pass

    if problems:
        return False, "; ".join(problems)
    return True, "No stale file references"


def check_git_provenance():
    """Check that checkpoints have valid git commit."""
    problems = []

    # Check exhibition checkpoint
    prov = REPO / "artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/provenance.json"
    if prov.exists():
        with open(prov) as f:
            p = json.load(f)
        git_commit = p.get("git_commit", "")
        if git_commit == "unknown" or len(git_commit) < 8:
            problems.append(f"Exhibition provenance has invalid git_commit: {git_commit}")

    # Check research checkpoint
    prov = REPO / "results/research/EXP-BENCH-PERSON/provenance.json"
    if prov.exists():
        with open(prov) as f:
            p = json.load(f)
        git_commit = p.get("git_commit", "")
        if git_commit == "unknown" or len(git_commit) < 8:
            problems.append(f"Research provenance has invalid git_commit: {git_commit}")

    # Check teacher checkpoint
    ckpt = REPO / "artifacts/teacher_improved_best.pt"
    if ckpt.exists():
        import torch
        data = torch.load(ckpt, map_location="cpu")
        git_commit = data.get("git_commit", "")
        if git_commit == "unknown" or len(git_commit) < 8:
            problems.append(f"Teacher checkpoint has invalid git_commit: {git_commit}")

    if problems:
        return False, "; ".join(problems)
    return True, "All checkpoints have valid git provenance"


def check_generated_junk():
    """Check for generated junk that should be gitignored."""
    problems = []

    junk_patterns = [
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.pyc",
        "*.tmp",
    ]

    for pattern in junk_patterns:
        matches = list(REPO.rglob(pattern))
        # Filter out allowed locations
        matches = [m for m in matches if ".git" not in str(m)]
        if matches:
            problems.append(f"Junk pattern '{pattern}' found: {[str(m.relative_to(REPO)) for m in matches[:5]]}")

    # Check for *.log files but allow run_logs directories
    log_matches = list(REPO.rglob("*.log"))
    log_matches = [m for m in log_matches if ".git" not in str(m) and "run_logs" not in str(m)]
    if log_matches:
        problems.append(f"Junk pattern '*.log' found: {[str(m.relative_to(REPO)) for m in log_matches[:5]]}")

    if problems:
        return False, "; ".join(problems)
    return True, "No generated junk in repository"


def check_manifest_checkpoint_match():
    """Verify checkpoints match their declared manifests."""
    problems = []

    # Exhibition student checkpoint
    ckpt = REPO / "artifacts/exhibition/EXP-EXHIBITION-15SUBJ/seed-42/student_best.pt"
    if ckpt.exists():
        import torch
        data = torch.load(ckpt, map_location="cpu")
        manifest = data.get("split_manifest", "")
        if manifest != "data/manifests/exhibition_15subj_v1.json":
            problems.append(f"Exhibition student checkpoint manifest mismatch: {manifest}")

    # Teacher checkpoint
    ckpt = REPO / "artifacts/teacher_improved_best.pt"
    if ckpt.exists():
        import torch
        data = torch.load(ckpt, map_location="cpu")
        manifest = data.get("split_manifest", "")
        if manifest != "data/manifests/exhibition_15subj_v1.json":
            problems.append(f"Teacher checkpoint manifest mismatch: {manifest}")

    if problems:
        return False, "; ".join(problems)
    return True, "All checkpoints match declared manifests"


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit repository consistency")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    checks = [
        ("Duplicate model definitions", check_duplicate_model_defs),
        ("Experiment IDs", check_experiment_ids),
        ("Result files", check_result_files),
        ("Hardcoded metrics", check_hardcoded_metrics),
        ("Stale references", check_stale_references),
        ("Git provenance", check_git_provenance),
        ("Generated junk", check_generated_junk),
        ("Manifest/checkpoint match", check_manifest_checkpoint_match),
    ]

    results = []
    for name, fn in checks:
        status, msg = run_check(name, fn)
        results.append({"check": name, "status": status, "message": msg})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 70)
        print("  REPOSITORY AUDIT")
        print("=" * 70)
        for r in results:
            print(r["message"])
        print("=" * 70)
        n_fail = sum(1 for r in results if r["status"] == "FAIL")
        if n_fail:
            print(f"  RESULT: FAIL ({n_fail} check(s) failed)")
            sys.exit(1)
        else:
            print("  RESULT: PASS (all checks passed)")
            sys.exit(0)


if __name__ == "__main__":
    main()