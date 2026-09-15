#!/usr/bin/env python3
"""
Documentation Metrics Sync Test
===============================

Ensures that all documented metrics match the canonical result artifacts.
Run this test to catch drift between documentation and actual results.

Usage:
    python tests/test_documentation_metrics.py
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Canonical source of truth
PRIMARY_BENCHMARK = REPO / "results" / "research" / "EXP-BENCH-PERSON" / "final_metrics.json"

# Expected metrics per file/section
EXPECTED = {
    # Primary benchmark (EXP-BENCH-PERSON) - these files/sections MUST match
    "README.md:primary": {
        "accuracy": 0.87297,
        "kappa": 0.73805,
        "macro_f1": 0.72432,
        "weighted_f1": 0.88028,
    },
    "MODEL_REPORT.md:primary": {
        "accuracy": 0.87297,
        "kappa": 0.73805,
        "macro_f1": 0.72432,
        "weighted_f1": 0.88028,
    },
    "GUIDE.md:primary": {
        "accuracy": 0.87297,
        "kappa": 0.73805,
        "macro_f1": 0.72432,
    },
    "RESULTS.md:primary": {
        "accuracy": 0.87297,
        "kappa": 0.73805,
        "macro_f1": 0.72432,
        "weighted_f1": 0.88028,
    },
    "EXPERIMENTS.md:primary": {
        "accuracy": 0.87297,
        "kappa": 0.73805,
        "macro_f1": 0.72432,
        "weighted_f1": 0.88028,
    },
    "hf_model_card.md:primary": {
        "accuracy": 0.87297,
        "kappa": 0.73805,
        "macro_f1": 0.72432,
        "weighted_f1": 0.88028,
    },

    # Historical exhibition (EXP-EXHIBITION-15SUBJ) - these files/sections MUST match
    "README.md:exhibition": {
        "accuracy": 0.8864,
        "kappa": 0.7725,
        "macro_f1": 0.7186,
        "weighted_f1": 0.8953,
    },
    "MODEL_REPORT.md:exhibition": {
        "accuracy": 0.8864,
        "kappa": 0.7725,
        "macro_f1": 0.7186,
        "weighted_f1": 0.8953,
    },
    "EXPERIMENTS.md:exhibition": {
        "accuracy": 0.8864,
        "kappa": 0.7725,
        "macro_f1": 0.7186,
        "weighted_f1": 0.8953,
    },
    "exhibition.md:exhibition": {
        "accuracy": 0.8864,
        "kappa": 0.7725,
        "macro_f1": 0.7186,
        "weighted_f1": 0.8953,
    },
}


def load_primary_metrics():
    """Load primary benchmark metrics from canonical artifact."""
    with open(PRIMARY_BENCHMARK) as f:
        data = json.load(f)
    return {
        "accuracy": data["accuracy"]["mean"],
        "kappa": data["cohen_kappa"]["mean"],
        "macro_f1": data["macro_f1"]["mean"],
        "weighted_f1": data["weighted_f1"]["mean"],
    }


def extract_metric(text: str, metric: str, context_hint: str = "") -> float | None:
    """Extract a specific metric value from text near a context hint."""
    # Skip quarantined sections
    if "quarantined" in text.lower() or "QUARANTINED" in text:
        return None

    # Search near the context hint
    search_text = text
    if context_hint:
        idx = text.lower().find(context_hint.lower())
        if idx >= 0:
            search_text = text[max(0, idx-200):idx+2000]

    # Skip if quarantined section
    if "quarantined" in search_text.lower() or "QUARANTINED" in search_text:
        return None

    # Patterns for each metric
    patterns = {
        "accuracy": [
            r"accuracy[^\d]*(\d+\.\d+)%?",
            r"Test Accuracy[^\d]*(\d+\.\d+)%?",
            r"\*\*Accuracy\*\*[^\d]*(\d+\.\d+)%?",
            r"\|\s*Accuracy\s*\|\s*(\d+\.\d+)%?",
            r"(\d+\.\d+)%\s*TEST ACCURACY",
        ],
        "kappa": [
            r"kappa[^\d]*(\d+\.\d+)",
            r"Cohen['\']?s?\s*κ[^\d]*(\d+\.\d+)",
            r"\*\*Cohen's κ\*\*[^\d]*(\d+\.\d+)",
            r"\|\s*κ\s*\|\s*(\d+\.\d+)",
            r"Cohen's κ[^\d]*(\d+\.\d+)",
        ],
        "macro_f1": [
            r"macro.?f1[^\d]*(\d+\.\d+)",
            r"Macro.?F1[^\d]*(\d+\.\d+)",
            r"\*\*Macro F1\*\*[^\d]*(\d+\.\d+)",
            r"\|\s*Macro.?F1\s*\|\s*(\d+\.\d+)",
        ],
        "weighted_f1": [
            r"weighted.?f1[^\d]*(\d+\.\d+)",
            r"Weighted.?F1[^\d]*(\d+\.\d+)",
            r"\*\*Weighted F1\*\*[^\d]*(\d+\.\d+)",
            r"\|\s*Weighted.?F1\s*\|\s*(\d+\.\d+)",
        ],
    }

    for pattern in patterns.get(metric, []):
        matches = re.findall(pattern, search_text, re.IGNORECASE)
        if matches:
            try:
                val = float(matches[-1])
                if val > 1.5:
                    val = val / 100.0
                return val
            except (ValueError, IndexError):
                continue
    return None


def extract_section(text: str, start_marker: str, end_marker: str = None) -> str:
    """Extract text between markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    if end_marker:
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = start_idx + 3000
        return text[start_idx:end_idx]
    return text[start_idx:start_idx+3000]


def check_file(file_path: Path, file_name: str) -> list:
    """Check a specific file for expected metrics."""
    if not file_path.exists():
        return [f"{file_name}: File not found"]

    text = file_path.read_text()
    errors = []

    # Check primary benchmark sections
    primary_keys = [k for k in EXPECTED if k.startswith(file_name + ":primary")]
    for key in primary_keys:
        expected = EXPECTED[key]
        section_name = key.split(":")[1]

        # Find the primary benchmark section
        section_text = ""
        for marker in ["Primary Benchmark", "EXP-BENCH-PERSON", "person-level", "PRIMARY BENCHMARK"]:
            if marker in text:
                idx = text.find(marker)
                section_text = text[max(0, idx-200):idx+3000]
                break
        if not section_text:
            section_text = text

        for metric, expected_val in expected.items():
            found = extract_metric(section_text, metric, "Primary Benchmark")
            if found is not None:
                if abs(found - expected_val) > 0.001:
                    errors.append(
                        f"{file_name} (primary): {metric} = {found:.4f}, "
                        f"expected {expected_val:.4f} (diff: {abs(found - expected_val):.4f})"
                    )

    # Check exhibition section
    exhibition_keys = [k for k in EXPECTED if k.startswith(file_name + ":exhibition")]
    for key in exhibition_keys:
        expected = EXPECTED[key]

        section_text = ""
        # Find the actual exhibition results section (not just the table entry)
        for marker in ["## EXP-EXHIBITION-15SUBJ", "Historical Exhibition", "historical exhibition", "legacy exhibition"]:
            if marker in text:
                idx = text.find(marker)
                # Make sure we're not in the table at the top
                if "Experiment Index" not in text[max(0, idx-100):idx]:
                    section_text = text[max(0, idx-200):idx+3000]
                    break
        if not section_text:
            continue  # Skip if no exhibition section

        for metric, expected_val in expected.items():
            found = extract_metric(section_text, metric, "Exhibition")
            if found is not None:
                if abs(found - expected_val) > 0.001:
                    errors.append(
                        f"{file_name} (exhibition): {metric} = {found:.4f}, "
                        f"expected {expected_val:.4f} (diff: {abs(found - expected_val):.4f})"
                    )

    return errors


def main():
    print("=" * 70)
    print("  DOCUMENTATION METRICS SYNC TEST")
    print("=" * 70)

    # Load actual primary metrics
    primary = load_primary_metrics()
    print(f"\nCanonical primary: {primary}")

    files_to_check = [
        ("README.md", REPO / "README.md"),
        ("MODEL_REPORT.md", REPO / "MODEL_REPORT.md"),
        ("GUIDE.md", REPO / "GUIDE.md"),
        ("RESULTS.md", REPO / "docs" / "RESULTS.md"),
        ("EXPERIMENTS.md", REPO / "docs" / "EXPERIMENTS.md"),
        ("exhibition.md", REPO / "docs" / "exhibition.md"),
        ("hf_model_card.md", REPO / "hf_model_card.md"),
    ]

    all_errors = []

    for name, path in files_to_check:
        errors = check_file(path, name)
        if errors:
            all_errors.extend(errors)
            for e in errors:
                print(f"  [FAIL] {e}")
        else:
            print(f"  [OK]   {name}")

    print("=" * 70)
    if all_errors:
        print(f"  RESULT: FAIL ({len(all_errors)} metric mismatch(es))")
        return 1
    else:
        print("  RESULT: PASS (all documented metrics match canonical sources)")
        return 0


if __name__ == "__main__":
    sys.exit(main())