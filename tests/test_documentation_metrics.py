#!/usr/bin/env python3
"""
Documentation Metrics Sync Test
===============================

Ensures that documented metrics match the canonical result artifacts.

Each expected entry maps a file to a (section marker, canonical value
variants) pair. The test locates the corresponding markdown section and
asserts that the canonical numbers are quoted verbatim (any of the
accepted variants). This catches drift between docs and results without
fragile numeric parsing.

Usage:
    python tests/test_documentation_metrics.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Header keywords used to isolate each named section within a document
# (first matching '## '/'### ' header wins).
SECTION_MARKERS = {
    "standalone": ["Standalone", "standalone_99k", "Deployed"],
    "freeze": ["Final Submission", "EXP-FULL-AUG30", "Final Protocol",
               "Full Corpus", "full corpus", "FULL-CORPUS"],
}

# Canonical value variants per experiment tier (any substring counts).
CANONICAL = {
    "standalone": {
        "accuracy": ["90.57%", "0.9057"],
        "kappa": ["0.8080", "0.808"],
        "macro_f1": ["0.7490", "0.749"],
        "weighted_f1": ["0.9115", "0.912"],
    },
    "freeze": {
        # Final submission run EXP-FULL-AUG30 (results/final/final_metrics.json)
        "accuracy": ["90.48%", "0.9048"],
        "kappa": ["0.8283"],
        "macro_f1": ["0.7899"],
        "weighted_f1": ["0.9089"],
        # Fit diagnosis (results/final/fit_diagnosis.json — NB05 §14)
        "fit_train_accuracy": ["92.50%", "0.9250"],
        "fit_train_val_gap": ["5.12 pp"],
    },
}

# Metrics that each file's tier section must quote (exactly what the
# document publishes). Tiers: freeze = final submission EXP-FULL-AUG30,
# standalone = deployed notebook-pipeline run. The historical
# EXP-BENCH-PERSON benchmark results were removed from all docs.
EXPECTED = {
    "README.md": {
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
        "standalone": ["accuracy", "kappa", "macro_f1", "weighted_f1"],
    },
    "MODEL_REPORT.md": {
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
        "standalone": ["accuracy", "kappa", "macro_f1", "weighted_f1"],
    },
    "GUIDE.md": {
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
    },
    "docs/RESULTS.md": {
        "standalone": ["accuracy", "kappa", "macro_f1", "weighted_f1"],
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
    },
    "docs/results.md": {
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
    },
    "docs/EXPERIMENTS.md": {
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
        "standalone": ["accuracy", "kappa", "macro_f1", "weighted_f1"],
    },
    "hf_model_card.md": {
        "freeze": ["accuracy", "kappa", "macro_f1", "weighted_f1",
                   "fit_train_accuracy", "fit_train_val_gap"],
    },
}


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (header, body) sections on '## ' boundaries.

    The body includes any nested '### ' subsections; those nested headers
    are also considered when matching (see ``find_section``).
    """
    sections: list[tuple[str, str]] = []
    current_header = "<preamble>"
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((current_header, "\n".join(current_lines)))
            current_header = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_header, "\n".join(current_lines)))
    return sections


def find_section(text: str, tier: str) -> str:
    """Return the body of the first '## ' section matching the named tier.

    A section matches when the keyword appears in its '## ' header or in
    any nested '### ' header inside its body.
    """
    for keyword in SECTION_MARKERS.get(tier, []):
        kw = keyword.lower()
        for header, body in split_sections(text):
            if kw in header.lower():
                return body
            nested = [
                line.lstrip("#").strip().lower()
                for line in body.splitlines()
                if line.startswith("### ")
            ]
            if any(kw in h for h in nested):
                return body
    return ""


def check_file(file_path: Path, file_name: str, tier_metrics: dict) -> list:
    """Check that each tier's canonical numbers are quoted in the file."""
    if not file_path.exists():
        return [f"{file_name}: File not found"]

    text = file_path.read_text()
    errors = []

    for tier, metrics in tier_metrics.items():
        section_text = find_section(text, tier)
        if not section_text:
            errors.append(f"{file_name}: no section found for tier '{tier}'")
            continue

        for metric in metrics:
            variants = CANONICAL[tier][metric]
            if not any(v in section_text for v in variants):
                errors.append(
                    f"{file_name} ({tier}): canonical {metric} not quoted — "
                    f"expected one of {variants}"
                )

    return errors


def main():
    print("=" * 70)
    print("  DOCUMENTATION METRICS SYNC TEST")
    print("=" * 70)

    all_errors = []
    for file_name, tier_metrics in EXPECTED.items():
        path = REPO / file_name
        errors = check_file(path, file_name, tier_metrics)
        if errors:
            all_errors.extend(errors)
            for e in errors:
                print(f"  [FAIL] {e}")
        else:
            print(f"  [OK]   {file_name}")

    print("=" * 70)
    if all_errors:
        print(f"  RESULT: FAIL ({len(all_errors)} metric mismatch(es))")
        return 1
    print("  RESULT: PASS (all documented metrics match canonical sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
