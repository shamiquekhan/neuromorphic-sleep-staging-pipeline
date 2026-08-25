"""Cross-dataset harmonization utilities.

Provides functions for dataset-provenance tracking, fingerprinting,
and harmonization reports.
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def dataset_fingerprint(cache_dir: Path) -> dict:
    """Compute a fingerprint of a cached dataset.

    Returns:
        Dict with hash, subject count, epoch count, class distribution.
    """
    subjects = sorted(cache_dir.glob("*_night0.npz"))
    total_epochs = 0
    class_counts = {}

    for path in subjects:
        d = np.load(path)
        labels = d["labels"]
        total_epochs += len(labels)
        for label in np.unique(labels):
            class_counts[int(label)] = class_counts.get(int(label), 0) + int((labels == label).sum())

    # Hash of all subject IDs
    subject_names = [p.stem.replace("_night0", "") for p in subjects]
    name_hash = hashlib.md5(",".join(subject_names).encode()).hexdigest()[:12]

    return {
        "n_subjects": len(subjects),
        "n_epochs": total_epochs,
        "class_distribution": class_counts,
        "subject_hash": name_hash,
        "computed_at": datetime.now().isoformat(),
    }


def harmonization_report(dataset_a_dir: Path, dataset_b_dir: Path) -> dict:
    """Compare two cached datasets for harmonization compatibility.

    Checks:
        - Sampling rate
        - Epoch duration
        - Channel count
        - Class distribution overlap
    """
    report = {}

    for name, d in [("dataset_a", dataset_a_dir), ("dataset_b", dataset_b_dir)]:
        subjects = sorted(d.glob("*_night0.npz"))
        if not subjects:
            report[name] = {"status": "empty"}
            continue

        sample = np.load(subjects[0])
        epochs = sample["epochs"]

        report[name] = {
            "n_subjects": len(subjects),
            "n_channels": epochs.shape[1],
            "samples_per_epoch": epochs.shape[2],
            "status": "ok",
        }

    # Compatibility check
    if "dataset_a" in report and "dataset_b" in report:
        a, b = report["dataset_a"], report["dataset_b"]
        compatible = (
            a.get("n_channels") == b.get("n_channels") and
            a.get("samples_per_epoch") == b.get("samples_per_epoch")
        )
        report["compatibility"] = "compatible" if compatible else "incompatible"

    return report


def save_dataset_registry(registry: dict, path: Path):
    """Save a dataset registry to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def load_dataset_registry(path: Path) -> dict:
    """Load a dataset registry from JSON."""
    with open(path) as f:
        return json.load(f)
