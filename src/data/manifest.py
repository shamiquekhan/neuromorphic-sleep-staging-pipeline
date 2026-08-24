"""Sleep-EDF dataset manifest building and subject-level splitting."""

import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_subject_night(filename: str):
    """Extract (subject_id, night) from a Sleep-EDF filename like 'SC4001E0-PSG.edf'."""
    base = os.path.basename(filename)
    m = re.match(r"(SC|ST)(\d{2})(\d)", base)
    if not m:
        return None, None
    cohort, subj, night = m.groups()
    return f"{cohort}{subj}", int(night)


def build_manifest(psg_files: list, hyp_files: list) -> pd.DataFrame:
    """Pair PSG files with hypnograms by subject+night prefix."""
    hyp_index = {}
    for h in hyp_files:
        subj, night = parse_subject_night(h)
        if subj is not None:
            hyp_index[(subj, night)] = h

    rows = []
    for p in psg_files:
        subj, night = parse_subject_night(p)
        hyp = hyp_index.get((subj, night))
        rows.append({
            "subject_id": subj,
            "night": night,
            "psg": p,
            "hypnogram": hyp,
            "matched": hyp is not None,
        })
    return pd.DataFrame(rows).sort_values(["subject_id", "night"]).reset_index(drop=True)


def subject_level_split(manifest: pd.DataFrame, seed: int = 42,
                        train_frac: float = 0.70, val_frac: float = 0.15):
    """Assign train/val/test splits at the subject level to prevent leakage."""
    subjects = sorted(manifest["subject_id"].unique())

    train_subj, temp_subj = train_test_split(
        subjects, test_size=1.0 - train_frac, random_state=seed
    )
    relative_val = val_frac / (1.0 - train_frac)
    val_subj, test_subj = train_test_split(
        temp_subj, test_size=1.0 - relative_val, random_state=seed
    )

    split_map = {s: "train" for s in train_subj}
    split_map.update({s: "val" for s in val_subj})
    split_map.update({s: "test" for s in test_subj})

    manifest = manifest.copy()
    manifest["split"] = manifest["subject_id"].map(split_map)

    assert set(train_subj) & set(val_subj) == set()
    assert set(train_subj) & set(test_subj) == set()
    assert set(val_subj) & set(test_subj) == set()

    return manifest, {
        "train": train_subj,
        "val": val_subj,
        "test": test_subj,
    }


def save_manifest(manifest: pd.DataFrame, manifest_dir: Path):
    """Save manifest and subject splits to CSV."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "sleep_edf.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path
