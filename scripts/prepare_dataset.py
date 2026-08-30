#!/usr/bin/env python
"""DEPRECATED: This script is broken — imports reference non-existent modules.

Use the working preprocessing pipeline instead:
    python scripts/preprocess_sleep_edf_expanded.py

This script is kept for reference only. The import paths below
(src.data.manifest, src.preprocessing.filters) do not exist in the
current codebase. The canonical modules are:
    src/sleep_staging/data/manifest.py
    src/sleep_staging/preprocessing/filters.py
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.manifest import build_manifest, subject_level_split, save_manifest
from src.preprocessing.filters import filter_signal, normalize_epoch, qc_flag

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    mne = None


def find_edf_files(raw_dir: Path):
    """Find all EDF files recursively."""
    edf_files = sorted(glob.glob(str(raw_dir / "**" / "*.edf"), recursive=True))
    psg_files = [f for f in edf_files if "PSG" in os.path.basename(f)]
    hyp_files = [f for f in edf_files if "Hypnogram" in os.path.basename(f)]
    return psg_files, hyp_files


def load_and_cache_recording(psg_path, hyp_path, cache_dir, row, cfg):
    """Load a single PSG/Hypnogram pair, filter, normalize, and cache."""
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    channels = cfg["dataset"]["channels"]
    missing = [ch for ch in channels.values() if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    raw.pick_channels(list(channels.values()))
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, emit_warning=False)

    aasm_map = {k: v for k, v in cfg["dataset"]["aasm_map"].items() if v is not None}
    events, _ = mne.events_from_annotations(
        raw, event_id=lambda lbl: aasm_map.get(lbl, None), chunk_duration=30
    )
    valid = set(aasm_map.values())
    events = events[np.isin(events[:, 2], list(valid))]

    eps = mne.Epochs(
        raw, events, event_id=None, tmin=0,
        tmax=30 - 1.0 / raw.info["sfreq"],
        baseline=None, preload=True, on_missing="ignore",
    )
    data = eps.get_data()
    labels = events[:, 2][:len(data)]

    fs = raw.info["sfreq"]
    bp = (cfg["preprocessing"]["bandpass"]["low_hz"], cfg["preprocessing"]["bandpass"]["high_hz"])
    notch = cfg["preprocessing"]["notch"]["freq_hz"]

    filt = np.stack([filter_signal(data[i], fs, bandpass=bp, notch_hz=notch) for i in range(len(data))])
    norm = np.stack([normalize_epoch(filt[i]) for i in range(len(filt))])
    flags = np.array([qc_flag(norm[i]) for i in range(len(norm))])

    out_file = cache_dir / f"{row['subject_id']}_night{row['night']}.npz"
    np.savez_compressed(
        out_file, epochs=norm.astype(np.float32), labels=labels.astype(np.int64),
        qc_flag=flags, fs=float(fs),
        subject_id=str(row["subject_id"]), night=int(row["night"]),
        split=str(row["split"]),
    )
    return out_file, len(norm), int(flags.sum())


def main():
    parser = argparse.ArgumentParser(description="Prepare Sleep-EDF dataset")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Dataset config")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--raw-dir", default="data/raw/sleep_edf")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--manifest-dir", default="data/manifests")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.preprocessing_config) as f:
        preprocessing_cfg = yaml.safe_load(f)
    cfg.update(preprocessing_cfg)

    raw_dir = Path(args.raw_dir)
    cache_dir = Path(args.cache_dir)
    manifest_dir = Path(args.manifest_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Finding EDF files...")
    psg_files, hyp_files = find_edf_files(raw_dir)
    print(f"  PSG: {len(psg_files)}, Hypnogram: {len(hyp_files)}")

    print("Building manifest...")
    manifest = build_manifest(psg_files, hyp_files)
    manifest = manifest[manifest["matched"]].drop(columns=["matched"]).reset_index(drop=True)

    print("Splitting subjects...")
    seed = cfg["splits"]["seed"]
    manifest, splits = subject_level_split(manifest, seed=seed)

    for split_name, subj_list in splits.items():
        print(f"  {split_name}: {len(subj_list)} subjects")

    save_manifest(manifest, manifest_dir)
    print(f"  Saved manifest to {manifest_dir / 'sleep_edf.csv'}")

    print("Preprocessing and caching...")
    cache_records = []
    for _, row in manifest.iterrows():
        out_file, n_epochs, n_flagged = load_and_cache_recording(
            row["psg"], row["hypnogram"], cache_dir, row, cfg
        )
        cache_records.append({
            "subject_id": row["subject_id"], "night": row["night"],
            "split": row["split"], "n_epochs": n_epochs,
            "n_flagged": n_flagged, "cache_path": str(out_file),
        })
        print(f"  {row['subject_id']} night{row['night']}: {n_epochs} epochs, {n_flagged} flagged")

    cache_df = pd.DataFrame(cache_records)
    cache_df.to_csv(cache_dir / "cache_index.csv", index=False)
    print(f"\nDone. Total: {len(cache_df)} recordings, {cache_df['n_epochs'].sum():,} epochs")


if __name__ == "__main__":
    main()
