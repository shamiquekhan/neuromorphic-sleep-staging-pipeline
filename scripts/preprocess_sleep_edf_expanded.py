"""Preprocess Sleep-EDF Expanded recordings into cached NPZ format.

Reads raw EDF files from data/raw/sleep_edf/, applies canonical preprocessing,
and saves per-subject NPZ files to data/cache/sleep_edf/.

Usage:
    python scripts/preprocess_sleep_edf_expanded.py [--subjects SC4001 ...] [--limit 10]

Requires: mne, numpy, scipy
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sleep_edf"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "sleep_edf"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
CHECKSUM_FILE = PROJECT_ROOT / "data" / "cache" / "sleep_edf" / "checksums.json"

# ── Canonical preprocessing parameters ──────────────────────────────────

SAMPLING_RATE = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLING_RATE * EPOCH_SECONDS  # 3000
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 35.0
NOTCH_FREQ = 50.0

# ── Channel configuration ───────────────────────────────────────────────

# Channels to extract (in canonical order: EEG1, EEG2, EOG, EMG)
CHANNEL_MAP = {
    "EEG Fpz-Cz": 0,
    "EEG Pz-Oz": 1,
    "EOG horizontal": 2,
    "EMG submental": 3,
}

# ── Label mapping (Rechtschaffen & Kales → canonical) ──────────────────

SLEEP_EDF_LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # R&K stage 4 → AASM N3
    "Sleep stage R": 4,
}

VALID_LABELS = {0, 1, 2, 3, 4}


def load_raw_recording(psg_path: Path, hyp_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a raw Sleep-EDF recording using MNE.

    Returns:
        (epochs, labels, fs) where:
            epochs: [n_epochs, n_channels, n_samples]
            labels: [n_epochs] integer canonical labels
            fs: sampling rate
    """
    import mne

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    fs = float(raw.info["sfreq"])

    # Select channels in canonical order
    selected = []
    for ch_name in CHANNEL_MAP:
        if ch_name in raw.ch_names:
            selected.append(ch_name)
        else:
            log.warning("%s: channel %s not found", psg_path.name, ch_name)

    if len(selected) < 2:
        raise ValueError(f"Too few channels in {psg_path.name}: {selected}")

    raw.pick_channels(selected)

    # Resample to target rate
    if abs(fs - SAMPLING_RATE) > 1.0:
        raw.resample(SAMPLING_RATE)
        fs = SAMPLING_RATE

    # Load hypnogram annotations
    raw.set_annotations(mne.read_annotations(str(hyp_path)), emit_warning=False)

    # Create events from annotations
    events, _ = mne.events_from_annotations(
        raw,
        event_id=lambda label: SLEEP_EDF_LABEL_MAP.get(label, None),
        chunk_duration=EPOCH_SECONDS,
    )

    # Keep only valid labels
    keep = np.isin(events[:, 2], list(VALID_LABELS))
    events = events[keep]

    # Extract fixed-length epochs
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=0,
        tmax=EPOCH_SECONDS - 1.0 / fs,
        baseline=None, preload=True, on_missing="ignore", verbose=False,
    )

    data = epochs.get_data()  # [n_epochs, n_channels, n_samples]
    labels = events[: len(data), 2].astype(np.int64)

    # Ensure correct shape
    if data.shape[2] < SAMPLES_PER_EPOCH:
        # Pad if short
        pad_width = SAMPLES_PER_EPOCH - data.shape[2]
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_width)))
    elif data.shape[2] > SAMPLES_PER_EPOCH:
        data = data[:, :, :SAMPLES_PER_EPOCH]

    return data, labels, fs


def apply_preprocessing(epochs: np.ndarray) -> np.ndarray:
    """Apply canonical preprocessing to epoch arrays.

    Input: [n_epochs, n_channels, n_samples]
    Output: [n_epochs, n_channels, n_samples]
    """
    from scipy.signal import butter, filtfilt, iirnotch

    n_epochs, n_channels, n_samples = epochs.shape
    result = np.zeros_like(epochs, dtype=np.float32)

    # Bandpass filter
    nyq = SAMPLING_RATE / 2.0
    b_bp, a_bp = butter(4, [BANDPASS_LOW / nyq, BANDPASS_HIGH / nyq], btype="band")

    # Notch filter
    b_notch, a_notch = iirnotch(NOTCH_FREQ, 30.0, SAMPLING_RATE)

    for i in range(n_epochs):
        for c in range(n_channels):
            x = epochs[i, c].astype(np.float64)
            # Bandpass
            x = filtfilt(b_bp, a_bp, x)
            # Notch
            x = filtfilt(b_notch, a_notch, x)
            # Z-score normalization
            std = x.std()
            if std > 1e-8:
                x = (x - x.mean()) / std
            result[i, c] = x.astype(np.float32)

    return result


def compute_class_distribution(labels: np.ndarray) -> dict:
    """Compute per-class epoch counts."""
    names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    counts = {}
    for code, name in names.items():
        counts[name] = int((labels == code).sum())
    counts["total"] = len(labels)
    return counts


def find_hypnogram_file(raw_dir: Path, subject_id: str) -> Path:
    """Find the hypnogram file for a subject."""
    for hyp in raw_dir.glob(f"{subject_id}*Hypnogram.edf"):
        return hyp
    raise FileNotFoundError(f"No hypnogram found for {subject_id} in {raw_dir}")


def find_psg_file(raw_dir: Path, subject_id: str) -> Path:
    """Find the PSG file for a subject."""
    for psg in raw_dir.glob(f"{subject_id}*PSG.edf"):
        return psg
    raise FileNotFoundError(f"No PSG found for {subject_id} in {raw_dir}")


def process_subject(subject_id: str, raw_dir: Path, cache_dir: Path) -> dict:
    """Process one subject: load, preprocess, cache, return metadata."""
    psg_path = find_psg_file(raw_dir, subject_id)
    hyp_path = find_hypnogram_file(raw_dir, subject_id)

    log.info("Processing %s: PSG=%s, Hyp=%s", subject_id, psg_path.name, hyp_path.name)

    t0 = time.time()
    epochs, labels, fs = load_raw_recording(psg_path, hyp_path)
    t_load = time.time() - t0

    t0 = time.time()
    epochs = apply_preprocessing(epochs)
    t_preproc = time.time() - t0

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{subject_id}_night0.npz"
    np.savez_compressed(cache_path, epochs=epochs, labels=labels)

    # Checksum
    sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    dist = compute_class_distribution(labels)

    return {
        "subject_id": subject_id,
        "n_epochs": len(labels),
        "n_channels": int(epochs.shape[1]),
        "samples_per_epoch": int(epochs.shape[2]),
        "sampling_rate": fs,
        "class_distribution": dist,
        "load_time_s": round(t_load, 2),
        "preprocess_time_s": round(t_preproc, 2),
        "cache_path": str(cache_path),
        "sha256": sha256[:16],
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF Expanded")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    raw = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    cache = Path(args.cache_dir) if args.cache_dir else CACHE_DIR

    if not raw.exists():
        log.error("Raw directory not found: %s", raw)
        log.error("Run scripts/download_sleep_edf_expanded.py first")
        sys.exit(1)

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(
            p.stem.split("E")[0]
            for p in raw.glob("SC*PSG.edf")
        ) + sorted(
            p.stem.split("E")[0]
            for p in raw.glob("ST*PSG.edf")
        )

    if args.limit:
        subjects = subjects[: args.limit]

    if not subjects:
        log.error("No subjects found in %s", raw)
        sys.exit(1)

    log.info("Found %d subjects to process", len(subjects))
    log.info("Cache directory: %s", cache)

    results = []
    errors = []
    t_start = time.time()

    for i, sid in enumerate(subjects, 1):
        try:
            log.info("\n[%d/%d] %s", i, len(subjects), sid)
            meta = process_subject(sid, raw, cache)
            results.append(meta)
            log.info(
                "  %d epochs, %d channels, N1=%d, REM=%d (%.1fs)",
                meta["n_epochs"], meta["n_channels"],
                meta["class_distribution"]["N1"],
                meta["class_distribution"]["REM"],
                meta["load_time_s"] + meta["preprocess_time_s"],
            )
        except Exception as e:
            log.error("  FAILED: %s", e)
            errors.append({"subject_id": sid, "error": str(e)})

    elapsed = time.time() - t_start

    # Save manifest
    manifest = {
        "dataset": "sleep_edf_expanded",
        "version": "1.0.0",
        "n_subjects": len(results),
        "n_errors": len(errors),
        "processing_time_s": round(elapsed, 1),
        "parameters": {
            "sampling_rate": SAMPLING_RATE,
            "epoch_seconds": EPOCH_SECONDS,
            "samples_per_epoch": SAMPLES_PER_EPOCH,
            "bandpass": [BANDPASS_LOW, BANDPASS_HIGH],
            "notch": NOTCH_FREQ,
        },
        "subjects": results,
        "errors": errors,
    }

    manifest_path = MANIFEST_DIR / "sleep_edf_expanded.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save checksums
    checksums = {r["subject_id"]: r["sha256"] for r in results}
    with open(CHECKSUM_FILE, "w") as f:
        json.dump(checksums, f, indent=2)

    # Summary
    total_epochs = sum(r["n_epochs"] for r in results)
    total_n1 = sum(r["class_distribution"]["N1"] for r in results)
    total_rem = sum(r["class_distribution"]["REM"] for r in results)
    total_wake = sum(r["class_distribution"]["Wake"] for r in results)
    total_n2 = sum(r["class_distribution"]["N2"] for r in results)
    total_n3 = sum(r["class_distribution"]["N3"] for r in results)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete: {len(results)}/{len(subjects)} subjects")
    print(f"{'='*60}")
    print(f"Total epochs:  {total_epochs:,}")
    print(f"  Wake:        {total_wake:,} ({100*total_wake/total_epochs:.1f}%)")
    print(f"  N1:          {total_n1:,} ({100*total_n1/total_epochs:.1f}%)")
    print(f"  N2:          {total_n2:,} ({100*total_n2/total_epochs:.1f}%)")
    print(f"  N3:          {total_n3:,} ({100*total_n3/total_epochs:.1f}%)")
    print(f"  REM:         {total_rem:,} ({100*total_rem/total_epochs:.1f}%)")
    print(f"Time:          {elapsed:.0f}s")
    print(f"Manifest:      {manifest_path}")
    if errors:
        print(f"Errors:        {len(errors)} subjects failed")


if __name__ == "__main__":
    main()
