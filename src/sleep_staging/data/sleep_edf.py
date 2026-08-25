"""Sleep-EDF Expanded dataset adapter.

Handles downloading, preprocessing, and caching of the full Sleep-EDF Expanded
collection (197 whole-night PSG recordings from PhysioNet).

Dataset: https://www.physionet.org/content/sleep-edfx/1.0.0/
"""

import logging
from pathlib import Path

import numpy as np

from .labels import SLEEP_EDF_MAP, CANONICAL_LIST, N_CLASSES

log = logging.getLogger(__name__)

# ── Sleep-EDF Expanded subject list ──────────────────────────────────────
# The full dataset has 197 recordings. The naming convention is:
#   SC4xxxxG (healthy controls) and ST7xxxxG (insomniacs)
# We use only healthy controls for the primary benchmark.

SLEEP_EDF_SUBJECTS = [
    # Healthy controls (SC4xxxx) — 153 subjects from Sleep-EDF Expanded
    # NOTE: SC4013 was lost due to a failing cassette, per documentation
    "SC4001", "SC4002", "SC4011", "SC4012", "SC4021", "SC4022",
    "SC4031", "SC4032", "SC4041", "SC4042", "SC4051", "SC4052",
    "SC4061", "SC4062", "SC4071", "SC4072", "SC4081", "SC4082",
    "SC4091", "SC4092", "SC4101", "SC4102", "SC4111", "SC4112",
    "SC4121", "SC4122", "SC4131", "SC4141", "SC4142",
    "SC4151", "SC4152", "SC4161", "SC4162", "SC4171", "SC4172",
    "SC4181", "SC4182", "SC4191", "SC4192", "SC4201", "SC4202",
    "SC4211", "SC4212", "SC4221", "SC4222", "SC4231", "SC4232",
    "SC4241", "SC4242", "SC4251", "SC4252", "SC4261", "SC4262",
    "SC4271", "SC4272", "SC4281", "SC4282", "SC4291", "SC4292",
    "SC4301", "SC4302", "SC4311", "SC4312", "SC4321", "SC4322",
    "SC4331", "SC4332", "SC4341", "SC4342", "SC4351", "SC4352",
    "SC4362", "SC4371", "SC4372", "SC4381", "SC4382",
    "SC4401", "SC4402", "SC4411", "SC4412", "SC4421", "SC4422",
    "SC4431", "SC4432", "SC4441", "SC4442", "SC4451", "SC4452",
    "SC4461", "SC4462", "SC4471", "SC4472", "SC4481", "SC4482",
    "SC4491", "SC4492", "SC4501", "SC4502", "SC4511", "SC4512",
    "SC4522", "SC4531", "SC4532", "SC4541", "SC4542",
    "SC4551", "SC4552", "SC4561", "SC4562", "SC4571", "SC4572",
    "SC4581", "SC4582", "SC4591", "SC4592", "SC4601", "SC4602",
    "SC4611", "SC4612", "SC4621", "SC4622", "SC4631", "SC4632",
    "SC4641", "SC4642", "SC4651", "SC4652", "SC4661", "SC4662",
    "SC4671", "SC4672", "SC4701", "SC4702", "SC4711", "SC4712",
    "SC4721", "SC4722", "SC4731", "SC4732", "SC4741", "SC4742",
    "SC4751", "SC4752", "SC4761", "SC4762", "SC4771", "SC4772",
    "SC4801", "SC4802", "SC4811", "SC4812", "SC4821", "SC4822",
    # Insomniacs (ST7xxxx) — 30 subjects
    "ST7011", "ST7012", "ST7021", "ST7022", "ST7031", "ST7032",
    "ST7041", "ST7042", "ST7051", "ST7052", "ST7061", "ST7062",
    "ST7071", "ST7072", "ST7081", "ST7082", "ST7091", "ST7092",
    "ST7101", "ST7102", "ST7111", "ST7112", "ST7121", "ST7122",
    "ST7131", "ST7132", "ST7141", "ST7142", "ST7151", "ST7152",
]

# ── Channel configuration (same as current pipeline) ─────────────────────

CHANNELS = {
    "EEG Fpz-Cz": "EEG Fpz-Cz",
    "EEG Pz-Oz": "EEG Pz-Oz",
    "EOG horizontal": "EOG horizontal",
    "EMG submental": "EMG submental",
}

SAMPLING_RATE = 100  # Hz (after resampling)
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLING_RATE * EPOCH_SECONDS  # 3000


def get_physionet_paths(subject_id: str, raw_dir: Path) -> dict:
    """Resolve PhysioNet file paths for a subject.

    PhysioNet naming convention:
        PSG: {subject_id}-{recording_id}-PSG.edf
        Hyp: {subject_id}-{recording_id}-Hypnogram.edf

    Returns:
        Dict with 'psg' and 'hyp' Path objects.
    """
    psg_files = sorted(raw_dir.glob(f"{subject_id}-PSG.edf"))
    hyp_files = sorted(raw_dir.glob(f"{subject_id}-Hypnogram.edf"))

    if not psg_files:
        # Try alternate naming: {subject_id}G PSG
        psg_files = sorted(raw_dir.glob(f"{subject_id}*-PSG.edf"))
    if not hyp_files:
        hyp_files = sorted(raw_dir.glob(f"{subject_id}*-Hypnogram.edf"))

    if not psg_files:
        raise FileNotFoundError(f"No PSG file found for {subject_id} in {raw_dir}")
    if not hyp_files:
        raise FileNotFoundError(f"No Hypnogram file found for {subject_id} in {raw_dir}")

    return {"psg": psg_files[0], "hyp": hyp_files[0]}


def load_recording(psg_path: Path, hyp_path: Path, target_fs: int = 100):
    """Load one Sleep-EDF recording using MNE.

    Returns:
        Tuple of (epochs, labels, fs) where:
            epochs: [n_epochs, n_channels, n_samples]
            labels: [n_epochs] integer canonical labels
            fs: sampling rate
    """
    import mne

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    fs = float(raw.info["sfreq"])

    # Select channels
    available = [ch for ch in CHANNELS.values() if ch in raw.ch_names]
    if len(available) < 4:
        missing = [ch for ch in CHANNELS.values() if ch not in raw.ch_names]
        log.warning("%s: missing channels %s", psg_path.name, missing)
    raw.pick_channels(available)

    # Resample to target rate
    if fs != target_fs:
        raw.resample(target_fs)
        fs = target_fs

    # Add annotations and create events
    raw.set_annotations(mne.read_annotations(str(hyp_path)), emit_warning=False)

    events, _ = mne.events_from_annotations(
        raw,
        event_id=lambda label: SLEEP_EDF_MAP.get(label, None),
        chunk_duration=EPOCH_SECONDS,
    )

    # Keep only valid labels
    valid_codes = np.array([0, 1, 2, 3, 4])
    keep = np.isin(events[:, 2], valid_codes)
    events = events[keep]

    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=0,
        tmax=EPOCH_SECONDS - 1.0 / fs,
        baseline=None, preload=True, on_missing="ignore", verbose=False,
    )

    data = epochs.get_data()  # [n_epochs, n_channels, n_samples]
    labels = events[: len(data), 2].astype(np.int64)

    return data, labels, fs


def compute_class_distribution(labels: np.ndarray) -> dict:
    """Compute per-class epoch counts."""
    counts = {}
    for i in range(N_CLASSES):
        name = CANONICAL_LIST[i]
        counts[name] = int((labels == i).sum())
    counts["total"] = len(labels)
    return counts
