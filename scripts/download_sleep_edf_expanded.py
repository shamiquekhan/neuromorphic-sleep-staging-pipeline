"""Download Sleep-EDF Expanded from PhysioNet.

Downloads the full Sleep-EDF Expanded collection (197 recordings) to data/raw/sleep_edf/.

Usage:
    python scripts/download_sleep_edf_expanded.py [--subjects SC4001 SC4002 ...]

Requires: requests, tqdm
"""

import argparse
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── PhysioNet URLs ───────────────────────────────────────────────────────

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0"

# SC files are in sleep-cassette/, ST files in sleep-telemetry/
def get_subdir(subject_id: str) -> str:
    if subject_id.startswith("SC"):
        return "sleep-cassette"
    elif subject_id.startswith("ST"):
        return "sleep-telemetry"
    raise ValueError(f"Unknown subject prefix: {subject_id}")

PSG_PATTERN = "{subject_id}E0-PSG.edf"
HYP_PATTERN = "{subject_id}E?-Hypnogram.edf"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sleep_edf"

# All subjects in the Sleep-EDF Expanded dataset (extracted from PhysioNet listing)
# NOTE: SC4013 was lost due to a failing cassette, per documentation
SC_SUBJECTS = [
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
]

ST_SUBJECTS = [
    "ST7011", "ST7012", "ST7021", "ST7022", "ST7031", "ST7032",
    "ST7041", "ST7042", "ST7051", "ST7052", "ST7061", "ST7062",
    "ST7071", "ST7072", "ST7081", "ST7082", "ST7091", "ST7092",
    "ST7101", "ST7102", "ST7111", "ST7112", "ST7121", "ST7122",
    "ST7131", "ST7132", "ST7141", "ST7142", "ST7151", "ST7152",
]

ALL_SUBJECTS = SC_SUBJECTS + ST_SUBJECTS

# Hypnogram suffix varies by subject
HYP_SUFFIXES = ["EC", "EH", "EO", "EM", "EC", "EH", "EO", "EM"]


def find_physionet_file(subject_id: str, file_type: str) -> tuple[str, str]:
    """Find the correct PhysioNet filename for a subject.

    Returns:
        Tuple of (subdirectory, filename).
    """
    subdir = get_subdir(subject_id)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if file_type == "psg":
        psg_name = PSG_PATTERN.format(subject_id=subject_id)
        return subdir, psg_name

    elif file_type == "hypnogram":
        # Try each suffix until we find the file
        for suffix in HYP_SUFFIXES:
            hyp_name = f"{subject_id}{suffix}-Hypnogram.edf"
            hyp_path = RAW_DIR / hyp_name
            if hyp_path.exists():
                return subdir, hyp_name
        # Return the first suffix as expected name
        return subdir, f"{subject_id}EC-Hypnogram.edf"


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file from PhysioNet with progress bar."""
    if dest.exists() and not force:
        return True

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Sleep-EDF Expanded from PhysioNet")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject IDs to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print download plan without executing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of subjects to download",
    )
    args = parser.parse_args()

    subjects = args.subjects or ALL_SUBJECTS
    if args.limit:
        subjects = subjects[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    psg_urls = []
    hyp_urls = []
    for sid in subjects:
        psg_subdir, psg_name = find_physionet_file(sid, "psg")
        hyp_subdir, hyp_name = find_physionet_file(sid, "hypnogram")
        psg_urls.append((sid, f"{BASE_URL}/{psg_subdir}/{psg_name}", RAW_DIR / psg_name))
        hyp_urls.append((sid, f"{BASE_URL}/{hyp_subdir}/{hyp_name}", RAW_DIR / hyp_name))

    total = len(psg_urls) * 2  # PSG + hypnogram per subject
    print(f"Subjects: {len(subjects)}")
    print(f"Files to download: {total}")
    print()

    if args.dry_run:
        print("Dry run — files that would be downloaded:")
        for sid, url, dest in psg_urls[:5]:
            exists = "EXISTS" if dest.exists() else "MISSING"
            print(f"  [{exists}] {dest.name}")
        for sid, url, dest in hyp_urls[:5]:
            exists = "EXISTS" if dest.exists() else "MISSING"
            print(f"  [{exists}] {dest.name}")
        if len(subjects) > 5:
            print(f"  ... and {len(subjects) - 5} more")
        return

    # Download PSG files
    print("=== Downloading PSG files ===")
    psg_success = 0
    psg_failed = []
    for sid, url, dest in tqdm(psg_urls, desc="PSG"):
        if download_file(url, dest, force=args.force):
            psg_success += 1
        else:
            psg_failed.append(sid)
        time.sleep(0.5)  # Be polite to PhysioNet

    # Download hypnogram files
    print("\n=== Downloading Hypnogram files ===")
    hyp_success = 0
    hyp_failed = []
    for sid, url, dest in tqdm(hyp_urls, desc="Hyp"):
        if download_file(url, dest, force=args.force):
            hyp_success += 1
        else:
            hyp_failed.append(sid)
        time.sleep(0.5)

    # Summary
    print(f"\n=== Summary ===")
    print(f"PSG files:      {psg_success}/{len(psg_urls)} downloaded")
    print(f"Hypnogram files: {hyp_success}/{len(hyp_urls)} downloaded")
    if psg_failed:
        print(f"PSG failures:   {', '.join(psg_failed)}")
    if hyp_failed:
        print(f"Hyp failures:   {', '.join(hyp_failed)}")


if __name__ == "__main__":
    main()
