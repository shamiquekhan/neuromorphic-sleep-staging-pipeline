"""Download Sleep-EDF Expanded from PhysioNet.

Downloads the full Sleep-EDF Expanded collection (197 recordings) to data/raw/sleep_edf/.

Usage:
    python scripts/download_sleep_edf_expanded.py [--subjects SC4001 SC4002 ...]
    python scripts/download_sleep_edf_expanded.py --stagger 3   # every 3rd subject for spread
    python scripts/download_sleep_edf_expanded.py --workers 6   # parallel downloads
    python scripts/download_sleep_edf_expanded.py --print-urls  # print URLs for aria2c

Requires: requests, tqdm
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Hypnogram suffix varies by subject (all observed suffixes in dataset)
HYP_SUFFIXES = ["EC", "EA", "EJ", "EU", "EV", "EW", "EG", "EH", "EP"]


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
        # Try each suffix until we find the file locally
        for suffix in HYP_SUFFIXES:
            hyp_name = f"{subject_id}{suffix}-Hypnogram.edf"
            hyp_path = RAW_DIR / hyp_name
            if hyp_path.exists() and hyp_path.stat().st_size > 1000:
                return subdir, hyp_name
        # Not found locally — return first suffix as default
        # (download_file will try HEAD to verify, but we need a name)
        return subdir, f"{subject_id}{HYP_SUFFIXES[0]}-Hypnogram.edf"


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file from PhysioNet with resume support and completeness check.

    Checks actual file size against Content-Length to detect incomplete downloads.
    Supports HTTP Range resume for interrupted transfers.
    """
    if force and dest.exists():
        dest.unlink()

    # Get expected size via HEAD request
    try:
        head = requests.head(url, timeout=30)
        head.raise_for_status()
        expected_size = int(head.headers.get("content-length", 0))
    except Exception:
        expected_size = 0

    # Check if file is already complete
    if dest.exists():
        current_size = dest.stat().st_size
        if expected_size and current_size == expected_size:
            return True  # genuinely complete
        elif expected_size and current_size > expected_size:
            dest.unlink()  # corrupt/oversized, start over
        elif current_size > 0:
            # Resume from where it left off
            headers = {"Range": f"bytes={current_size}-"}
            mode = "ab"
            initial = current_size
        else:
            headers, mode, initial = {}, "wb", 0
    else:
        headers, mode, initial = {}, "wb", 0

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        # If server doesn't support range requests, start fresh
        if headers and response.status_code == 200:
            mode, initial = "wb", 0

        with open(dest, mode) as f, tqdm(
            total=expected_size or None,
            initial=initial,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            disable=not sys.stderr.isatty(),
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        # Verify completeness
        if expected_size and dest.stat().st_size != expected_size:
            return False
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}", file=sys.stderr)
        return False


def download_one(args_tuple):
    """Download a single file. Used by ThreadPoolExecutor."""
    url, dest, force = args_tuple
    return download_file(url, dest, force)


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
        help="Re-download even if file exists and is complete",
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
    parser.add_argument(
        "--stagger",
        type=int,
        default=None,
        help="Take every Nth subject for better demographic spread (e.g. --stagger 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel download workers (default: 6)",
    )
    parser.add_argument(
        "--print-urls",
        action="store_true",
        help="Print URL list for aria2c and exit",
    )
    args = parser.parse_args()

    subjects = args.subjects or ALL_SUBJECTS
    if args.stagger:
        subjects = subjects[:: args.stagger]
    if args.limit:
        subjects = subjects[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    psg_urls = []
    hyp_urls = []
    for sid in subjects:
        psg_subdir, psg_name = find_physionet_file(sid, "psg")
        psg_urls.append((sid, f"{BASE_URL}/{psg_subdir}/{psg_name}", RAW_DIR / psg_name))
        # For hypnograms: check if any suffix already downloaded
        hyp_downloaded = False
        for suffix in HYP_SUFFIXES:
            hyp_name = f"{sid}{suffix}-Hypnogram.edf"
            hyp_dest = RAW_DIR / hyp_name
            if hyp_dest.exists() and hyp_dest.stat().st_size > 1000:
                hyp_downloaded = True
                break
        if not hyp_downloaded:
            # Add all possible hypnogram URLs — download_hypnogram will try each
            for suffix in HYP_SUFFIXES:
                hyp_name = f"{sid}{suffix}-Hypnogram.edf"
                hyp_urls.append((sid, f"{BASE_URL}/sleep-cassette/{hyp_name}", RAW_DIR / hyp_name))

    total = len(psg_urls) + len(hyp_urls)  # PSG + hypnogram attempts

    # --print-urls: output for aria2c
    if args.print_urls:
        print(f"# {len(subjects)} subjects, {total} files")
        print(f"# Usage: aria2c -x 8 -s 8 -c -i urls.txt")
        print(f"# -x 8 = 8 connections per file, -s 8 = split into 8 parts, -c = resume")
        print()
        for sid, url, dest in psg_urls + hyp_urls:
            print(url)
            print(f"  dir={RAW_DIR}")
            print(f"  out={dest.name}")
        return

    print(f"Subjects: {len(subjects)}")
    print(f"Files to download: {total}")
    print(f"Workers: {args.workers}")
    print()

    if args.dry_run:
        print("Dry run - files that would be downloaded:")
        for sid, url, dest in psg_urls[:5]:
            status = "COMPLETE" if dest.exists() else "MISSING"
            if dest.exists() and dest.stat().st_size > 0:
                try:
                    head = requests.head(url, timeout=10)
                    expected = int(head.headers.get("content-length", 0))
                    if expected and dest.stat().st_size == expected:
                        status = "COMPLETE"
                    else:
                        status = f"PARTIAL ({dest.stat().st_size}/{expected})"
                except Exception:
                    status = "EXISTS (size unknown)"
            print(f"  [{status}] {dest.name}")
        for sid, url, dest in hyp_urls[:5]:
            exists = "EXISTS" if dest.exists() else "MISSING"
            print(f"  [{exists}] {dest.name}")
        if len(subjects) > 5:
            print(f"  ... and {len(subjects) - 5} more")
        return

    # Build download tasks
    tasks = []
    for sid, url, dest in psg_urls + hyp_urls:
        tasks.append((url, dest, args.force))

    # Download with thread pool
    success = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, t): t for t in tasks}
        for future in as_completed(futures):
            url, dest, _ = futures[future]
            ok = future.result()
            if ok:
                success += 1
            else:
                failed.append(dest.name)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Downloaded: {success}/{total}")
    print(f"Failed:     {len(failed)}")
    if failed:
        print(f"Failed files:")
        for name in failed[:20]:
            print(f"  {name}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
