"""Build a small demo cache so the Streamlit app works without Notebook 02.

The full cache (``data/cache_full``, 197 recordings, ~20 GB) and the legacy
cache are gitignored, so the deployed Streamlit Cloud app has no epoch data
to browse. This script crops a few recordings to the first ``--epochs``
epochs and writes them to ``data/cache_demo/``, which IS committed (see the
negation rules in ``.gitignore``).

Usage:
    python scripts/build_demo_cache.py                 # defaults: 3 recordings x 300 epochs
    python scripts/build_demo_cache.py --epochs 400 --max 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "cache_full"
DEFAULT_OUT = REPO / "data" / "cache_demo"

SPLIT_PRIORITY = {"train": 0, "test": 1, "val": 2}


def read_index() -> list[dict]:
    path = SRC / "cache_index.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run Notebook 02 first (full cache required "
            "as the source)."
        )
    with open(path) as f:
        return list(csv.DictReader(f))


def pick_recordings(rows: list[dict], max_n: int) -> list[dict]:
    """Deterministically pick up to ``max_n`` recordings: one per cohort
    first (train split preferred), then fill with distinct-subject rows."""
    ordered = sorted(
        rows,
        key=lambda r: (
            SPLIT_PRIORITY.get(r["split"], 9),
            r["cohort"],
            r["subject_id"],
            r["night"],
        ),
    )
    picked: list[dict] = []
    seen_cohorts: set[str] = set()
    seen_subjects: set[str] = set()
    for r in ordered:
        if len(picked) >= max_n:
            break
        if r["cohort"] in seen_cohorts:
            continue
        picked.append(r)
        seen_cohorts.add(r["cohort"])
        seen_subjects.add(r["subject_id"])
    for r in ordered:
        if len(picked) >= max_n:
            break
        if r["subject_id"] in seen_subjects:
            continue
        picked.append(r)
        seen_subjects.add(r["subject_id"])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=300,
                    help="epochs to keep per recording (default 300 = 2.5 h)")
    ap.add_argument("--max", type=int, default=3,
                    help="max recordings to include (default 3)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = read_index()
    chosen = pick_recordings(rows, args.max)
    args.out.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for r in chosen:
        rid = f"{r['subject_id']}_{r['night']}"
        epochs_path = SRC / f"{rid}_epochs.npy"
        meta_path = SRC / f"{rid}_meta.npz"
        if not epochs_path.exists() or not meta_path.exists():
            raise SystemExit(f"Missing cache files for {rid}, skipping.")

        epochs = np.load(epochs_path, mmap_mode="r")
        meta = dict(np.load(meta_path))
        n = min(args.epochs, epochs.shape[0], len(meta["labels"]))

        np.save(args.out / f"{rid}_epochs.npy", np.asarray(epochs[:n]))
        np.savez(
            args.out / f"{rid}_meta.npz",
            labels=np.asarray(meta["labels"][:n]),
            onsets=np.asarray(meta["onsets"][:n]),
            qc_flag=np.asarray(meta["qc_flag"][:n]),
            fs=np.asarray(meta["fs"]),
        )
        out_rows.append({
            **r,
            "n_epochs": str(n),
            "cache_path": str(args.out / f"{rid}.npz"),
        })
        size_mb = (args.out / f"{rid}_epochs.npy").stat().st_size / 1e6
        print(f"  {rid}: {n} epochs ({size_mb:.1f} MB) [{r['cohort']}/{r['split']}]")

    index_path = args.out / "cache_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    total_mb = sum(
        p.stat().st_size for p in args.out.glob("*.npy")
    ) / 1e6
    print(f"Done: {len(out_rows)} recordings, {total_mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
