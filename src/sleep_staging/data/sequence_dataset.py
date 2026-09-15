"""Subject-safe sequence construction for PSG sleep staging.

Fixes the two evaluation-protocol flaws documented in the 2026 audit:

1. **Subject-boundary leakage (P0.1).** The legacy `SequenceDataset`
   windows over a concatenation of all subjects
   (``load_subjects`` → ``np.concatenate``), so the final epochs of one
   subject share a training/evaluation window with the first epochs of
   the next. `SubjectSequenceDataset` builds windows *per subject*, so a
   window can never span two subjects.

2. **Temporal-gap leakage (P0.1b).** Preprocessing drops epochs with
   unlabeled annotations ("?", movement time) *before* caching, so
   consecutive rows in a subject cache may be temporally discontiguous.
   When a cache stores ``orig_epoch_idx`` (the epoch index within the
   raw 30 s grid of the recording), both datasets exclude windows that
   span such gaps: training windows are simply not built across them,
   and epochs whose causal context would span a gap are reported as
   unscoreable (``CausalEvalDataset.gap_excluded``) rather than being
   scored on spliced context.

3. **Epoch double-counting (P0.6).** The legacy evaluation used
   ``stride=5`` with all-position supervision, so most epochs were
   scored multiple times and metrics were computed over window-position
   pairs. `CausalEvalDataset` uses stride-1 windows and supervises only
   the final epoch: each scored epoch receives exactly one prediction,
   from the 5 minutes of PSG context immediately preceding it.

Datasets
--------
``SubjectSequenceDataset``
    Training dataset. Windows never cross a subject boundary or an
    unlabeled-epoch gap. All positions are supervised (training signal,
    not reporting). Returns ``(x, y)`` only.

``CausalEvalDataset``
    Evaluation dataset. One stride-1 window per subject per target epoch,
    supervising only the last position. Returns
    ``(x, y_last, subject_id, epoch_index)`` so results can be reported
    per unique scored epoch with full provenance.

Both accept dicts as returned by
`sleep_staging.data.loader.load_cached_subject`, optionally extended
with an ``orig_epoch_idx`` key (see
``scripts/preprocess_sleep_edf_expanded.py``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "SubjectSequenceDataset",
    "CausalEvalDataset",
    "make_subject_list",
]


def make_subject_list(
    subject_ids: list[str],
    cache_dir,
) -> list[dict]:
    """Load per-subject caches into the dict format the datasets expect.

    Unlike ``load_subjects`` (which concatenates), this preserves the
    subject as the unit of sequence construction.

    Raises ``FileNotFoundError`` on the first missing subject rather than
    silently skipping, so fold manifests can never silently shrink.
    """
    from .loader import load_cached_subject

    out = []
    for sid in subject_ids:
        data = load_cached_subject(sid, cache_dir)
        out.append({
            "subject_id": data["subject_id"],
            "epochs": data["epochs"],
            "labels": data["labels"],
            "orig_epoch_idx": data.get("orig_epoch_idx"),
        })
    return out


def _gap_safe_starts(
    orig_epoch_idx: np.ndarray | None,
    seq_len: int,
    stride: int,
    n_epochs: int,
) -> list[int]:
    """Starts whose window contains no dropped (unlabeled) epochs.

    If ``orig_epoch_idx`` is unavailable (legacy caches), we cannot
    detect gaps and fall back to all array-contiguous windows; the
    benchmark prints a warning in that case so the weaker guarantee is
    never silent.
    """
    starts = list(range(0, n_epochs - seq_len + 1, stride))
    if orig_epoch_idx is None:
        return starts

    orig = np.asarray(orig_epoch_idx)
    if orig.ndim != 1 or len(orig) != n_epochs:
        raise ValueError(
            f"orig_epoch_idx must be 1-D of length {n_epochs}, "
            f"got shape {orig.shape}"
        )

    # A window is temporally contiguous iff the original epoch indices
    # inside it are consecutive integers.
    good = []
    for start in starts:
        window = orig[start : start + seq_len]
        if len(window) == seq_len and np.array_equal(
            window, window[0] + np.arange(seq_len)
        ):
            good.append(start)
    return good


class SubjectSequenceDataset(Dataset):
    """Training windows built strictly within one subject/night.

    Every window:
      * contains epochs from exactly one subject, and
      * (when ``orig_epoch_idx`` is available) spans no dropped epochs.

    ``subject_id`` per window is retained for the boundary test
    (``tests/test_sequence_dataset.py``) even though ``__getitem__``
    returns ``(x, y)`` for DataLoader compatibility.
    """

    def __init__(
        self,
        subjects: list[dict],
        seq_len: int = 10,
        stride: int = 5,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.seq_len = seq_len
        self.stride = stride

        # Tensors are stored once per subject; samples only hold
        # (subject_idx, start) so windows share memory instead of
        # duplicating the 3000-sample epoch arrays.
        self._epochs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self.subject_ids: list[str] = []  # subject id per sample
        self.samples: list[tuple[int, int]] = []  # (subject_idx, start)

        for item in subjects:
            epochs = torch.from_numpy(np.ascontiguousarray(item["epochs"])).float()
            labels = torch.from_numpy(np.ascontiguousarray(item["labels"])).long()
            if len(labels) != epochs.shape[0]:
                raise ValueError(
                    f"subject {item['subject_id']}: "
                    f"{epochs.shape[0]} epochs vs {len(labels)} labels"
                )

            n = len(labels)
            if n < seq_len:
                continue  # too short to contribute any window

            starts = _gap_safe_starts(
                item.get("orig_epoch_idx"), seq_len, stride, n,
            )
            if not starts:
                continue

            subj_idx = len(self._epochs)
            self._epochs.append(epochs)
            self._labels.append(labels)
            for start in starts:
                self.samples.append((subj_idx, start))
                self.subject_ids.append(item["subject_id"])

        if not self.samples:
            raise ValueError(
                "SubjectSequenceDataset is empty — no subject has "
                f"{seq_len} contiguous epochs"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj_idx, start = self.samples[idx]
        end = start + self.seq_len
        return (
            self._epochs[subj_idx][start:end],
            self._labels[subj_idx][start:end],
        )

    def sample_subject_ids(self) -> list[str]:
        """Subject id per sample — used by the boundary-safety test."""
        return list(self.subject_ids)


class CausalEvalDataset(Dataset):
    """One prediction per unique scored epoch (causal protocol, P0.6).

    For subject epochs ``e[0] .. e[n-1]`` and ``seq_len=k``:
    every epoch ``t >= k-1`` receives exactly one prediction, produced
    from window ``[t-k+1 .. t]`` (the k epochs ending at t, i.e. the
    ``seq_len * 30 s`` of context immediately preceding and including
    the scored epoch). The first ``k-1`` epochs of each subject cannot be
    scored under a causal protocol and are excluded.

    Epochs whose context window spans a dropped (unlabeled) epoch are
    also excluded at construction time — they cannot be scored under a
    strict contiguous-context protocol. Exclusion counts are exposed via
    ``gap_excluded`` so reports can state exactly how many epochs were
    not scoreable and why.

    ``__getitem__`` returns ``(x, y_last, subject_id, epoch_index)``:
    exactly what the per-epoch result artifact requires.
    """

    def __init__(
        self,
        subjects: list[dict],
        seq_len: int = 10,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.seq_len = seq_len
        self.subjects: list[dict] = []
        self.samples: list[tuple[int, int]] = []  # (subject_idx, start)
        self.gap_excluded: dict[str, int] = {}    # subject → n excluded

        for item in subjects:
            n = len(item["labels"])
            if n < seq_len:
                continue

            subj_idx = len(self.subjects)
            self.subjects.append({
                "subject_id": item["subject_id"],
                "epochs": item["epochs"],
                "labels": item["labels"],
                "orig_epoch_idx": item.get("orig_epoch_idx"),
            })

            orig = item.get("orig_epoch_idx")
            if orig is None:
                # Legacy cache: all array-contiguous windows accepted.
                for start in range(0, n - seq_len + 1):
                    self.samples.append((subj_idx, start))
            else:
                orig = np.asarray(orig)
                if orig.ndim != 1 or len(orig) != n:
                    raise ValueError(
                        f"subject {item['subject_id']}: orig_epoch_idx must "
                        f"be 1-D of length {n}, got shape {orig.shape}"
                    )
                excluded = 0
                # Target t = start + seq_len - 1 is scoreable iff its
                # full context [start .. t] is temporally contiguous on
                # the raw grid.
                for start in range(0, n - seq_len + 1):
                    window = orig[start : start + seq_len]
                    if np.array_equal(
                        window, window[0] + np.arange(seq_len)
                    ):
                        self.samples.append((subj_idx, start))
                    else:
                        excluded += 1
                if excluded:
                    self.gap_excluded[item["subject_id"]] = excluded

        if not self.samples:
            raise ValueError(
                "CausalEvalDataset is empty — no subject has "
                f"{seq_len} contiguous epochs"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj_idx, start = self.samples[idx]
        s = self.subjects[subj_idx]
        t = start + self.seq_len - 1  # scored (last) position

        x = torch.from_numpy(
            np.ascontiguousarray(s["epochs"][start : start + self.seq_len])
        ).float()
        y_last = int(s["labels"][t])
        epoch_index = (
            int(s["orig_epoch_idx"][t])
            if s["orig_epoch_idx"] is not None
            else t
        )
        return x, y_last, s["subject_id"], epoch_index

    def subject_ids_with_gaps(self) -> list[str]:
        """Subjects whose cached arrays have unknown temporal gaps.

        Non-empty means the run is on legacy caches without
        ``orig_epoch_idx``; the caller should print a warning.
        """
        return [s["subject_id"] for s in self.subjects if s["orig_epoch_idx"] is None]
