"""Integrity tests for the subject-safe + causal evaluation protocol.

Covers the three audit findings these tests must make machine-checkable:

  P0.1  no training/eval window spans two subjects
  P0.1b no window spans an unlabeled-epoch temporal gap
  P0.6  every scored epoch receives exactly one prediction
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.data.sequence_dataset import (
    CausalEvalDataset,
    SubjectSequenceDataset,
    _gap_safe_starts,
)


def make_subject(subject_id, n_epochs, gap_after=None, n_channels=4, seq_seed=0,
                 n_samples=3000):
    """Synthetic subject; ``gap_after`` simulates a dropped epoch.

    Defaults match the real input contract (4 channels × 3000 samples
    per 30 s epoch) so windows feed the actual ImprovedStudent in
    end-to-end protocol tests.
    """
    rng = np.random.RandomState(seq_seed)
    epochs = rng.randn(n_epochs, n_channels, n_samples).astype(np.float32)
    labels = rng.randint(0, 5, size=n_epochs).astype(np.int64)

    orig_epoch_idx = np.arange(n_epochs, dtype=np.int64)
    if gap_after is not None:
        # Simulate preprocessing having dropped the epoch at
        # ``gap_after``: subsequent rows shift back one grid slot.
        orig_epoch_idx[gap_after + 1:] += 1

    return {
        "subject_id": subject_id,
        "epochs": epochs,
        "labels": labels,
        "orig_epoch_idx": orig_epoch_idx,
    }


# ── P0.1: subject-boundary safety ────────────────────────────────────────


class TestSubjectBoundarySafety:
    """The core audit test: a window must contain epochs from exactly one subject.

    The legacy protocol concatenated subjects via np.concatenate and then
    windowed, so windows like A[2645:2649] + B[0:6] were possible.
    """

    def test_no_window_crosses_subject_boundary(self):
        subjects = [make_subject("A", 30), make_subject("B", 30)]
        ds = SubjectSequenceDataset(subjects, seq_len=10, stride=5)

        per_window_subjects = ds.sample_subject_ids()
        for window_idx, sid in enumerate(per_window_subjects):
            assert sid in ("A", "B"), (
                f"window {window_idx} attributed to unknown subject {sid}"
            )

        # Both subjects contribute windows and no window blends them:
        # window content is drawn from a single subject's arrays by
        # construction; verify the sample bookkeeping is consistent.
        n_a = per_window_subjects.count("A")
        n_b = per_window_subjects.count("B")
        assert n_a > 0 and n_b > 0
        assert len(ds) == n_a + n_b

        # A 30-epoch subject with seq_len 10 / stride 5 yields exactly
        # starts {0, 5, 10, 15, 20} = 5 windows.
        assert n_a == 5 and n_b == 5

    def test_window_content_is_single_subject(self):
        """Labels inside a window all come from the window's own subject.

        Give each subject a constant label so mixing is detectable.
        """
        a = make_subject("A", 20)
        a["labels"][:] = 0
        b = make_subject("B", 20)
        b["labels"][:] = 4

        ds = SubjectSequenceDataset([a, b], seq_len=10, stride=5)
        for i in range(len(ds)):
            x, y = ds[i]
            assert len(set(y.tolist())) == 1, (
                f"window {i} mixes subjects: labels {y.tolist()}"
            )

    def test_short_subjects_are_skipped_not_merged(self):
        """A subject with < seq_len epochs must contribute nothing —
        and must not be bridged into the next subject's windows."""
        subjects = [
            make_subject("short", 4),        # < seq_len
            make_subject("long", 25),
        ]
        ds = SubjectSequenceDataset(subjects, seq_len=10, stride=5)
        assert set(ds.sample_subject_ids()) == {"long"}
        assert len(ds) == 4  # starts 0,5,10,15

    def test_legacy_concatenation_leak_is_gone(self):
        """Reproduce the exact legacy failure mode and show it cannot happen.

        Legacy: np.concatenate([A, B]) then windowing produced a window
        containing A's tail and B's head. Here we verify the Subject
        SequenceDataset window count equals the per-subject sum — no
        extra seam windows exist.
        """
        a = make_subject("A", 15)
        b = make_subject("B", 15)
        concat_len = 15 + 15
        legacy_starts = list(range(0, concat_len - 10 + 1, 5))  # includes seam
        seam_windows = [
            s for s in legacy_starts if s + 10 > 15  # window spans the seam
        ]
        assert seam_windows, "test setup: legacy windows must span the seam"

        ds = SubjectSequenceDataset([a, b], seq_len=10, stride=5)
        # 15 epochs, seq 10, stride 5 → starts {0, 5} = 2 windows/subject
        assert len(ds) == 4, (
            f"expected 4 subject-safe windows, got {len(ds)} "
            f"(legacy would give {len(legacy_starts)})"
        )


# ── P0.1b: temporal-gap safety ────────────────────────────────────────────


class TestGapSafety:
    def test_gap_safe_starts_excludes_gap_windows(self):
        orig = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10])  # gap at 4
        starts = _gap_safe_starts(orig, seq_len=4, stride=1, n_epochs=10)
        # Windows containing both array positions 3 and 4 straddle the
        # hole (grid 3→5) and must be excluded: starts 1, 2, 3.
        spanning = {s for s in range(10) if s <= 3 and s + 4 > 4}
        bad = spanning & set(starts)
        assert not bad, f"gap-spanning windows allowed: {bad}"
        # Windows fully before or after the hole are kept.
        assert 0 in starts
        assert {5, 6} <= set(starts)

    def test_dataset_skips_gap_windows(self):
        subj = make_subject("G", 20, gap_after=9)  # epoch "10" dropped
        ds = SubjectSequenceDataset([subj], seq_len=10, stride=5)
        # Without gap info: starts {0,5,10} = 3 windows. With it: the
        # window at start=5 spans the gap (grid 5..14 with a hole at
        # 10) and start=10 is fine (grid 11..20 consecutive).
        starts = [s for _, s in ds.samples]
        assert 5 not in starts, "window spanning dropped epoch was allowed"
        assert 0 in starts and 10 in starts

    def test_legacy_cache_without_index_falls_back(self):
        """Caches predating orig_epoch_idx still work (no gap guard)."""
        subj = make_subject("L", 20)
        subj["orig_epoch_idx"] = None
        ds = SubjectSequenceDataset([subj], seq_len=10, stride=5)
        assert len(ds) == 3  # starts 0, 5, 10 — all accepted, no crash

    def test_bad_index_shape_raises(self):
        subj = make_subject("X", 20)
        subj["orig_epoch_idx"] = np.arange(5)
        with pytest.raises(ValueError, match="orig_epoch_idx"):
            SubjectSequenceDataset([subj], seq_len=10, stride=5)


# ── P0.6: causal unique-epoch protocol ──────────────────────────────────


class TestCausalEvalDataset:
    def test_one_prediction_per_epoch(self):
        n = 25
        subj = make_subject("C", n)
        ds = CausalEvalDataset([subj], seq_len=10)

        # n - seq_len + 1 scored epochs, each exactly once
        assert len(ds) == n - 10 + 1

        seen = {}
        for i in range(len(ds)):
            x, y_last, sid, epoch_index = ds[i]
            assert x.shape == (10, 4, 3000)
            assert sid == "C"
            key = (sid, int(epoch_index))
            assert key not in seen, f"epoch {key} predicted twice"
            seen[key] = int(y_last)
        assert len(seen) == n - 9

    def test_epoch_index_provenance(self):
        """epoch_index follows the raw grid, not the array position,
        when orig_epoch_idx is present."""
        subj = make_subject("P", 20, gap_after=4)  # grid hole at 5
        ds = CausalEvalDataset([subj], seq_len=10)
        # The final array epoch (19) maps to grid epoch 20 — its context
        # (array 10..19 = grid 11..20) is contiguous, so it is scoreable
        # and must be the last sample.
        x, y, sid, idx = ds[len(ds) - 1]
        assert idx == 20
        # A hole invalidates every window that spans it: windows
        # containing array positions 4 and 5 together (grid 4 and 6)
        # are starts 0..4 → 5 targets excluded.
        assert ds.gap_excluded == {"P": 5}
        # 11 potential targets − 5 gap-spanning = 6 scoreable.
        assert len(ds) == 6

    def test_gap_context_epochs_are_excluded_not_fatal(self):
        """Epochs whose 5-min context spans a dropped epoch cannot be
        scored under a contiguous-context protocol — they are excluded
        at construction (and reported), never spliced or fatal."""
        subj = make_subject("R", 20, gap_after=9)  # grid hole at 10
        ds = CausalEvalDataset([subj], seq_len=10)
        # Windows spanning array positions 9 and 10 (grid 9 and 11)
        # are starts 1..9 → 9 targets excluded; 11 − 9 = 2 scoreable:
        # the fully-before window (start 0) and fully-after (start 10).
        assert len(ds) == 2
        assert ds.gap_excluded == {"R": 9}
        # All remaining items retrievable, unique, and correctly on the
        # grid: start 0 scores grid 9; start 10 scores grid 20.
        keys = sorted((ds[i][2], int(ds[i][3])) for i in range(len(ds)))
        assert keys == [("R", 9), ("R", 20)]

    def test_no_predictions_before_context_available(self):
        """The first seq_len-1 epochs of a subject cannot be scored
        causally and must not appear."""
        subj = make_subject("Q", 30)
        ds = CausalEvalDataset([subj], seq_len=10)
        indices = [int(ds[i][3]) for i in range(len(ds))]
        assert min(indices) == 9  # first scoreable epoch
        assert max(indices) == 29
        assert len(indices) == 21

    def test_multi_subject_no_cross_attribution(self):
        a = make_subject("A", 15)
        b = make_subject("B", 15)
        ds = CausalEvalDataset([a, b], seq_len=10)
        assert len(ds) == 12  # 6 + 6
        ids = {ds[i][2] for i in range(len(ds))}
        assert ids == {"A", "B"}


# ── evaluate_causal end-to-end guarantee ─────────────────────────────────


class TestEvaluateCausal:
    def test_unique_epoch_guarantee_end_to_end(self):
        """A deterministic model + the protocol library must produce
        exactly one prediction per (subject, epoch) pair."""
        from sleep_staging.evaluation.protocol import evaluate_causal
        from sleep_staging.models.improved_student import ImprovedStudent

        subjects = [make_subject("A", 15), make_subject("B", 15)]
        ds = CausalEvalDataset(subjects, seq_len=10)
        model = ImprovedStudent()
        metrics = evaluate_causal(model, ds, device="cpu", batch_size=4)

        preds = metrics["predictions"]
        keys = list(zip(preds["subject_id"], preds["epoch_index"]))
        assert len(keys) == len(set(map(tuple, keys)))
        assert metrics["n_unique_epochs"] == len(keys) == 12
        assert metrics["protocol"] == "causal_unique_epoch"
        assert metrics["n_subjects"] == 2

    def test_metrics_complete(self):
        from sleep_staging.evaluation.protocol import evaluate_causal
        from sleep_staging.models.improved_student import ImprovedStudent

        ds = CausalEvalDataset([make_subject("A", 15)], seq_len=10)
        metrics = evaluate_causal(ImprovedStudent(), ds, device="cpu")
        for key in ("accuracy", "kappa", "macro_f1", "weighted_f1", "mgm",
                    "per_class_accuracy", "confusion_matrix"):
            assert key in metrics, f"missing {key}"
        # per-class accuracy over 5 stages
        assert set(metrics["per_class_accuracy"]) == {
            "Wake", "N1", "N2", "N3", "REM",
        }
