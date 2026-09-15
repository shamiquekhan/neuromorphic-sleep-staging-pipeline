"""Tests for the post-audit P1 model fixes.

Covers:
  * config contract — every layer built from StudentConfig (P1 #8)
  * Gabor band constraint — frequencies bounded in [0.5, 30] Hz by
    construction, under training and under legacy-checkpoint loads
    (P1 #9)
  * input contract — shape/dtype/finiteness validation (P1 #10)
  * legacy checkpoint compatibility — old state dicts load and map
    onto the constrained reparameterization
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sleep_staging.config import StudentConfig
from sleep_staging.models.improved_student import (
    GABOR_FREQ_MAX_HZ,
    GABOR_FREQ_MIN_HZ,
    ImprovedStudent,
    count_parameters,
)


@pytest.fixture
def model():
    return ImprovedStudent()


class TestConfigContract:
    def test_param_count_matches_trained_architecture(self, model):
        # The canonical checkpoint architecture; must not drift.
        assert count_parameters(model) == 99_477

    def test_layers_follow_config(self):
        cfg = StudentConfig(stem_width=12, gabor_out_dim=24)
        m = ImprovedStudent(cfg)
        # Stem widths
        assert m.stem_s[0].out_channels == 12
        assert m.stem_l[0].out_channels == 12
        # Encoder built from config
        assert m.enc["0"]["pw"].out_channels == cfg.encoder_channels[0]
        assert m.enc["1"]["pw"].out_channels == cfg.encoder_channels[1]
        # Gabor projection and feature dim follow config
        assert m.gab_proj.out_features == 24
        assert m.feature_dim == cfg.encoder_channels[1] * 8 + 24

    def test_config_change_changes_architecture(self):
        """The pre-fix bug: config edits did nothing. Now they must."""
        base = ImprovedStudent(StudentConfig())
        wide = ImprovedStudent(StudentConfig(stem_width=16))
        assert count_parameters(wide) != count_parameters(base)

    def test_config_defaults_describe_trained_model(self):
        """Defaults must equal the trained architecture so a bare
        ImprovedStudent() loads canonical checkpoints."""
        cfg = StudentConfig()
        assert cfg.stem_width == 8
        assert cfg.gabor_out_dim == 16
        assert cfg.encoder_channels == (32, 32)
        assert cfg.gabor_n_filters == 8


class TestGaborConstraint:
    def test_band_by_construction(self, model):
        f = model.gabor_freq_hz
        assert f.min() >= GABOR_FREQ_MIN_HZ - 1e-6
        assert f.max() <= GABOR_FREQ_MAX_HZ + 1e-6

    def test_band_survives_training(self):
        """Even extreme optimization cannot escape the band."""
        m = ImprovedStudent()
        opt = torch.optim.SGD(m.parameters(), lr=100.0)
        x = torch.randn(1, 10, 4, 3000)
        for _ in range(3):
            loss = m(x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert m.gabor_freq_hz.min() >= GABOR_FREQ_MIN_HZ - 1e-4
        assert m.gabor_freq_hz.max() <= GABOR_FREQ_MAX_HZ + 1e-4
        assert (m.gabor_sigma > 0).all()

    def test_fresh_init_matches_legacy_values(self, model):
        """A fresh model must reproduce the legacy linear init:
        linspace(0.5, 30, 8) Hz and sigma = 0.02. Interior points match
        exactly; the band endpoints saturate the sigmoid by up to
        ~0.003 Hz (logit clamp), which is negligible."""
        freqs = model.gabor_freq_hz
        expected = torch.linspace(0.5, 30.0, 8)
        torch.testing.assert_close(
            freqs, expected, rtol=0, atol=0.005,
        )
        torch.testing.assert_close(
            model.gabor_sigma, torch.full((8,), 0.02 + 1e-4),
            rtol=1e-3, atol=1e-4,
        )

    def test_legacy_out_of_band_checkpoint_is_clamped(self):
        """Legacy checkpoints with drifted frequencies must load and
        land inside the band, with the clamp recorded."""
        m = ImprovedStudent()
        # Simulate a legacy state dict with the observed drift:
        # −4.8 Hz and +39.4 Hz (normalized by fs=100).
        sd = m.state_dict()
        sd.pop("gabor_freq_raw")
        sd.pop("gabor_sigma_raw")
        sd["gabor_freq"] = torch.tensor(
            [-0.048, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.394]
        )
        sd["gabor_sigma"] = torch.full((8,), 0.02)
        m2 = ImprovedStudent()
        m2.load_state_dict(sd)
        assert m2.gabor_freq_hz.min() >= GABOR_FREQ_MIN_HZ - 1e-6
        assert m2.gabor_freq_hz.max() <= GABOR_FREQ_MAX_HZ + 1e-6
        assert getattr(m2, "_legacy_gabor_clamped", False)

    def test_legacy_in_band_checkpoint_loads_unclamped(self):
        m = ImprovedStudent()
        sd = m.state_dict()
        sd.pop("gabor_freq_raw")
        sd.pop("gabor_sigma_raw")
        # Strictly interior frequencies: no clamping can occur, and
        # the round-trip (Hz → normalized → raw → Hz) must be exact.
        freq_hz = torch.linspace(1.0, 29.0, 8)
        sd["gabor_freq"] = freq_hz / 100.0
        sd["gabor_sigma"] = torch.full((8,), 0.02)
        m2 = ImprovedStudent()
        m2.load_state_dict(sd)
        assert not getattr(m2, "_legacy_gabor_clamped", False)
        torch.testing.assert_close(m2.gabor_freq_hz, freq_hz, rtol=1e-5, atol=1e-5)


class TestInputContract:
    def test_rejects_wrong_ndim(self, model):
        with pytest.raises(ValueError, match="4 dims"):
            model(torch.randn(10, 4, 3000))

    def test_rejects_wrong_shape(self, model):
        with pytest.raises(ValueError, match=r"Expected \[B,10,4,3000\]"):
            model(torch.randn(2, 10, 2, 3000))  # wrong channel count
        with pytest.raises(ValueError, match=r"Expected \[B,10,4,3000\]"):
            model(torch.randn(2, 5, 4, 3000))  # wrong seq_len

    def test_rejects_nan_and_inf(self, model):
        x = torch.randn(1, 10, 4, 3000)
        x[0, 0, 0, 5] = float("nan")
        with pytest.raises(ValueError, match="NaN/Inf"):
            model(x)
        x[0, 0, 0, 5] = float("inf")
        with pytest.raises(ValueError, match="NaN/Inf"):
            model(x)

    def test_accepts_valid_batch(self, model):
        out = model(torch.randn(3, 10, 4, 3000))
        assert out.shape == (3, 10, 5)
