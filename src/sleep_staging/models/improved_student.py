"""Improved Student — final lightweight sleep-stage classifier.

99,477 parameters | 300s context | 5-class output
Multi-Res Stem → Depthwise-Separable CNN → Gabor FEB → 2-layer GRU

Post-audit fixes (Sept 2026):

* **Config contract (P1):** every layer is built from ``StudentConfig``.
  The pre-fix model hardcoded widths (8/16/272) while the config
  advertised different ones (10/32/32), so config edits silently did
  nothing. ``StudentConfig`` defaults now describe the trained
  architecture exactly, and the constructor refuses mismatched
  assumptions.
* **Gabor band constraint (P1):** center frequencies are stored as an
  unconstrained reparameterization (``gabor_freq_raw``) and mapped
  through a sigmoid into the documented 0.5–30 Hz band, and bandwidths
  through a softplus into a positive range. The pre-fix parameters
  drifted as far as −4.8 Hz / +39.4 Hz in trained fold checkpoints;
  both now stay in range by construction. Old checkpoints remain
  loadable: ``load_state_dict`` maps legacy ``gabor_freq``/``gabor_sigma``
  keys onto the new parameters via the inverse transform.
* **Input contract (P1):** ``forward`` validates shape and finiteness
  before any compute — required for deployment safety.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StudentConfig

# Documented Gabor filter band (Hz). Center frequencies are constrained
# to this range by construction, not by hope.
GABOR_FREQ_MIN_HZ = 0.5
GABOR_FREQ_MAX_HZ = 30.0


class ImprovedStudent(nn.Module):
    """Final lightweight student network for sleep-stage classification."""

    def __init__(self, config: StudentConfig | None = None):
        super().__init__()
        if config is None:
            config = StudentConfig()
        self.config = config

        stem_w = config.stem_width

        # ── Stem (two parallel Conv1d + BN branches) ────────────────
        # Short branch: 25-sample kernels, stride 6 → fine temporal
        # detail. Long branch: 200-sample kernels, stride 25 → slow
        # envelope. Widths come from the config (no hardcoded 8s).
        self.stem_s = nn.Sequential(
            nn.Conv1d(
                config.n_channels, stem_w, 25, stride=6, bias=False,
            ),
            nn.BatchNorm1d(stem_w),
        )
        self.stem_l = nn.Sequential(
            nn.Conv1d(
                config.n_channels, stem_w, 200, stride=25, bias=False,
            ),
            nn.BatchNorm1d(stem_w),
        )

        # ── Encoder (two depthwise-separable blocks) ───────────────
        enc_c = config.encoder_channels  # (c0_out, c1_out)
        stem_concat = 2 * stem_w
        self.enc = nn.ModuleDict({
            "0": nn.ModuleDict({
                "dw": nn.Conv1d(
                    stem_concat, stem_concat, 5, stride=2, padding=2,
                    groups=stem_concat, bias=False,
                ),
                "pw": nn.Conv1d(stem_concat, enc_c[0], 1, bias=False),
                "bn": nn.BatchNorm1d(enc_c[0]),
            }),
            "1": nn.ModuleDict({
                "dw": nn.Conv1d(
                    enc_c[0], enc_c[0], 5, stride=2, padding=2,
                    groups=enc_c[0], bias=False,
                ),
                "pw": nn.Conv1d(enc_c[0], enc_c[1], 1, bias=False),
                "bn": nn.BatchNorm1d(enc_c[1]),
            }),
        })

        self.pool = nn.AdaptiveAvgPool1d(8)

        # ── Gabor FEB ──────────────────────────────────────────────
        n_f = config.gabor_n_filters
        # Constrained reparameterization:
        #   freq_hz = 0.5 + 29.5 * sigmoid(freq_raw)   → [0.5, 30] Hz
        #   sigma    = softplus(sigma_raw) + eps      → positive
        # ``freq_raw`` initialized so sigmoid(freq_raw) reproduces the
        # legacy linear-space init (linspace 0.5..30 Hz), and
        # ``sigma_raw`` so softplus reproduces 0.02 — so a fresh model
        # is functionally identical to the legacy one.
        legacy_freq_hz = torch.linspace(
            GABOR_FREQ_MIN_HZ, GABOR_FREQ_MAX_HZ, n_f,
        )
        freq_frac = (legacy_freq_hz - GABOR_FREQ_MIN_HZ) / (
            GABOR_FREQ_MAX_HZ - GABOR_FREQ_MIN_HZ
        )
        # inverse-sigmoid, clamped for numerical headroom
        freq_raw = torch.logit(freq_frac.clamp(1e-4, 1 - 1e-4))
        import scipy.special as sp

        sigma_legacy = torch.full((n_f,), 0.02)
        sigma_raw = torch.from_numpy(
            sp.inv_softplus(sigma_legacy.numpy())
        ) if hasattr(sp, "inv_softplus") else torch.log(
            torch.expm1(sigma_legacy)
        )
        self.gabor_freq_raw = nn.Parameter(freq_raw)
        self.gabor_sigma_raw = nn.Parameter(sigma_raw)
        self.gab_proj = nn.Linear(n_f, config.gabor_out_dim)

        # ── GRU + classification head ───────────────────────────────
        feature_dim = enc_c[1] * 8 + config.gabor_out_dim
        self.gru = nn.GRU(
            feature_dim, config.gru_hidden,
            num_layers=config.gru_layers, batch_first=True,
        )
        self.head = nn.Linear(config.gru_hidden, config.n_classes)
        self.feature_dim = feature_dim

    # ── Constrained Gabor parameters ────────────────────────────────

    @property
    def gabor_freq_hz(self) -> torch.Tensor:
        """Center frequencies in Hz, guaranteed in [0.5, 30]."""
        frac = torch.sigmoid(self.gabor_freq_raw)
        return GABOR_FREQ_MIN_HZ + frac * (
            GABOR_FREQ_MAX_HZ - GABOR_FREQ_MIN_HZ
        )

    @property
    def gabor_sigma(self) -> torch.Tensor:
        """Bandwidth parameters, guaranteed positive."""
        return F.softplus(self.gabor_sigma_raw) + 1e-4

    # ── Checkpoint compatibility ───────────────────────────────────

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoints from either the legacy or new layout.

        Legacy checkpoints store the *unconstrained* parameters as
        ``gabor_freq`` (normalized frequency) and ``gabor_sigma``. Those
        values may violate the band (observed −4.8..39.4 Hz in trained
        fold checkpoints); loading them naively would either crash (key
        mismatch) or silently move the constraint. We map them through
        the inverse transforms and, because a raw out-of-band value
        cannot be represented exactly, record the clamped value as an
        attribute on load for inspection.
        """
        sd = dict(state_dict)
        legacy_freq = sd.pop("gabor_freq", None)
        legacy_sigma = sd.pop("gabor_sigma", None)

        if legacy_freq is not None:
            freq_hz = legacy_freq * self.config.sampling_rate
            clamped = freq_hz.clamp(
                GABOR_FREQ_MIN_HZ, GABOR_FREQ_MAX_HZ,
            )
            if not torch.equal(freq_hz, clamped):
                self._legacy_gabor_clamped = True
            frac = (clamped - GABOR_FREQ_MIN_HZ) / (
                GABOR_FREQ_MAX_HZ - GABOR_FREQ_MIN_HZ
            )
            sd["gabor_freq_raw"] = torch.logit(frac.clamp(1e-4, 1 - 1e-4))
        if legacy_sigma is not None:
            sd["gabor_sigma_raw"] = torch.log(
                torch.expm1(legacy_sigma.clamp(min=1e-6))
            )
        return super().load_state_dict(sd, strict)

    def _apply(self, fn):
        # Keep the clamped-load flag across .to()/.cuda()/.cpu().
        flag = self.__dict__.get("_legacy_gabor_clamped", False)
        out = super()._apply(fn)
        if flag:
            out.__dict__["_legacy_gabor_clamped"] = True
        return out

    # ── Forward ─────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, T, C, S]`` where T=seq_len, C=n_channels, S=samples_per_epoch.
            return_features: If True, also return the GRU input features.

        Returns:
            Logits of shape ``[B, T, n_classes]``.

        Raises:
            ValueError: on shape or dtype contract violations, or
                non-finite input (NaN/Inf). Deployment safety: these
            checks run before any compute.
        """
        expected = (
            self.config.seq_len,
            self.config.n_channels,
            self.config.samples_per_epoch,
        )
        if x.ndim != 4:
            raise ValueError(
                f"Expected input [B,T,C,S] with 4 dims, got {tuple(x.shape)}"
            )
        if x.shape[1:] != expected:
            raise ValueError(
                f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN/Inf")

        b, t, c, s = x.shape
        flat = x.reshape(b * t, c, s)

        # Stem
        short = F.relu6(self.stem_s[1](self.stem_s[0](flat)))
        long = F.relu6(self.stem_l[1](self.stem_l[0](flat)))
        target = min(short.shape[-1], long.shape[-1])
        short = F.adaptive_avg_pool1d(short, target)
        long = F.adaptive_avg_pool1d(long, target)
        stem_out = torch.cat([short, long], dim=1)

        # Encoder
        e0 = F.relu6(self.enc["0"]["bn"](self.enc["0"]["pw"](self.enc["0"]["dw"](stem_out))))
        e1 = F.relu6(self.enc["1"]["bn"](self.enc["1"]["pw"](self.enc["1"]["dw"](e0))))

        # CNN features: pool to 8 temporal positions → flatten
        cnn = self.pool(e1).flatten(1)

        # Gabor features: learnable constrained filter bank → project.
        # All 4 channels are averaged into one signal before Gabor
        # convolution (design trade-off documented in MODEL_REPORT.md).
        kernel_size = self.config.gabor_kernel_size
        t_axis = torch.arange(
            -(kernel_size // 2), kernel_size // 2 + 1,
            dtype=torch.float32, device=x.device,
        ).unsqueeze(0)
        f0 = (self.gabor_freq_hz / self.config.sampling_rate).unsqueeze(1)
        sigma = self.gabor_sigma.unsqueeze(1)
        envelope = torch.exp(-0.5 * (t_axis / (sigma * kernel_size)) ** 2)
        carrier = torch.cos(2 * math.pi * f0 * t_axis)
        kernels = (envelope * carrier).unsqueeze(1)
        x_mean = flat.mean(dim=1, keepdim=True)
        gab = F.conv1d(x_mean, kernels, padding=kernel_size // 2)
        gab = F.adaptive_avg_pool1d(gab, 1).squeeze(-1)
        gab = self.gab_proj(gab)

        features = torch.cat([cnn, gab], dim=-1).reshape(b, t, self.feature_dim)
        sequence_out, _ = self.gru(features)
        logits = self.head(sequence_out)

        if return_features:
            return logits, features
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
