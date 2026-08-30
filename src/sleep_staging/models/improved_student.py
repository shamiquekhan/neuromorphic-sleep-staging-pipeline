"""Improved Student — final lightweight sleep-stage classifier.

99,477 parameters | 300s context | 5-class output
Multi-Res Stem → Depthwise-Separable CNN → Gabor FEB → 2-layer GRU

Module attribute names are chosen to match the official checkpoint state_dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StudentConfig


class ImprovedStudent(nn.Module):
    """Final lightweight student network for sleep-stage classification."""

    def __init__(self, config: StudentConfig | None = None):
        super().__init__()
        if config is None:
            config = StudentConfig()
        self.config = config

        # ── Stem (two parallel Conv1d + BN branches) ────────────────────
        self.stem_s = nn.Sequential(
            nn.Conv1d(4, 8, 25, stride=6, bias=False),
            nn.BatchNorm1d(8),
        )
        self.stem_l = nn.Sequential(
            nn.Conv1d(4, 8, 200, stride=25, bias=False),
            nn.BatchNorm1d(8),
        )

        # ── Encoder (two depthwise-separable blocks) ────────────────────
        self.enc = nn.ModuleDict({
            "0": nn.ModuleDict({
                "dw": nn.Conv1d(16, 16, 5, stride=2, padding=2, groups=16, bias=False),
                "pw": nn.Conv1d(16, 32, 1, bias=False),
                "bn": nn.BatchNorm1d(32),
            }),
            "1": nn.ModuleDict({
                "dw": nn.Conv1d(32, 32, 5, stride=2, padding=2, groups=32, bias=False),
                "pw": nn.Conv1d(32, 32, 1, bias=False),
                "bn": nn.BatchNorm1d(32),
            }),
        })

        self.pool = nn.AdaptiveAvgPool1d(8)

        # ── Gabor FEB ──────────────────────────────────────────────────
        self.gabor_freq = nn.Parameter(
            torch.linspace(0.5, 30.0, config.gabor_n_filters) / config.sampling_rate
        )
        self.gabor_sigma = nn.Parameter(
            torch.full((config.gabor_n_filters,), 0.02)
        )
        self.gab_proj = nn.Linear(config.gabor_n_filters, 16)

        # ── GRU + classification head ───────────────────────────────────
        feature_dim = 32 * 8 + 16  # 272
        self.gru = nn.GRU(
            feature_dim, config.gru_hidden,
            num_layers=config.gru_layers, batch_first=True,
        )
        self.head = nn.Linear(config.gru_hidden, config.n_classes)
        self.feature_dim = feature_dim

    def forward(
        self, x: torch.Tensor, return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, T, C, S]`` where T=seq_len, C=n_channels, S=samples_per_epoch.
            return_features: If True, also return the GRU input features.

        Returns:
            Logits of shape ``[B, T, n_classes]``.
        """
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

        # CNN features: pool to 8 temporal positions → flatten → 256
        cnn = self.pool(e1).flatten(1)

        # Gabor features: learnable filter bank → project → 16
        # Note: We average all 4 channels (2 EEG, EOG, EMG) into a single
        # signal before Gabor convolution. This is a design trade-off:
        # - Pro: Reduces parameter count and computation
        # - Pro: Gabor filters learn frequency patterns common across channels
        # - Con: Muddies channel-specific features (e.g., EMG artifacts vs EEG alpha)
        # Alternative: per-channel Gabor with channel-wise attention weighting
        kernel_size = 51
        t_axis = torch.arange(
            -(kernel_size // 2), kernel_size // 2 + 1,
            dtype=torch.float32, device=x.device,
        ).unsqueeze(0)
        f0 = self.gabor_freq.unsqueeze(1)
        sigma = (self.gabor_sigma.abs() + 1e-4).unsqueeze(1)
        envelope = torch.exp(-0.5 * (t_axis / (sigma * kernel_size)) ** 2)
        carrier = torch.cos(2 * 3.14159265 * f0 * t_axis)
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
