"""Final Improved Student architecture for sleep-stage classification.

99,477 parameters | 300s context | 5-class output
Multi-Resolution Stem → Depthwise-Separable CNN → Gabor FEB → 2-layer GRU
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution to reduce parameter cost."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel, stride, kernel // 2,
            groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class LiteMultiResolutionStem(nn.Module):
    """Parallel short/long receptive-field stems for multi-scale feature capture."""

    def __init__(self, in_ch: int = 4, width: int = 10, fs: int = 100):
        super().__init__()
        small_kernel, small_stride = fs // 4, max(1, fs // 16)
        large_kernel, large_stride = fs, max(1, fs // 4)

        self.short = DepthwiseSeparableConv1d(in_ch, width, small_kernel, small_stride)
        self.long = DepthwiseSeparableConv1d(in_ch, width, large_kernel, large_stride)
        self.out_ch = width * 2

    def forward(self, x):
        short = self.short(x)
        long = self.long(x)
        target = min(short.shape[-1], long.shape[-1])
        short = F.adaptive_avg_pool1d(short, target)
        long = F.adaptive_avg_pool1d(long, target)
        return torch.cat([short, long], dim=1)


class ParametricGabor1D(nn.Module):
    """Parametric Gabor-like 1D filter bank for frequency-localized feature extraction."""

    def __init__(self, in_ch: int = 4, n_filters: int = 8, kernel_size: int = 51, fs: int = 100):
        super().__init__()
        self.in_ch = in_ch
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.center_freq = nn.Parameter(torch.linspace(0.5, 30.0, n_filters) / fs)
        self.bandwidth = nn.Parameter(torch.full((n_filters,), 0.02))

        t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("t", t)

    def forward(self, x):
        t = self.t.unsqueeze(0)
        f0 = self.center_freq.unsqueeze(1)
        sigma = (self.bandwidth.abs() + 1e-4).unsqueeze(1)

        envelope = torch.exp(-0.5 * (t / (sigma * self.kernel_size)) ** 2)
        carrier = torch.cos(2 * math.pi * f0 * t)
        kernels = (envelope * carrier).unsqueeze(1)

        b, c, s = x.shape
        y = F.conv1d(x.reshape(b * c, 1, s), kernels, padding=self.kernel_size // 2)
        return y.reshape(b, c * self.n_filters, -1)


class CompactGaborFEB(nn.Module):
    """Compact Gabor Feature Extraction Block."""

    def __init__(self, in_ch: int = 4, n_filters: int = 8, out_dim: int = 32):
        super().__init__()
        self.gabor = ParametricGabor1D(in_ch=in_ch, n_filters=n_filters)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch * n_filters, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        x = self.pool(x).squeeze(-1)
        return F.relu6(self.proj(x))


class ImprovedStudent(nn.Module):
    """Final lightweight student network for sleep-stage classification.

    Architecture: Multi-Res Stem → Depthwise-Separable CNN → Gabor FEB → GRU → 5-class
    Parameters: ~99,477
    Context: 10 epochs × 30s = 300s
    """

    def __init__(self, n_classes: int = 5, gru_hidden: int = 64,
                 n_channels: int = 4, fs: int = 100):
        super().__init__()
        self.n_channels = n_channels
        self.fs = fs

        self.stem = LiteMultiResolutionStem(in_ch=n_channels, width=10, fs=fs)

        self.encoder = nn.Sequential(
            DepthwiseSeparableConv1d(self.stem.out_ch, 32, kernel=5, stride=2),
            DepthwiseSeparableConv1d(32, 32, kernel=5, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gabor_feb = CompactGaborFEB(in_ch=n_channels, n_filters=8, out_dim=32)

        feature_dim = 32 + 32
        self.gru = nn.GRU(feature_dim, gru_hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(gru_hidden, n_classes)
        self.feature_dim = feature_dim

    def forward(self, x, return_features=False):
        b, t, c, s = x.shape
        flat = x.reshape(b * t, c, s)

        cnn = self.pool(self.encoder(self.stem(flat))).squeeze(-1)
        gabor = self.gabor_feb(flat)

        features = torch.cat([cnn, gabor], dim=-1).reshape(b, t, self.feature_dim)
        sequence_out, _ = self.gru(features)
        logits = self.head(sequence_out)

        if return_features:
            return logits, features
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
