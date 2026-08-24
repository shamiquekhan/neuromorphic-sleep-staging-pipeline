"""Reusable building blocks for the Improved Student architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteMultiResolutionStem(nn.Module):
    """Parallel short/long receptive-field stems.

    Uses plain Conv1d + BatchNorm1d (not DepthwiseSeparableConv1d).
    """

    def __init__(self, in_ch: int = 4, width: int = 8, fs: int = 100):
        super().__init__()
        small_kernel, small_stride = fs // 4, max(1, fs // 16)
        large_kernel, large_stride = fs * 2, max(1, fs // 4)

        self.stem_s = nn.Sequential(
            nn.Conv1d(in_ch, width, small_kernel, stride=small_stride, bias=False),
            nn.BatchNorm1d(width),
        )
        self.stem_l = nn.Sequential(
            nn.Conv1d(in_ch, width, large_kernel, stride=large_stride, bias=False),
            nn.BatchNorm1d(width),
        )
        self.out_ch = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = F.relu6(self.stem_s[1](self.stem_s[0](x)))
        long = F.relu6(self.stem_l[1](self.stem_l[0](x)))
        target = min(short.shape[-1], long.shape[-1])
        short = F.adaptive_avg_pool1d(short, target)
        long = F.adaptive_avg_pool1d(long, target)
        return torch.cat([short, long], dim=1)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.dw = nn.Conv1d(
            in_ch, in_ch, kernel, stride=stride, padding=padding,
            groups=in_ch, bias=False,
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(self.bn(self.pw(self.dw(x))))


class ParametricGaborFEB(nn.Module):
    """Parametric Gabor Feature Extraction Block.

    Averages across channels, applies learnable Gabor filter bank,
    pools to a single vector, and projects to output dimension.
    """

    def __init__(self, n_filters: int = 8, out_dim: int = 16,
                 kernel_size: int = 51, fs: int = 100):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.gabor_freq = nn.Parameter(
            torch.linspace(0.5, 30.0, n_filters) / fs
        )
        self.gabor_sigma = nn.Parameter(torch.full((n_filters,), 0.02))

        t = torch.arange(
            -(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32
        )
        self.register_buffer("t", t)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gab_proj = nn.Linear(n_filters, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape [B, C, S]. Returns: [B, out_dim]."""
        t = self.t.unsqueeze(0)
        f0 = self.gabor_freq.unsqueeze(1)
        sigma = (self.gabor_sigma.abs() + 1e-4).unsqueeze(1)

        envelope = torch.exp(-0.5 * (t / (sigma * self.kernel_size)) ** 2)
        carrier = torch.cos(2 * math.pi * f0 * t)
        kernels = (envelope * carrier).unsqueeze(1)

        x_mean = x.mean(dim=1, keepdim=True)
        y = F.conv1d(x_mean, kernels, padding=self.kernel_size // 2)
        y = self.pool(y).squeeze(-1)
        return self.gab_proj(y)
