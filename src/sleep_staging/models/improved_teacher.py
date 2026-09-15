"""Improved Teacher — larger model for knowledge distillation.

Multi-resolution stem → residual encoder → learned spectral filterbank
→ gated fusion → 2-layer transformer encoder.

Used only during training; not exported for deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 1)),
            nn.ReLU(),
            nn.Linear(max(ch // r, 1), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.se = SEBlock1D(ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.net(x)) + x)


class ImprovedTeacher(nn.Module):
    """Teacher model for knowledge distillation."""

    def __init__(self, in_ch=4, n_cls=5, fs=100):
        super().__init__()
        self.stem_short = nn.Sequential(
            nn.Conv1d(in_ch, 16, fs // 4, stride=fs // 8, padding=fs // 8, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.stem_long = nn.Sequential(
            nn.Conv1d(in_ch, 16, fs * 2, stride=fs // 2, padding=fs, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ResBlock1D(32),
            ResBlock1D(32),
            nn.AdaptiveAvgPool1d(32),
        )
        self.fb_filters = nn.Parameter(torch.randn(32, 129))
        self.fb_proj = nn.Linear(32, 64)
        self.gate_w = nn.Linear(32 + 64, 96)
        enc_layer = nn.TransformerEncoderLayer(96, 4, 192, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, 2)
        self.head = nn.Linear(96, n_cls)
        self.feature_dim = 96

    def forward(self, x, return_features=False):
        B, T, C, S = x.shape
        xf = x.reshape(B * T, C, S)
        ts = self.stem_short(xf)
        tl = self.stem_long(xf)
        L = min(ts.shape[-1], tl.shape[-1])
        t_feat = self.encoder(torch.cat([ts[..., :L], tl[..., :L]], 1)).reshape(B * T, 32, 32).mean(-1)
        spec = torch.fft.rfft(xf[:, 0, :], dim=-1).abs()[:, :129]
        fb = F.softmax(self.fb_filters, -1)
        f_feat = self.fb_proj((spec.unsqueeze(1) * fb.unsqueeze(0)).sum(-1))
        gate = torch.sigmoid(self.gate_w(torch.cat([t_feat, f_feat], -1)))
        fused = torch.cat([t_feat, f_feat], -1) * gate
        seq = fused.reshape(B, T, 96)
        out = self.transformer(seq)
        logits = self.head(out)
        if return_features:
            return logits, fused.reshape(B, T, 96)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)