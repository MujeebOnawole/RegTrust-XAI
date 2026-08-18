"""Standalone copy of RegTrust-XAI's model.py for the public demo app --
architecture and occlusion-attribution code only, no config.py dependency
(checkpoints carry their own arch dict, so nothing here needs project-wide
constants at inference time). Kept byte-for-byte equivalent to the source
project's model.py; see that file for the full design rationale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CHANNELS = 4      # one-hot ACGT
POOL_EVERY = 2       # matches config.POOL_EVERY in the source project


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class Seq2AccessibilityCNN(nn.Module):
    """Input: (batch, N_CHANNELS, seq_len) one-hot sequence.
    Output: (batch,) predicted accessibility scalar (standardized target)."""

    def __init__(self, channels, kernel, dropout):
        super().__init__()
        blocks = []
        in_ch = N_CHANNELS
        for i, out_ch in enumerate(channels):
            blocks.append(ConvBlock(in_ch, out_ch, kernel, pool=(i % POOL_EVERY == POOL_EVERY - 1)))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, x):
        h = self.blocks(x)
        h = self.pool(h).squeeze(-1)
        return self.head(h).squeeze(-1)

    def embed(self, x):
        """(batch, in_ch) pooled pre-head feature vector -- the latent space
        the applicability-domain axis measures cosine distance in."""
        h = self.blocks(x)
        return self.pool(h).squeeze(-1)


def occlusion_attribution(model, x, window_stride=32, window_width=32, device="cpu"):
    """Per-position attribution by sliding-window occlusion. x: (N_CHANNELS,
    seq_len) single example, one-hot. Returns (n_windows,) attribution array.
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        base_pred = model(x.unsqueeze(0)).item()
        seq_len = x.shape[-1]
        uniform = torch.full_like(x[:, :window_width], 0.25)
        attrs = []
        for start in range(0, seq_len - window_width + 1, window_stride):
            x_occ = x.clone()
            x_occ[:, start:start + window_width] = uniform
            p = model(x_occ.unsqueeze(0)).item()
            attrs.append(base_pred - p)
    return attrs
