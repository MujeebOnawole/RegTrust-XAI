"""
RegTrust-XAI phase 1 model: a compact 1D CNN over one-hot DNA sequence,
predicting a single pooled chromatin-accessibility scalar per window.

WHY THIS ARCHITECTURE, NOT A PORT FROM ProtTrust-XAI OR CancerTrust-XAI.
Unlike those two ports, nothing here is reused code — a seq2func model has no
graph, no contact map, no fingerprint. This is new architecture, budgeted as
such (see project_status.md). Deliberately ChromBPNet-scale (a handful of
conv layers, ~kb receptive field) rather than Enformer/Borzoi-scale
(transformer trunk, ~100kb receptive field, hundreds of millions of
parameters): the review this project cites (Nagai et al. 2026) notes
"architectural comparisons rarely produce decisive winners" and that
well-tuned CNNs match transformer-based models on this task class. A single-
GPU-trainable model is the right choice for a phase-1 demonstration of the
trust framework, not a chase for state-of-the-art accuracy.

OUTPUT. Phase 1 predicts one pooled scalar (mean accessibility signal over
the window), not a base-pair-resolution profile. A profile head is a phase-2
extension if the scalar model shows the ceiling matters, the same arbiter
logic CancerTrust-XAI and ProtTrust-XAI both use before adding model
complexity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CONV_CHANNELS, CONV_KERNEL, DROPOUT, N_CHANNELS, POOL_EVERY


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
    """Input: (batch, N_CHANNELS, WINDOW_BP) one-hot sequence.
    Output: (batch,) predicted accessibility scalar (standardized target,
    matching the z-scoring convention ProtTrust-XAI and CancerTrust-XAI both
    use — fit on the training pool only, never on validation/test)."""

    def __init__(self, channels=None, kernel=CONV_KERNEL, dropout=DROPOUT):
        super().__init__()
        channels = channels or CONV_CHANNELS
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
        the applicability-domain axis (trust.nn_distance) measures cosine
        distance in, same role as ProtTrust-XAI's embedding-distance-to-
        nearest-training-example formula, ported unchanged (trust.py)."""
        h = self.blocks(x)
        return self.pool(h).squeeze(-1)


def occlusion_attribution(model, x, window_stride=32, window_width=32, device="cpu"):
    """Per-position attribution by sliding-window occlusion (set a window of
    positions to the all-zero/uniform base-composition vector, record the
    change in prediction), the same causal, architecture-agnostic strategy
    ProtTrust-XAI (residue occlusion) and the Taste RGCN (scaffold masking)
    both use, rather than a gradient-based method — for the same reason
    stated in the taste paper: a perturbation-based signal does not depend on
    any assumption that gradients/attention are meaningful in the first
    place. window_width=32 matches config.BIN_BP, a motif-scale rather than
    single-nucleotide granularity, since single-base occlusion is expensive
    and motif-scale is what the coherence axis and the perturbation-agreement
    axis both operate on.

    x: (N_CHANNELS, WINDOW_BP) single example, one-hot.
    Returns: (n_windows,) attribution array, one value per occluded window.
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
