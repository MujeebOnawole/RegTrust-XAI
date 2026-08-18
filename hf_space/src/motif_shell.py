"""Standalone copy of RegTrust-XAI's motif_shell.py for the public demo app
-- the K562-relevant JASPAR TF panel and motif-scan functions the coherence
axis needs, with the config.py/GENOME_2BIT-dependent __main__ self-test
removed (that check runs against the full training genome, not something the
demo app ships). See the source project's motif_shell.py for the full
rationale behind the panel choice and the reverse-complement-scanning bug it
documents.
"""
from __future__ import annotations

import numpy as np
from Bio import motifs
from Bio.Seq import Seq

K562_TF_PANEL = [
    "GATA1", "GATA1::TAL1", "TAL1::TCF3", "KLF1", "NFE2", "MAF::NFE2",
    "GATA2", "RUNX1", "MYB", "STAT5A::STAT5B",
]

MOTIF_FPR = 0.001
MOTIF_PSEUDOCOUNT = 0.1
MOTIF_DISTRIBUTION_PRECISION = 10 ** 3


def load_k562_pssms(jaspar_path, panel=K562_TF_PANEL,
                     pseudocount=MOTIF_PSEUDOCOUNT, fpr=MOTIF_FPR,
                     precision=MOTIF_DISTRIBUTION_PRECISION):
    """name -> (forward_pssm, reverse_complement_pssm, threshold)."""
    with open(jaspar_path) as f:
        recs = list(motifs.parse(f, "jaspar"))
    by_name_ci = {m.name.upper(): m for m in recs}

    out = {}
    missing = []
    for tf in panel:
        m = by_name_ci.get(tf.upper())
        if m is None:
            missing.append(tf)
            continue
        pwm = m.counts.normalize(pseudocounts=pseudocount)
        fwd = pwm.log_odds()
        rev = fwd.reverse_complement()
        threshold = fwd.distribution(precision=precision).threshold_fpr(fpr)
        out[tf] = (fwd, rev, float(threshold))
    if missing:
        print(f"  WARNING: {len(missing)} K562_TF_PANEL entries not found in "
              f"{jaspar_path}: {missing}")
    return out


def motif_hits(seq: str, pssms: dict):
    """List of (tf_name, strand, start, end) for every motif hit above that
    TF's FPR=0.001 threshold -- used by the app to show WHICH transcription
    factors matched, not just a coverage boolean."""
    seq_obj = Seq(seq.upper())
    n = len(seq)
    hits = []
    for name, (fwd, rev, threshold) in pssms.items():
        length = fwd.length
        if length > n:
            continue
        for strand, pssm in (("+", fwd), ("-", rev)):
            scores = pssm.calculate(seq_obj)
            scores = np.atleast_1d(np.asarray(scores, dtype=np.float64))
            for start in np.where(scores >= threshold)[0]:
                hits.append((name, strand, int(start), int(start) + length))
    return hits


def motif_hit_positions_from_hits(hits, n):
    covered = np.zeros(n, dtype=bool)
    for _name, _strand, start, end in hits:
        covered[start:end] = True
    return covered


def motif_coverage_by_bin(seq: str, pssms: dict, bin_width: int, bin_stride: int):
    """Indices of occlusion bins that overlap at least one motif hit -- the
    coherence axis's "shell". Returns (shell_bin_indices, raw_hits)."""
    hits = motif_hits(seq, pssms)
    n = len(seq)
    covered = motif_hit_positions_from_hits(hits, n)
    shell = []
    bin_idx = 0
    for start in range(0, n - bin_width + 1, bin_stride):
        if covered[start:start + bin_width].any():
            shell.append(bin_idx)
        bin_idx += 1
    return shell, hits
