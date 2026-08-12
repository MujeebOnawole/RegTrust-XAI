#!/usr/bin/env python
"""Phase-1 feature builder for RegTrust-XAI: DNA sequence windows -> chromatin
accessibility label, from the ENCODE K562 ATAC-seq data verified in
notes/data_sources.md.

WHAT THIS PRODUCES. A lightweight windows table (chrom, start, end, label,
is_positive), NOT pre-materialized one-hot sequence arrays. One-hot encoding
a genome-wide window set up front would be enormous for no benefit (~500k
windows x 2048bp x 4 channels x 4 bytes is several GB), so sequence is read
on demand from the 2bit genome at train time -- the same cache/read-in-worker
discipline ENZYME_XAI's data_module.py uses for AlphaFold structures, applied
here to genome coordinates instead.

DEPENDENCIES: py2bit, pyBigWig. Confirmed installable and correctly reading
the real downloaded files locally 2026-08-11 (see notes/data_sources.md) --
py2bit opened hg38.2bit with correct chromosome lengths and returned the
expected telomeric repeat at a test coordinate; pyBigWig read a real fold-
change value consistent with the peaks file's own signalValue at the same
locus. NEITHER IS YET CONFIRMED PRESENT in whatever container this project
ends up using on Bunya -- the preflight import check below fails fast if
missing, rather than partway through the slow step, the same pattern as
every slurm preflight in CancerTrust-XAI and ENZYME_XAI.

POSITIVE WINDOWS: centered on each ENCODE narrowPeak's summit
(start + column-10 offset, or the peak midpoint if that column is -1), width
WINDOW_BP, dropped if the window would run off the end of its chromosome
rather than clipped (clipping would silently shrink the window and break the
one-hot shape downstream).

NEGATIVE WINDOWS: sampled uniformly at random from the same primary
chromosomes, rejected if they overlap ANY peak interval (not just the
positive windows actually built from it), so a negative is genuinely
non-accessible, not merely off-summit. config.NEG_TO_POS_RATIO negatives per
positive, rejection-sampled with a bounded number of tries so a
pathologically peak-dense genome cannot hang the job silently.

LABEL: mean bigWig fold-change-over-control signal across the window (a
continuous regression target), not the binary peak/no-peak call -- the
Nagai et al. 2026 review is explicit that quantitative profile supervision
captures more of the cis-regulatory signal than a binary label does, and
this matches the convention ENCODE's own ChromBPNet pipeline uses.

RUN AS A BATCH JOB (once slurm/1_build_features.sh exists): reading the
bigWig at ~500k window coordinates is the slow step; local testing on a
truncated peak set showed no correctness issues (see project_status.md), but
full-genome wall time has not been measured yet.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

from config import (
    ACCESSIBILITY_PEAKS_BED,
    ACCESSIBILITY_SIGNAL_BW,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    NEG_TO_POS_RATIO,
    PRIMARY_CHROMS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    WINDOW_BP,
)


# --------------------------------------------------------------------------------------
# peaks
# --------------------------------------------------------------------------------------
def load_peaks(path=ACCESSIBILITY_PEAKS_BED, chroms=PRIMARY_CHROMS):
    """ENCODE narrowPeak, confirmed by direct inspection (notes/data_sources.md):
    chrom  start  end  name  score  strand  signalValue  pValue  qValue  summit_offset
    (10 columns, tab-separated, no header)."""
    chroms = set(chroms)
    peaks = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            if chrom not in chroms:
                continue
            summit_offset = int(parts[9]) if len(parts) > 9 and parts[9] not in ("-1", "") else (end - start) // 2
            summit = start + summit_offset
            peaks.append((chrom, start, end, summit))
    return peaks


def peak_intervals_by_chrom(peaks):
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end, _ in peaks:
        by_chrom.setdefault(chrom, []).append((start, end))
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    return by_chrom


def overlaps_any(chrom, start, end, intervals_by_chrom):
    """Linear scan with an early sorted-order break. This runs once per
    negative-window CANDIDATE during preprocessing, not in a training loop,
    so it does not need to be an interval tree at this scale (a few hundred
    thousand peaks)."""
    for s, e in intervals_by_chrom.get(chrom, []):
        if s > end:
            break  # sorted by start; nothing further can overlap
        if start < e and end > s:
            return True
    return False


# --------------------------------------------------------------------------------------
# window construction
# --------------------------------------------------------------------------------------
def positive_windows(peaks, window_bp, chrom_sizes):
    out = []
    half = window_bp // 2
    dropped = 0
    for chrom, _, _, summit in peaks:
        start = summit - half
        end = start + window_bp
        size = chrom_sizes.get(chrom)
        if size is None or start < 0 or end > size:
            dropped += 1
            continue
        out.append((chrom, start, end))
    if dropped:
        print(f"  dropped {dropped} peaks whose window would run off a chromosome end")
    return out


def negative_windows(n_needed, chroms, chrom_sizes, intervals_by_chrom, window_bp, seed):
    import random
    rng = random.Random(seed)
    chroms = [c for c in chroms if chrom_sizes.get(c, 0) > window_bp]
    out = []
    tries, max_tries = 0, max(1000, n_needed * 50)
    while len(out) < n_needed and tries < max_tries:
        tries += 1
        chrom = rng.choice(chroms)
        size = chrom_sizes[chrom]
        start = rng.randint(0, size - window_bp)
        end = start + window_bp
        if overlaps_any(chrom, start, end, intervals_by_chrom):
            continue
        out.append((chrom, start, end))
    if len(out) < n_needed:
        print(f"  WARNING: only sampled {len(out)}/{n_needed} negatives after {tries} tries")
    return out


# --------------------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------------------
def label_windows(windows, bw_path):
    import pyBigWig
    bw = pyBigWig.open(str(bw_path))
    labels = np.empty(len(windows), dtype=np.float32)
    for i, (chrom, start, end) in enumerate(windows):
        v = bw.stats(chrom, start, end, type="mean")[0]
        labels[i] = float(v) if v is not None else 0.0
    bw.close()
    return labels


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--peaks", type=Path, default=ACCESSIBILITY_PEAKS_BED,
                     help="override for smoke-testing on a truncated peak file")
    args = ap.parse_args()

    print("preflight: checking required inputs and imports...")
    for p in (args.peaks, ACCESSIBILITY_SIGNAL_BW, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; see notes/data_sources.md.")
    import py2bit
    import pyBigWig  # noqa: F401  (checked here; actually used inside label_windows)

    tb = py2bit.open(str(GENOME_2BIT))
    chrom_sizes = {c: s for c, s in tb.chroms().items() if c in PRIMARY_CHROMS}
    tb.close()
    print(f"  primary chromosomes with sizes resolved: {len(chrom_sizes)}/{len(PRIMARY_CHROMS)}")

    print(f"loading peaks from {args.peaks} ...")
    peaks = load_peaks(path=args.peaks)
    print(f"  {len(peaks)} peaks on primary chromosomes")
    intervals_by_chrom = peak_intervals_by_chrom(peaks)

    pos = positive_windows(peaks, WINDOW_BP, chrom_sizes)
    print(f"  {len(pos)} positive windows")

    n_neg = int(round(len(pos) * NEG_TO_POS_RATIO))
    neg = negative_windows(n_neg, list(chrom_sizes.keys()), chrom_sizes, intervals_by_chrom,
                            WINDOW_BP, args.seed)
    print(f"  {len(neg)} negative windows sampled")

    windows = pos + neg
    is_positive = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8)

    print(f"reading bigWig signal for {len(windows)} windows (the slow step)...")
    labels = label_windows(windows, ACCESSIBILITY_SIGNAL_BW)

    chrom_arr = np.array([w[0] for w in windows])
    start_arr = np.array([w[1] for w in windows], dtype=np.int64)
    end_arr = np.array([w[2] for w in windows], dtype=np.int64)

    n_holdout = int(np.isin(chrom_arr, HOLDOUT_CHROMS).sum())
    print(f"windows on holdout chromosomes {HOLDOUT_CHROMS}: {n_holdout}/{len(windows)}")
    print(f"label stats: mean={labels.mean():.4f} std={labels.std():.4f} "
          f"min={labels.min():.4f} max={labels.max():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        chrom=chrom_arr,
        start=start_arr,
        end=end_arr,
        label=labels,
        is_positive=is_positive,
        window_bp=WINDOW_BP,
        genome_2bit_path=str(GENOME_2BIT),
    )
    n_pos, n_neg_actual = int(is_positive.sum()), int((1 - is_positive).sum())
    print(f"Saved -> {args.out} ({len(windows)} windows: {n_pos} positive, {n_neg_actual} negative)")


if __name__ == "__main__":
    main()
