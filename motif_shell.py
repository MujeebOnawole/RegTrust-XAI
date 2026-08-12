"""
Motif-based coherence shell for RegTrust-XAI's accessibility model.

WHY THIS EXISTS. trust.py's localization_coherence() was flagged PROVISIONAL:
it only measured whether |attribution| concentrates in a small subset of the
window, with no external prior for what that subset SHOULD be -- weaker than
ProtTrust-XAI's buried-core shell or CancerTrust-XAI's target-gene shell,
both of which check attribution against something known to matter, not just
"is it concentrated". This closes that gap the same way CancerTrust-XAI's
coherence shell was strengthened on 2026-08-11 (see its PHASE1_MODEL_SCOPE.md
"Coherence shell network expansion"): give the axis a real, external,
verifiable prior -- known TF binding motif occurrences -- instead of a
self-referential statistic.

THE SHELL: JASPAR PWM matches for a curated, K562-relevant TF panel, not the
full ~1,000-motif JASPAR vertebrate collection. Scanning for every
vertebrate TF's motif regardless of whether it is expressed in this cell
type would make "some motif present somewhere" nearly guaranteed in any
window by chance, diluting the prior to meaninglessness -- the same reason
CancerTrust-XAI's network-expanded shell is capped rather than unbounded.
K562 is a CML-derived, BCR-ABL-positive erythroleukemia line; its
accessibility landscape is expected to be organised substantially around
the erythroid/megakaryocytic master regulators, not an arbitrary TF sample.

K562_TF_PANEL is deliberately short and named for a reason each entry is
included, not a "grab everything with a plausible name" list:
  GATA1, GATA1::TAL1  -- the master erythroid regulator and its core dimer
                          partner; JASPAR carries the GATA1::TAL1 composite
                          motif directly, which is more specific than either
                          alone.
  TAL1::TCF3          -- TAL1's other major dimerisation partner.
  KLF1                -- erythroid Kruppel-like factor, GATA1 cofactor.
  NFE2, MAF::NFE2      -- erythroid/megakaryocytic bZIP regulator.
  GATA2                -- upstream of GATA1 in the erythroid program,
                          relevant to K562's progenitor-like state.
  RUNX1                -- megakaryocytic/haematopoietic regulator, and
                          clinically relevant to K562's own lineage biology.
  MYB                  -- haematopoietic progenitor regulator.
  STAT5A::STAT5B        -- downstream of BCR-ABL signalling, the defining
                          lesion of the K562 line itself.

REVERSE-COMPLEMENT SCANNING IS MANDATORY, NOT OPTIONAL, verified 2026-08-11
by a real bug that would otherwise have silently returned near-zero
coverage: JASPAR stores each PFM in whatever orientation it was originally
reported in, not necessarily the "biological" reading direction. GATA1's
own JASPAR matrix (MA0035) has consensus CTAATCT on the strand it is
stored, and only its reverse complement (AGATTAG orientation) scores above
threshold against a canonical WGATAR-type site -- confirmed directly: a
synthetic sequence containing "TGATAAGG" scored -5.4 on the forward PSSM
and +8.4 on the reverse-complement PSSM (FPR=0.001 threshold was 6.95).
Every scan below computes both strands and takes the max, the same
correctness requirement any real motif scanner (FIMO, MOODS, etc.) enforces
by default.
"""
from __future__ import annotations

import numpy as np
from Bio import motifs
from Bio.Seq import Seq

from config import GENOME_2BIT

K562_TF_PANEL = [
    "GATA1", "GATA1::TAL1", "TAL1::TCF3", "KLF1", "NFE2", "MAF::NFE2",
    "GATA2", "RUNX1", "MYB", "STAT5A::STAT5B",
]

MOTIF_FPR = 0.001            # standard false-positive-rate threshold convention (FIMO/MOODS-like)
MOTIF_PSEUDOCOUNT = 0.1
MOTIF_DISTRIBUTION_PRECISION = 10 ** 3   # controls threshold_fpr's accuracy; 1e3 is fast (~0.1s/motif)
                                            # and sufficient at FPR=0.001 -- not pushed higher without need


def load_k562_pssms(jaspar_path, panel=K562_TF_PANEL,
                     pseudocount=MOTIF_PSEUDOCOUNT, fpr=MOTIF_FPR,
                     precision=MOTIF_DISTRIBUTION_PRECISION):
    """name -> (forward_pssm, reverse_complement_pssm, threshold).

    Only TFs actually found in the JASPAR file are returned; a missing panel
    entry shrinks the shell, it is never silently substituted with something
    else. Verified 2026-08-11: all 10 panel entries resolve against
    JASPAR2026_CORE_vertebrates_non-redundant_pfms.txt.
    """
    with open(jaspar_path) as f:
        recs = list(motifs.parse(f, "jaspar"))
    # JASPAR names are NOT reliably all-caps (verified 2026-08-11: RUNX1 is
    # stored as "Runx1", STAT5A::STAT5B as "Stat5a::Stat5b") -- an exact-case
    # lookup silently dropped 2 of 10 panel entries. Match case-insensitively,
    # keep the file's own original casing for reporting/provenance.
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


def motif_hit_positions(seq: str, pssms: dict) -> np.ndarray:
    """Boolean array, length len(seq), True at any position covered by a
    motif hit (either strand, any panel TF) scoring above that TF's own
    FPR=0.001 threshold. A hit at position i covers [i, i+motif_length)."""
    seq_obj = Seq(seq.upper())
    n = len(seq)
    covered = np.zeros(n, dtype=bool)
    for name, (fwd, rev, threshold) in pssms.items():
        length = fwd.length
        if length > n:
            continue
        for pssm in (fwd, rev):
            scores = pssm.calculate(seq_obj)
            scores = np.atleast_1d(np.asarray(scores, dtype=np.float64))
            hits = np.where(scores >= threshold)[0]
            for start in hits:
                covered[start:start + length] = True
    return covered


def motif_coverage_by_bin(seq: str, pssms: dict, bin_width: int, bin_stride: int) -> list[int]:
    """The actual coherence "shell": indices of occlusion bins (see
    model.py's occlusion_attribution, same bin_width/bin_stride convention)
    that overlap at least one motif hit. Returned as a plain index list, the
    same shape trust.localization_coherence expects (mirrors CancerTrust-
    XAI's shell_indices -> localization_coherence(attr, shell_indices))."""
    covered = motif_hit_positions(seq, pssms)
    n = len(seq)
    shell = []
    bin_idx = 0
    for start in range(0, n - bin_width + 1, bin_stride):
        if covered[start:start + bin_width].any():
            shell.append(bin_idx)
        bin_idx += 1
    return shell


if __name__ == "__main__":
    import random

    import py2bit
    from scipy import stats as sstats

    from build_features import load_peaks, negative_windows, peak_intervals_by_chrom
    from config import ACCESSIBILITY_PEAKS_BED, BIN_BP, JASPAR_PFM_PATH, PRIMARY_CHROMS

    print("loading K562 TF panel PSSMs...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    print("loading real peaks and a CLEAN negative control (build_features.py's own "
          "any-peak-overlap rejection, not just avoiding the sampled peak itself)...")
    peaks = load_peaks(path=ACCESSIBILITY_PEAKS_BED)
    intervals_by_chrom = peak_intervals_by_chrom(peaks)
    tb = py2bit.open(str(GENOME_2BIT))
    chrom_sizes = {c: s for c, s in tb.chroms().items() if c in PRIMARY_CHROMS}

    rng = random.Random(1)
    sample_peaks = rng.sample(peaks, 500)
    neg_windows = negative_windows(500, list(chrom_sizes.keys()), chrom_sizes,
                                    intervals_by_chrom, window_bp=400, seed=1)

    def frac_bins_covered(chrom, start, end, width=400):
        mid = (start + end) // 2
        seq = tb.sequence(chrom, max(0, mid - width // 2), mid + width // 2)
        shell = motif_coverage_by_bin(seq, pssms, bin_width=BIN_BP, bin_stride=BIN_BP)
        n_bins = max(1, len(seq) // BIN_BP)
        return len(shell) / n_bins

    peak_cov = [frac_bins_covered(c, s, e) for c, s, e, _ in sample_peaks]
    neg_cov = [frac_bins_covered(c, s, e) for c, s, e in neg_windows]
    tb.close()

    t, p = sstats.ttest_ind(peak_cov, neg_cov)
    print(f"\nmean motif-bin coverage, real peak-centered regions (n={len(peak_cov)}): "
          f"{np.mean(peak_cov):.3f} +/- {np.std(peak_cov):.3f}")
    print(f"mean motif-bin coverage, clean negative windows, no peak overlap (n={len(neg_cov)}): "
          f"{np.mean(neg_cov):.3f} +/- {np.std(neg_cov):.3f}")
    print(f"two-sample t-test: t={t:.2f}, p={p:.2e}")
