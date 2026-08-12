"""
k-mer composition and divergence utilities for item 9 rung 1a (composition-
divergence stratification of the chromosome-holdout test set).

WHY THIS EXISTS, SEPARATE FROM model.embed()'S AD DISTANCE. Rung 1a needs a
severity axis that does NOT depend on the model's own learned representation
-- using AD distance to prove AD distance predicts error would be circular.
k-mer frequency composition is a plain, model-independent property of the
raw sequence, computable before any model exists, giving an independent
check on whether "sequence divergence from the training distribution"
predicts error at all, not just whether this one model's embedding does.
"""
from __future__ import annotations

import itertools

import numpy as np

ALPHABET = "ACGT"


def kmer_index(k: int) -> dict[str, int]:
    """All 4**k k-mers -> a stable integer index, lexicographic order."""
    return {"".join(t): i for i, t in enumerate(itertools.product(ALPHABET, repeat=k))}


def kmer_counts(seq: str, k: int, index: dict[str, int]) -> np.ndarray:
    """(4**k,) raw counts of each k-mer in seq (sliding window, stride 1).
    k-mers containing an ambiguous/N base are skipped, not counted as any
    bin -- same "no information here" convention data_module.one_hot_encode
    uses for N bases, rather than silently corrupting a real bin's count."""
    counts = np.zeros(len(index), dtype=np.float64)
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        idx = index.get(seq[i:i + k])
        if idx is not None:
            counts[idx] += 1
    return counts


def kmer_distribution(seq: str, k: int, index: dict[str, int], pseudocount: float = 1.0) -> np.ndarray:
    """(4**k,) probability distribution, Laplace-smoothed (pseudocount) so a
    window that happens to miss a k-mer entirely doesn't produce a zero that
    breaks Jensen-Shannon divergence -- a real property of short windows at
    k=4+ (256 possible 4-mers, a 2048bp window has ~2045 draws, so some
    4-mers are legitimately absent by chance, not a data problem)."""
    counts = kmer_counts(seq, k, index) + pseudocount
    return counts / counts.sum()


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Base-2 JSD, bounded [0, 1] -- symmetric, always finite (unlike raw KL
    divergence, which is why JSD rather than KL is used to compare a
    window's own k-mer distribution against the training-pool's aggregate
    one: some pool-rare k-mers legitimately have near-zero mass, and KL
    diverges to infinity on any such mismatch, which JSD does not)."""
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


if __name__ == "__main__":
    # Quick self-check: a window drawn from the same composition as the
    # reference should score near-zero JSD; a deliberately skewed window
    # (e.g. poly-A) should score high. Not a unit test framework, just a
    # sanity check run once before trusting this module, same discipline
    # motif_shell.py's own __main__ block uses.
    rng = np.random.default_rng(0)
    idx = kmer_index(4)
    real_like = "".join(rng.choice(list(ALPHABET), size=2048, p=[0.3, 0.2, 0.2, 0.3]))
    same_dist = "".join(rng.choice(list(ALPHABET), size=2048, p=[0.3, 0.2, 0.2, 0.3]))
    poly_a = "A" * 2048

    ref_dist = kmer_distribution(real_like, 4, idx)
    same_jsd = jensen_shannon_divergence(kmer_distribution(same_dist, 4, idx), ref_dist)
    polya_jsd = jensen_shannon_divergence(kmer_distribution(poly_a, 4, idx), ref_dist)
    print(f"same-composition JSD: {same_jsd:.4f} (expect small)")
    print(f"poly-A JSD: {polya_jsd:.4f} (expect large, near max)")
    assert same_jsd < 0.05, "same-composition control should score near zero"
    assert polya_jsd > same_jsd * 5, "poly-A should score much higher than same-composition control"
    print("sanity checks passed")
