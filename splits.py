"""
Chromosome-level OOD splitting for genomic sequence windows.

A random window split lets adjacent, highly similar windows (overlapping or
near-adjacent genomic positions share compositional and regulatory structure)
land on both sides of a fold, which is exactly the leakage Nagai et al. 2026
warn inflates reported seq2func generalization. Splitting at the chromosome
level is the DNA analogue of ProtTrust-XAI's PIDE20 protein-identity-cluster
split and CancerTrust-XAI's OncotreeLineage split: the grouping variable
changes, the discipline (hold out whole groups, never split within one) does
not.

config.HOLDOUT_CHROMS is the internal test. Cross-validation folds are drawn
from the remaining chromosomes, grouped by chromosome so no fold's validation
windows share a chromosome with its training windows.
"""
from __future__ import annotations

import random

from config import CV_FOLDS, HOLDOUT_CHROMS, SPLIT_SEED


def make_test_split(window_chroms: list[str], holdout=HOLDOUT_CHROMS):
    """window_chroms: per-window chromosome label, same length/order as the
    window array. Returns (pool_indices, test_indices)."""
    holdout_set = set(holdout)
    pool = [i for i, c in enumerate(window_chroms) if c not in holdout_set]
    test = [i for i, c in enumerate(window_chroms) if c in holdout_set]
    return pool, test


def cluster_kfold(pool_indices: list[int], window_chroms: list[str],
                   n_folds=CV_FOLDS, seed=SPLIT_SEED):
    """Yield (train_indices, val_indices) folds, grouped by chromosome over the
    pool (post test-holdout). Every fold's validation chromosomes are disjoint
    from its training chromosomes."""
    pool_chroms = sorted({window_chroms[i] for i in pool_indices})
    rng = random.Random(seed)
    rng.shuffle(pool_chroms)
    folds = [set(pool_chroms[i::n_folds]) for i in range(n_folds)]
    for k in range(n_folds):
        val_chroms = folds[k]
        val_idx = [i for i in pool_indices if window_chroms[i] in val_chroms]
        train_idx = [i for i in pool_indices if window_chroms[i] not in val_chroms]
        yield train_idx, val_idx


def split_summary(window_chroms: list[str]) -> str:
    n_windows = len(window_chroms)
    n_chroms = len(set(window_chroms))
    return (f"{n_windows:,} windows across {n_chroms} chromosomes "
            f"(mean {n_windows / max(n_chroms, 1):.1f}/chromosome)")
