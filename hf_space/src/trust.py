"""Standalone copy of the trust-axis formulas RegTrust-XAI's public demo
needs at inference time (consensus, coherence, applicability domain, and
scenario labelling) -- a subset of the source project's trust.py, with the
config.py-dependent defaults inlined as explicit call-site arguments instead.
Formulas are unchanged; see the source project's trust.py for full derivation
notes and the validation results behind each cutoff.
"""
from __future__ import annotations

import numpy as np


def cv_ratio_from_stats(ensemble_std, pop_std):
    return ensemble_std / pop_std if pop_std > 1e-8 else ensemble_std


def localization_coherence(window_attr, motif_shell_bins):
    """Mean percentile rank of |attribution| over motif_shell_bins. Returns
    None if motif_shell_bins is empty (no motif hit anywhere in this window)
    -- callers must treat coherence, and therefore the scenario label, as
    unavailable rather than default it to a value."""
    if not motif_shell_bins:
        return None
    a = np.abs(np.asarray(window_attr))
    n = len(a)
    if n <= 1:
        return 0.5
    order = a.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    pct = ranks / (n - 1)
    vals = [pct[i] for i in motif_shell_bins if 0 <= i < n]
    return float(np.mean(vals)) if vals else 0.5


def l2_normalize_rows(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)


def nn_distance(query_vec, ref_matrix, exclude_idx=None):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    sims = ref_matrix @ q
    if exclude_idx is not None:
        sims = sims.copy()
        sims[exclude_idx] = -np.inf
    return float(1.0 - sims.max())


def scenario_labels(cv_ratio, coherence, cons_cut, agr_cut):
    return np.where((cv_ratio <= cons_cut) & (coherence >= agr_cut), "A",
           np.where(cv_ratio <= cons_cut, "B",
           np.where(coherence >= agr_cut, "C", "D")))
