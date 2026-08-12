"""
The three trust axes for RegTrust-XAI, plus the fourth, project-specific axis
(explanation-experiment agreement) that differentiates this from a naive port.
See ../TRUST_AWARE_PREDICTION_GUIDE.md before changing any formula here.

Do not "improve" cv_ratio_from_stats or the any-far-is-OOD combine rule
without re-deriving the justification; they are ported unchanged, the same
"port the quantity, not the instrument" discipline used for CancerTrust-XAI.
"""
from __future__ import annotations

import numpy as np

from config import AD_EMBED_PERCENTILE, PERTURBATION_AGREEMENT_TOP_K_PERCENT


# --------------------------------------------------------------------------------
# consensus (axis 1): identical formula to every other port
# --------------------------------------------------------------------------------
def cv_ratio_from_stats(ensemble_std, pop_std):
    return ensemble_std / pop_std if pop_std > 1e-8 else ensemble_std


# --------------------------------------------------------------------------------
# coherence (axis 2) — now has a real external prior (added 2026-08-11)
# --------------------------------------------------------------------------------
def localization_coherence(window_attr, motif_shell_bins):
    """Mean percentile rank of |attribution| over motif_shell_bins (occlusion
    bin indices overlapping a real K562-relevant TF motif hit, see
    motif_shell.py). Same formula shape as CancerTrust-XAI's
    localization_coherence(attr, shell_indices) — the quantity ("does
    attribution concentrate on a known mechanistic prior") is ported
    unchanged across domains; only the shell's construction differs
    (target-gene network neighbourhood there, motif occurrence here).

    REPLACES an earlier, weaker instrument that only measured whether
    attribution concentrates in SOME small subset of the window, with no
    external prior for what that subset should be (see git history / prior
    revision of this function). Verified 2026-08-11 (motif_shell.py's
    standalone check): real ENCODE peak-centered windows show significantly
    higher motif-shell coverage than a clean negative control drawn the same
    way build_features.py samples negatives (t=4.70, p=3.0e-06, n=500 each).
    The effect is real but MODEST in size (62.6% vs 57.2% mean bin coverage)
    -- background motif occurrence at the FPR=0.001 per-motif threshold,
    scanned across ~10 motifs x 2 strands x many positions per window, is
    non-trivial even in genuinely closed chromatin, because DNA sequence
    alone cannot distinguish "motif present" from "motif present AND
    accessible". This is an honest, expected limitation of a sequence-only
    prior, not a bug to fix by tightening the threshold until the gap looks
    better -- see motif_shell.py's module docstring.

    Returns None if motif_shell_bins is empty (no motif hit anywhere in this
    window) -- a real "no shell" case, not an error, and callers must treat
    coherence as unavailable rather than default it to a value, the same
    convention every other shell-based port in this codebase uses.
    """
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


# --------------------------------------------------------------------------------
# applicability domain (axis 3): direct port of the CancerTrust-XAI formulas
# --------------------------------------------------------------------------------
def l2_normalize_rows(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)


def nn_distance(query_vec, ref_matrix, exclude_idx=None):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    sims = ref_matrix @ q
    if exclude_idx is not None:
        sims = sims.copy()
        sims[exclude_idx] = -np.inf
    return float(1.0 - sims.max())


def calibrate_nn_cutoff(ref_matrix, calib_indices, percentile=AD_EMBED_PERCENTILE):
    dists = [nn_distance(ref_matrix[i], ref_matrix, exclude_idx=i) for i in calib_indices]
    return float(np.percentile(dists, percentile)), dists


def is_ood(dist, cutoff):
    return bool(dist > cutoff)


# --------------------------------------------------------------------------------
# EXTERNAL VALIDATION CRITERION (not an inference-time trust axis) — cross-assay
# attribution-activity correlation, see the RENAME/RESTRUCTURE note below.
# --------------------------------------------------------------------------------
# RENAMED AND RECLASSIFIED 2026-08-11, in response to two correct, independent
# points raised about the original "axis 4: explanation-experiment agreement"
# framing (project_status.md previously listed this alongside consensus/
# coherence/AD as a fourth trust axis):
#
# 1. NAMING WAS TOO CAUSAL FOR WHAT THE DATA ACTUALLY SUPPORTS. The lentiMPRA
#    data (notes/data_sources.md) is 53,989 CANDIDATE ELEMENTS, each an
#    independent 200bp sequence with ONE measured log2 activity value in an
#    episomal reporter assay. It is NOT a set of reference/mutant sequence
#    PAIRS demonstrating "mutating base i changed activity by delta y" the
#    way a true saturation-mutagenesis MPRA (e.g. Kircher et al.) would be.
#    Calling agreement with this data "explanation-experiment agreement" or
#    describing it as checking whether "a perturbation actually changed
#    measured activity" overclaims the causal structure actually present.
#    What this data CAN support is a cross-assay correlation: does
#    attribution-weighted importance track which SEQUENCES have high
#    measured activity, not which BASE EDITS caused an activity change.
#    That is still useful and still the least-reproduced part of this
#    project relative to a naive coherence-only port, but it is a WEAKER,
#    correctly-scoped claim, and the function below is renamed to match.
#    Also worth stating plainly (this is exactly the label-shift/cross-assay
#    warning the Nagai et al. 2026 review itself makes): ATAC-seq
#    accessibility and MPRA episomal reporter activity are RELATED but
#    DIFFERENT biological readouts, measured by different assays with
#    different noise structure, so a correlation here is cross-assay
#    evidence, not same-assay validation.
# 2. STRUCTURALLY THIS CANNOT BE A 4TH INFERENCE-TIME AXIS. Consensus,
#    coherence and applicability-domain are all computable for a genuinely
#    NEW query sequence with no known label. This function needs
#    mpra_effect_by_window, which by definition does not exist for a novel
#    deployment-time query. It is therefore an EXTERNAL VALIDATION
#    CRITERION, computed once on the lentiMPRA validation cohort to decide
#    whether the (inference-time-only) coherence axis should be trusted at
#    all -- the same relationship ProtTrust-XAI's four explanation-validity
#    criteria have to its deployed A/B/C/D label: the criteria are computed
#    once, offline, against ground truth, to decide whether the label-
#    producing axis is trustworthy; they are not themselves part of the
#    per-prediction label. The deployed trust taxonomy in this project is
#    and remains THREE axes (consensus, coherence, applicability domain).
def cross_assay_attribution_activity_correlation(
        element_attribution, element_activity, top_k_percent=PERTURBATION_AGREEMENT_TOP_K_PERCENT):
    """Cross-assay validation, NOT an inference-time signal: does attribution
    mass on an MPRA element predict that element's independently measured
    (different-assay) regulatory activity? See the module-level note above
    for exactly what this can and cannot claim.

    THE COMPARISON UNIT IS ONE MPRA ELEMENT, NOT ONE OCCLUSION BIN -- clarified
    2026-08-11 while writing build_mpra_features.py, because the original
    parameter names (window_attr / mpra_activity_by_window) were ambiguous
    between "per-bin within one shared window" and "per-element across many
    elements", and only the second is the operation that makes sense: an MPRA
    element is a standalone 200bp construct tested in isolation, not a labelled
    sub-span of a larger window that shares bins with other elements.

    element_attribution: (n_elements,) ONE scalar per MPRA element, computed
    (once a trained model exists) as the mean |attribution| over the
    occlusion bins that element's 200bp span covers within its own
    element-centred WINDOW_BP window -- see build_mpra_features.py's
    `bin_start`/`bin_end` columns, which record exactly which bins those are.
    element_activity: (n_elements,) that element's measured lentiMPRA log2
    activity (data/raw/lentimpra_k562_ENCFF802FUV.bed column 7, staged per-
    element by build_mpra_features.py -> data/processed/mpra_elements.npz).
    Both arrays must be in the same element order; build_mpra_features.py's
    output preserves a stable element_id column for exactly this join.

    Returns rank correlation (Spearman) between per-element attribution and
    measured activity, restricted to the top-k% of ELEMENTS BY MEASURED
    ACTIVITY MAGNITUDE (the elements with the strongest measured signal,
    mirroring the "top-magnitude blocks" restriction TrustPGS uses and the
    reason given there: asking every element to agree dilutes the signal,
    restricting to where the real signal lives does not). Compute this
    separately per subgroup (K562-native vs HepG2-/WTC11-designed, see
    build_mpra_features.py) rather than pooling all elements, since pooling
    hides exactly the distribution-shift comparison the subgroup split
    exists to enable.
    """
    from scipy import stats
    eff = np.asarray(element_activity)
    attr = np.asarray(element_attribution)
    n_top = max(2, int(np.ceil(len(eff) * top_k_percent / 100.0)))
    top_idx = np.argsort(np.abs(eff))[-n_top:]
    if len(top_idx) < 2:
        return None
    rho = stats.spearmanr(attr[top_idx], eff[top_idx]).statistic
    return float(rho) if np.isfinite(rho) else None


# --------------------------------------------------------------------------------
# generic severity-quantile risk stratification -- shared by validate_trust_axes.py
# (AD distance) and item 9's rung 1a (composition-divergence), any continuous
# "how far/unusual is this window" score against error. Extracted 2026-08-12
# rather than duplicated a second time -- see validate_trust_axes.py's original
# ad_quantile_stratification, now a thin wrapper around this.
# --------------------------------------------------------------------------------
def quantile_stratification(severity, err, edges, thresholds, id_ood_cutoff=None):
    """Bins `severity` (any continuous per-window score, e.g. AD distance or
    composition-divergence JSD) into percentile bands defined by `edges`
    (e.g. [0,20,40,60,80,95,100]) and reports MAE/RMSE/precision-at-threshold
    per band. A single correlation coefficient is a weak way to communicate
    what a severity axis is good for; a monotonic per-band error table is
    what a reader acts on directly.

    id_ood_cutoff, if given, also reports a single in-domain-vs-out-of-domain
    comparison at that cutoff value (severity <= cutoff is "ID")."""
    severity = np.asarray(severity, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    pct_edges = np.percentile(severity, edges)
    bands = []
    for lo_pct, hi_pct, lo_val, hi_val in zip(edges[:-1], edges[1:], pct_edges[:-1], pct_edges[1:]):
        mask = ((severity >= lo_val) & (severity <= hi_val) if hi_pct == edges[-1]
                else (severity >= lo_val) & (severity < hi_val))
        n = int(mask.sum())
        band = {"band_pct": f"{lo_pct:g}-{hi_pct:g}", "n": n,
                "severity_range": [float(lo_val), float(hi_val)]}
        if n > 0:
            band["mae"] = float(np.mean(err[mask]))
            band["rmse"] = float(np.sqrt(np.mean(err[mask] ** 2)))
            band["precision"] = {f"{thr:g}": float((err[mask] <= thr).mean()) for thr in thresholds}
        bands.append(band)
        print(f"  band {lo_pct:>5.0f}-{hi_pct:<5.0f}pct  n={n:>6}  "
              f"MAE {band.get('mae', float('nan')):.4f}  RMSE {band.get('rmse', float('nan')):.4f}")

    out = {"bands": bands}
    if id_ood_cutoff is not None:
        id_mask, ood_mask = severity <= id_ood_cutoff, severity > id_ood_cutoff
        id_vs_ood = {
            "n_id": int(id_mask.sum()), "n_ood": int(ood_mask.sum()),
            "mae_id": float(err[id_mask].mean()) if id_mask.sum() else None,
            "mae_ood": float(err[ood_mask].mean()) if ood_mask.sum() else None,
            "precision": {
                f"{thr:g}": {
                    "id": float((err[id_mask] <= thr).mean()) if id_mask.sum() else None,
                    "ood": float((err[ood_mask] <= thr).mean()) if ood_mask.sum() else None,
                } for thr in thresholds
            },
        }
        print(f"  ID (n={id_vs_ood['n_id']}) MAE {id_vs_ood['mae_id']:.4f}  vs  "
              f"OOD (n={id_vs_ood['n_ood']}) MAE {id_vs_ood['mae_ood']:.4f}")
        out["id_vs_ood"] = id_vs_ood
    return out


# --------------------------------------------------------------------------------
# scenario labelling and scoring — identical shape to every other port
# --------------------------------------------------------------------------------
def scenario_labels(cv_ratio, coherence, cons_cut, agr_cut):
    return np.where((cv_ratio <= cons_cut) & (coherence >= agr_cut), "A",
           np.where(cv_ratio <= cons_cut, "B",
           np.where(coherence >= agr_cut, "C", "D")))


def enrichment(err, mask, thr):
    base = float((err <= thr).mean())
    n = int(mask.sum())
    if n == 0 or base <= 0:
        return {"n": n, "ef": None, "precision": None, "base_rate": base}
    prec = float((err[mask] <= thr).mean())
    return {"n": n, "ef": prec / base, "precision": prec, "base_rate": base,
            "unstable": n < 5}


def trust_block(cv_ratio, coherence, err, cons_cut, agr_cut, thresholds, tag):
    """Ported from CancerTrust-XAI's trust_block, same aggregation logic
    (scenario counts/coverage plus enrichment at each error threshold, for A
    alone and for the A+B / C+D bands), with one deliberate signature
    change: also RETURNS the per-window scenario array, not just the
    summary dict, because xai.py needs that array immediately afterward for
    its predictions dump and CancerTrust-XAI's ridge_baseline.py otherwise
    has to call scenario_labels() a second time to get it -- fixed here
    rather than carried over. Added 2026-08-11 when xai.py needed it;
    CancerTrust-XAI keeps this function in trust.py rather than in its
    per-domain scoring script, so it is added here too rather than
    duplicated inline in xai.py."""
    scen = scenario_labels(cv_ratio, coherence, cons_cut, agr_cut)
    counts = {s: int((scen == s).sum()) for s in "ABCD"}
    out = {"cutoffs": {"consensus": float(cons_cut), "agreement": float(agr_cut)},
           "counts": counts,
           "coverage_pct": {s: 100.0 * counts[s] / len(scen) for s in "ABCD"},
           "enrichment": {}}
    for thr in thresholds:
        out["enrichment"][f"{thr:g}"] = {
            "A": enrichment(err, scen == "A", thr),
            "A+B": enrichment(err, np.isin(scen, ["A", "B"]), thr),
            "C+D": enrichment(err, np.isin(scen, ["C", "D"]), thr),
        }
    cov = out["coverage_pct"]
    print(f"  {tag:<26} A {cov['A']:5.1f}%  B {cov['B']:5.1f}%  C {cov['C']:5.1f}%  D {cov['D']:5.1f}%")
    return out, scen
