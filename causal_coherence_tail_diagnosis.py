#!/usr/bin/env python
"""Why does causal coherence's own top decile invert (mean|error| 0.519 vs
the rest's 0.412, see matched_coherence_comparison_results.json's N=1367
row)? Candidate hypothesis, not yet tested: the largest shell-knockout
effects may partly reflect prediction SENSITIVITY/VOLATILITY to a motif
span rather than pure grounding-accuracy -- i.e. causal coherence may be
non-monotonic (low = not using the expected biology; moderate/high =
biologically grounded and accurate; extreme = the model is fragile to that
region, not necessarily correctly so).

Pure local analysis, no Bunya time -- joins three already-computed .npz
files (motif_causal_occlusion_predictions.npz for causal_coherence/
n_instances/n_modules, xai_predictions.npz for the same window population's
error, trust_validation_predictions.npz for AD distance) on
(chrom, start, end).

Two views:
1. Full 10-decile profile of causal_coherence within the fixed A+B
   (high-consensus) population -- mean|error|, mean n_instances, mean
   n_modules, mean |y_true|, mean AD distance per decile. Characterizes the
   FULL shape, not just top-10%-vs-rest, so a genuine non-monotonic (U- or
   J-shaped) pattern is visible directly rather than assumed from one
   binary split.
2. Spearman corr(causal_coherence, each candidate variable) computed BOTH
   over the whole A+B population and restricted to the top decile alone --
   if the top decile's own internal ranking still correlates with error in
   the normal (negative) direction, the inversion is about the tail's
   ABSOLUTE level, not a breakdown of the metric's internal ordering, an
   important distinction for how to handle it (a cap/upper-bound rule vs.
   treating the metric as unreliable at the extreme).

Do not tune the decile count or candidate-variable list to manufacture a
cleaner story -- report whichever way it comes out.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from config import RESULTS, XAI_PREDICTIONS_NPZ

CAUSAL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_predictions.npz"
AD_PREDICTIONS_NPZ = RESULTS / "trust_validation_predictions.npz"
OUT_JSON = RESULTS / "causal_coherence_tail_diagnosis_results.json"
N_DECILES = 10


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined():
    bin_d = np.load(XAI_PREDICTIONS_NPZ)
    causal_d = np.load(CAUSAL_PREDICTIONS_NPZ)
    ad_d = np.load(AD_PREDICTIONS_NPZ)

    resolved = causal_d["sampled_mask"] & ~np.isnan(causal_d["causal_coherence"])
    causal_key = _keys(causal_d["chrom"][resolved], causal_d["start"][resolved], causal_d["end"][resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}
    ad_key = _keys(ad_d["chrom"], ad_d["start"], ad_d["end"])
    ad_idx = {k: i for i, k in enumerate(ad_key)}

    bin_key = _keys(bin_d["chrom"], bin_d["start"], bin_d["end"])
    common_mask = np.array([(k in causal_idx) and (k in ad_idx) for k in bin_key])
    print(f"joined on (chrom,start,end): {common_mask.sum()}")

    causal_order = [causal_idx[k] for k in bin_key[common_mask]]
    ad_order = [ad_idx[k] for k in bin_key[common_mask]]

    y_true = bin_d["y_true"][common_mask]
    y_pred = bin_d["y_pred"][common_mask]
    cv_ratio = bin_d["cv_ratio"][common_mask]
    cons_cut = float(bin_d["cons_cut"])

    causal_coh = causal_d["causal_coherence"][resolved][causal_order]
    n_instances = causal_d["n_instances"][resolved][causal_order]
    n_modules = causal_d["n_modules"][resolved][causal_order]
    ad_distance = ad_d["ad_distance"][ad_order]

    err = np.abs(y_pred - y_true)
    ab_mask = cv_ratio <= cons_cut
    return {
        "err": err, "y_true": y_true, "causal_coherence": causal_coh,
        "bin_coherence": bin_d["coherence"][common_mask],
        "n_instances": n_instances, "n_modules": n_modules, "ad_distance": ad_distance,
        "ab_mask": ab_mask,
    }


def decile_profile(coherence, err, y_true, n_instances, n_modules, ad_distance, mask):
    idx = np.where(mask)[0]
    order = idx[np.argsort(coherence[idx])]  # ascending
    n = len(order)
    bounds = np.linspace(0, n, N_DECILES + 1).astype(int)
    rows = []
    for i in range(N_DECILES):
        sel = order[bounds[i]:bounds[i + 1]]
        rows.append({
            "decile": i + 1, "n": int(len(sel)),
            "causal_coherence_range": [float(coherence[sel].min()), float(coherence[sel].max())],
            "mean_abs_error": float(np.abs(err[sel]).mean()) if len(sel) else None,
            "mean_n_instances": float(n_instances[sel].mean()) if len(sel) else None,
            "mean_n_modules": float(n_modules[sel].mean()) if len(sel) else None,
            "mean_abs_y_true": float(np.abs(y_true[sel]).mean()) if len(sel) else None,
            "mean_ad_distance": float(ad_distance[sel].mean()) if len(sel) else None,
        })
        print(f"  decile {i+1:2d} (n={len(sel):5d}, coherence "
              f"[{rows[-1]['causal_coherence_range'][0]:+.3f}, {rows[-1]['causal_coherence_range'][1]:+.3f}]): "
              f"mean|err| {rows[-1]['mean_abs_error']:.4f}  "
              f"n_inst {rows[-1]['mean_n_instances']:.2f}  n_mod {rows[-1]['mean_n_modules']:.2f}  "
              f"|y_true| {rows[-1]['mean_abs_y_true']:.4f}  AD-dist {rows[-1]['mean_ad_distance']:.4f}")
    return rows


def correlations(label, coherence, err, y_true, n_instances, n_modules, ad_distance, mask):
    idx = np.where(mask)[0]
    out = {}
    for name, var in (("error", np.abs(err[idx])), ("n_instances", n_instances[idx]),
                       ("n_modules", n_modules[idx]), ("abs_y_true", np.abs(y_true[idx])),
                       ("ad_distance", ad_distance[idx])):
        rho, p = stats.spearmanr(coherence[idx], var)
        out[name] = {"rho": float(rho), "p": float(p)}
    print(f"  [{label}, n={len(idx)}] " + "  ".join(f"{k} rho={v['rho']:+.3f} p={v['p']:.2g}" for k, v in out.items()))
    return out


def main():
    d = load_joined()
    coh, err, y_true = d["causal_coherence"], d["err"], d["y_true"]
    n_inst, n_mod, ad_dist, ab_mask = d["n_instances"], d["n_modules"], d["ad_distance"], d["ab_mask"]

    print(f"\nA+B (high-consensus) population: n={int(ab_mask.sum())}")

    print("\n=== full 10-decile profile of causal_coherence within A+B ===")
    deciles = decile_profile(coh, err, y_true, n_inst, n_mod, ad_dist, ab_mask)

    print("\n=== Spearman corr(causal_coherence, candidate variable) ===")
    print("whole A+B population:")
    corr_all = correlations("whole A+B", coh, err, y_true, n_inst, n_mod, ad_dist, ab_mask)

    idx_ab = np.where(ab_mask)[0]
    order = idx_ab[np.argsort(-coh[idx_ab])]
    top_decile_idx = order[:len(order) // 10]
    top_mask = np.zeros(len(coh), dtype=bool)
    top_mask[top_decile_idx] = True
    print("top decile only (does internal ranking still track error normally within the tail?):")
    corr_top = correlations("top decile only", coh, err, y_true, n_inst, n_mod, ad_dist, top_mask)

    results = {
        "n_ab_population": int(ab_mask.sum()),
        "decile_profile": deciles,
        "spearman_whole_ab": corr_all,
        "spearman_top_decile_only": corr_top,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
