#!/usr/bin/env python
"""Does causal coherence's raw corr(coherence, |error|) (rho -0.183 within
the fixed A+B population, see matched_coherence_comparison_results.json)
survive once the two confounds causal_coherence_tail_diagnosis.py surfaced
are controlled for? That script found the top-decile error inversion
coincides with (a) a large jump in |y_true| magnitude (regression-to-mean:
this project's own 2026-08-19 finding that the model underpredicts the
highest accessibility quintile by -0.54) and (b) causal_coherence itself
correlating fairly strongly with AD distance across the WHOLE population
(rho -0.51) -- meaning causal coherence may partly be re-expressing
applicability-domain signal rather than adding fully independent
information. n_instances (motif-instance count) also drifts with coherence
(rho -0.098) and is a third plausible confound.

Same discipline as this project's other confound-control passes (mpra_eval.
py's partial-Spearman-controlling-model_pred design, repeat_blindspot_
analysis.py's logistic regression controlling AD-distance-z/GC): report
whether the raw signal survives, do not tune the control set to make it
survive or fail. Two views, on the SAME fixed A+B (high-consensus)
population causal_coherence_tail_diagnosis.py used:

1. Whole A+B population -- multi-control partial Spearman (rank-residualize
   causal_coherence and |error| against {|y_true|, ad_distance, n_instances}
   jointly, correlate the residuals) plus an OLS |error| ~ standardized
   predictors decomposition, so the causal_coherence coefficient's own
   sign/significance is visible alongside the raw rho.
2. Top-decile-only population -- same two checks, restricted to the top
   10% of causal_coherence values (where the raw, unconditional correlation
   INVERTS). If the inversion survives after controlling for |y_true| and
   AD distance, that supports a genuine sensitivity/fragility signal in the
   metric's extreme tail. If it does not survive, the inversion is better
   explained by the two confounds already on record than by a new
   independent phenomenon.

Pure local analysis (reuses causal_coherence_tail_diagnosis.load_joined
unchanged, no re-computation of the join), no Bunya time.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from causal_coherence_tail_diagnosis import load_joined
from config import RESULTS

OUT_JSON = RESULTS / "causal_coherence_confound_controlled_results.json"


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def ols_with_se(X, y):
    """Plain OLS via lstsq with analytic SE/t/p -- same minimal
    implementation mpra_eval.py already uses for its own confound-controlled
    decomposition, reproduced here rather than imported to avoid pulling in
    mpra_eval's module-level torch/model.py dependencies for a pure
    numpy/scipy analysis script."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float((resid @ resid) / dof) if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return beta, se, t, p, r2


def multi_control_partial_spearman(x, y, controls, control_names):
    """Rank-transform x, y, and every column of `controls`; regress ranked x
    and ranked y each on the ranked controls (+ intercept); partial
    Spearman = Pearson correlation of the two residual series. Generalizes
    mpra_eval.cross_assay_block's single-control closed-form partial-rho
    formula to an arbitrary number of simultaneous controls."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rC = np.column_stack([stats.rankdata(c) for c in controls])
    X = np.column_stack([np.ones(len(x)), rC])
    beta_x, _, _, _ = np.linalg.lstsq(X, rx, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(X, ry, rcond=None)
    resid_x = rx - X @ beta_x
    resid_y = ry - X @ beta_y
    rho, p = stats.pearsonr(resid_x, resid_y)
    return {
        "controls": list(control_names),
        "n": int(len(x)),
        "partial_rho": float(rho),
        "partial_p": float(p),
    }


def ols_block(err, causal_coherence, y_true, ad_distance, n_instances):
    X = np.column_stack([
        np.ones(len(err)),
        zscore(causal_coherence), zscore(np.abs(y_true)),
        zscore(ad_distance), zscore(n_instances),
    ])
    beta, se, t, p, r2 = ols_with_se(X, np.abs(err).astype(np.float64))
    names = ["intercept", "causal_coherence_z", "abs_y_true_z", "ad_distance_z", "n_instances_z"]
    return {
        "coefficients": {n: {"beta": float(b), "se": float(s), "p": float(pp)}
                          for n, b, s, pp in zip(names, beta, se, p)},
        "r_squared": float(r2),
    }


def analyze(label, err, causal_coherence, y_true, ad_distance, n_instances):
    raw_rho, raw_p = stats.spearmanr(causal_coherence, err)
    partial = multi_control_partial_spearman(
        causal_coherence, err, [np.abs(y_true), ad_distance, n_instances],
        ["abs_y_true", "ad_distance", "n_instances"])
    ols = ols_block(err, causal_coherence, y_true, ad_distance, n_instances)
    cc_row = ols["coefficients"]["causal_coherence_z"]
    print(f"\n  [{label}] n={len(err)}")
    print(f"    raw corr(causal_coherence, |error|):     rho {raw_rho:+.4f}  p {raw_p:.3g}")
    print(f"    partial rho (ctrl |y_true|/AD/n_inst):   rho {partial['partial_rho']:+.4f}  p {partial['partial_p']:.3g}")
    print(f"    OLS causal_coherence_z coefficient:      beta {cc_row['beta']:+.4f}  p {cc_row['p']:.3g}  (R2={ols['r_squared']:.4f})")
    return {"n": int(len(err)), "raw_spearman": {"rho": float(raw_rho), "p": float(raw_p)},
            "partial_spearman": partial, "ols": ols}


def main():
    d = load_joined()
    err = d["err"]
    y_true, ad_dist, n_inst = d["y_true"], d["ad_distance"], d["n_instances"]
    ab_mask = d["ab_mask"]

    results = {"note": ("Confound-controlled follow-up to causal_coherence_tail_diagnosis.py's "
                         "decile profile. Controls: |y_true| (regression-to-mean confound), "
                         "AD distance (causal_coherence correlates rho -0.51 with AD distance "
                         "across the whole A+B population), n_instances (motif-instance count). "
                         "Bin-overlap coherence run through the identical control set for a fair "
                         "side-by-side comparison -- does EITHER axis's raw advantage survive, "
                         "or does controlling for these confounds erase the gap between them?")}

    for metric_key, coh in (("causal", d["causal_coherence"]), ("bin_overlap", d["bin_coherence"])):
        print(f"\n########## {metric_key} ##########")
        results[metric_key] = {}
        print("=== whole A+B population ===")
        results[metric_key]["whole_ab"] = analyze(
            f"{metric_key} / whole A+B", err[ab_mask], coh[ab_mask], y_true[ab_mask],
            ad_dist[ab_mask], n_inst[ab_mask])

        idx_ab = np.where(ab_mask)[0]
        order = idx_ab[np.argsort(-coh[idx_ab])]
        top_idx = order[:len(order) // 10]
        print("\n=== top decile of this metric only ===")
        results[metric_key]["top_decile"] = analyze(
            f"{metric_key} / top decile", err[top_idx], coh[top_idx], y_true[top_idx],
            ad_dist[top_idx], n_inst[top_idx])

    causal_partial = results["causal"]["whole_ab"]["partial_spearman"]["partial_rho"]
    bin_partial = results["bin_overlap"]["whole_ab"]["partial_spearman"]["partial_rho"]
    causal_raw = results["causal"]["whole_ab"]["raw_spearman"]["rho"]
    bin_raw = results["bin_overlap"]["whole_ab"]["raw_spearman"]["rho"]
    print(f"\n########## summary ##########")
    print(f"  causal:      raw rho {causal_raw:+.4f}  ->  partial rho {causal_partial:+.4f} "
          f"({100*(1 - abs(causal_partial)/abs(causal_raw)):.0f}% shrinkage)")
    print(f"  bin-overlap: raw rho {bin_raw:+.4f}  ->  partial rho {bin_partial:+.4f} "
          f"({100*(1 - abs(bin_partial)/abs(bin_raw)):.0f}% shrinkage)")

    top_partial = results["causal"]["top_decile"]["partial_spearman"]
    verdict = ("inversion SURVIVES confound control (partial rho still positive/significant in "
               "the top decile) -- supports a genuine sensitivity/fragility signal"
               if top_partial["partial_rho"] > 0 and top_partial["partial_p"] < 0.05
               else "inversion does NOT clearly survive confound control -- better explained by "
                    "the |y_true|/AD-distance/n_instances confounds already on record than by a "
                    "new independent phenomenon")
    results["verdict"] = verdict
    print(f"\nVERDICT (top-decile inversion): {verdict}")
    results["whole_ab_comparison"] = {
        "causal_raw_rho": float(causal_raw), "causal_partial_rho": float(causal_partial),
        "bin_overlap_raw_rho": float(bin_raw), "bin_overlap_partial_rho": float(bin_partial),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
