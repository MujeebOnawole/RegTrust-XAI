#!/usr/bin/env python
"""Does causal coherence's top-decile residual signal -- confirmed real and
statistically significant genome-wide even after controlling for |y_true|,
AD distance, and n_instances jointly (partial rho +0.067, p=8.3e-18, see
causal_coherence_confound_controlled_full_dataset_results.json) -- survive
adding GC content and repeat-derived content as two more controls?

Motivated directly by causal_coherence_tail_diagnosis_full_dataset.py's own
"not yet started" next step: the three-control model does not fully explain
the top-decile signal, and this project already has both GC content and
RepeatMasker-derived repeat fraction computed per-window for the full
517,790-window dataset (repeat_blindspot_analysis.py, 2026-08-19, used for
the Section 3.8/Table S9 representational-blind-spot finding) -- no new
computation needed, this is a pure local join and an extended version of the
same confound-controlled machinery causal_coherence_confound_controlled.py
already established.

results/repeat_blindspot_predictions.npz is INDEX-ALIGNED with
results/xai_full_dataset_predictions.npz (both computed in the same
(chrom, start, end) row order, verified directly: chrom/start/end arrays are
np.array_equal across the two files) -- so no separate key-based join is
needed for this third file; the same xai_resolved/common_mask boolean
indexing causal_coherence_confound_controlled_full_dataset.py already uses
against xai_d applies unchanged to the repeat/GC file.

Two views, matching every other confound-control pass in this project:
1. Whole full-dataset A+B (high-consensus) population -- 5-control partial
   Spearman + OLS decomposition (|y_true|, AD distance, n_instances, GC,
   repeat_frac), causal coherence and bin-overlap coherence both run through
   the identical extended control set for a fair comparison.
2. Top decile of causal_coherence only -- the actual test this script exists
   to run: does the residual partial correlation survive, shrink, or vanish
   once GC/repeat content are added on top of the existing three controls?
A holdout-test-subset crosscheck is NOT meaningful here in the usual sense
(no test-set-only predecessor script/result file exists yet for this
5-control model) -- reported as a full-dataset-only, design-facing result,
with the holdout subset broken out separately as its own reliability-scoped
view instead, following the Trust/Scenario Reporting Rule's partition logic
directly rather than by crosschecking against a nonexistent prior file.

Do not tune the control set or decile definition to chase a cleaner or
messier result -- report whichever way it comes out.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from causal_coherence_confound_controlled import multi_control_partial_spearman, ols_with_se, zscore
from causal_coherence_confound_controlled_full_dataset import load_joined_full_dataset
from config import RESULTS

REPEAT_GC_NPZ = RESULTS / "repeat_blindspot_predictions.npz"
XAI_FULL_PREDICTIONS_NPZ = RESULTS / "xai_full_dataset_predictions.npz"
CAUSAL_FULL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz"
OUT_JSON = RESULTS / "causal_coherence_confound_controlled_gc_repeat_results.json"

CONTROL_NAMES = ["abs_y_true", "ad_distance", "n_instances", "gc", "repeat_frac"]


def load_joined_with_repeat_gc():
    """Re-derives the exact same xai_resolved/common_mask boolean indices
    load_joined_full_dataset() uses internally (that function doesn't expose
    them), then applies them identically to repeat_blindspot_predictions.npz
    -- valid only because that file is index-aligned with
    xai_full_dataset_predictions.npz (verified: chrom/start/end arrays
    np.array_equal across the two files, same 517,790-row order)."""
    d = load_joined_full_dataset()

    xai_d = np.load(XAI_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    causal_d = np.load(CAUSAL_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    repeat_d = np.load(REPEAT_GC_NPZ, allow_pickle=True)

    if not (np.array_equal(repeat_d["chrom"], xai_d["chrom"])
            and np.array_equal(repeat_d["start"], xai_d["start"])
            and np.array_equal(repeat_d["end"], xai_d["end"])):
        raise RuntimeError(f"{REPEAT_GC_NPZ} is not index-aligned with {XAI_FULL_PREDICTIONS_NPZ} -- "
                            "the row-order assumption this script depends on no longer holds; "
                            "fall back to a proper (chrom,start,end) key join instead.")

    xai_resolved = xai_d["have_shell"]
    causal_resolved = causal_d["have_shell"]
    xai_key = np.array([f"{c}:{s}:{e}" for c, s, e in
                         zip(xai_d["chrom"][xai_resolved], xai_d["start"][xai_resolved], xai_d["end"][xai_resolved])])
    causal_key = np.array([f"{c}:{s}:{e}" for c, s, e in
                            zip(causal_d["chrom"][causal_resolved], causal_d["start"][causal_resolved], causal_d["end"][causal_resolved])])
    causal_idx = {k: i for i, k in enumerate(causal_key)}
    common_mask = np.array([k in causal_idx for k in xai_key])

    gc = repeat_d["gc"][xai_resolved][common_mask]
    repeat_frac = repeat_d["repeat_frac"][xai_resolved][common_mask]
    n_frac = repeat_d["n_frac"][xai_resolved][common_mask]

    if len(gc) != len(d["err"]):
        raise RuntimeError(f"length mismatch: gc/repeat_frac n={len(gc)} vs. joined n={len(d['err'])} -- "
                            "the xai_resolved/common_mask re-derivation above does not match "
                            "load_joined_full_dataset()'s own internal join.")

    d["gc"] = gc
    d["repeat_frac"] = repeat_frac
    d["valid_gc"] = ~np.isnan(gc) & (n_frac < 0.5)
    return d


def ols_block_extended(err, coherence, y_true, ad_distance, n_instances, gc, repeat_frac):
    X = np.column_stack([
        np.ones(len(err)),
        zscore(coherence), zscore(np.abs(y_true)), zscore(ad_distance),
        zscore(n_instances), zscore(gc), zscore(repeat_frac),
    ])
    beta, se, t, p, r2 = ols_with_se(X, np.abs(err).astype(np.float64))
    names = ["intercept", "coherence_z", "abs_y_true_z", "ad_distance_z", "n_instances_z", "gc_z", "repeat_frac_z"]
    return {
        "coefficients": {n: {"beta": float(b), "se": float(s), "p": float(pp)}
                          for n, b, s, pp in zip(names, beta, se, p)},
        "r_squared": float(r2),
    }


def analyze_extended(label, err, coherence, y_true, ad_distance, n_instances, gc, repeat_frac):
    raw_rho, raw_p = stats.spearmanr(coherence, err)
    partial5 = multi_control_partial_spearman(
        coherence, err, [np.abs(y_true), ad_distance, n_instances, gc, repeat_frac], CONTROL_NAMES)
    ols = ols_block_extended(err, coherence, y_true, ad_distance, n_instances, gc, repeat_frac)
    coh_row = ols["coefficients"]["coherence_z"]
    print(f"\n  [{label}] n={len(err)}")
    print(f"    raw corr(coherence, |error|):                    rho {raw_rho:+.4f}  p {raw_p:.3g}")
    print(f"    5-control partial rho (+ GC, repeat_frac):       rho {partial5['partial_rho']:+.4f}  p {partial5['partial_p']:.3g}")
    print(f"    OLS coherence_z coefficient (6-predictor model): beta {coh_row['beta']:+.4f}  p {coh_row['p']:.3g}  (R2={ols['r_squared']:.4f})")
    return {"n": int(len(err)), "raw_spearman": {"rho": float(raw_rho), "p": float(raw_p)},
            "partial_spearman_5control": partial5, "ols_6predictor": ols}


def main():
    for p in (XAI_FULL_PREDICTIONS_NPZ, CAUSAL_FULL_PREDICTIONS_NPZ, REPEAT_GC_NPZ):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run xai_full_dataset.py, motif_causal_occlusion_"
                              f"full_dataset.py, and repeat_blindspot_analysis.py first.")

    d = load_joined_with_repeat_gc()
    err, y_true, ad_dist, n_inst = d["err"], d["y_true"], d["ad_distance"], d["n_instances"]
    gc, repeat_frac, valid_gc = d["gc"], d["repeat_frac"], d["valid_gc"]
    ab_mask = d["ab_mask"] & valid_gc
    is_holdout = d["is_holdout"]

    print(f"full-dataset A+B population with valid GC/repeat data: n={int(ab_mask.sum())} "
          f"(dropped {int((d['ab_mask'] & ~valid_gc).sum())} A+B windows with N-heavy/NaN GC)")

    results = {"note": ("Extends causal_coherence_confound_controlled_full_dataset.py's 3-control "
                         "model (|y_true|, AD distance, n_instances) with two more controls already "
                         "computed elsewhere in this project (GC content, RepeatMasker-derived repeat "
                         "fraction, both from repeat_blindspot_predictions.npz) -- tests whether "
                         "causal_coherence_tail_diagnosis_full_dataset.py's still-unexplained "
                         "top-decile residual (partial rho +0.067, p=8.3e-18 under the 3-control "
                         "model) is absorbed by these two additional candidates."),
        "controls": CONTROL_NAMES,
    }

    for metric_key, coh in (("causal", d["causal_coherence"]), ("bin_overlap", d["bin_coherence"])):
        print(f"\n########## {metric_key} ##########")
        results[metric_key] = {}

        print("=== whole A+B population (GC-valid subset) ===")
        results[metric_key]["whole_ab"] = analyze_extended(
            f"{metric_key} / whole A+B", err[ab_mask], coh[ab_mask], y_true[ab_mask],
            ad_dist[ab_mask], n_inst[ab_mask], gc[ab_mask], repeat_frac[ab_mask])

        idx_ab = np.where(ab_mask)[0]
        order = idx_ab[np.argsort(-coh[idx_ab])]
        top_idx = order[:len(order) // 10]
        print("\n=== top decile of this metric only ===")
        results[metric_key]["top_decile"] = analyze_extended(
            f"{metric_key} / top decile", err[top_idx], coh[top_idx], y_true[top_idx],
            ad_dist[top_idx], n_inst[top_idx], gc[top_idx], repeat_frac[top_idx])

        ho_mask = ab_mask & is_holdout
        print("\n=== holdout-only subset (reliability-scoped view, not a crosscheck of a prior file) ===")
        results[metric_key]["holdout_only"] = analyze_extended(
            f"{metric_key} / holdout only", err[ho_mask], coh[ho_mask], y_true[ho_mask],
            ad_dist[ho_mask], n_inst[ho_mask], gc[ho_mask], repeat_frac[ho_mask])

        ho_idx = np.where(ho_mask)[0]
        ho_order = ho_idx[np.argsort(-coh[ho_idx])]
        ho_top_idx = ho_order[:max(1, len(ho_order) // 10)]
        results[metric_key]["holdout_top_decile"] = analyze_extended(
            f"{metric_key} / holdout top decile", err[ho_top_idx], coh[ho_top_idx], y_true[ho_top_idx],
            ad_dist[ho_top_idx], n_inst[ho_top_idx], gc[ho_top_idx], repeat_frac[ho_top_idx])

    causal_3ctrl_top_p = 8.26e-18  # causal_coherence_confound_controlled_full_dataset_results.json, top_decile
    causal_5ctrl_top = results["causal"]["top_decile"]["partial_spearman_5control"]
    verdict = ("top-decile residual SURVIVES adding GC/repeat_frac as controls "
               f"(5-control partial rho {causal_5ctrl_top['partial_rho']:+.4f}, p={causal_5ctrl_top['partial_p']:.3g}) "
               "-- GC/repeat content do not explain the remaining signal"
               if causal_5ctrl_top["partial_p"] < 0.05
               else "top-decile residual is ABSORBED once GC/repeat_frac are added as controls "
                    f"(5-control partial rho {causal_5ctrl_top['partial_rho']:+.4f}, p={causal_5ctrl_top['partial_p']:.3g}, "
                    "no longer significant) -- GC and/or repeat-derived content explain a meaningful "
                    "share of what the 3-control model left unexplained")
    results["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print(f"(for reference, the 3-control model's own top-decile partial p was {causal_3ctrl_top_p:.3g})")
    results["three_control_top_decile_p_reference"] = causal_3ctrl_top_p

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
