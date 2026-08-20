#!/usr/bin/env python
"""FULL-DATASET companion to causal_coherence_confound_controlled.py --
does the test-set finding (causal coherence's raw ~3x advantage over
bin-overlap coherence shrinks 84% after controlling for |y_true|, AD
distance, and n_instances, ending up essentially tied with bin-overlap's
own 36%-shrunk signal) replicate genome-wide, or is it itself a
held-out-chromosome artifact?

MUST BE RUN AFTER xai_full_dataset.py [already run, provides full-dataset
AD distance + bin-overlap coherence] and motif_causal_occlusion_full_
dataset.py [provides full-dataset causal coherence/n_instances/n_modules].
Pure local join, no Bunya time, reuses causal_coherence_confound_
controlled.py's own analyze()/multi_control_partial_spearman()/ols_with_se
functions unchanged so the two runs are directly comparable in format.

Reports three views, same discipline as every other full-dataset script in
this project:
1. Whole full-dataset A+B (high-consensus) population -- design-facing.
2. Top decile of each metric, full dataset -- design-facing.
3. Holdout-test-subset crosscheck -- restricts the full-dataset run to the
   held-out chromosomes only and compares against causal_coherence_
   confound_controlled_results.json's own already-reported numbers, the
   same crosscheck convention xai_full_dataset.py established.

Do not tune the control set or decile definition to chase a cleaner or
messier replication -- report whichever way it comes out.
"""
from __future__ import annotations

import json

import numpy as np

from causal_coherence_confound_controlled import analyze
from config import RESULTS

XAI_FULL_PREDICTIONS_NPZ = RESULTS / "xai_full_dataset_predictions.npz"
CAUSAL_FULL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz"
TEST_SET_RESULTS_JSON = RESULTS / "causal_coherence_confound_controlled_results.json"
OUT_JSON = RESULTS / "causal_coherence_confound_controlled_full_dataset_results.json"


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined_full_dataset():
    xai_d = np.load(XAI_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    causal_d = np.load(CAUSAL_FULL_PREDICTIONS_NPZ, allow_pickle=True)

    xai_resolved = xai_d["have_shell"]
    causal_resolved = causal_d["have_shell"]
    xai_key = _keys(xai_d["chrom"][xai_resolved], xai_d["start"][xai_resolved], xai_d["end"][xai_resolved])
    causal_key = _keys(causal_d["chrom"][causal_resolved], causal_d["start"][causal_resolved], causal_d["end"][causal_resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}

    common_mask = np.array([k in causal_idx for k in xai_key])
    print(f"bin-overlap shell-resolved windows: {len(xai_key)}")
    print(f"causal shell-resolved windows:      {len(causal_key)}")
    print(f"joined on (chrom,start,end):        {common_mask.sum()}")
    order = [causal_idx[k] for k in xai_key[common_mask]]

    y_true = xai_d["y_true"][xai_resolved][common_mask]
    y_pred = xai_d["y_pred"][xai_resolved][common_mask]
    cv_ratio = xai_d["cv_ratio"][xai_resolved][common_mask]
    bin_coherence = xai_d["coherence"][xai_resolved][common_mask]
    ad_distance = xai_d["ad_distance"][xai_resolved][common_mask]
    is_holdout = xai_d["is_holdout"][xai_resolved][common_mask]
    cons_cut = float(xai_d["cons_cut"])

    causal_coherence = causal_d["causal_coherence"][causal_resolved][order]
    n_instances = causal_d["n_instances"][causal_resolved][order]

    y_true_causal_check = causal_d["y_true"][causal_resolved][order]
    if not np.allclose(y_true, y_true_causal_check):
        raise RuntimeError("y_true mismatch between xai_full_dataset and motif_causal_occlusion_"
                            "full_dataset on joined windows -- should be identical (same checkpoint, "
                            "same windows table). Investigate before trusting anything below.")

    err = np.abs(y_pred - y_true)
    ab_mask = cv_ratio <= cons_cut
    return {
        "err": err, "y_true": y_true, "causal_coherence": causal_coherence,
        "bin_coherence": bin_coherence, "n_instances": n_instances, "ad_distance": ad_distance,
        "ab_mask": ab_mask, "is_holdout": is_holdout,
    }


def run_both_metrics(label, d, mask):
    out = {}
    for metric_key, coh in (("causal", d["causal_coherence"]), ("bin_overlap", d["bin_coherence"])):
        print(f"\n---- {metric_key}, {label} ----")
        idx = np.where(mask)[0]
        if len(idx) < 50:
            print(f"  too few windows (n={len(idx)}) for a meaningful report -- skipping.")
            out[metric_key] = {"n": int(len(idx)), "note": "too few windows"}
            continue
        out[metric_key] = {}
        out[metric_key]["whole"] = analyze(
            f"{metric_key} / {label} / whole", d["err"][idx], coh[idx], d["y_true"][idx],
            d["ad_distance"][idx], d["n_instances"][idx])

        order = idx[np.argsort(-coh[idx])]
        top_idx = order[:max(1, len(order) // 10)]
        out[metric_key]["top_decile"] = analyze(
            f"{metric_key} / {label} / top decile", d["err"][top_idx], coh[top_idx], d["y_true"][top_idx],
            d["ad_distance"][top_idx], d["n_instances"][top_idx])
    return out


def main():
    for p in (XAI_FULL_PREDICTIONS_NPZ, CAUSAL_FULL_PREDICTIONS_NPZ):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run xai_full_dataset.py and "
                              f"motif_causal_occlusion_full_dataset.py (in that order, plus "
                              f"motif_causal_occlusion.py before it) first.")

    d = load_joined_full_dataset()
    print(f"\nfull dataset A+B (high-consensus) population: n={int(d['ab_mask'].sum())}")

    results = {"note": ("Design-facing full-dataset extension of causal_coherence_confound_"
                         "controlled.py. Reliability-type comparisons belong to that held-out-"
                         "test-set script only; the holdout_test_subset_crosscheck block below "
                         "should reproduce it and is the number to trust for reliability claims.")}

    print("\n=== FULL DATASET (design-facing) ===")
    results["full_dataset"] = run_both_metrics("full dataset", d, d["ab_mask"])

    print("\n=== HOLDOUT-TEST-SUBSET CROSSCHECK (should reproduce causal_coherence_confound_controlled_results.json) ===")
    holdout_ab_mask = d["ab_mask"] & d["is_holdout"]
    results["holdout_test_subset_crosscheck"] = run_both_metrics("holdout crosscheck", d, holdout_ab_mask)

    if TEST_SET_RESULTS_JSON.exists():
        print(f"\n  (compare the block above by hand against {TEST_SET_RESULTS_JSON} -- should "
              "match to a few significant figures; small mismatches are expected only from "
              "floating-point/window-order differences, not from a different cutoff or population)")
        results["crosscheck_reference_file"] = str(TEST_SET_RESULTS_JSON)

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
