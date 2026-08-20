#!/usr/bin/env python
"""FULL-DATASET companion to causal_coherence_tail_diagnosis.py, same
relationship xai_full_dataset.py has to xai.py: does the test-set-only
10-decile profile and candidate-confound correlation picture (U-shaped
mean|error| across causal_coherence deciles, causal_coherence correlated
with AD distance rho -0.512 and with |y_true| in the top decile) replicate
genome-wide, design-facing, not just on the held-out chromosomes?

Motivated directly by causal_coherence_confound_controlled_full_dataset.py's
own finding (2026-08-20): the top-decile partial correlation that read as
NOT statistically real on the test set alone (p=0.356) turned out to be real
genome-wide (p=8.3e-18) once power increased. This script re-runs the
DESCRIPTIVE decile-profile/candidate-correlation diagnosis that originally
motivated the confound-controlled check, at the same full-dataset scale, to
see whether the same candidate confounds (|y_true|, AD distance) still look
like the right explanation for the now-confirmed top-decile signal, or
whether something else is left over.

Pure local analysis, no Bunya time -- joins two already-computed .npz files:
xai_full_dataset_predictions.npz (bin-overlap coherence, AD distance, all
already present as full 517,790-length arrays, no separate AD file needed
here unlike the test-set script) and motif_causal_occlusion_full_dataset_
predictions.npz (causal_coherence/n_instances/n_modules), on
(chrom, start, end).

Reuses decile_profile() and correlations() from causal_coherence_tail_
diagnosis.py UNCHANGED (imported, not duplicated), so the full-dataset and
test-set outputs are directly comparable format-for-format.

Same standard as every other full-dataset script in this project: reports
(1) full-dataset, design-facing, and (2) a holdout-test-subset crosscheck
against causal_coherence_tail_diagnosis_results.json, so nobody accidentally
quotes the full-dataset numbers as a reliability claim -- the Trust/Scenario
Reporting Rule's own distinction.

Do not tune the decile count or candidate-variable list to manufacture a
cleaner story -- report whichever way it comes out.
"""
from __future__ import annotations

import json

import numpy as np

from config import RESULTS
from causal_coherence_tail_diagnosis import correlations, decile_profile

XAI_FULL_PREDICTIONS_NPZ = RESULTS / "xai_full_dataset_predictions.npz"
CAUSAL_FULL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz"
TEST_SET_TAIL_RESULTS_JSON = RESULTS / "causal_coherence_tail_diagnosis_results.json"
OUT_JSON = RESULTS / "causal_coherence_tail_diagnosis_full_dataset_results.json"


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined():
    bin_d = np.load(XAI_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    causal_d = np.load(CAUSAL_FULL_PREDICTIONS_NPZ, allow_pickle=True)

    bin_resolved = bin_d["have_shell"]
    causal_resolved = causal_d["have_shell"] & ~np.isnan(causal_d["causal_coherence"])

    causal_key = _keys(causal_d["chrom"][causal_resolved], causal_d["start"][causal_resolved], causal_d["end"][causal_resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}

    bin_key = _keys(bin_d["chrom"][bin_resolved], bin_d["start"][bin_resolved], bin_d["end"][bin_resolved])
    common_mask = np.array([k in causal_idx for k in bin_key])
    print(f"bin-overlap shell-resolved windows: {len(bin_key)}")
    print(f"causal shell-resolved windows:      {len(causal_key)}")
    print(f"joined on (chrom,start,end):        {common_mask.sum()}")

    causal_order = [causal_idx[k] for k in bin_key[common_mask]]

    y_true = bin_d["y_true"][bin_resolved][common_mask]
    y_pred = bin_d["y_pred"][bin_resolved][common_mask]
    cv_ratio = bin_d["cv_ratio"][bin_resolved][common_mask]
    cons_cut = float(bin_d["cons_cut"])
    ad_distance = bin_d["ad_distance"][bin_resolved][common_mask]
    is_holdout = bin_d["is_holdout"][bin_resolved][common_mask]

    causal_coh = causal_d["causal_coherence"][causal_resolved][causal_order]
    n_instances = causal_d["n_instances"][causal_resolved][causal_order]
    n_modules = causal_d["n_modules"][causal_resolved][causal_order]

    if not np.allclose(y_true, causal_d["y_true"][causal_resolved][causal_order]):
        raise RuntimeError("y_true mismatch between xai_full_dataset and motif_causal_occlusion_full_dataset "
                            "on joined windows -- these should be identical (same checkpoint, same windows table).")

    err = np.abs(y_pred - y_true)
    ab_mask = cv_ratio <= cons_cut
    return {
        "err": err, "y_true": y_true, "causal_coherence": causal_coh,
        "n_instances": n_instances, "n_modules": n_modules, "ad_distance": ad_distance,
        "ab_mask": ab_mask, "is_holdout": is_holdout,
    }


def run_block(label, coh, err, y_true, n_inst, n_mod, ad_dist, mask):
    print(f"\n{label} population: n={int(mask.sum())}")
    print(f"\n=== full 10-decile profile of causal_coherence within {label} ===")
    deciles = decile_profile(coh, err, y_true, n_inst, n_mod, ad_dist, mask)

    print(f"\n=== Spearman corr(causal_coherence, candidate variable), {label} ===")
    print("whole population:")
    corr_all = correlations(f"{label} whole", coh, err, y_true, n_inst, n_mod, ad_dist, mask)

    idx = np.where(mask)[0]
    order = idx[np.argsort(-coh[idx])]
    top_decile_idx = order[:len(order) // 10]
    top_mask = np.zeros(len(coh), dtype=bool)
    top_mask[top_decile_idx] = True
    print("top decile only:")
    corr_top = correlations(f"{label} top decile", coh, err, y_true, n_inst, n_mod, ad_dist, top_mask)

    return {
        "n_population": int(mask.sum()),
        "decile_profile": deciles,
        "spearman_whole": corr_all,
        "spearman_top_decile_only": corr_top,
    }


def main():
    for p in (XAI_FULL_PREDICTIONS_NPZ, CAUSAL_FULL_PREDICTIONS_NPZ):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run xai_full_dataset.py and "
                              f"motif_causal_occlusion_full_dataset.py first.")

    d = load_joined()
    coh, err, y_true = d["causal_coherence"], d["err"], d["y_true"]
    n_inst, n_mod, ad_dist = d["n_instances"], d["n_modules"], d["ad_distance"]
    ab_mask, is_holdout = d["ab_mask"], d["is_holdout"]

    results = {
        "note": ("Design-facing full-dataset extension of causal_coherence_tail_diagnosis.py. "
                 "Reliability-type diagnosis belongs to that held-out-test-set script only; this "
                 "file is for design-facing mining plus the holdout-subset crosscheck against it, "
                 "computed below."),
    }
    results["full_dataset"] = run_block("full dataset (A+B)", coh, err, y_true, n_inst, n_mod, ad_dist, ab_mask)

    print("\n=== holdout-test-subset crosscheck (should reproduce causal_coherence_tail_diagnosis_results.json) ===")
    ho_mask = ab_mask & is_holdout
    results["holdout_test_subset_crosscheck"] = run_block("holdout crosscheck (A+B)", coh, err, y_true, n_inst, n_mod, ad_dist, ho_mask)

    if TEST_SET_TAIL_RESULTS_JSON.exists():
        print(f"\n  (compare the crosscheck block above by hand against {TEST_SET_TAIL_RESULTS_JSON} -- "
              "should match to a few significant figures; small mismatches are expected only from "
              "floating-point/random-control-draw differences, not from a different cutoff or population)")
        results["crosscheck_reference_file"] = str(TEST_SET_TAIL_RESULTS_JSON)

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
