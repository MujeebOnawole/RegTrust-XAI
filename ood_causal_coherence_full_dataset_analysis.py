#!/usr/bin/env python
"""FULL-DATASET companion to ood_causal_coherence_analysis.py, same
relationship xai_full_dataset.py has to xai.py and motif_causal_occlusion_
full_dataset.py has to motif_causal_occlusion.py: does either coherence
axis's ID/OOD collapse (see ood_causal_coherence_analysis.py -- both axes'
error-discrimination essentially vanishes outside the applicability domain
on the held-out test set) also hold genome-wide, design-facing, not just on
the held-out chromosomes?

MUST BE RUN AFTER (1) xai_full_dataset.py [already run, provides full-
dataset AD distance AND bin-overlap coherence in xai_full_dataset_
predictions.npz -- no new embedding/occlusion pass needed here] and (2)
motif_causal_occlusion_full_dataset.py [provides full-dataset causal
coherence in motif_causal_occlusion_full_dataset_predictions.npz]. Pure
local join of two already-computed .npz files -- no model pass, no Bunya
time, runs in seconds once both inputs exist.

SAME STANDARD as every other full-dataset script in this project:
- Reuses the already-calibrated AD cutoff from results/trust_validation_
  results.json (identical cutoff ood_causal_coherence_analysis.py uses for
  the test set) -- does NOT recalibrate for the full dataset, so "OOD" means
  the same thing here as it does on the test set.
- Reuses each coherence definition's own already-calibrated cutoffs (from
  xai_results.json / motif_causal_occlusion_results.json, both already
  baked into the two predictions.npz files as cons_cut/agr_cut) -- not
  recalibrated per ID/OOD subgroup, matching ood_causal_coherence_analysis.py.
- Every window tagged `pool` (`holdout_test` / `train_pool`); the headline
  numbers this script prints are design-facing full-dataset ones, but a
  holdout-test-only crosscheck against ood_causal_coherence_results.json
  (this project's own reliability-scoped result) is computed and reported
  alongside, so nobody accidentally quotes the full-dataset ID/OOD split as
  a reliability claim -- the exact confusion the Trust/Scenario Reporting
  Rule exists to prevent.

Do not tune any cutoff to manufacture a bigger or cleaner ID/OOD gap --
report whichever way it comes out, per this project's standing
no-p-hacking discipline.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from config import RESULTS, XAI_ERROR_THRESHOLDS
from trust import trust_block
from validate_trust_axes import coherence_scenario_a_vs_b

XAI_FULL_PREDICTIONS_NPZ = RESULTS / "xai_full_dataset_predictions.npz"
CAUSAL_FULL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz"
AD_RESULTS_JSON = RESULTS / "trust_validation_results.json"
TEST_SET_OOD_RESULTS_JSON = RESULTS / "ood_causal_coherence_results.json"
OUT_JSON = RESULTS / "ood_causal_coherence_full_dataset_results.json"


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined():
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

    # "scenario" is pre-filtered to shell-resolved windows only (502,898 long) in
    # xai_full_dataset_predictions.npz, unlike every other field here (517,790 long,
    # indexed by have_shell) -- excluded, and unused downstream (load_joined()'s
    # return dict never includes it).
    xai_sub = {k: (xai_d[k][xai_resolved][common_mask] if xai_d[k].ndim > 0 else xai_d[k]) for k in xai_d.files if k not in ("ad_distance", "scenario")}
    ad_distance = xai_d["ad_distance"][xai_resolved][common_mask]
    # same pre-filtered-"scenario" issue as xai_d above, see comment there.
    causal_sub = {k: (causal_d[k][causal_resolved][order] if causal_d[k].ndim > 0 else causal_d[k]) for k in causal_d.files if k != "scenario"}

    if not np.allclose(xai_sub["y_true"], causal_sub["y_true"]):
        raise RuntimeError("y_true mismatch between xai_full_dataset and motif_causal_occlusion_full_dataset "
                            "on joined windows -- these should be identical (same checkpoint, same windows table).")

    ad_results = json.loads(AD_RESULTS_JSON.read_text())
    ad_cutoff = float(ad_results["distance_vs_error"]["ad_cutoff"])

    err = np.abs(xai_sub["y_pred"] - xai_sub["y_true"])
    return {
        "err": err, "ad_distance": ad_distance, "ad_cutoff": ad_cutoff,
        "cv_ratio": xai_sub["cv_ratio"], "is_holdout": xai_sub["is_holdout"],
        "bin_coherence": xai_sub["coherence"], "bin_cons_cut": float(xai_d["cons_cut"]), "bin_agr_cut": float(xai_d["agr_cut"]),
        "causal_coherence": causal_sub["causal_coherence"], "causal_cons_cut": float(causal_d["cons_cut"]), "causal_agr_cut": float(causal_d["agr_cut"]),
    }


def report_subgroup(label, coherence, err, cv_ratio, cons_cut, agr_cut, mask):
    n = int(mask.sum())
    print(f"\n  [{label}] n={n}")
    if n < 10:
        print("    too few windows for a meaningful subgroup report -- skipping.")
        return {"n": n, "note": "too few windows for a meaningful report"}
    rho, p = stats.spearmanr(coherence[mask], err[mask])
    print(f"    corr(coherence, |error|): rho {rho:+.4f}  p {p:.3g}")
    block, scen = trust_block(cv_ratio[mask], coherence[mask], err[mask], cons_cut, agr_cut, XAI_ERROR_THRESHOLDS, label)
    out = {"n": n, "coherence_vs_error": {"rho": float(rho), "p": float(p)}, "trust": block}
    if (scen == "A").sum() >= 5 and (scen == "B").sum() >= 5:
        out["scenario_a_vs_b_conditional"] = coherence_scenario_a_vs_b(scen, err[mask])
    else:
        print(f"    Scenario A/B too small within this subgroup (A={int((scen=='A').sum())}, "
              f"B={int((scen=='B').sum())}) for the conditional test -- skipped, not defaulted to a number.")
    return out


def main():
    for p in (XAI_FULL_PREDICTIONS_NPZ, CAUSAL_FULL_PREDICTIONS_NPZ, AD_RESULTS_JSON):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run xai_full_dataset.py and "
                              f"motif_causal_occlusion_full_dataset.py (in that order, plus "
                              f"motif_causal_occlusion.py before it) first.")

    d = load_joined()
    err, ad_distance, ad_cutoff = d["err"], d["ad_distance"], d["ad_cutoff"]
    is_holdout = d["is_holdout"]
    id_mask, ood_mask = ad_distance <= ad_cutoff, ad_distance > ad_cutoff
    print(f"\nAD cutoff (reused from {AD_RESULTS_JSON.name}): {ad_cutoff:.4f}")
    print(f"full dataset -- ID: n={int(id_mask.sum())} ({id_mask.mean():.1%})   "
          f"OOD: n={int(ood_mask.sum())} ({ood_mask.mean():.1%})")

    results = {
        "note": ("Design-facing full-dataset extension of ood_causal_coherence_analysis.py. "
                 "Reliability-type ID/OOD comparisons belong to that held-out-test-set script "
                 "only; this file is for design-facing mining plus the holdout-subset crosscheck "
                 "against it, computed below."),
        "ad_cutoff": ad_cutoff,
        "n_total": int(len(err)), "n_id": int(id_mask.sum()), "n_ood": int(ood_mask.sum()),
        "ood_rate": float(ood_mask.mean()),
    }

    print("\n=== bin-overlap coherence, FULL DATASET (design-facing) ===")
    results["bin_overlap"] = {
        "id": report_subgroup("bin-overlap / full / ID", d["bin_coherence"], err, d["cv_ratio"], d["bin_cons_cut"], d["bin_agr_cut"], id_mask),
        "ood": report_subgroup("bin-overlap / full / OOD", d["bin_coherence"], err, d["cv_ratio"], d["bin_cons_cut"], d["bin_agr_cut"], ood_mask),
    }
    print("\n=== causal knockout coherence, FULL DATASET (design-facing) ===")
    results["causal"] = {
        "id": report_subgroup("causal / full / ID", d["causal_coherence"], err, d["cv_ratio"], d["causal_cons_cut"], d["causal_agr_cut"], id_mask),
        "ood": report_subgroup("causal / full / OOD", d["causal_coherence"], err, d["cv_ratio"], d["causal_cons_cut"], d["causal_agr_cut"], ood_mask),
    }

    print("\n=== holdout-test-subset crosscheck (should reproduce ood_causal_coherence_results.json) ===")
    ho_id, ho_ood = id_mask & is_holdout, ood_mask & is_holdout
    crosscheck = {
        "bin_overlap": {
            "id": report_subgroup("bin-overlap / holdout crosscheck / ID", d["bin_coherence"], err, d["cv_ratio"], d["bin_cons_cut"], d["bin_agr_cut"], ho_id),
            "ood": report_subgroup("bin-overlap / holdout crosscheck / OOD", d["bin_coherence"], err, d["cv_ratio"], d["bin_cons_cut"], d["bin_agr_cut"], ho_ood),
        },
        "causal": {
            "id": report_subgroup("causal / holdout crosscheck / ID", d["causal_coherence"], err, d["cv_ratio"], d["causal_cons_cut"], d["causal_agr_cut"], ho_id),
            "ood": report_subgroup("causal / holdout crosscheck / OOD", d["causal_coherence"], err, d["cv_ratio"], d["causal_cons_cut"], d["causal_agr_cut"], ho_ood),
        },
    }
    results["holdout_test_subset_crosscheck"] = crosscheck
    if TEST_SET_OOD_RESULTS_JSON.exists():
        ref = json.loads(TEST_SET_OOD_RESULTS_JSON.read_text())
        print(f"\n  (compare the block above by hand against {TEST_SET_OOD_RESULTS_JSON} -- "
              "should match to a few significant figures; small mismatches are expected only "
              "from floating-point/window-order differences, not from a different cutoff or population)")
        results["crosscheck_reference_file"] = str(TEST_SET_OOD_RESULTS_JSON)

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
