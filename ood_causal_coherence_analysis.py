#!/usr/bin/env python
"""Does the coherence axis (either definition -- bin-overlap or the new
causal-knockout one) still track error OUTSIDE the model's applicability
domain, or does it only work inside the region the model was actually
trained on? Motivated directly by the "design principles" question this
project's causal-occlusion work raised: if coherence's error-discrimination
falls apart in the OOD region, that is itself a real, reportable model
limitation (the explanation stops being trustworthy exactly where it would
matter most); if it holds up, that is a real strength worth stating too.

SAME STANDARD as every other axis-validation script in this project:
- Reuses the already-calibrated AD cutoff (results/trust_validation_results.
  json's distance_vs_error.ad_cutoff, 95th percentile of reference-pool
  self-distances) -- does NOT recalibrate a separate OOD threshold for this
  analysis, so "OOD" here means exactly what it means everywhere else in
  this project.
- Reuses each coherence definition's own already-calibrated consensus/
  agreement cutoffs (bin-overlap from results/xai_results.json, causal from
  results/motif_causal_occlusion_results.json) -- does NOT recalibrate
  per-subgroup (ID-only or OOD-only), which would make the ID/OOD scenario
  labels incomparable to the pooled numbers already reported elsewhere.
- Pure local join of three already-computed .npz files (trust_validation_
  predictions.npz for AD distance, xai_predictions.npz for bin-overlap
  coherence, motif_causal_occlusion_predictions.npz for causal coherence) --
  no model pass, no Bunya time, all three files already exist from prior
  pipeline steps. Runs in seconds.
- Held-out INTERNAL TEST SET only, matching this project's own standing
  Trust/Scenario Reporting Rule: this is a reliability-type question ("does
  the axis work"), not a design-facing full-dataset mining question, so it
  belongs on the test set. See ood_causal_coherence_full_dataset_analysis.py
  for the full-dataset, design-facing companion (which reuses this script's
  own ID/OOD numbers as its crosscheck baseline, the same relationship
  xai_full_dataset.py has to xai.py).

For each coherence definition and each of ID/OOD:
- corr(coherence, |error|), cutoff-free (the single cleanest number)
- the full trust_block scenario table (A/B/C/D coverage + enrichment),
  scored with the SAME pooled cutoff, restricted to that subgroup
- the Scenario-A-vs-B conditional test, restricted to that subgroup (only
  meaningful where both A and B are non-trivially populated within the
  subgroup -- OOD is ~4.9% of the test set, so small-n caveats apply and are
  reported via each scenario count, not hidden)

Do not tune the AD cutoff or either coherence cutoff to manufacture a bigger
ID/OOD gap -- report whichever way it comes out, per this project's standing
no-p-hacking discipline.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from config import RESULTS, XAI_ERROR_THRESHOLDS, XAI_PREDICTIONS_NPZ, XAI_RESULTS_JSON
from trust import trust_block
from validate_trust_axes import coherence_scenario_a_vs_b

AD_PREDICTIONS_NPZ = RESULTS / "trust_validation_predictions.npz"
AD_RESULTS_JSON = RESULTS / "trust_validation_results.json"
CAUSAL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_predictions.npz"
CAUSAL_RESULTS_JSON = RESULTS / "motif_causal_occlusion_results.json"
OUT_JSON = RESULTS / "ood_causal_coherence_results.json"


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined():
    """Join AD distance (all 42,844 test windows) against both coherence
    definitions (each restricted to its own shell-resolved subset, 41,379
    windows) on (chrom, start, end). Returns arrays aligned to the
    shell-resolved intersection, since both coherence values are required."""
    ad_d = np.load(AD_PREDICTIONS_NPZ)
    bin_d = np.load(XAI_PREDICTIONS_NPZ)
    causal_d = np.load(CAUSAL_PREDICTIONS_NPZ)

    resolved = causal_d["sampled_mask"] & ~np.isnan(causal_d["causal_coherence"])
    causal_key = _keys(causal_d["chrom"][resolved], causal_d["start"][resolved], causal_d["end"][resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}

    bin_key = _keys(bin_d["chrom"], bin_d["start"], bin_d["end"])
    ad_key = _keys(ad_d["chrom"], ad_d["start"], ad_d["end"])
    ad_idx = {k: i for i, k in enumerate(ad_key)}

    common_mask = np.array([(k in causal_idx) and (k in ad_idx) for k in bin_key])
    print(f"bin-overlap shell-resolved windows: {len(bin_key)}")
    print(f"causal shell-resolved windows:      {resolved.sum()}")
    print(f"AD-distance-scored windows:         {len(ad_key)}")
    print(f"joined on (chrom,start,end):        {common_mask.sum()}")

    causal_order = [causal_idx[k] for k in bin_key[common_mask]]
    ad_order = [ad_idx[k] for k in bin_key[common_mask]]

    bin_sub = {k: (bin_d[k][common_mask] if bin_d[k].ndim > 0 else bin_d[k]) for k in bin_d.files}
    causal_sub = {}
    for k, v in causal_d.items():
        if v.ndim != 1:
            continue
        if len(v) == len(resolved):
            causal_sub[k] = v[resolved][causal_order]
        elif len(v) == int(resolved.sum()):
            causal_sub[k] = v[causal_order]
    ad_distance = ad_d["ad_distance"][ad_order]
    ad_cutoff = float(ad_d["ad_cutoff"])

    if not np.allclose(bin_sub["y_true"], causal_sub["y_true"]):
        raise RuntimeError("y_true mismatch between bin-overlap and causal files on joined windows.")
    if not np.allclose(bin_sub["y_true"], ad_d["y_true"][ad_order]):
        raise RuntimeError("y_true mismatch between bin-overlap and AD-distance files on joined windows.")

    err = np.abs(bin_sub["y_pred"] - bin_sub["y_true"])
    return {
        "err": err, "ad_distance": ad_distance, "ad_cutoff": ad_cutoff,
        "cv_ratio": bin_sub["cv_ratio"],
        "bin_coherence": bin_sub["coherence"], "bin_cons_cut": float(bin_d["cons_cut"]), "bin_agr_cut": float(bin_d["agr_cut"]),
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
    d = load_joined()
    err, ad_distance, ad_cutoff = d["err"], d["ad_distance"], d["ad_cutoff"]
    id_mask = ad_distance <= ad_cutoff
    ood_mask = ~id_mask
    print(f"\nAD cutoff (reused from {AD_RESULTS_JSON.name}): {ad_cutoff:.4f}")
    print(f"ID: n={int(id_mask.sum())} ({id_mask.mean():.1%})   OOD: n={int(ood_mask.sum())} ({ood_mask.mean():.1%})")

    results = {
        "note": ("OOD defined by the already-calibrated AD cutoff from "
                  f"{AD_RESULTS_JSON.name}, not recalibrated here; each coherence "
                  "definition's own already-calibrated cutoffs are reused unchanged, "
                  "not recalibrated per ID/OOD subgroup."),
        "ad_cutoff": ad_cutoff,
        "n_total": int(len(err)), "n_id": int(id_mask.sum()), "n_ood": int(ood_mask.sum()),
        "ood_rate": float(ood_mask.mean()),
    }

    print("\n=== bin-overlap coherence ===")
    results["bin_overlap"] = {
        "id": report_subgroup("bin-overlap / ID", d["bin_coherence"], err, d["cv_ratio"], d["bin_cons_cut"], d["bin_agr_cut"], id_mask),
        "ood": report_subgroup("bin-overlap / OOD", d["bin_coherence"], err, d["cv_ratio"], d["bin_cons_cut"], d["bin_agr_cut"], ood_mask),
    }

    print("\n=== causal knockout coherence ===")
    results["causal"] = {
        "id": report_subgroup("causal / ID", d["causal_coherence"], err, d["cv_ratio"], d["causal_cons_cut"], d["causal_agr_cut"], id_mask),
        "ood": report_subgroup("causal / OOD", d["causal_coherence"], err, d["cv_ratio"], d["causal_cons_cut"], d["causal_agr_cut"], ood_mask),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
