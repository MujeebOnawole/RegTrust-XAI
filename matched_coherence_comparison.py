#!/usr/bin/env python
"""Matched-population check: is causal-knockout coherence's larger A-vs-B
separation (see motif_causal_occlusion_results.json vs xai_results.json /
trust_validation_results.json) a real property of the causal metric, or an
artifact of the two metrics' independently-calibrated agreement cutoffs
carving out different-sized Scenario A populations (12.55% vs 10.06% of the
test set)?

Pure local analysis over two already-computed .npz files -- no model pass,
no new occlusion run, nothing that needs Bunya. Runs in seconds.

Two checks, in increasing order of how much they remove the cutoff choice
from the comparison:

1. Sanity check: confirm both runs' Scenario A+B population (cv_ratio <=
   consensus cutoff, i.e. everything BEFORE either coherence cutoff is
   applied) is the identical set of windows. If true, the two runs differ
   only in how they split that one fixed population into A/B, not in which
   windows are even eligible -- the trust.scenario_labels formula guarantees
   this whenever the same consensus cutoff is reused (it is: both files
   report the same cons_cut), so this is a hard check, not a soft one.

2. Cutoff-free comparison: within that one fixed A+B population, Spearman
   corr(coherence, -|error|) for each metric directly -- no agreement cutoff
   at all, so no population-size confound can exist. This is the cleanest
   answer to "is the causal axis intrinsically a better error-discriminator
   within the same population," independent of any calibration choice.

3. Matched-N comparison: within the same fixed A+B population, rank windows
   by each metric independently and take the top N as "A-matched" for a grid
   of N values (10/20/30/40/50% of the population, plus each metric's own
   original A count for direct comparability to the numbers already reported
   in xai_results.json / motif_causal_occlusion_results.json). Reuses
   validate_trust_axes.coherence_scenario_a_vs_b's exact statistical
   machinery (Mann-Whitney, Welch t, precision ratios) unchanged, just fed a
   locally-built "A"/"B" label array instead of trust.scenario_labels'
   consensus+agreement scenario, so results are directly comparable in
   format to every other coherence-vs-error number this project has already
   reported.

Do not tune N or re-run this with a different N grid to chase a bigger gap
for one metric -- report whichever way it comes out, per this project's
standing no-p-hacking discipline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import RESULTS, XAI_PREDICTIONS_NPZ
from validate_trust_axes import coherence_scenario_a_vs_b

CAUSAL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_predictions.npz"
OUT_JSON = RESULTS / "matched_coherence_comparison_results.json"
N_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined():
    bin_d = np.load(XAI_PREDICTIONS_NPZ)
    causal_d = np.load(CAUSAL_PREDICTIONS_NPZ)

    # restrict the causal file to shell-resolved windows only, matching
    # xai_predictions.npz's own row population
    resolved = causal_d["sampled_mask"] & ~np.isnan(causal_d["causal_coherence"])
    causal_key = _keys(causal_d["chrom"][resolved], causal_d["start"][resolved], causal_d["end"][resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}

    bin_key = _keys(bin_d["chrom"], bin_d["start"], bin_d["end"])
    common_mask = np.array([k in causal_idx for k in bin_key])
    print(f"bin-overlap shell-resolved windows: {len(bin_key)}")
    print(f"causal shell-resolved windows:      {resolved.sum()}")
    print(f"joined on (chrom,start,end):        {common_mask.sum()}")
    if common_mask.sum() != len(bin_key) or common_mask.sum() != resolved.sum():
        print("  WARNING: the two shell-resolved populations are not identical -- "
              "proceeding on the intersection only, but this itself is worth noting.")

    order = [causal_idx[k] for k in bin_key[common_mask]]
    # causal_d has two row-length conventions: arrays over ALL sampled windows
    # (len == len(resolved), need [resolved] first) and arrays already
    # restricted to shell-resolved windows only (len == resolved.sum(),
    # e.g. "scenario", already in the same relative order as [resolved]).
    causal_sub = {}
    for k, v in causal_d.items():
        if v.ndim != 1:
            continue
        if len(v) == len(resolved):
            causal_sub[k] = v[resolved][order]
        elif len(v) == int(resolved.sum()):
            causal_sub[k] = v[order]

    bin_sub = {k: (bin_d[k][common_mask] if bin_d[k].ndim > 0 else bin_d[k]) for k in bin_d.files}

    # sanity: same checkpoint, same test windows -> y_true/y_pred must match exactly
    if not np.allclose(bin_sub["y_true"], causal_sub["y_true"]) or not np.allclose(bin_sub["y_pred"], causal_sub["y_pred"]):
        raise RuntimeError("y_true/y_pred mismatch between the two files on joined windows -- "
                            "these should be identical (same checkpoint, same test set). Investigate before trusting anything below.")
    if not np.isclose(float(bin_d["cons_cut"]), float(causal_d["cons_cut"])):
        raise RuntimeError(f"consensus cutoffs differ ({float(bin_d['cons_cut'])} vs {float(causal_d['cons_cut'])}) -- "
                            "the whole point of this check assumes the same consensus cutoff was reused.")

    err = np.abs(bin_sub["y_pred"] - bin_sub["y_true"])
    cv_ratio = bin_sub["cv_ratio"]
    cons_cut = float(bin_d["cons_cut"])
    return {
        "err": err, "cv_ratio": cv_ratio, "cons_cut": cons_cut,
        "bin_coherence": bin_sub["coherence"], "bin_scenario": bin_sub["scenario"],
        "causal_coherence": causal_sub["causal_coherence"], "causal_scenario": causal_sub["scenario"],
    }


def matched_topN(coherence, err, ab_mask, n):
    """Within ab_mask (the fixed A+B population), label the top n windows by
    `coherence` as 'A' and the rest as 'B'. Same shape/semantics as
    trust.scenario_labels' A/B split, just rank-based instead of
    cutoff-based, so coherence_scenario_a_vs_b can be reused unchanged."""
    idx = np.where(ab_mask)[0]
    order = idx[np.argsort(-coherence[idx])]
    scen = np.full(len(coherence), "X", dtype="<U1")
    scen[order[:n]] = "A"
    scen[order[n:]] = "B"
    return scen


def main():
    d = load_joined()
    err, cv_ratio, cons_cut = d["err"], d["cv_ratio"], d["cons_cut"]
    ab_mask = cv_ratio <= cons_cut
    n_ab = int(ab_mask.sum())
    print(f"\nfixed A+B (high-consensus) population: n={n_ab} "
          f"(bin-overlap run reports {int((d['bin_scenario']=='A').sum() + (d['bin_scenario']=='B').sum())}, "
          f"causal run reports {int((d['causal_scenario']=='A').sum() + (d['causal_scenario']=='B').sum())})")

    results = {"n_ab_population": n_ab, "cons_cut": cons_cut}

    print("\n--- check 2: cutoff-free Spearman corr(coherence, -|error|) within fixed A+B population ---")
    from scipy import stats
    bin_rho, bin_p = stats.spearmanr(d["bin_coherence"][ab_mask], -err[ab_mask])
    causal_rho, causal_p = stats.spearmanr(d["causal_coherence"][ab_mask], -err[ab_mask])
    print(f"  bin-overlap coherence: rho {bin_rho:+.4f}  p {bin_p:.3g}")
    print(f"  causal coherence:      rho {causal_rho:+.4f}  p {causal_p:.3g}")
    results["cutoff_free_spearman_within_ab"] = {
        "bin_overlap": {"rho": float(bin_rho), "p": float(bin_p)},
        "causal": {"rho": float(causal_rho), "p": float(causal_p)},
    }

    print("\n--- check 3: matched-N top/bottom split within fixed A+B population ---")
    n_grid = sorted(set([int(round(f * n_ab)) for f in N_FRACTIONS]) |
                     {int((d["bin_scenario"] == "A").sum()), int((d["causal_scenario"] == "A").sum())})
    matched = {}
    for n in n_grid:
        if n <= 0 or n >= n_ab:
            continue
        print(f"\n  N={n} ({n/n_ab:.1%} of A+B population):")
        print("  bin-overlap:")
        bin_scen = matched_topN(d["bin_coherence"], err, ab_mask, n)
        bin_out = coherence_scenario_a_vs_b(bin_scen, err)
        print("  causal:")
        causal_scen = matched_topN(d["causal_coherence"], err, ab_mask, n)
        causal_out = coherence_scenario_a_vs_b(causal_scen, err)
        matched[str(n)] = {"bin_overlap": bin_out, "causal": causal_out}
    results["matched_topN"] = matched

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
