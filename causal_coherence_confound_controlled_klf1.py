#!/usr/bin/env python
"""Does the top-decile residual signal that survived FIVE prior controls
(|y_true|, AD distance, n_instances, GC, repeat_frac -- 5-control partial
rho +0.057, p=2.5e-13 genome-wide, see causal_coherence_confound_controlled_
gc_repeat_results.json) get absorbed once KLF1's own per-window instance
count is added as a sixth control?

Motivated directly by causal_coherence_top_decile_tf_breakdown_results.json
(this session): KLF1 is the one outlier among the 10 K562 panel TFs -- every
other TF's instance count is LOWER in the top decile than in the rest of
A+B, but KLF1's is 3.6x higher (17.6 vs 4.9 mean instances/window), and
within the top decile alone KLF1's own count is the single strongest
correlate of causal_coherence (rho +0.460) and even correlates positively
with |error| there (rho +0.199). KLF1's own PSSM (8bp, FPR=0.001 threshold
3.43, far more permissive relative to its length than any other panel entry
-- GATA1 is 7bp at threshold 6.95) is consistent with a degenerate GC-box-
type motif racking up disproportionate chance hits in GC-rich windows,
which would make it partly redundant with GC content already tested -- but
GC only partially absorbed the residual, so this is a genuinely distinct
test, not an assumption.

Requires a small amount of NEW local computation (a KLF1-only motif rescan,
much faster than the 10-TF scan already run this session: benchmarked at
~0.23ms/window locally, so ~38s for the full ~164,585-window A+B
population, no caching needed). Everything else reuses already-computed
.npz files (xai_full_dataset_predictions.npz, motif_causal_occlusion_full_
dataset_predictions.npz, repeat_blindspot_predictions.npz) via one
self-contained loader here (does not chain through the other confound
scripts' own loaders, to guarantee row-for-row alignment between
chrom/start/end -- needed here to drive the KLF1 rescan -- and the
already-joined coherence/GC/repeat arrays, rather than trusting two
independently-implemented loaders to produce bit-identical ordering).

Same discipline as every confound-control pass in this project: report
whether the residual survives, do not tune the control set to make it
survive or fail.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import py2bit
from scipy import stats

from causal_coherence_confound_controlled import multi_control_partial_spearman, ols_with_se, zscore
from config import GENOME_2BIT, JASPAR_PFM_PATH, RESULTS
from motif_causal_occlusion import contiguous_spans
from motif_shell import load_k562_pssms, motif_hit_positions_by_tf

XAI_FULL_PREDICTIONS_NPZ = RESULTS / "xai_full_dataset_predictions.npz"
CAUSAL_FULL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz"
REPEAT_GC_NPZ = RESULTS / "repeat_blindspot_predictions.npz"
OUT_JSON = RESULTS / "causal_coherence_confound_controlled_klf1_results.json"

CONTROL_NAMES = ["abs_y_true", "ad_distance", "n_instances", "gc", "repeat_frac", "klf1_instances"]


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined_full():
    """Self-contained join of all three files, kept independent of the other
    confound-control scripts' own loaders (rather than chaining through
    them) so chrom/start/end -- needed here to drive the KLF1 rescan -- are
    guaranteed row-for-row aligned with every other array, not merely
    assumed to match some other script's own internal ordering."""
    xai_d = np.load(XAI_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    causal_d = np.load(CAUSAL_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    repeat_d = np.load(REPEAT_GC_NPZ, allow_pickle=True)

    if not (np.array_equal(repeat_d["chrom"], xai_d["chrom"])
            and np.array_equal(repeat_d["start"], xai_d["start"])
            and np.array_equal(repeat_d["end"], xai_d["end"])):
        raise RuntimeError(f"{REPEAT_GC_NPZ} is not index-aligned with {XAI_FULL_PREDICTIONS_NPZ}.")

    xai_resolved = xai_d["have_shell"]
    causal_resolved = causal_d["have_shell"]
    xai_key = _keys(xai_d["chrom"][xai_resolved], xai_d["start"][xai_resolved], xai_d["end"][xai_resolved])
    causal_key = _keys(causal_d["chrom"][causal_resolved], causal_d["start"][causal_resolved], causal_d["end"][causal_resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}
    common_mask = np.array([k in causal_idx for k in xai_key])
    order = [causal_idx[k] for k in xai_key[common_mask]]
    print(f"joined on (chrom,start,end): {common_mask.sum()}")

    chrom = xai_d["chrom"][xai_resolved][common_mask]
    start = xai_d["start"][xai_resolved][common_mask]
    end = xai_d["end"][xai_resolved][common_mask]
    y_true = xai_d["y_true"][xai_resolved][common_mask]
    y_pred = xai_d["y_pred"][xai_resolved][common_mask]
    cv_ratio = xai_d["cv_ratio"][xai_resolved][common_mask]
    bin_coherence = xai_d["coherence"][xai_resolved][common_mask]
    ad_distance = xai_d["ad_distance"][xai_resolved][common_mask]
    is_holdout = xai_d["is_holdout"][xai_resolved][common_mask]
    cons_cut = float(xai_d["cons_cut"])
    gc = repeat_d["gc"][xai_resolved][common_mask]
    repeat_frac = repeat_d["repeat_frac"][xai_resolved][common_mask]
    n_frac = repeat_d["n_frac"][xai_resolved][common_mask]

    causal_coherence = causal_d["causal_coherence"][causal_resolved][order]
    n_instances = causal_d["n_instances"][causal_resolved][order]
    y_true_check = causal_d["y_true"][causal_resolved][order]
    if not np.allclose(y_true, y_true_check):
        raise RuntimeError("y_true mismatch between xai_full_dataset and motif_causal_occlusion_"
                            "full_dataset on joined windows -- investigate before trusting anything below.")

    err = np.abs(y_pred - y_true)
    ab_mask = (cv_ratio <= cons_cut) & ~np.isnan(gc) & (n_frac < 0.5)
    return {
        "chrom": chrom, "start": start, "end": end, "err": err, "y_true": y_true,
        "causal_coherence": causal_coherence, "bin_coherence": bin_coherence,
        "n_instances": n_instances, "ad_distance": ad_distance, "gc": gc,
        "repeat_frac": repeat_frac, "is_holdout": is_holdout, "ab_mask": ab_mask,
    }


def scan_klf1(chrom, start, end, klf1_pssm, twobit_path):
    tb = py2bit.open(str(twobit_path))
    n = len(chrom)
    counts = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    pssm_dict = {"KLF1": klf1_pssm}
    for i in range(n):
        seq = tb.sequence(str(chrom[i]), int(start[i]), int(end[i]))
        counts[i] = len(contiguous_spans(motif_hit_positions_by_tf(seq, pssm_dict)["KLF1"]))
        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  scanned {i+1}/{n} ({rate:.0f}/s, ~{(n - i - 1) / rate:.0f}s remaining)")
    tb.close()
    return counts


def ols_block_6(err, coherence, y_true, ad_distance, n_instances, gc, repeat_frac, klf1):
    X = np.column_stack([
        np.ones(len(err)), zscore(coherence), zscore(np.abs(y_true)), zscore(ad_distance),
        zscore(n_instances), zscore(gc), zscore(repeat_frac), zscore(klf1),
    ])
    beta, se, t, p, r2 = ols_with_se(X, np.abs(err).astype(np.float64))
    names = ["intercept", "coherence_z", "abs_y_true_z", "ad_distance_z", "n_instances_z", "gc_z", "repeat_frac_z", "klf1_z"]
    return {"coefficients": {n: {"beta": float(b), "se": float(s), "p": float(pp)}
                              for n, b, s, pp in zip(names, beta, se, p)},
            "r_squared": float(r2)}


def analyze_6(label, err, coherence, y_true, ad_distance, n_instances, gc, repeat_frac, klf1):
    raw_rho, raw_p = stats.spearmanr(coherence, err)
    partial6 = multi_control_partial_spearman(
        coherence, err, [np.abs(y_true), ad_distance, n_instances, gc, repeat_frac, klf1], CONTROL_NAMES)
    ols = ols_block_6(err, coherence, y_true, ad_distance, n_instances, gc, repeat_frac, klf1)
    coh_row = ols["coefficients"]["coherence_z"]
    print(f"\n  [{label}] n={len(err)}")
    print(f"    raw corr(coherence, |error|):                       rho {raw_rho:+.4f}  p {raw_p:.3g}")
    print(f"    6-control partial rho (+ GC, repeat_frac, KLF1):    rho {partial6['partial_rho']:+.4f}  p {partial6['partial_p']:.3g}")
    print(f"    OLS coherence_z coefficient (7-predictor model):    beta {coh_row['beta']:+.4f}  p {coh_row['p']:.3g}  (R2={ols['r_squared']:.4f})")
    return {"n": int(len(err)), "raw_spearman": {"rho": float(raw_rho), "p": float(raw_p)},
            "partial_spearman_6control": partial6, "ols_7predictor": ols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=None, help="Smoke-test override.")
    args = ap.parse_args()

    for p in (XAI_FULL_PREDICTIONS_NPZ, CAUSAL_FULL_PREDICTIONS_NPZ, REPEAT_GC_NPZ, GENOME_2BIT, JASPAR_PFM_PATH):
        if not p.exists():
            raise SystemExit(f"Missing {p}.")

    d = load_joined_full()
    ab_mask = d["ab_mask"]
    idx_ab = np.where(ab_mask)[0]
    if args.max_windows is not None:
        idx_ab = idx_ab[:args.max_windows]
        print(f"SMOKE TEST: restricted to first {len(idx_ab)} A+B windows.")
    print(f"A+B (high-consensus, GC-valid) population to scan: n={len(idx_ab)}")

    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print("scanning KLF1 instance counts...")
    klf1_count = scan_klf1(d["chrom"][idx_ab], d["start"][idx_ab], d["end"][idx_ab], pssms["KLF1"], GENOME_2BIT)

    err = d["err"][idx_ab]
    y_true = d["y_true"][idx_ab]
    ad_dist = d["ad_distance"][idx_ab]
    n_inst = d["n_instances"][idx_ab]
    gc = d["gc"][idx_ab]
    repeat_frac = d["repeat_frac"][idx_ab]
    is_holdout = d["is_holdout"][idx_ab]
    causal_coh = d["causal_coherence"][idx_ab]
    bin_coh = d["bin_coherence"][idx_ab]

    results = {"note": ("Extends the 5-control model (|y_true|, AD distance, n_instances, GC, "
                         "repeat_frac) with KLF1's own per-window instance count as a 6th control, "
                         "motivated by causal_coherence_top_decile_tf_breakdown_results.json's own "
                         "finding that KLF1 is the one panel TF enriched (not depleted) in the "
                         "top decile and the strongest single within-top-decile correlate of "
                         "causal_coherence."),
        "controls": CONTROL_NAMES, "n_scanned": int(len(idx_ab))}

    for metric_key, coh in (("causal", causal_coh), ("bin_overlap", bin_coh)):
        print(f"\n########## {metric_key} ##########")
        results[metric_key] = {}

        print("=== whole A+B population ===")
        results[metric_key]["whole_ab"] = analyze_6(
            f"{metric_key} / whole A+B", err, coh, y_true, ad_dist, n_inst, gc, repeat_frac, klf1_count)

        order = np.argsort(-coh)
        top_idx = order[:len(order) // 10]
        print("\n=== top decile of this metric only ===")
        results[metric_key]["top_decile"] = analyze_6(
            f"{metric_key} / top decile", err[top_idx], coh[top_idx], y_true[top_idx], ad_dist[top_idx],
            n_inst[top_idx], gc[top_idx], repeat_frac[top_idx], klf1_count[top_idx])

        ho_mask = is_holdout
        if ho_mask.sum() >= 50:
            print("\n=== holdout-only subset (reliability-scoped) ===")
            results[metric_key]["holdout_only"] = analyze_6(
                f"{metric_key} / holdout only", err[ho_mask], coh[ho_mask], y_true[ho_mask], ad_dist[ho_mask],
                n_inst[ho_mask], gc[ho_mask], repeat_frac[ho_mask], klf1_count[ho_mask])
            ho_idx = np.where(ho_mask)[0]
            ho_order = ho_idx[np.argsort(-coh[ho_idx])]
            ho_top = ho_order[:max(1, len(ho_order) // 10)]
            results[metric_key]["holdout_top_decile"] = analyze_6(
                f"{metric_key} / holdout top decile", err[ho_top], coh[ho_top], y_true[ho_top], ad_dist[ho_top],
                n_inst[ho_top], gc[ho_top], repeat_frac[ho_top], klf1_count[ho_top])

    causal_5ctrl_top_rho, causal_5ctrl_top_p = 0.0570, 2.54e-13  # causal_coherence_confound_controlled_gc_repeat_results.json
    causal_6ctrl_top = results["causal"]["top_decile"]["partial_spearman_6control"]
    verdict = ("top-decile residual SURVIVES adding KLF1 count as a 6th control "
               f"(6-control partial rho {causal_6ctrl_top['partial_rho']:+.4f}, p={causal_6ctrl_top['partial_p']:.3g}) "
               "-- KLF1 density is not the (sole) missing explanation"
               if causal_6ctrl_top["partial_p"] < 0.05
               else "top-decile residual is ABSORBED once KLF1 count is added as a 6th control "
                    f"(6-control partial rho {causal_6ctrl_top['partial_rho']:+.4f}, p={causal_6ctrl_top['partial_p']:.3g}, "
                    "no longer significant) -- KLF1's disproportionate instance density in the top decile "
                    "explains a meaningful share of what the 5-control model left unexplained")
    results["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print(f"(for reference, the 5-control model's own top-decile partial rho/p was "
          f"{causal_5ctrl_top_rho:+.4f} / {causal_5ctrl_top_p:.3g})")
    results["five_control_top_decile_reference"] = {"rho": causal_5ctrl_top_rho, "p": causal_5ctrl_top_p}

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
