#!/usr/bin/env python
"""Per-panel-TF breakdown of the causal-coherence top-decile residual --
the one candidate flagged as untested after `n_instances`, `|y_true|`, AD
distance, GC content, and repeat-derived content (all tested, all partial
explanations, none fully absorbing the confound-controlled top-decile
signal: partial rho +0.057, p=2.5e-13 genome-wide even after all five
controls, see causal_coherence_confound_controlled_gc_repeat_results.json).

Question: is the top-decile signal driven by aggregate motif density alone
(already captured by n_instances), or does ONE specific K562 panel TF
(GATA1, GATA1::TAL1, TAL1::TCF3, KLF1, NFE2, MAF::NFE2, GATA2, RUNX1, MYB,
STAT5A::STAT5B) disproportionately drive it -- e.g. STAT5A::STAT5B
(downstream of the BCR-ABL lesion defining this cell line) or a
GATA1-dimer motif behaving differently than the rest of the panel?

Requires NEW local computation (motif re-scanning per window), unlike every
other script in this session which was a pure .npz join -- but cheap: a
per-window per-TF scan (motif_shell.motif_hit_positions_by_tf, added this
session, additive, does not touch the existing merged-scan function every
other script's coherence numbers depend on) benchmarked at ~2.6ms/window
locally, no GPU needed. Full A+B population (n=164,585) is ~7 minutes single-
threaded; --max-windows lets a smoke test run in seconds first, per this
project's own "local runs stay small/fast" discipline -- SMOKE-TESTED before
committing to the full run.

Two comparisons, both restricted to the A+B (high-consensus) population,
same fixed cutoff causal_coherence_tail_diagnosis.py / causal_coherence_
confound_controlled.py already use:
1. Top decile (by causal_coherence) vs. the remaining 90% of A+B -- for each
   panel TF: hit-rate (chi2), mean instance count (Mann-Whitney), and
   (within the top decile only) Spearman corr(per-TF instance count,
   causal_coherence) and corr(per-TF instance count, |error|) -- does this
   TF's count track the tail's own internal ranking or its elevated error?
2. Top decile vs. the single SAFEST decile (lowest mean|error|, decile 8 in
   the already-recorded profile) -- a sharper contrast specifically aimed at
   "what differs between the safest and the most error-prone-despite-high-
   coherence windows", not just "top decile vs. everything else".

Do not stop scanning TFs early or drop panel entries to manufacture a
cleaner story -- report all 10 panel TFs' numbers, whichever way they land.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import py2bit
from scipy import stats

from config import GENOME_2BIT, JASPAR_PFM_PATH, RESULTS
from motif_causal_occlusion import contiguous_spans
from motif_shell import K562_TF_PANEL, load_k562_pssms, motif_hit_positions_by_tf

XAI_FULL_PREDICTIONS_NPZ = RESULTS / "xai_full_dataset_predictions.npz"
CAUSAL_FULL_PREDICTIONS_NPZ = RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz"
OUT_JSON = RESULTS / "causal_coherence_top_decile_tf_breakdown_results.json"


def _keys(chrom, start, end):
    return np.array([f"{c}:{s}:{e}" for c, s, e in zip(chrom, start, end)])


def load_joined_with_coords():
    """Same join as causal_coherence_confound_controlled_full_dataset.
    load_joined_full_dataset(), but also keeps chrom/start/end -- needed here
    to re-fetch each window's own sequence for per-TF scanning, which no
    prior script in this session needed since they were all pure-.npz joins."""
    xai_d = np.load(XAI_FULL_PREDICTIONS_NPZ, allow_pickle=True)
    causal_d = np.load(CAUSAL_FULL_PREDICTIONS_NPZ, allow_pickle=True)

    xai_resolved = xai_d["have_shell"]
    causal_resolved = causal_d["have_shell"]
    xai_key = _keys(xai_d["chrom"][xai_resolved], xai_d["start"][xai_resolved], xai_d["end"][xai_resolved])
    causal_key = _keys(causal_d["chrom"][causal_resolved], causal_d["start"][causal_resolved], causal_d["end"][causal_resolved])
    causal_idx = {k: i for i, k in enumerate(causal_key)}
    common_mask = np.array([k in causal_idx for k in xai_key])
    order = [causal_idx[k] for k in xai_key[common_mask]]

    chrom = xai_d["chrom"][xai_resolved][common_mask]
    start = xai_d["start"][xai_resolved][common_mask]
    end = xai_d["end"][xai_resolved][common_mask]
    y_true = xai_d["y_true"][xai_resolved][common_mask]
    y_pred = xai_d["y_pred"][xai_resolved][common_mask]
    cv_ratio = xai_d["cv_ratio"][xai_resolved][common_mask]
    cons_cut = float(xai_d["cons_cut"])

    causal_coherence = causal_d["causal_coherence"][causal_resolved][order]
    y_true_check = causal_d["y_true"][causal_resolved][order]
    if not np.allclose(y_true, y_true_check):
        raise RuntimeError("y_true mismatch between xai_full_dataset and motif_causal_occlusion_"
                            "full_dataset on joined windows -- investigate before trusting anything below.")

    err = np.abs(y_pred - y_true)
    ab_mask = cv_ratio <= cons_cut
    return {"chrom": chrom, "start": start, "end": end, "err": err,
            "causal_coherence": causal_coherence, "ab_mask": ab_mask}


def scan_windows(chrom, start, end, pssms, twobit_path):
    """Per-window, per-TF instance count and hit-flag. Returns dict
    tf_name -> {"instance_count": array, "has_hit": array}."""
    tb = py2bit.open(str(twobit_path))
    n = len(chrom)
    counts = {tf: np.zeros(n, dtype=np.int32) for tf in pssms}
    t0 = time.time()
    for i in range(n):
        seq = tb.sequence(str(chrom[i]), int(start[i]), int(end[i]))
        per_tf = motif_hit_positions_by_tf(seq, pssms)
        for tf, covered in per_tf.items():
            counts[tf][i] = len(contiguous_spans(covered))
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  scanned {i+1}/{n} windows ({rate:.0f}/s, {elapsed:.0f}s elapsed, "
                  f"~{(n - i - 1) / rate:.0f}s remaining)")
    tb.close()
    return {tf: {"instance_count": counts[tf], "has_hit": (counts[tf] > 0)} for tf in pssms}


def compare_group(label, tf_scan, mask_top, mask_other):
    print(f"\n=== {label} ===")
    out = {}
    for tf in tf_scan:
        ic = tf_scan[tf]["instance_count"]
        hit = tf_scan[tf]["has_hit"]
        top_ic, other_ic = ic[mask_top], ic[mask_other]
        top_hit, other_hit = hit[mask_top], hit[mask_other]

        top_hit_rate = float(top_hit.mean())
        other_hit_rate = float(other_hit.mean())
        table = [[int(top_hit.sum()), int((~top_hit).sum())],
                 [int(other_hit.sum()), int((~other_hit).sum())]]
        # A near-universal-hit-rate TF (e.g. a short/permissive motif hitting
        # ~100% of windows either way) produces a degenerate table with a
        # zero-expected-frequency cell, which chi2_contingency rejects
        # outright rather than returning a (correctly non-significant) p --
        # caught directly during the smoke test. Fall back to Fisher's exact
        # test for any 2x2 table with a zero cell, which handles this
        # correctly instead of crashing.
        if any(c == 0 for row in table for c in row):
            _, chi2_p = stats.fisher_exact(table)
        else:
            _, chi2_p, _, _ = stats.chi2_contingency(table)

        mw_u, mw_p = stats.mannwhitneyu(top_ic, other_ic, alternative="two-sided") if top_ic.std() + other_ic.std() > 0 else (np.nan, 1.0)

        row = {
            "top_hit_rate": top_hit_rate, "other_hit_rate": other_hit_rate,
            "hit_rate_ratio": float(top_hit_rate / other_hit_rate) if other_hit_rate > 0 else float("inf"),
            "chi2_p": float(chi2_p),
            "top_mean_instances": float(top_ic.mean()), "other_mean_instances": float(other_ic.mean()),
            "mann_whitney_p": float(mw_p),
        }
        out[tf] = row
        print(f"  {tf:20s} hit-rate {top_hit_rate:.3f} vs {other_hit_rate:.3f} (chi2 p={chi2_p:.3g})  "
              f"mean-inst {top_ic.mean():.3f} vs {other_ic.mean():.3f} (MWU p={mw_p:.3g})")
    return out


def within_top_decile_correlations(tf_scan, mask_top, causal_coherence, err):
    print("\n=== within top decile: does this TF's count track causal_coherence's own "
          "ranking, or the top decile's elevated error, within the tail itself? ===")
    out = {}
    for tf in tf_scan:
        ic = tf_scan[tf]["instance_count"][mask_top]
        rho_coh, p_coh = stats.spearmanr(ic, causal_coherence[mask_top])
        rho_err, p_err = stats.spearmanr(ic, err[mask_top])
        out[tf] = {"corr_with_causal_coherence": {"rho": float(rho_coh), "p": float(p_coh)},
                    "corr_with_error": {"rho": float(rho_err), "p": float(p_err)}}
        print(f"  {tf:20s} vs causal_coherence: rho {rho_coh:+.3f} (p={p_coh:.3g})   "
              f"vs |error|: rho {rho_err:+.3f} (p={p_err:.3g})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=None,
                     help="Smoke-test override: scan only the first N A+B windows (by array order, "
                          "not by coherence rank) instead of the full population.")
    args = ap.parse_args()

    for p in (XAI_FULL_PREDICTIONS_NPZ, CAUSAL_FULL_PREDICTIONS_NPZ, GENOME_2BIT, JASPAR_PFM_PATH):
        if not p.exists():
            raise SystemExit(f"Missing {p}.")

    d = load_joined_with_coords()
    ab_mask = d["ab_mask"]
    print(f"A+B (high-consensus) population: n={int(ab_mask.sum())}")

    idx_ab = np.where(ab_mask)[0]
    if args.max_windows is not None:
        idx_ab = idx_ab[:args.max_windows]
        print(f"SMOKE TEST: restricted to first {len(idx_ab)} A+B windows.")

    coh = d["causal_coherence"]
    order = idx_ab[np.argsort(-coh[idx_ab])]
    n_decile = max(1, len(order) // 10)
    top_idx = order[:n_decile]
    rest_idx = order[n_decile:]

    # deciles, ascending, to find the single safest decile for the sharper contrast
    order_asc = idx_ab[np.argsort(coh[idx_ab])]
    bounds = np.linspace(0, len(order_asc), 11).astype(int)
    decile_slices = [order_asc[bounds[i]:bounds[i + 1]] for i in range(10)]
    decile_mean_err = [float(d["err"][sl].mean()) if len(sl) else np.nan for sl in decile_slices]
    safest_decile_i = int(np.argmin(decile_mean_err))
    safest_idx = decile_slices[safest_decile_i]
    print(f"decile mean|err| profile (this scan's own subset): "
          + "  ".join(f"d{i+1}={e:.3f}" for i, e in enumerate(decile_mean_err)))
    print(f"safest decile: {safest_decile_i + 1} (n={len(safest_idx)}, mean|err|={decile_mean_err[safest_decile_i]:.4f})")

    scan_idx = np.union1d(top_idx, np.union1d(rest_idx, safest_idx))
    print(f"\nloading K562 panel PSSMs and scanning {len(scan_idx)} windows "
          f"(top decile n={len(top_idx)}, rest n={len(rest_idx)}, safest decile n={len(safest_idx)})...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    scan_full = scan_windows(d["chrom"][scan_idx], d["start"][scan_idx], d["end"][scan_idx], pssms, GENOME_2BIT)

    # remap scan results (indexed 0..len(scan_idx)) back onto boolean masks over idx_ab's own index space
    n_scan = len(scan_idx)
    scan_pos = {int(gi): li for li, gi in enumerate(scan_idx)}
    top_local = np.array([scan_pos[int(gi)] for gi in top_idx])
    rest_local = np.array([scan_pos[int(gi)] for gi in rest_idx])
    safest_local = np.array([scan_pos[int(gi)] for gi in safest_idx])

    mask_top = np.zeros(n_scan, dtype=bool); mask_top[top_local] = True
    mask_rest = np.zeros(n_scan, dtype=bool); mask_rest[rest_local] = True
    mask_safest = np.zeros(n_scan, dtype=bool); mask_safest[safest_local] = True

    causal_coherence_scan = coh[scan_idx]
    err_scan = d["err"][scan_idx]

    results = {
        "note": ("Per-panel-TF breakdown of the causal-coherence top-decile residual, the last "
                 "untested candidate after n_instances/|y_true|/AD-distance/GC/repeat_frac (see "
                 "causal_coherence_confound_controlled_gc_repeat_results.json's still-significant "
                 "5-control partial rho +0.057, p=2.5e-13). New local motif-rescan computation "
                 "(motif_shell.motif_hit_positions_by_tf, added this session), not a pure .npz join."),
        "n_ab_population": int(len(idx_ab)), "n_top_decile": int(len(top_idx)),
        "n_scanned": int(n_scan), "safest_decile_index": safest_decile_i + 1,
        "decile_mean_err_profile": decile_mean_err,
        "panel": list(pssms.keys()),
    }

    results["top_vs_rest"] = compare_group("top decile vs. rest of A+B", scan_full, mask_top, mask_rest)
    results["top_vs_safest_decile"] = compare_group(
        f"top decile vs. safest decile (decile {safest_decile_i + 1})", scan_full, mask_top, mask_safest)
    results["within_top_decile_correlations"] = within_top_decile_correlations(
        scan_full, mask_top, causal_coherence_scan, err_scan)

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
