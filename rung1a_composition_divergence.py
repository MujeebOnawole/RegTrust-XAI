#!/usr/bin/env python
"""Item 9, rung 1a: composition-divergence stratification of the chromosome-
holdout test set (project_status.md, "Graded distribution-shift design").

WHAT THIS TESTS, AND WHY IT IS NOT JUST A SECOND COPY OF THE AD ANALYSIS.
validate_trust_axes.py already showed error grows with the MODEL'S OWN
embedding distance from the training pool. That is a real result, but on
its own it cannot rule out a circularity concern: the model's embedding is
learned FROM this same task, so "distance in the model's own space predicts
its own error" is a weaker claim than "distance under a plain, model-
independent description of the sequence predicts error." This script
answers the model-independent version: bin the test set by k-mer (tetra-
nucleotide) composition divergence from the training pool's own aggregate
composition (sequence_composition.py's Jensen-Shannon divergence, computed
before any model touches the sequence) and ask whether error still grows
with that severity axis, and whether this severity axis agrees with AD
distance at all (they could, in principle, be measuring different things).

Same assay, real labels throughout -- unlike rung 1b (motif-shuffled
synthetic controls), which has no real labels to test error against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import py2bit
from scipy import stats

from config import (
    AD_QUANTILE_EDGES,
    COMPOSITION_KMER,
    COMPOSITION_REF_POOL_SIZE,
    EVAL_PREDICTIONS_NPZ,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    RUNG1A_PREDICTIONS_NPZ,
    RUNG1A_RESULTS_JSON,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    TRUST_VALIDATION_PREDICTIONS_NPZ,
    XAI_ERROR_THRESHOLDS,
)
from sequence_composition import jensen_shannon_divergence, kmer_counts, kmer_distribution, kmer_index
from trust import quantile_stratification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--eval-predictions", type=Path, default=EVAL_PREDICTIONS_NPZ)
    ap.add_argument("--trust-validation-predictions", type=Path, default=TRUST_VALIDATION_PREDICTIONS_NPZ,
                     help="for the AD-distance cross-check; optional, skipped if missing")
    ap.add_argument("--out", type=Path, default=RUNG1A_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=RUNG1A_PREDICTIONS_NPZ)
    ap.add_argument("--kmer", type=int, default=COMPOSITION_KMER)
    ap.add_argument("--ref-pool-size", type=int, default=COMPOSITION_REF_POOL_SIZE,
                     help="training-pool reference sample for the aggregate composition; "
                          "override for a quick local smoke test")
    ap.add_argument("--thresholds", type=float, nargs="+", default=XAI_ERROR_THRESHOLDS)
    ap.add_argument("--max-test-windows", type=int, default=None,
                     help="cap the test set for a quick local smoke test; omit for the full run")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, args.eval_predictions, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps first.")

    idx = kmer_index(args.kmer)
    print(f"k={args.kmer}, {len(idx)} possible k-mers")

    tb = py2bit.open(str(GENOME_2BIT))

    print("\nbuilding the training-pool AGGREGATE k-mer distribution (the reference "
          "every test window is compared against)...")
    d = np.load(args.windows_npz, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    ref_size = min(args.ref_pool_size, len(pool_idx))
    ref_idx = rng.choice(pool_idx, size=ref_size, replace=False)
    ref_counts = np.zeros(len(idx), dtype=np.float64)
    for i in ref_idx:
        seq = tb.sequence(str(chrom_all[i]), int(d["start"][i]), int(d["end"][i]))
        ref_counts += kmer_counts(seq, args.kmer, idx)
    ref_dist = (ref_counts + 1.0) / (ref_counts.sum() + len(idx))
    print(f"  reference pool: {ref_size} windows, aggregate {args.kmer}-mer distribution built")

    print("\ncomputing per-window k-mer divergence for the internal test set "
          "(all windows, same set validate_trust_axes.py scored)...")
    ep = np.load(args.eval_predictions, allow_pickle=True)
    test_chrom, test_start, test_end = ep["chrom"], ep["start"], ep["end"]
    y_true, y_pred = ep["y_true"], ep["y_pred"]
    if args.max_test_windows is not None:
        n = min(args.max_test_windows, len(test_chrom))
        test_chrom, test_start, test_end = test_chrom[:n], test_start[:n], test_end[:n]
        y_true, y_pred = y_true[:n], y_pred[:n]
        print(f"  --max-test-windows override: scoring only the first {n} test windows")
    divergence = np.empty(len(test_chrom), dtype=np.float64)
    for i in range(len(test_chrom)):
        seq = tb.sequence(str(test_chrom[i]), int(test_start[i]), int(test_end[i]))
        window_dist = kmer_distribution(seq, args.kmer, idx, pseudocount=1.0)
        divergence[i] = jensen_shannon_divergence(window_dist, ref_dist)
    tb.close()
    print(f"  {len(divergence)} test windows scored; divergence range "
          f"[{divergence.min():.4f}, {divergence.max():.4f}]")

    err_test = np.abs(y_pred.astype(np.float64) - y_true.astype(np.float64))

    results = {"kmer": args.kmer, "ref_pool_size": ref_size}

    print("\n=== corr(composition divergence, |error|) ===")
    res = stats.spearmanr(divergence, err_test)
    results["divergence_vs_error"] = {"n": int(len(divergence)), "rho": float(res.statistic), "p": float(res.pvalue)}
    print(f"  n={len(divergence):>6}  rho {res.statistic:+.4f}  p {res.pvalue:.3g}")

    print("\n=== composition-divergence quantile risk stratification ===")
    results["quantile_stratification"] = quantile_stratification(
        divergence, err_test, AD_QUANTILE_EDGES, args.thresholds)

    ad_distance = None
    if Path(args.trust_validation_predictions).exists():
        tvp = np.load(args.trust_validation_predictions, allow_pickle=True)
        same_order = (len(tvp["chrom"]) == len(test_chrom) and
                      np.array_equal(tvp["chrom"], test_chrom) and
                      np.array_equal(tvp["start"], test_start))
        if same_order:
            ad_distance = tvp["ad_distance"].astype(np.float64)
            print("\n=== composition divergence vs AD distance (do the two severity "
                  "axes agree?) ===")
            res2 = stats.spearmanr(divergence, ad_distance)
            results["divergence_vs_ad_distance"] = {
                "n": int(len(divergence)), "rho": float(res2.statistic), "p": float(res2.pvalue)}
            print(f"  n={len(divergence):>6}  rho {res2.statistic:+.4f}  p {res2.pvalue:.3g}")
        else:
            print("\n  SKIPPED divergence-vs-AD-distance: chrom/start order does not match "
                  f"{args.trust_validation_predictions} (rerun validate_trust_axes.py first if needed)")
    else:
        print(f"\n  SKIPPED divergence-vs-AD-distance: {args.trust_validation_predictions} not found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    save_kwargs = dict(
        chrom=test_chrom, start=test_start, end=test_end,
        y_true=y_true, y_pred=y_pred, divergence=divergence,
    )
    if ad_distance is not None:
        save_kwargs["ad_distance"] = ad_distance
    np.savez(args.pred_out, **save_kwargs)
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
