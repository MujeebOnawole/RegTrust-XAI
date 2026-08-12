#!/usr/bin/env python
"""Item 8 (project_status.md NEXT UP): validate the coherence and
applicability-domain axes against actual error, rather than reporting either
as meaningful just because xai.py's enrichment-factor framing looked right.

Two correlations, both Spearman, both computed on the internal chromosome-
holdout test set:

1. corr(coherence, |error|) -- does high motif-shell coherence (trust.py's
   localization_coherence) actually mark more accurate predictions, or does
   it only separate "motif signal present" from "motif signal absent"
   without that tracking error? Uses xai_predictions.npz directly (already
   has coherence + y_true/y_pred for the 41,379 shell-covered test windows)
   -- no new model pass needed.

2. corr(distance, |error|) -- does embedding distance from a query window to
   its nearest training-pool neighbor (trust.nn_distance, in the attribution
   checkpoint's pooled pre-head feature space, model.embed()) actually track
   error, the way applicability-domain axes are assumed to? This axis has
   never been run in this project before (project_status.md: "trust.py's AD
   formulas are written but nothing in the pipeline calls them yet") -- this
   script is the first time it is computed against real data.

Both are diagnostic, not confirmatory by construction: a null result here is
exactly as reportable as a positive one (see project_status.md's discipline
against tuning thresholds to manufacture a bigger gap). Do not re-run this
with a different embedding layer or reference pool size to chase significance.

3. AD-quantile risk stratification -- rho +0.278 is real but a single
   coefficient is a weak way to communicate what the AD axis is actually
   good for. Bins the test set into AD-distance percentile bands (nearest
   20% / 20-40 / 40-60 / 60-80 / 80-95 / top-5% "OOD" by the same
   AD_EMBED_PERCENTILE cutoff distance_vs_error already computes) and
   reports MAE/RMSE/precision per band, plus a single ID-vs-OOD precision
   comparison at each XAI_ERROR_THRESHOLDS value -- turns the correlation
   into a monotonic table a reader can act on directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from config import (
    AD_EMBED_PERCENTILE,
    AD_QUANTILE_EDGES,
    AD_REF_POOL_SIZE,
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_PREDICTIONS_NPZ,
    HOLDOUT_CHROMS,
    NUM_WORKERS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    TRUST_VALIDATION_PREDICTIONS_NPZ,
    TRUST_VALIDATION_RESULTS_JSON,
    XAI_ERROR_THRESHOLDS,
    XAI_PREDICTIONS_NPZ,
)
from data_module import WindowDataset, worker_init_fn
from trust import l2_normalize_rows, quantile_stratification
from xai import load_best_checkpoint


def embed_loader(model, loader, device):
    """(n, feat_dim) pooled pre-head embeddings, one row per window, in
    loader order -- same batching pattern final_eval.predict_ensemble uses,
    but a single forward through model.embed() (no head, no ensemble)."""
    out = []
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            out.append(model.embed(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def coherence_scenario_a_vs_b(scenario, err):
    """The CORRECT test of coherence's value, per this portfolio's own
    precedent (TRUST_FRAMEWORK_PRIOR_PAPERS.md, the Taste paper's reading of
    its own A=B accuracy tie): does coherence separate Scenario A from
    Scenario B WITHIN the already-high-consensus population, not the pooled
    corr(coherence, |error|) above, which is diluted by consensus already
    doing most of the work (D alone is ~half the test set on low consensus
    regardless of coherence). A pooled correlation understates this axis if
    reported alone -- see project_status.md's 2026-08-12 correction."""
    a, b = err[scenario == "A"], err[scenario == "B"]
    u = stats.mannwhitneyu(a, b, alternative="less")
    t = stats.ttest_ind(a, b, equal_var=False)
    out = {
        "n_a": int(len(a)), "n_b": int(len(b)),
        "mean_abs_error_a": float(a.mean()), "mean_abs_error_b": float(b.mean()),
        "mann_whitney_a_lt_b_p": float(u.pvalue),
        "welch_t_p": float(t.pvalue),
        "precision_ratio_a_over_b": {
            f"{thr:g}": float((a <= thr).mean() / (b <= thr).mean()) for thr in (0.1, 0.2, 0.3)
        },
    }
    print(f"  A (n={out['n_a']}) mean|err| {out['mean_abs_error_a']:.4f}  vs  "
          f"B (n={out['n_b']}) mean|err| {out['mean_abs_error_b']:.4f}  "
          f"(Mann-Whitney A<B p={out['mann_whitney_a_lt_b_p']:.3g})")
    return out


def spearman_report(x, y, label):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    res = stats.spearmanr(x, y)
    rho, p = float(res.statistic), float(res.pvalue)
    print(f"  {label:<32} n={mask.sum():>6}  rho {rho:+.4f}  p {p:.3g}")
    return {"n": int(mask.sum()), "rho": rho, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--eval-predictions", type=Path, default=EVAL_PREDICTIONS_NPZ)
    ap.add_argument("--xai-predictions", type=Path, default=XAI_PREDICTIONS_NPZ)
    ap.add_argument("--out", type=Path, default=TRUST_VALIDATION_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=TRUST_VALIDATION_PREDICTIONS_NPZ)
    ap.add_argument("--ref-pool-size", type=int, default=AD_REF_POOL_SIZE,
                     help="training-pool reference sample for the AD nearest-neighbor "
                          "distance; override for a quick local smoke test")
    ap.add_argument("--thresholds", type=float, nargs="+", default=XAI_ERROR_THRESHOLDS)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, args.eval_predictions, args.xai_predictions):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results = {}

    # ---- 1. corr(coherence, |error|) -- no model pass needed, already computed --------
    print("\n=== coherence vs error (xai_predictions.npz, shell-covered test windows) ===")
    xp = np.load(args.xai_predictions, allow_pickle=True)
    coh = xp["coherence"].astype(np.float64)
    err_coh = np.abs(xp["y_pred"].astype(np.float64) - xp["y_true"].astype(np.float64))
    results["coherence_vs_error"] = spearman_report(coh, err_coh, "corr(coherence, |error|)")

    print("\n=== coherence: Scenario A vs B conditional test (the correct test, see docstring) ===")
    results["coherence_scenario_a_vs_b_conditional"] = coherence_scenario_a_vs_b(xp["scenario"], err_coh)

    # ---- 2. corr(distance, |error|) -- needs embeddings, first run of this axis -------
    print("\nloading attribution checkpoint (same one xai.py used for coherence)...")
    model, _label_mean, _label_std = load_best_checkpoint(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("\nbuilding training-pool reference sample for nearest-neighbor distance...")
    d = np.load(args.windows_npz, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    ref_size = min(args.ref_pool_size, len(pool_idx))
    ref_idx = rng.choice(pool_idx, size=ref_size, replace=False)
    ref_ds = WindowDataset(d, ref_idx)
    ref_loader = DataLoader(ref_ds, batch_size=128, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    ref_embed = embed_loader(model, ref_loader, device)
    ref_embed_n = l2_normalize_rows(ref_embed)
    print(f"  reference pool: {ref_size} windows, embedding dim {ref_embed.shape[1]}")

    print("\nembedding the internal test set (all windows, not just shell-covered)...")
    ep = np.load(args.eval_predictions, allow_pickle=True)
    test_windows = {
        "chrom": ep["chrom"], "start": ep["start"], "end": ep["end"],
        "label": ep["y_true"],
    }
    test_ds = WindowDataset(test_windows, np.arange(len(ep["chrom"])))
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    test_embed = embed_loader(model, test_loader, device)
    test_embed_n = l2_normalize_rows(test_embed)
    print(f"  test set: {len(test_embed)} windows")

    print("\ncomputing nearest-neighbor cosine distance (test -> reference pool)...")
    sims = test_embed_n @ ref_embed_n.T          # (n_test, n_ref)
    nn_dist_test = 1.0 - sims.max(axis=1)

    # leave-one-out reference-pool self-distances, for the AD_EMBED_PERCENTILE cutoff --
    # same calibration procedure trust.calibrate_nn_cutoff uses, done here in one matmul
    # instead of the O(n^2) python loop that function uses (equivalent result, vectorized).
    ref_sims = ref_embed_n @ ref_embed_n.T
    np.fill_diagonal(ref_sims, -np.inf)
    nn_dist_ref = 1.0 - ref_sims.max(axis=1)
    ad_cutoff = float(np.percentile(nn_dist_ref, AD_EMBED_PERCENTILE))
    ood_rate = float((nn_dist_test > ad_cutoff).mean())
    print(f"  AD cutoff ({AD_EMBED_PERCENTILE:.0f}th pct of ref self-distances): "
          f"{ad_cutoff:.4f}; test-set OOD rate at this cutoff: {ood_rate:.1%}")

    err_test = np.abs(ep["y_pred"].astype(np.float64) - ep["y_true"].astype(np.float64))
    print("\n=== distance vs error (all internal test windows) ===")
    results["distance_vs_error"] = spearman_report(nn_dist_test, err_test, "corr(distance, |error|)")
    results["distance_vs_error"]["ad_cutoff"] = ad_cutoff
    results["distance_vs_error"]["ad_embed_percentile"] = AD_EMBED_PERCENTILE
    results["distance_vs_error"]["ood_rate_at_cutoff"] = ood_rate
    results["distance_vs_error"]["ref_pool_size"] = ref_size
    results["distance_vs_error"]["embed_dim"] = int(ref_embed.shape[1])

    print("\n=== AD-quantile risk stratification ===")
    results["ad_quantile_stratification"] = quantile_stratification(
        nn_dist_test, err_test, AD_QUANTILE_EDGES, args.thresholds, id_ood_cutoff=ad_cutoff)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        chrom=ep["chrom"], start=ep["start"], end=ep["end"],
        y_true=ep["y_true"], y_pred=ep["y_pred"],
        ad_distance=nn_dist_test, ad_cutoff=ad_cutoff,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
