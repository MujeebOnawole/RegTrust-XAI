#!/usr/bin/env python
"""FUTURE WORK, not part of the current pipeline DAG and not used by the
JBI manuscript. Extends xai.py's trust taxonomy from the held-out internal
test set (n=42,844) to the FULL 517,790-window dataset (training pool +
holdout combined), per the Trust/Scenario Reporting Rule this author's other
projects follow: compute XAI on the full dataset for design-facing insight
(e.g. which genome-wide regions/motif contexts the model treats as
reliable), while reliability claims (Scenario coverage/accuracy percentages
reported in the manuscript) stay test-set-only, exactly as already done in
xai.py/xai_results.json. This script does NOT recompute or touch that
test-set result.

REUSES the already-calibrated cutoffs from results/xai_results.json (the
same consensus/agreement percentile cutoffs the manuscript reports) rather
than re-deriving them here -- scoring the full dataset against a different
calibration would make the two runs' scenario labels incomparable.

COST WARNING: occlusion attribution is 64 forward passes per window and is
this pipeline's dominant cost. The existing xai.py run covers ~44,844
windows (42,844 test + 2,000 calibration). This script covers all 517,790
windows, ~11.5x that count -- wall time has NOT been profiled at this scale
and the SLURM script's --time is an unverified guess (see its own header).
Run a short --n-windows smoke test first if wall time is a concern before
trusting the full submission.

Every window gets a `pool` tag (`"holdout_test"` if its chromosome is in
config.HOLDOUT_CHROMS, else `"train_pool"`) so downstream analysis can break
out design-facing findings by partition without re-deriving the split, and
so nobody accidentally quotes a full-dataset scenario-coverage number as a
reliability claim, the exact confusion the Trust/Scenario Reporting Rule
exists to prevent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import py2bit
import torch
from torch.utils.data import DataLoader

from config import (
    AD_REF_POOL_SIZE,
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_RESULTS_JSON,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    JASPAR_PFM_PATH,
    NUM_WORKERS,
    RESULTS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    XAI_ERROR_THRESHOLDS,
    XAI_RESULTS_JSON,
)
from data_module import WindowDataset, worker_init_fn
from final_eval import load_ensemble, predict_ensemble
from motif_shell import K562_TF_PANEL, load_k562_pssms
from trust import cv_ratio_from_stats, l2_normalize_rows, trust_block
from xai import load_best_checkpoint, window_coherence


def embed_loader(model, loader, device):
    out = []
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            out.append(model.embed(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--xai-results", type=Path, default=XAI_RESULTS_JSON,
                     help="already-calibrated consensus/agreement cutoffs are read from here")
    ap.add_argument("--eval-results", type=Path, default=EVAL_RESULTS_JSON,
                     help="pop_std is read from here, matching the test-set run exactly")
    ap.add_argument("--ref-pool-size", type=int, default=AD_REF_POOL_SIZE)
    ap.add_argument("--out", type=Path, default=RESULTS / "xai_full_dataset_results.json")
    ap.add_argument("--pred-out", type=Path, default=RESULTS / "xai_full_dataset_predictions.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=XAI_ERROR_THRESHOLDS)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--n-windows", type=int, default=None,
                     help="score only the first N windows -- smoke test / wall-time profiling only")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, args.xai_results, args.eval_results, GENOME_2BIT, JASPAR_PFM_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps (including xai.py) first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    xai_results = json.loads(args.xai_results.read_text())
    cons_cut = xai_results["trust"]["internal_test"]["cutoffs"]["consensus"]
    agr_cut = xai_results["trust"]["internal_test"]["cutoffs"]["agreement"]
    print(f"reusing calibrated cutoffs from {args.xai_results}: "
          f"consensus {cons_cut:.4f}, agreement {agr_cut:.4f}")

    eval_results = json.loads(args.eval_results.read_text())
    pop_std = float(eval_results["pop_std"])
    print(f"reusing pop_std from {args.eval_results}: {pop_std:.4f}")

    print("\nloading full windows table...")
    d = np.load(args.windows_npz, allow_pickle=True)
    n_total = len(d["chrom"]) if args.n_windows is None else min(args.n_windows, len(d["chrom"]))
    all_idx = np.arange(n_total)
    chrom_all, start_all, end_all = d["chrom"][all_idx], d["start"][all_idx], d["end"][all_idx]
    is_holdout = np.isin(chrom_all, HOLDOUT_CHROMS)
    print(f"  {n_total} windows total ({is_holdout.sum()} holdout / {(~is_holdout).sum()} train-pool)")

    print("\nloading 5-model ensemble (for consensus)...")
    ensemble, ens_label_means, ens_label_stds = load_ensemble(args.checkpoint_dir, ENSEMBLE_SIZE, device)
    full_ds = WindowDataset(d, all_idx)
    full_loader = DataLoader(full_ds, batch_size=128, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    print("  running ensemble forward pass over the full dataset...")
    y_pred, y_pred_std, y_true = predict_ensemble(ensemble, full_loader, device, ens_label_means, ens_label_stds)
    err_all = np.abs(y_pred - y_true)
    cvr_all = cv_ratio_from_stats(y_pred_std, pop_std)

    print("\nloading attribution checkpoint (same one xai.py used)...")
    best_model, best_label_mean, best_label_std = load_best_checkpoint(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("loading K562 TF panel PSSMs...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    print("\nbuilding applicability-domain reference pool (same seed/size as validate_trust_axes.py)...")
    pool_idx_for_ref = np.where(~np.isin(d["chrom"], HOLDOUT_CHROMS))[0]
    ref_size = min(args.ref_pool_size, len(pool_idx_for_ref))
    ref_idx = rng.choice(pool_idx_for_ref, size=ref_size, replace=False)
    ref_ds = WindowDataset(d, ref_idx)
    ref_loader = DataLoader(ref_ds, batch_size=128, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    ref_embed_n = l2_normalize_rows(embed_loader(best_model, ref_loader, device))
    print(f"  reference pool: {ref_size} windows")

    print("\nembedding the full dataset (for AD distance)...")
    full_embed_n = l2_normalize_rows(embed_loader(best_model, full_loader, device))
    sims = full_embed_n @ ref_embed_n.T
    ad_distance_all = 1.0 - sims.max(axis=1)

    print("\ncomputing coherence on the FULL dataset (occlusion attribution, the slow step -- "
          f"{n_total} windows x 64 forward passes)...")
    tb = py2bit.open(str(GENOME_2BIT))
    coh_all = np.empty(n_total, dtype=np.float64)
    have_shell = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        c = window_coherence(best_model, tb, chrom_all[i], start_all[i], end_all[i], pssms, device)
        coh_all[i] = c if c is not None else np.nan
        have_shell[i] = c is not None
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n_total} windows scored ({have_shell[:i + 1].sum()} with a resolvable shell)")
    tb.close()
    print(f"  full dataset windows with a resolvable motif shell: {have_shell.sum()}/{n_total}")

    print("\n=== TRUST TAXONOMY, FULL DATASET (design-facing only -- NOT a reliability claim) ===")
    results = {
        "attribution_checkpoint_label_mean": best_label_mean,
        "attribution_checkpoint_label_std": best_label_std,
        "pop_std": pop_std,
        "cutoffs_reused_from": str(args.xai_results),
        "cutoffs": {"consensus": cons_cut, "agreement": agr_cut},
        "n_total": n_total,
        "n_holdout_test": int(is_holdout.sum()),
        "n_train_pool": int((~is_holdout).sum()),
        "thresholds": args.thresholds,
        "trust": {},
    }
    scen_all = np.array([])
    valid = have_shell
    if valid.sum() > 0:
        block, scen_all = trust_block(
            cvr_all[valid], coh_all[valid], err_all[valid], cons_cut, agr_cut,
            args.thresholds, "full dataset (shell-covered only)")
        results["trust"]["full_dataset"] = block
        results["trust"]["coverage_note"] = (
            f"{valid.sum()}/{n_total} full-dataset windows have a resolvable motif shell; "
            f"the rest are excluded from trust labelling, not defaulted to a scenario."
        )

        # same breakdown, restricted to the holdout test partition only, as a sanity check
        # that this script reproduces xai.py's own reported numbers when filtered the same way
        test_valid = valid & is_holdout
        if test_valid.sum() > 0:
            test_block, _ = trust_block(
                cvr_all[test_valid], coh_all[test_valid], err_all[test_valid], cons_cut, agr_cut,
                args.thresholds, "full-dataset run, holdout-test subset only (cross-check vs xai.py)")
            results["trust"]["holdout_test_subset_crosscheck"] = test_block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        chrom=chrom_all, start=start_all, end=end_all,
        y_true=y_true, y_pred=y_pred,
        cv_ratio=cvr_all, coherence=coh_all, ad_distance=ad_distance_all,
        have_shell=have_shell, is_holdout=is_holdout,
        scenario=scen_all if valid.sum() > 0 else np.array([]),
        scenario_valid_mask=valid,
        cons_cut=cons_cut, agr_cut=agr_cut,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
