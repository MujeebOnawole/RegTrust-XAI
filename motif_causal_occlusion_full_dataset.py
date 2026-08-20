#!/usr/bin/env python
"""FULL-DATASET companion to motif_causal_occlusion.py, same relationship
xai_full_dataset.py has to xai.py: extends the causal, motif-span-exact
occlusion-coherence ablation from the held-out internal test set
(n=42,844) to the FULL 517,790-window dataset (training pool + holdout
combined), design-facing only, per the same Trust/Scenario Reporting Rule
xai_full_dataset.py already documents -- reliability-type numbers (does the
ablation change the manuscript's Scenario coverage/accuracy) are the
TEST-SET run's job (motif_causal_occlusion.py / motif_causal_occlusion_
results.json); this script answers the design-facing question (does the
causal-coherence result and the context-sensitivity/additivity finding hold
up genome-wide, not just on the held-out chromosomes).

MUST BE RUN AFTER motif_causal_occlusion.py, NOT INSTEAD OF IT. Reuses that
script's OWN calibrated causal-coherence agreement cutoff (read from
results/motif_causal_occlusion_results.json), the same "reuse the already-
calibrated cutoff, do not re-derive it" discipline xai_full_dataset.py uses
for xai_results.json's cutoffs -- scoring the full dataset against a
DIFFERENT calibration would make the two runs' scenario labels
incomparable. The consensus cutoff is still reused from results/
xai_results.json directly (unchanged from every other script in this
project).

COST. Same per-window cost as motif_causal_occlusion.py (~10-15 forward
passes/window, versus xai_full_dataset.py's fixed 64/window), applied to
~11.5x more windows (517,790 vs ~44,844). Run --n-windows first for a wall-
time estimate before committing to the full submission, the same discipline
slurm/11's own header recommends.
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
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_RESULTS_JSON,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    JASPAR_PFM_PATH,
    RESULTS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    XAI_ERROR_THRESHOLDS,
    XAI_RESULTS_JSON,
)
from data_module import WindowDataset, worker_init_fn
from final_eval import load_ensemble, predict_ensemble
from motif_causal_occlusion import (
    MODULE_GAP_BP,
    N_RANDOM_CONTROLS,
    additivity_summary,
    run_over_indices,
)
from motif_shell import K562_TF_PANEL, load_k562_pssms
from trust import cv_ratio_from_stats, trust_block
from xai import load_best_checkpoint

CAUSAL_TEST_RESULTS_JSON = RESULTS / "motif_causal_occlusion_results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--xai-results", type=Path, default=XAI_RESULTS_JSON,
                     help="consensus cutoff is read from here, same convention as xai_full_dataset.py")
    ap.add_argument("--causal-test-results", type=Path, default=CAUSAL_TEST_RESULTS_JSON,
                     help="the causal-coherence agreement cutoff is read from here -- run "
                          "motif_causal_occlusion.py first")
    ap.add_argument("--eval-results", type=Path, default=EVAL_RESULTS_JSON,
                     help="pop_std is read from here, matching the test-set run exactly")
    ap.add_argument("--out", type=Path, default=RESULTS / "motif_causal_occlusion_full_dataset_results.json")
    ap.add_argument("--pred-out", type=Path,
                     default=RESULTS / "motif_causal_occlusion_full_dataset_predictions.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=XAI_ERROR_THRESHOLDS)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--n-random-controls", type=int, default=N_RANDOM_CONTROLS)
    ap.add_argument("--module-gap-bp", type=int, default=MODULE_GAP_BP)
    ap.add_argument("--n-windows", type=int, default=None,
                     help="score only the first N windows -- smoke test / wall-time profiling only")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, args.xai_results, args.causal_test_results, args.eval_results,
              GENOME_2BIT, JASPAR_PFM_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps (including xai.py and "
                              f"motif_causal_occlusion.py) first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    xai_results = json.loads(args.xai_results.read_text())
    cons_cut = float(xai_results["trust"]["internal_test"]["cutoffs"]["consensus"])
    print(f"reusing consensus cutoff from {args.xai_results}: {cons_cut:.4f}")

    causal_test_results = json.loads(args.causal_test_results.read_text())
    agr_cut = float(causal_test_results["causal_agreement_cutoff"])
    print(f"reusing causal-coherence agreement cutoff from {args.causal_test_results}: {agr_cut:.4f}")

    eval_results = json.loads(args.eval_results.read_text())
    pop_std = float(eval_results["pop_std"])
    print(f"reusing pop_std from {args.eval_results}: {pop_std:.4f}")

    print("\nloading K562 TF panel PSSMs...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    print("loading attribution checkpoint (same one xai.py/motif_causal_occlusion.py used)...")
    model, label_mean, label_std = load_best_checkpoint(args.checkpoint_dir, ENSEMBLE_SIZE, device)

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
                              num_workers=4, worker_init_fn=worker_init_fn)
    print("  running ensemble forward pass over the full dataset (for consensus/error only)...")
    y_pred, y_pred_std, y_true = predict_ensemble(ensemble, full_loader, device, ens_label_means, ens_label_stds)
    err_all = np.abs(y_pred - y_true)
    cvr_all = cv_ratio_from_stats(y_pred_std, pop_std)

    print("\ncomputing causal (motif-span-exact) occlusion on the FULL dataset "
          f"({n_total} windows, ~10-15 forward passes/window -- the slow step)...")
    tb = py2bit.open(str(GENOME_2BIT))
    causal_all, n_inst_all, n_mod_all, additivity_records = run_over_indices(
        model, tb, chrom_all, start_all, end_all, all_idx, pssms, device, rng,
        args.n_random_controls, args.module_gap_bp, "full dataset")
    tb.close()
    have_shell = ~np.isnan(causal_all)
    print(f"  full dataset windows with a resolvable motif shell: {have_shell.sum()}/{n_total}")

    print("\n=== TRUST TAXONOMY, causal-coherence variant, FULL DATASET "
          "(design-facing only -- NOT a reliability claim) ===")
    results = {
        "note": ("Design-facing full-dataset extension of motif_causal_occlusion.py's causal-"
                 "coherence ablation, matching xai_full_dataset.py's own relationship to xai.py. "
                 "Reliability-type comparisons against xai_results.json/trust_validation_results.json "
                 "belong to the held-out-test-set run only (motif_causal_occlusion_results.json); "
                 "this file is for design-facing mining and the holdout-subset crosscheck below."),
        "consensus_cutoff_reused_from": str(args.xai_results),
        "causal_agreement_cutoff_reused_from": str(args.causal_test_results),
        "consensus_cutoff": cons_cut,
        "causal_agreement_cutoff": agr_cut,
        "module_gap_bp": args.module_gap_bp,
        "n_random_controls": args.n_random_controls,
        "n_total": n_total,
        "n_holdout_test": int(is_holdout.sum()),
        "n_train_pool": int((~is_holdout).sum()),
        "thresholds": args.thresholds,
    }

    scen_all = np.array([])
    valid = have_shell
    if valid.sum() > 0:
        block, scen_all = trust_block(
            cvr_all[valid], causal_all[valid], err_all[valid], cons_cut, agr_cut,
            args.thresholds, "causal coherence, full dataset (shell-covered)")
        results["trust"] = {"full_dataset": block}
        results["trust"]["coverage_note"] = (
            f"{valid.sum()}/{n_total} full-dataset windows have a resolvable motif shell; "
            f"the rest are excluded from trust labelling, not defaulted to a scenario."
        )

        # cross-check: restricting this full-dataset run to the holdout-test partition only
        # should reproduce motif_causal_occlusion.py's own reported numbers -- same purpose as
        # xai_full_dataset.py's holdout_test_subset_crosscheck block.
        test_valid = valid & is_holdout
        if test_valid.sum() > 0:
            test_block, _ = trust_block(
                cvr_all[test_valid], causal_all[test_valid], err_all[test_valid], cons_cut, agr_cut,
                args.thresholds, "full-dataset run, holdout-test subset only (cross-check)")
            results["trust"]["holdout_test_subset_crosscheck"] = test_block
    else:
        results["trust"] = None

    print("\n=== context-sensitivity / additivity, FULL DATASET ===")
    results["additivity"] = additivity_summary(additivity_records)
    a = results["additivity"]
    if a.get("n_multi_instance_modules", 0) > 0:
        print(f"  n={a['n_multi_instance_modules']}  mean|joint| {a['mean_abs_joint_delta']:.5f}  "
              f"mean|summed instances| {a['mean_abs_summed_instance_deltas']:.5f}  "
              f"superadditive {a['frac_superadditive']:.1%}  subadditive {a['frac_subadditive']:.1%}  "
              f"Wilcoxon p {a['wilcoxon_joint_vs_summed_p']:.3g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        chrom=chrom_all, start=start_all, end=end_all,
        y_true=y_true, y_pred=y_pred,
        cv_ratio=cvr_all, causal_coherence=causal_all,
        n_instances=n_inst_all, n_modules=n_mod_all,
        have_shell=have_shell, is_holdout=is_holdout,
        scenario=scen_all, cons_cut=cons_cut, agr_cut=agr_cut,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
