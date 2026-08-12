#!/usr/bin/env python
"""Trust taxonomy labelling for RegTrust-XAI: occlusion attribution,
consensus, coherence, and A/B/C/D scenario labels on the internal test set.
Same role as ProtTrust-XAI's xai.py.

ATTRIBUTION RUNS ON ONE CHECKPOINT, NOT THE FULL ENSEMBLE -- same design as
ProtTrust-XAI (occlusion cost scales with the ensemble size, and ProtTrust-
XAI's own robustness check, Table S8, found the deployed conclusions
invariant to which member is chosen even though per-residue attributions
are not). The member with the highest validation Spearman across the
train_cv.py folds is used, read directly from each checkpoint's own stored
val_spearman -- not re-derived, not re-ranked by any other criterion.

CONSENSUS reuses final_eval.py's ALREADY-COMPUTED per-window ensemble
mean/std and pop_std from eval_predictions.npz, rather than re-running the
5-model ensemble here -- that forward pass is expensive and final_eval.py
already paid for it once; xai.py's own expensive step is occlusion
attribution on a single model, not ensemble inference.

COHERENCE uses motif_shell.py's real JASPAR-based K562 TF shell (see
trust.py's localization_coherence docstring for the verified, honest,
modest effect size this axis carries and why it is not tightened further).

CUTOFFS are calibrated on a POOL sample (config.TRUST_CALIB_MAX_SAMPLES
windows from outside config.HOLDOUT_CHROMS), never on the test set itself --
this needs its OWN ensemble forward pass (for consensus) and its own
occlusion-attribution pass (for coherence) over the calibration sample,
since eval_predictions.npz only carries per-window stats for the TEST set.

RUN AS A BATCH JOB (once slurm/4_xai.sh exists): needs the same GPU
container as train_cv.py/final_eval.py -- occlusion attribution (many
forward passes per window) is the slow step here, the same role bigWig
reading plays in build_features.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import py2bit
import torch

from config import (
    BIN_BP,
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_PREDICTIONS_NPZ,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    JASPAR_PFM_PATH,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    TRUST_AGREEMENT_PERCENTILE,
    TRUST_CALIB_MAX_SAMPLES,
    TRUST_CONSENSUS_PERCENTILE,
    WINDOW_BP,
    XAI_ERROR_THRESHOLDS,
    XAI_PREDICTIONS_NPZ,
    XAI_RESULTS_JSON,
)
from final_eval import load_ensemble, predict_ensemble
from data_module import WindowDataset, worker_init_fn
from model import Seq2AccessibilityCNN, occlusion_attribution
from motif_shell import K562_TF_PANEL, load_k562_pssms, motif_coverage_by_bin
from trust import cv_ratio_from_stats, trust_block


def load_best_checkpoint(checkpoint_dir, ensemble_size, device):
    """The single checkpoint attribution runs on: highest val_spearman
    across the ensemble, read directly from each checkpoint's own stored
    metadata (see module docstring for why this is the right member, not a
    re-derived criterion)."""
    ckpt_paths = sorted(Path(checkpoint_dir).glob("fold_*.pt"))[:ensemble_size]
    best_path, best_rho, best_ckpt = None, -np.inf, None
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        if ckpt["val_spearman"] > best_rho:
            best_path, best_rho, best_ckpt = p, ckpt["val_spearman"], ckpt
    arch = best_ckpt["arch"]
    model = Seq2AccessibilityCNN(channels=arch["channels"], kernel=arch["kernel"],
                                  dropout=arch["dropout"]).to(device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()
    print(f"  attribution checkpoint: {best_path.name} (val_spearman {best_rho:+.4f})")
    return model, best_ckpt["label_mean"], best_ckpt["label_std"]


def window_coherence(model, tb, chrom, start, end, pssms, device):
    """One window's full pipeline: sequence -> one-hot -> occlusion
    attribution -> motif shell -> localization_coherence. Imports
    localization_coherence lazily to avoid a module-level circular-looking
    import list; it is a plain function call, not a class, so this costs
    nothing at runtime."""
    from data_module import one_hot_encode
    from trust import localization_coherence

    seq = tb.sequence(str(chrom), int(start), int(end))
    x = torch.from_numpy(one_hot_encode(seq))
    attr = occlusion_attribution(model, x, window_stride=BIN_BP, window_width=BIN_BP, device=device)
    shell = motif_coverage_by_bin(seq, pssms, bin_width=BIN_BP, bin_stride=BIN_BP)
    return localization_coherence(attr, shell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--eval-predictions", type=Path, default=EVAL_PREDICTIONS_NPZ)
    ap.add_argument("--out", type=Path, default=XAI_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=XAI_PREDICTIONS_NPZ)
    ap.add_argument("--thresholds", type=float, nargs="+", default=XAI_ERROR_THRESHOLDS)
    ap.add_argument("--calib-size", type=int, default=TRUST_CALIB_MAX_SAMPLES,
                     help="pool calibration sample size; override for a quick smoke test")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, args.eval_predictions, GENOME_2BIT, JASPAR_PFM_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading K562 TF panel PSSMs...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    print("loading attribution checkpoint (highest val_spearman)...")
    best_model, best_label_mean, best_label_std = load_best_checkpoint(
        args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("loading eval_predictions.npz (test set, from final_eval.py)...")
    ep = np.load(args.eval_predictions, allow_pickle=True)
    test_chrom, test_start, test_end = ep["chrom"], ep["start"], ep["end"]
    y_true, y_pred, y_pred_std = ep["y_true"], ep["y_pred"], ep["y_pred_std"]
    pop_std = float(ep["pop_std"])
    err_test = np.abs(y_pred - y_true)
    cvr_test = cv_ratio_from_stats(y_pred_std, pop_std)
    print(f"  {len(y_true)} test windows, pop_std {pop_std:.4f}")

    tb = py2bit.open(str(GENOME_2BIT))

    print("\ncomputing coherence on the test set (occlusion attribution, the slow step)...")
    coh_test = np.array([
        window_coherence(best_model, tb, test_chrom[i], test_start[i], test_end[i], pssms, device)
        for i in range(len(test_chrom))
    ])
    have_shell_test = np.array([c is not None for c in coh_test])
    coh_test = np.array([c if c is not None else np.nan for c in coh_test])
    print(f"  test windows with a resolvable motif shell: {have_shell_test.sum()}/{len(test_chrom)}")

    # ---- calibration: a pool sample, NEVER the test set --------------------------------
    print("\ncalibrating cutoffs on a training-pool sample...")
    d = np.load(args.windows_npz, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    calib_size = min(args.calib_size, len(pool_idx))
    calib_idx = rng.choice(pool_idx, size=calib_size, replace=False)
    print(f"  calibration sample: {calib_size} pool windows")

    # Fold-local scalers (train_cv.py, fixed 2026-08-11) mean ensemble members legitimately
    # carry different label_mean/label_std -- predict_ensemble de-standardizes each member
    # with its own scaler, so there is nothing to assert equal here any more.
    ensemble, ens_label_means, ens_label_stds = load_ensemble(args.checkpoint_dir, ENSEMBLE_SIZE, device)
    from torch.utils.data import DataLoader
    calib_ds = WindowDataset(d, calib_idx)  # raw labels; only ensemble predictions need de-standardizing
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, worker_init_fn=worker_init_fn)
    _, calib_std, _ = predict_ensemble(ensemble, calib_loader, device, ens_label_means, ens_label_stds)
    cvr_calib = cv_ratio_from_stats(calib_std, pop_std)
    cons_cut = float(np.percentile(cvr_calib, TRUST_CONSENSUS_PERCENTILE))

    print("  computing coherence on the calibration sample (occlusion attribution again)...")
    coh_calib_raw = [
        window_coherence(best_model, tb, chrom_all[i], d["start"][i], d["end"][i], pssms, device)
        for i in calib_idx
    ]
    coh_calib = [c for c in coh_calib_raw if c is not None]
    agr_cut = float(np.percentile(coh_calib, TRUST_AGREEMENT_PERCENTILE)) if coh_calib else 0.5
    tb.close()

    print(f"  pool-calibrated cutoffs (n_calib={calib_size}, n_with_shell={len(coh_calib)}): "
          f"consensus {cons_cut:.4f}, agreement {agr_cut:.4f}")

    # ---- apply cutoffs to the test set ---------------------------------------------------
    print("\n=== TRUST TAXONOMY ===")
    valid = have_shell_test
    results = {
        "attribution_checkpoint_label_mean": best_label_mean,
        "attribution_checkpoint_label_std": best_label_std,
        "pop_std": pop_std,
        "thresholds": args.thresholds,
        "trust": {},
    }
    scen_test = np.array([])
    if valid.sum() > 0:
        block, scen_test = trust_block(
            cvr_test[valid], coh_test[valid], err_test[valid], cons_cut, agr_cut,
            args.thresholds, "internal test (shell-covered only)")
        results["trust"]["internal_test"] = block
        results["trust"]["coverage_note"] = (
            f"{valid.sum()}/{len(test_chrom)} internal test windows have a resolvable "
            f"motif shell; the rest are excluded from trust labelling, not defaulted "
            f"to a scenario."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    valid_chrom = test_chrom[valid] if valid.sum() > 0 else np.array([])
    np.savez(
        args.pred_out,
        chrom=valid_chrom,
        start=test_start[valid] if valid.sum() > 0 else np.array([]),
        end=test_end[valid] if valid.sum() > 0 else np.array([]),
        y_true=y_true[valid] if valid.sum() > 0 else np.array([]),
        y_pred=y_pred[valid] if valid.sum() > 0 else np.array([]),
        cv_ratio=cvr_test[valid] if valid.sum() > 0 else np.array([]),
        coherence=coh_test[valid] if valid.sum() > 0 else np.array([]),
        scenario=scen_test,
        cons_cut=cons_cut, agr_cut=agr_cut,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
