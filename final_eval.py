#!/usr/bin/env python
"""Score the deployed ENSEMBLE_SIZE-member ensemble (train_cv.py's fold
checkpoints) on the internal held-out test set (config.HOLDOUT_CHROMS'
windows), the same role ProtTrust-XAI's final_eval.py and CancerTrust-XAI's
internal-test scoring block both play. Also computes and saves the
per-window ensemble mean/std this project's eventual xai.py needs for the
consensus trust axis -- this script does NOT do trust labelling itself
(no cutoffs calibrated, no coherence computed here), only accuracy plus the
raw ingredients a future xai.py consumes, matching ProtTrust-XAI's
separation between final_eval.py (accuracy + ensemble selection) and xai.py
(trust taxonomy).

ARCH-AWARE CHECKPOINT LOADING. Each fold_k.pt carries its own "arch" dict
(channels/kernel/dropout) alongside the state dict, written by train_cv.py --
this script rebuilds each model from ITS OWN stored arch rather than
assuming config.py's current CONV_CHANNELS/CONV_KERNEL/DROPOUT still match
what a checkpoint was actually trained with. This matters the first time
those constants are ever changed after a training run exists; until then it
is a no-op, but the checkpoint is the source of truth either way, not the
current config file.

LABEL DE-STANDARDIZATION IS PER-MODEL, NOT SHARED. train_cv.py fits each
fold's label_mean/label_std on that fold's OWN training indices only (fixed
2026-08-11 to remove a validation-target leak into the scaler -- see
train_cv.py's design note), so ensemble members legitimately carry different
scalers now. predict_ensemble de-standardizes each model's z-scored
prediction with THAT model's own checkpoint scaler before averaging in raw
units -- never averages in z-space across models with different scalers,
and never assumes the checkpoints share one scaler the way the earlier
pool-wide-fit design did.

INTERPRETATION RULE, pre-registered, same convention as CancerTrust-XAI:
judge accuracy on skill over a constant-prediction null (predict the
training pool's own mean accessibility for every window), not raw RMSE/MAE
alone, since raw error on a range-compressed model mostly reads the label
distribution rather than genuine predictive skill.

RUN AS A BATCH JOB (once slurm/3_final_eval.sh exists): needs the same GPU
container as train_cv.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_N_BOOT,
    EVAL_POOL_SAMPLE_SIZE,
    EVAL_PREDICTIONS_NPZ,
    EVAL_RESULTS_JSON,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    NUM_WORKERS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
)
from data_module import WindowDataset, worker_init_fn
from model import Seq2AccessibilityCNN


def load_ensemble(checkpoint_dir, ensemble_size, device):
    """Returns (models, label_means, label_stds) -- one scaler PER model,
    since train_cv.py now fits each fold's scaler on that fold's own
    training indices only (see module docstring). No equality assertion:
    per-fold scalers legitimately differing is expected, not a corruption
    signal, as long as each model is de-standardized with its own scaler."""
    ckpt_paths = sorted(Path(checkpoint_dir).glob("fold_*.pt"))
    if len(ckpt_paths) < ensemble_size:
        raise SystemExit(
            f"Found {len(ckpt_paths)} checkpoints in {checkpoint_dir}, need "
            f"{ensemble_size}; run train_cv.py first."
        )
    ckpt_paths = ckpt_paths[:ensemble_size]

    models, label_means, label_stds = [], [], []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        arch = ckpt["arch"]
        m = Seq2AccessibilityCNN(channels=arch["channels"], kernel=arch["kernel"],
                                  dropout=arch["dropout"]).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        label_means.append(float(ckpt["label_mean"]))
        label_stds.append(float(ckpt["label_std"]))
        print(f"  loaded {p.name}: val_spearman {ckpt['val_spearman']:+.4f}, "
              f"best_epoch {ckpt['best_epoch']}, label_mean {ckpt['label_mean']:.4f}, "
              f"label_std {ckpt['label_std']:.4f}")

    return models, label_means, label_stds


def predict_ensemble(models, loader, device, label_means, label_stds):
    """Returns (mean_pred, std_pred, true), all in ORIGINAL accessibility
    units. Each model's z-scored prediction is de-standardized with ITS OWN
    (label_mean, label_std) before averaging across models -- never averages
    raw model outputs in a shared z-space, since fold-local scalers can
    differ. `loader` is expected to be built with WindowDataset's default
    raw (mean=0/std=1) labels, so `true` comes back unmodified. std_pred is
    the ensemble's own inter-member spread per window -- the consensus
    axis's raw ingredient."""
    all_preds = []  # (n_models, n_windows)
    true = None
    for m, label_mean, label_std in zip(models, label_means, label_stds):
        preds, truths = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                p = m(x).cpu().numpy()
                preds.append(p * label_std + label_mean)
                truths.append(y.numpy())
        all_preds.append(np.concatenate(preds))
        if true is None:
            true = np.concatenate(truths)
    all_preds = np.stack(all_preds)  # (n_models, n_windows)
    return all_preds.mean(axis=0), all_preds.std(axis=0), true


def skill_vs_constant(pred, true, const):
    return float(np.abs(const - true).mean() - np.abs(pred - true).mean())


def score(pred, true, const, n_boot, rng, label):
    res = {
        "n": int(len(true)),
        "spearman": float(stats.spearmanr(pred, true).statistic),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "bias": float(np.mean(pred - true)),
        "skill_vs_constant": skill_vs_constant(pred, true, const),
        "constant_null": float(const),
    }
    boots = [skill_vs_constant(pred[i], true[i], const)
             for i in (rng.integers(0, len(true), len(true)) for _ in range(n_boot))]
    res["skill_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    print(f"  {label:<24} n={res['n']:>6}  rho {res['spearman']:+.4f}  "
          f"MAE {res['mae']:.4f}  skill {res['skill_vs_constant']:+.4f} "
          f"[{res['skill_ci95'][0]:+.4f},{res['skill_ci95'][1]:+.4f}]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--out", type=Path, default=EVAL_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=EVAL_PREDICTIONS_NPZ)
    ap.add_argument("--n-boot", type=int, default=EVAL_N_BOOT)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run build_features.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading ensemble...")
    models, label_means, label_stds = load_ensemble(args.checkpoint_dir, ENSEMBLE_SIZE, device)
    print(f"  {len(models)} models loaded, per-fold label_mean range "
          f"[{min(label_means):.4f}, {max(label_means):.4f}], label_std range "
          f"[{min(label_stds):.4f}, {max(label_stds):.4f}]")

    d = np.load(args.windows_npz, allow_pickle=True)
    chrom = d["chrom"]
    is_holdout = np.isin(chrom, HOLDOUT_CHROMS)
    test_idx = np.where(is_holdout)[0]
    pool_idx = np.where(~is_holdout)[0]
    print(f"internal test: {len(test_idx)} windows on {HOLDOUT_CHROMS}; "
          f"training pool: {len(pool_idx)} windows")

    # ---- internal test -------------------------------------------------------------------
    print("\n=== ACCURACY ===")
    test_ds = WindowDataset(d, test_idx)  # raw labels; each model de-standardizes with its own scaler
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    mean_test, std_test, y_test = predict_ensemble(models, test_loader, device, label_means, label_stds)

    constant_null = float(np.mean(label_means))  # ensemble-average training-pool mean, for the skill-vs-constant null
    results = {
        "ensemble_size": len(models),
        "label_means": label_means, "label_stds": label_stds,
        "holdout_chroms": HOLDOUT_CHROMS,
        "cohorts": {},
    }
    results["cohorts"]["internal_test"] = score(
        mean_test, y_test, constant_null, args.n_boot, rng, "internal test (chrom holdout)")

    # ---- pop_std: ensemble prediction spread over a training-pool sample -----------------
    # The consensus axis (trust.cv_ratio_from_stats) normalizes by this, not by std_test --
    # a query's own consensus is judged against how much the ensemble's predictions vary
    # ACROSS THE POPULATION it was trained on, the same role pop_std plays in every other
    # port in this portfolio.
    print("\n=== POPULATION SPREAD (for the consensus axis) ===")
    sample_size = min(EVAL_POOL_SAMPLE_SIZE, len(pool_idx))
    pool_sample_idx = rng.choice(pool_idx, size=sample_size, replace=False)
    pool_ds = WindowDataset(d, pool_sample_idx)
    pool_loader = DataLoader(pool_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    pool_mean_pred, _, _ = predict_ensemble(models, pool_loader, device, label_means, label_stds)
    pop_std = float(pool_mean_pred.std())
    print(f"  pop_std (n={sample_size} pool windows sampled): {pop_std:.4f}")
    results["pop_std"] = pop_std
    results["pop_std_sample_size"] = sample_size

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    # ---- per-window prediction dump, for the eventual xai.py -----------------------------
    np.savez(
        args.pred_out,
        chrom=chrom[test_idx], start=d["start"][test_idx], end=d["end"][test_idx],
        y_true=y_test, y_pred=mean_test, y_pred_std=std_test,
        pop_std=pop_std,
    )
    print(f"Saved -> {args.pred_out} (per-window ensemble mean/std for the trust taxonomy)")


if __name__ == "__main__":
    main()
