#!/usr/bin/env python
"""Phase-1 chromosome-grouped CV training for RegTrust-XAI's accessibility
CNN. Produces the deployed ENSEMBLE_SIZE-member ensemble the same way
CancerTrust-XAI's ridge_baseline.py and ProtTrust-XAI's train_cv.py both do:
one model per cluster fold, no separate "final" retrain.

SPLIT DESIGN. `data/processed/phase1_sequence_windows.npz` (build_features.py)
carries every window's chromosome. Windows on config.HOLDOUT_CHROMS are the
internal test set and NEVER enter this script -- reserved for a later
final_eval.py, not touched here, the same separation ProtTrust-XAI and
CancerTrust-XAI both keep. The remaining pool is split into config.CV_FOLDS
chromosome-grouped folds (splits.cluster_kfold): no fold's validation
chromosomes overlap its training chromosomes, so each fold's validation
score is a genuine held-out-locus estimate, not an interpolation score.

LABEL STANDARDIZATION IS FIT PER FOLD, ON THAT FOLD'S OWN TRAINING INDICES
ONLY -- changed 2026-08-11 from an earlier pool-wide-fit design after review:
fitting mean/std on the whole pool before the fold split let each fold's
validation-chromosome labels contribute to the scaler its own training set
was standardized with, a real (if likely small at full scale) leak of the
validation target distribution into training preprocessing. Each fold now
gets its own (label_mean_k, label_std_k) from train_idx alone, stored in
that fold's checkpoint; downstream code (final_eval.py, xai.py) de-
standardizes each ensemble member's prediction with ITS OWN fold's scaler
before averaging in raw units, rather than assuming one shared scaler across
the ensemble. For the eventual deployed model retrained on the full
non-holdout pool (no CV), a single pool-wide scaler is fine again -- the
leak only exists when the scaler-fitting set and the validation set overlap.

SEQUENCE IS READ LAZILY from hg38.2bit by data_module.WindowDataset, not
pre-materialised -- see that module's docstring.

RUN AS A BATCH JOB (slurm/2_train_cv.sh): needs GPU, unlike
build_features.py/build_mpra_features.py. Confirm the Bunya GPU queue/account
before submitting; do not assume CancerTrust-XAI's CPU-job values carry over
(see project_status.md).

READS best_hyperparameters.json IF PRESENT (load_hparams() below) -- written
by hyper.py, which should run first (added 2026-08-11 to close a real gap:
this project had skipped the hyper step of the usual build -> hyper -> cv ->
final_eval -> xai pipeline every other port in this portfolio follows). Falls
back to config.py's LEARNING_RATE/WEIGHT_DECAY/BATCH_SIZE/CONV_CHANNELS/
CONV_KERNEL/DROPOUT if hyper.py has not been run yet, so this script still
works standalone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    BEST_HYPERPARAMETERS_JSON,
    CHECKPOINT_DIR,
    CONV_CHANNELS,
    CONV_KERNEL,
    CV_FOLDS,
    CV_SUMMARY_JSON,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    ENSEMBLE_SIZE,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_WORKERS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    WEIGHT_DECAY,
)
from data_module import WindowDataset, worker_init_fn
from model import Seq2AccessibilityCNN
from splits import cluster_kfold


def load_hparams():
    """Hyperparameter dict, from best_hyperparameters.json if hyper.py has run,
    else config.py defaults. hyper.py's channel_profile is a comma-separated
    string (e.g. "64,96,128"); parsed back to a list of ints here."""
    h = {}
    if BEST_HYPERPARAMETERS_JSON.exists():
        h = json.loads(BEST_HYPERPARAMETERS_JSON.read_text())
    channels = CONV_CHANNELS
    if "channel_profile" in h:
        channels = [int(c) for c in h["channel_profile"].split(",")]
    return {
        "lr": h.get("lr", LEARNING_RATE),
        "weight_decay": h.get("weight_decay", WEIGHT_DECAY),
        "batch_size": h.get("batch_size", BATCH_SIZE),
        "channels": channels,
        "kernel": h.get("kernel", CONV_KERNEL),
        "dropout": h.get("dropout", DROPOUT),
    }


def fit_one_fold(train_ds, val_ds, device, seed, max_epochs, patience, tag, hparams):
    torch.manual_seed(seed)
    model = Seq2AccessibilityCNN(channels=hparams["channels"], kernel=hparams["kernel"],
                                  dropout=hparams["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True,
                               num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=hparams["batch_size"], shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)

    best_val_rho = -np.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                val_preds.append(model(x).cpu().numpy())
                val_true.append(y.numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_rho = float(stats.spearmanr(val_preds, val_true).statistic)
        train_loss = train_loss_sum / max(n_train, 1)

        print(f"  [{tag}] epoch {epoch:>2}  train_mse {train_loss:.4f}  val_rho {val_rho:+.4f}")

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  [{tag}] early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs, best was epoch {best_epoch})")
                break

    return {
        "model_state": best_state,
        "val_spearman": best_val_rho,
        "best_epoch": best_epoch,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--out", type=Path, default=CV_SUMMARY_JSON)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run build_features.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    hparams = load_hparams()
    src = "best_hyperparameters.json" if BEST_HYPERPARAMETERS_JSON.exists() else "config.py defaults"
    print(f"hyperparameters (from {src}): {hparams}")

    d = np.load(args.windows_npz, allow_pickle=True)
    chrom = d["chrom"]
    is_holdout = np.isin(chrom, HOLDOUT_CHROMS)
    pool_idx = np.where(~is_holdout)[0].tolist()
    print(f"windows: {len(chrom)} total, {is_holdout.sum()} on holdout chromosomes "
          f"{HOLDOUT_CHROMS} (reserved, not used here), {len(pool_idx)} in the training pool")

    # ---- chromosome-grouped CV, label scaler fit fresh per fold -------------------------
    pool_chrom_map = {i: str(chrom[i]) for i in pool_idx}
    members = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for k, (train_idx, val_idx) in enumerate(
            cluster_kfold(pool_idx, pool_chrom_map, n_folds=CV_FOLDS, seed=args.seed)):
        print(f"\n=== fold {k}: {len(train_idx)} train windows, {len(val_idx)} val windows ===")
        fold_labels = d["label"][train_idx].astype(np.float64)
        label_mean, label_std = float(fold_labels.mean()), float(fold_labels.std())
        print(f"  fold-local label stats (train_idx only): mean {label_mean:.4f}, "
              f"std {label_std:.4f}")
        train_ds = WindowDataset(d, np.array(train_idx), label_mean, label_std)
        val_ds = WindowDataset(d, np.array(val_idx), label_mean, label_std)

        result = fit_one_fold(train_ds, val_ds, device, seed=args.seed + k,
                               max_epochs=args.max_epochs, patience=args.patience, tag=f"fold{k}",
                               hparams=hparams)

        ckpt_path = args.checkpoint_dir / f"fold_{k}.pt"
        torch.save({
            "model_state": result["model_state"],
            "arch": {"channels": hparams["channels"], "kernel": hparams["kernel"],
                     "dropout": hparams["dropout"]},
            "label_mean": label_mean, "label_std": label_std,
            "fold": k, "val_spearman": result["val_spearman"], "best_epoch": result["best_epoch"],
        }, ckpt_path)
        print(f"  saved -> {ckpt_path} (val_rho {result['val_spearman']:+.4f}, "
              f"best_epoch {result['best_epoch']})")

        members.append({
            "fold": k, "val_spearman": result["val_spearman"], "best_epoch": result["best_epoch"],
            "n_train": result["n_train"], "n_val": result["n_val"], "checkpoint": str(ckpt_path),
            "label_mean": label_mean, "label_std": label_std,
        })
        if len(members) >= ENSEMBLE_SIZE:
            break

    cv_rho = float(np.mean([m["val_spearman"] for m in members]))
    cv_std = float(np.std([m["val_spearman"] for m in members]))
    print(f"\nCV Spearman {cv_rho:+.4f} +/- {cv_std:.4f} across {len(members)} folds "
          f"({time.time() - t0:.0f}s total)")

    summary = {
        "cv_spearman": cv_rho, "cv_spearman_std": cv_std,
        "holdout_chroms": HOLDOUT_CHROMS,
        "members": members,
        "device": str(device),
        "elapsed_seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
