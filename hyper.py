#!/usr/bin/env python
"""
Optuna hyperparameter search for RegTrust-XAI's accessibility CNN.

Added 2026-08-11, after the project's initial build -> cv -> final_eval -> xai
pipeline skipped the hyper step every other port in this portfolio has
(ProtTrust-XAI: 40-trial Optuna search before its first real CV run). This
closes that gap, run BEFORE train_cv.py's real ensemble.

Searches lr / weight_decay / batch_size (optimizer) AND channel profile /
kernel / dropout (architecture) in one study -- config.py's CONV_CHANNELS/
CONV_KERNEL were never independently justified the way ProtTrust-XAI's RGCN
depth was, so both go in scope here, matching ProtTrust-XAI's own choice to
search num_rgcn_layers/node_dim alongside the optimizer.

Objective = validation Spearman on ONE chromosome-grouped fold
(splits.cluster_kfold's first fold over the non-holdout pool), trained for a
reduced HYPER_EPOCHS budget on a HYPER_MAX_TRAIN_WINDOWS subsample -- a fast
ranking signal across trials, not a converged accuracy estimate. The real,
reported ensemble comes from train_cv.py's full CV_FOLDS run afterwards,
which reads this script's output (best_hyperparameters.json) via
load_hparams().

Run (GPU, on Bunya): python hyper.py   (needs data/processed/
phase1_sequence_windows.npz from build_features.py first)
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from scipy import stats
from torch.utils.data import DataLoader

from config import (
    BEST_HYPERPARAMETERS_JSON,
    HOLDOUT_CHROMS,
    HYPER_BATCH_SIZES,
    HYPER_CONV_CHANNEL_PROFILES,
    HYPER_DROPOUT_RANGE,
    HYPER_EPOCHS,
    HYPER_KERNELS,
    HYPER_LR_RANGE,
    HYPER_MAX_TRAIN_WINDOWS,
    HYPER_N_TRIALS,
    HYPER_WEIGHT_DECAY_RANGE,
    NUM_WORKERS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
)
from data_module import WindowDataset, worker_init_fn
from model import Seq2AccessibilityCNN
from splits import cluster_kfold


def build_fold():
    """One chromosome-grouped fold over the non-holdout pool -- same split
    machinery train_cv.py uses, just one fold instead of CV_FOLDS, and the
    training side subsampled for per-trial speed."""
    d = np.load(SEQUENCE_WINDOWS_NPZ, allow_pickle=True)
    chrom = d["chrom"]
    is_holdout = np.isin(chrom, HOLDOUT_CHROMS)
    pool_idx = np.where(~is_holdout)[0].tolist()
    pool_chrom_map = {i: str(chrom[i]) for i in pool_idx}

    train_idx, val_idx = next(cluster_kfold(pool_idx, pool_chrom_map, n_folds=5, seed=SPLIT_SEED))
    if HYPER_MAX_TRAIN_WINDOWS and len(train_idx) > HYPER_MAX_TRAIN_WINDOWS:
        rng = random.Random(SPLIT_SEED)
        train_idx = rng.sample(train_idx, HYPER_MAX_TRAIN_WINDOWS)

    fold_labels = d["label"][train_idx].astype(np.float64)
    label_mean, label_std = float(fold_labels.mean()), float(fold_labels.std())
    train_ds = WindowDataset(d, np.array(train_idx), label_mean, label_std)
    val_ds = WindowDataset(d, np.array(val_idx), label_mean, label_std)
    return train_ds, val_ds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("preflight: checking required inputs...")
    if not Path(SEQUENCE_WINDOWS_NPZ).exists():
        raise SystemExit(f"Missing {SEQUENCE_WINDOWS_NPZ}; run build_features.py first.")

    train_ds, val_ds = build_fold()
    print(f"hyper fold: train={len(train_ds):,} (subsampled) val={len(val_ds):,}")

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)

    def objective(trial):
        lr = trial.suggest_float("lr", *HYPER_LR_RANGE, log=True)
        wd = trial.suggest_float("weight_decay", *HYPER_WEIGHT_DECAY_RANGE, log=True)
        bs = trial.suggest_categorical("batch_size", HYPER_BATCH_SIZES)
        channel_profile = trial.suggest_categorical("channel_profile", HYPER_CONV_CHANNEL_PROFILES)
        channels = [int(c) for c in channel_profile.split(",")]
        kernel = trial.suggest_categorical("kernel", HYPER_KERNELS)
        dropout = trial.suggest_float("dropout", *HYPER_DROPOUT_RANGE)

        torch.manual_seed(SPLIT_SEED)
        model = Seq2AccessibilityCNN(channels=channels, kernel=kernel, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = torch.nn.MSELoss()
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                   num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn,
                                   drop_last=True)

        best = -1.0
        t0 = time.time()
        for epoch in range(HYPER_EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    preds.append(model(x).cpu().numpy())
                    trues.append(y.numpy())
            score = float(stats.spearmanr(np.concatenate(preds), np.concatenate(trues)).statistic)
            trial.report(score, epoch)
            best = max(best, score)
            if trial.should_prune():
                break

        print(f"  trial {trial.number}: lr={lr:.2e} wd={wd:.3g} bs={bs} "
              f"channels=[{channel_profile}] kernel={kernel} drop={dropout:.2f} "
              f"-> Spearman {best:.4f} ({time.time() - t0:.0f}s)")
        return best

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SPLIT_SEED),
                                 pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=HYPER_N_TRIALS)

    best = study.best_params
    BEST_HYPERPARAMETERS_JSON.write_text(json.dumps(best, indent=2))
    print(f"\nBest: {best}  Spearman {study.best_value:.4f}")
    print(f"[written] {BEST_HYPERPARAMETERS_JSON}")

    # bound-pinning checks, same discipline as ENZYME_XAI/hyper.py
    lo, hi = HYPER_LR_RANGE
    if best["lr"] <= lo * 1.5:
        print(f"WATCH: best lr {best['lr']:.2e} pinned near floor {lo:.0e}; consider widening down.")
    if best["lr"] >= hi / 1.5:
        print(f"WATCH: best lr {best['lr']:.2e} pinned near ceiling {hi:.0e}; consider widening up.")
    if best["channel_profile"] == HYPER_CONV_CHANNEL_PROFILES[-1]:
        print(f"WATCH: best channel_profile pinned at the deepest/widest option "
              f"({best['channel_profile']}); consider adding a bigger profile to confirm the ceiling.")
    if best["kernel"] == max(HYPER_KERNELS):
        print(f"WATCH: best kernel={best['kernel']} pinned at range max; consider a wider kernel to confirm.")


if __name__ == "__main__":
    main()
