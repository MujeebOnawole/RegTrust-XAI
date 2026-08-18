#!/usr/bin/env python
"""One-off supplementary robustness check (not part of the pipeline DAG):
does the A/B/C/D trust-taxonomy result survive alternative choices of the
consensus/agreement calibration percentile?

Reuses xai.py's exact calibration procedure (same rng/seed, same pool
sample, same window_coherence/occlusion-attribution call) so the calibration
sample is IDENTICAL to the one xai.py itself used -- only the percentile
taken of that sample's cv_ratio/coherence distribution varies. The test
set's own cv_ratio/coherence/error arrays are reused directly from the
already-saved results/xai_predictions.npz (no re-inference needed there).

Baseline: consensus 30th pct, agreement 70th pct (config.py, unchanged).
Sweep: consensus in {20, 30, 40}, agreement in {60, 70, 80}, one axis
varied at a time with the other held at baseline (5 conditions total,
including the baseline itself as a sanity re-check).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import py2bit
import torch

from config import (
    CHECKPOINT_DIR, ENSEMBLE_SIZE, GENOME_2BIT, HOLDOUT_CHROMS, JASPAR_PFM_PATH,
    SEQUENCE_WINDOWS_NPZ, SPLIT_SEED, TRUST_CALIB_MAX_SAMPLES,
    XAI_ERROR_THRESHOLDS, XAI_PREDICTIONS_NPZ,
)
from data_module import WindowDataset, worker_init_fn
from final_eval import load_ensemble, predict_ensemble
from motif_shell import load_k562_pssms
from trust import cv_ratio_from_stats, trust_block
from xai import load_best_checkpoint, window_coherence

OUT_JSON = Path("results/sensitivity_thresholds.json")

CONSENSUS_SWEEP = [20.0, 30.0, 40.0]   # baseline 30.0
AGREEMENT_SWEEP = [60.0, 70.0, 80.0]   # baseline 70.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading test-set cv_ratio/coherence/error from xai_predictions.npz "
          "(already computed by xai.py, reused as-is)...")
    xp = np.load(XAI_PREDICTIONS_NPZ, allow_pickle=True)
    cvr_test = xp["cv_ratio"].astype(np.float64)
    coh_test = xp["coherence"].astype(np.float64)
    y_true = xp["y_true"].astype(np.float64)
    y_pred = xp["y_pred"].astype(np.float64)
    err_test = np.abs(y_pred - y_true)
    print(f"  {len(cvr_test)} test windows")

    print("\nreproducing xai.py's exact calibration sample "
          f"(seed={SPLIT_SEED}, size={TRUST_CALIB_MAX_SAMPLES})...")
    rng = np.random.default_rng(SPLIT_SEED)
    d = np.load(SEQUENCE_WINDOWS_NPZ, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    calib_size = min(TRUST_CALIB_MAX_SAMPLES, len(pool_idx))
    calib_idx = rng.choice(pool_idx, size=calib_size, replace=False)
    print(f"  calibration sample: {calib_size} pool windows (identical to the "
          f"one xai.py's own full-scale run used)")

    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    best_model, _, _ = load_best_checkpoint(CHECKPOINT_DIR, ENSEMBLE_SIZE, device)

    print("\nconsensus on the calibration sample (cheap, ensemble forward pass, "
          "no attribution)...")
    ensemble, ens_label_means, ens_label_stds = load_ensemble(CHECKPOINT_DIR, ENSEMBLE_SIZE, device)
    from torch.utils.data import DataLoader
    calib_ds = WindowDataset(d, calib_idx)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, worker_init_fn=worker_init_fn)
    ep = np.load("results/eval_predictions.npz", allow_pickle=True)
    pop_std = float(ep["pop_std"])
    _, calib_std, _ = predict_ensemble(ensemble, calib_loader, device, ens_label_means, ens_label_stds)
    cvr_calib = cv_ratio_from_stats(calib_std, pop_std)

    print("\ncoherence on the calibration sample (occlusion attribution, the "
          "slow step -- ~2,000 windows, same cost class as xai.py's own "
          "calibration pass)...")
    tb = py2bit.open(str(GENOME_2BIT))
    coh_calib_raw = [
        window_coherence(best_model, tb, chrom_all[i], d["start"][i], d["end"][i], pssms, device)
        for i in calib_idx
    ]
    tb.close()
    coh_calib = np.array([c for c in coh_calib_raw if c is not None])
    print(f"  {len(coh_calib)}/{calib_size} calibration windows resolved a motif shell")

    results = {
        "note": ("Supplementary robustness check: same calibration sample xai.py "
                 "used (identical seed/pool draw), swept across alternative "
                 "consensus/agreement calibration percentiles. Test-set "
                 "cv_ratio/coherence/error reused unchanged from xai_predictions.npz."),
        "baseline": {"consensus_percentile": 30.0, "agreement_percentile": 70.0},
        "conditions": [],
    }

    conditions = [("baseline", 30.0, 70.0)]
    for c in CONSENSUS_SWEEP:
        if c != 30.0:
            conditions.append((f"consensus_{c:g}pct", c, 70.0))
    for a in AGREEMENT_SWEEP:
        if a != 70.0:
            conditions.append((f"agreement_{a:g}pct", 30.0, a))

    print("\n=== sweeping calibration percentiles ===")
    for label, cons_pct, agr_pct in conditions:
        cons_cut = float(np.percentile(cvr_calib, cons_pct))
        agr_cut = float(np.percentile(coh_calib, agr_pct))
        block, _ = trust_block(cvr_test, coh_test, err_test, cons_cut, agr_cut,
                                XAI_ERROR_THRESHOLDS, label)
        block["consensus_percentile"] = cons_pct
        block["agreement_percentile"] = agr_pct
        results["conditions"].append({"label": label, **block})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
