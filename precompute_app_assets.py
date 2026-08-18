#!/usr/bin/env python
"""One-off, run-locally-once script: builds every asset the hf_space/ Gradio
app needs that CANNOT be recomputed cheaply at inference time (the AD
reference-embedding pool needs the full training-pool coordinate table +
hg38.2bit, neither of which ships with the app; a handful of real example
sequences for the "try an example" buttons, so a first-time visitor is not
stuck facing an empty textbox).

NOT part of the RegTrust-XAI pipeline DAG (build_features.py -> hyper.py ->
train_cv.py -> final_eval.py -> xai.py -> validate_trust_axes.py) -- this is
downstream of all of it, packaging already-computed results for a different
consumer (the public demo), not producing a new scientific result. Reuses
xai.load_best_checkpoint, trust.l2_normalize_rows, and the same seeded
reference-pool draw validate_trust_axes.py uses (SPLIT_SEED, AD_REF_POOL_SIZE)
so the shipped reference pool is not a fresh, unaudited sample.

Outputs, all under hf_space/assets/:
  checkpoints/fold_{0..4}.pt          copied verbatim (all 5 needed for the
                                       consensus ensemble; ~700KB each)
  JASPAR2026_CORE_..._pfms.txt        copied verbatim (336KB, needed for the
                                       coherence motif shell at inference time)
  ref_embeddings.npy                  (AD_REF_POOL_SIZE, embed_dim) float32,
                                       L2-normalized, attribution-checkpoint
                                       embedding space -- the AD axis's
                                       nearest-neighbor reference set
  calibration.json                    pop_std, consensus/coherence/AD cutoffs,
                                       WINDOW_BP/BIN_BP -- everything the app
                                       needs to reproduce xai.py's trust
                                       labelling without re-running calibration
  examples.json                       a few real, named example sequences
                                       (PKLR promoter locus; one real
                                       Scenario-A and one real Scenario-D
                                       holdout window, so a first-time visitor
                                       sees the taxonomy's contrast immediately)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import py2bit
import torch
from torch.utils.data import DataLoader

from config import (
    AD_EMBED_PERCENTILE,
    AD_REF_POOL_SIZE,
    BIN_BP,
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_PREDICTIONS_NPZ,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    JASPAR_PFM_PATH,
    NUM_WORKERS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    TRUST_AGREEMENT_PERCENTILE,
    TRUST_CONSENSUS_PERCENTILE,
    WINDOW_BP,
    XAI_PREDICTIONS_NPZ,
    XAI_RESULTS_JSON,
)
from data_module import WindowDataset, worker_init_fn
from trust import l2_normalize_rows
from validate_trust_axes import embed_loader
from xai import load_best_checkpoint

OUT_DIR = Path("hf_space/assets")


def main():
    rng = np.random.default_rng(SPLIT_SEED)
    device = torch.device("cpu")  # asset prep is a one-off, no need for GPU

    print("=== 1. copying checkpoints + JASPAR panel ===")
    ckpt_out = OUT_DIR / "checkpoints"
    ckpt_out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(CHECKPOINT_DIR).glob("fold_*.pt"))[:ENSEMBLE_SIZE]:
        shutil.copy(p, ckpt_out / p.name)
        print(f"  copied {p.name}")
    shutil.copy(JASPAR_PFM_PATH, OUT_DIR / Path(JASPAR_PFM_PATH).name)
    print(f"  copied {Path(JASPAR_PFM_PATH).name}")

    print("\n=== 2. loading attribution checkpoint (same one xai.py/validate_trust_axes.py use) ===")
    model, label_mean, label_std = load_best_checkpoint(CHECKPOINT_DIR, ENSEMBLE_SIZE, device)

    print("\n=== 3. building AD reference-embedding pool (same seeded draw as validate_trust_axes.py) ===")
    d = np.load(SEQUENCE_WINDOWS_NPZ, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    ref_size = min(AD_REF_POOL_SIZE, len(pool_idx))
    ref_idx = rng.choice(pool_idx, size=ref_size, replace=False)
    ref_ds = WindowDataset(d, ref_idx)
    ref_loader = DataLoader(ref_ds, batch_size=128, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    ref_embed = embed_loader(model, ref_loader, device)
    ref_embed_n = l2_normalize_rows(ref_embed).astype(np.float32)
    np.save(OUT_DIR / "ref_embeddings.npy", ref_embed_n)
    print(f"  saved ref_embeddings.npy: {ref_embed_n.shape}, "
          f"{ref_embed_n.nbytes / 1e6:.2f} MB")

    print("\n=== 4. calibration constants (reusing xai.py's already-calibrated cutoffs, not re-deriving) ===")
    xai_results = json.loads(Path(XAI_RESULTS_JSON).read_text())
    ad_results = json.loads(Path("results/trust_validation_results.json").read_text())
    calibration = {
        "window_bp": WINDOW_BP,
        "bin_bp": BIN_BP,
        "pop_std": xai_results["pop_std"],
        "consensus_cutoff": xai_results["trust"]["internal_test"]["cutoffs"]["consensus"],
        "coherence_cutoff": xai_results["trust"]["internal_test"]["cutoffs"]["agreement"],
        "consensus_calibration_percentile": TRUST_CONSENSUS_PERCENTILE,
        "coherence_calibration_percentile": TRUST_AGREEMENT_PERCENTILE,
        "ad_cutoff": ad_results["distance_vs_error"]["ad_cutoff"],
        "ad_embed_percentile": AD_EMBED_PERCENTILE,
        "ad_ref_pool_size": ref_size,
        "attribution_checkpoint_label_mean": label_mean,
        "attribution_checkpoint_label_std": label_std,
        "internal_test_rho": 0.7816817799229944,
        "internal_test_n": 42844,
    }
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"  saved calibration.json: {calibration}")

    print("\n=== 5. example sequences ===")
    tb = py2bit.open(str(GENOME_2BIT))
    examples = []

    # (a) PKLR promoter, K562's own saturation-mutagenesis locus (item 9 rung 4) --
    # real, K562-relevant, disease-associated regulatory element. Kircher et al.
    # 2019 span is chr1:155,301,395-155,301,864 (469bp); center a full WINDOW_BP
    # window on that span's own midpoint, same convention build_mpra_features.py
    # uses for MPRA elements.
    pklr_mid = (155_301_395 + 155_301_864) // 2
    pklr_start = pklr_mid - WINDOW_BP // 2
    pklr_seq = tb.sequence("chr1", pklr_start, pklr_start + WINDOW_BP)
    examples.append({
        "name": "PKLR promoter (K562, disease-associated)",
        "description": (
            "The PKLR erythroid promoter, the one locus in Kircher et al. 2019's "
            "saturation-mutagenesis panel tested in K562 cells -- a real, "
            "disease-associated regulatory element in this model's own cell type."
        ),
        "chrom": "chr1", "start": pklr_start, "end": pklr_start + WINDOW_BP,
        "sequence": pklr_seq,
    })

    # (b)/(c) one real Scenario-A and one real Scenario-D window from the actual
    # held-out test set (xai_predictions.npz), so a first-time visitor sees the
    # taxonomy's contrast on genuine data, not a synthetic sequence.
    xp = np.load(XAI_PREDICTIONS_NPZ, allow_pickle=True)
    ep = np.load(EVAL_PREDICTIONS_NPZ, allow_pickle=True)
    err = np.abs(ep["y_pred"] - ep["y_true"])
    # key ep windows by (chrom,start,end) to look up error alongside xp's scenario
    ep_key = {(str(c), int(s), int(e)): i for c, s, e, i in
              zip(ep["chrom"], ep["start"], ep["end"], range(len(ep["chrom"])))}

    def best_example(scenario_letter, label):
        idx = np.where(xp["scenario"] == scenario_letter)[0]
        # pick the lowest-error member of this scenario for A (cleanest illustration),
        # highest-error member for D (cleanest illustration of "do not trust this")
        cand_err = []
        for i in idx:
            key = (str(xp["chrom"][i]), int(xp["start"][i]), int(xp["end"][i]))
            j = ep_key.get(key)
            cand_err.append(err[j] if j is not None else np.nan)
        cand_err = np.array(cand_err)
        pick = idx[np.nanargmin(cand_err)] if scenario_letter == "A" else idx[np.nanargmax(cand_err)]
        chrom, start, end = str(xp["chrom"][pick]), int(xp["start"][pick]), int(xp["end"][pick])
        seq = tb.sequence(chrom, start, end)
        return {
            "name": label,
            "description": (
                f"A real held-out test window (chr8/chr9, never used in training) "
                f"labelled Scenario {scenario_letter} by the model's own trust taxonomy."
            ),
            "chrom": chrom, "start": start, "end": end, "sequence": seq,
        }

    examples.append(best_example("A", "Real held-out window: Scenario A (high consensus, high coherence)"))
    examples.append(best_example("D", "Real held-out window: Scenario D (low consensus, low coherence)"))
    tb.close()

    (OUT_DIR / "examples.json").write_text(json.dumps(examples, indent=2))
    print(f"  saved examples.json: {len(examples)} examples")

    print("\nDone. hf_space/assets/ is ready for the app.")


if __name__ == "__main__":
    main()
