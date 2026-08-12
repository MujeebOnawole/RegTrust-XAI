#!/usr/bin/env python
"""Item 9, rung 1b: motif-shuffled synthetic controls (project_status.md,
"Graded distribution-shift design").

WHAT THIS TESTS, AND WHY THERE IS NO ERROR NUMBER HERE. dinuc_shuffle.py
generates, for a sample of real test-set windows, a dinucleotide-count-
preserving shuffle -- same local base-pair statistics, destroyed motif
organization, the Nagai et al. 2026 review's own Table 1 "motif
rearrangement" covariate-shift category. A shuffled sequence does not exist
in the genome, so there is no measured accessibility to compare a
prediction against -- unlike rung 1a, this script can only ask whether the
model's OWN signals (AD distance, predicted accessibility) shift in the
expected direction under this synthetic perturbation, not whether error
grows (there is no error to compute).

Two paired (shuffled vs its own original) comparisons, both Wilcoxon
signed-rank (paired, not independent-samples, since each shuffled sequence
is compared against the one real sequence it came from):
1. AD distance: does the model's embedding distance to the training pool
   increase after motif organization is destroyed? (uses model.embed(),
   the identical embedding space validate_trust_axes.py's AD axis uses)
2. Predicted accessibility: does the model's own prediction drop when
   motif organization is destroyed but composition is held fixed? (a
   real, if indirect, test of whether the model has learned motif-
   dependent structure, not just base composition)
3. Motif-shell coverage (motif_shell.py): does the JASPAR-panel motif hit
   coverage actually drop after shuffling, confirming the perturbation did
   what it was designed to do, not just a sanity check assumed to hold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import py2bit
import torch
from scipy import stats

from config import (
    BIN_BP,
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    JASPAR_PFM_PATH,
    RUNG1B_PREDICTIONS_NPZ,
    RUNG1B_RESULTS_JSON,
    RUNG1B_SAMPLE_SIZE,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
)
from data_module import one_hot_encode
from dinuc_shuffle import dinuc_shuffle
from motif_shell import K562_TF_PANEL, load_k562_pssms, motif_coverage_by_bin
from trust import l2_normalize_rows, nn_distance
from xai import load_best_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--out", type=Path, default=RUNG1B_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=RUNG1B_PREDICTIONS_NPZ)
    ap.add_argument("--sample-size", type=int, default=RUNG1B_SAMPLE_SIZE,
                     help="real test-set windows to shuffle and re-score; override for a "
                          "quick local smoke test")
    ap.add_argument("--ref-pool-size", type=int, default=5000,
                     help="training-pool reference sample for AD nearest-neighbor distance")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, GENOME_2BIT, JASPAR_PFM_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading K562 TF panel PSSMs...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    print("\nloading attribution checkpoint (same one xai.py/validate_trust_axes.py use)...")
    model, label_mean, label_std = load_best_checkpoint(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    d = np.load(args.windows_npz, allow_pickle=True)
    chrom_all = d["chrom"]
    is_holdout = np.isin(chrom_all, HOLDOUT_CHROMS)
    test_idx_all = np.where(is_holdout)[0]
    pool_idx = np.where(~is_holdout)[0]

    sample_size = min(args.sample_size, len(test_idx_all))

    print("\nbuilding training-pool reference sample for AD nearest-neighbor distance...")
    ref_size = min(args.ref_pool_size, len(pool_idx))
    ref_idx = rng.choice(pool_idx, size=ref_size, replace=False)

    tb = py2bit.open(str(GENOME_2BIT))

    # dinuc_shuffle needs an ACGT-only sequence (see its own ValueError on non-ACGT
    # characters) -- real genomic windows occasionally contain N runs (assembly gaps),
    # a real, expected edge case, not an error. Oversample and skip N-containing
    # windows rather than crash or silently corrupt the shuffle.
    candidate_pool = rng.permutation(test_idx_all)
    sample_idx, n_skipped_n = [], 0
    for i in candidate_pool:
        if len(sample_idx) >= sample_size:
            break
        seq_check = tb.sequence(str(chrom_all[i]), int(d["start"][i]), int(d["end"][i]))
        if "N" in seq_check.upper():
            n_skipped_n += 1
            continue
        sample_idx.append(i)
    sample_idx = np.array(sample_idx)
    sample_size = len(sample_idx)
    print(f"\nsampled {sample_size} real test-set windows to shuffle and re-score "
          f"({n_skipped_n} candidates skipped for containing an N base)")

    def embed_one(seq):
        x = torch.from_numpy(one_hot_encode(seq)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.embed(x).cpu().numpy()[0]

    def predict_one(seq):
        x = torch.from_numpy(one_hot_encode(seq)).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(x).item()
        return z * label_std + label_mean

    print("embedding the training-pool reference sample...")
    ref_embed = np.stack([
        embed_one(tb.sequence(str(chrom_all[i]), int(d["start"][i]), int(d["end"][i])))
        for i in ref_idx
    ])
    ref_embed_n = l2_normalize_rows(ref_embed)

    print("\nfor each sampled window: original vs dinucleotide-shuffled -- "
          "AD distance, prediction, motif-shell coverage (the slow step)...")
    ad_dist_orig, ad_dist_shuf = np.empty(sample_size), np.empty(sample_size)
    pred_orig, pred_shuf = np.empty(sample_size), np.empty(sample_size)
    cov_orig, cov_shuf = np.empty(sample_size), np.empty(sample_size)
    n_bins_ref = None

    for j, i in enumerate(sample_idx):
        chrom, start, end = str(chrom_all[i]), int(d["start"][i]), int(d["end"][i])
        seq = tb.sequence(chrom, start, end)
        shuf_seq = dinuc_shuffle(seq, np.random.default_rng(int(rng.integers(0, 2**32 - 1))))

        e_orig, e_shuf = embed_one(seq), embed_one(shuf_seq)
        e_orig_n = e_orig / (np.linalg.norm(e_orig) + 1e-8)
        e_shuf_n = e_shuf / (np.linalg.norm(e_shuf) + 1e-8)
        ad_dist_orig[j] = 1.0 - float(np.max(ref_embed_n @ e_orig_n))
        ad_dist_shuf[j] = 1.0 - float(np.max(ref_embed_n @ e_shuf_n))

        pred_orig[j] = predict_one(seq)
        pred_shuf[j] = predict_one(shuf_seq)

        shell_orig = motif_coverage_by_bin(seq, pssms, bin_width=BIN_BP, bin_stride=BIN_BP)
        shell_shuf = motif_coverage_by_bin(shuf_seq, pssms, bin_width=BIN_BP, bin_stride=BIN_BP)
        n_bins = max(1, len(seq) // BIN_BP)
        n_bins_ref = n_bins
        cov_orig[j] = len(shell_orig) / n_bins
        cov_shuf[j] = len(shell_shuf) / n_bins

        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{sample_size} scored")

    tb.close()

    results = {"sample_size": sample_size, "ref_pool_size": ref_size, "n_bins_per_window": n_bins_ref}

    print("\n=== AD distance: original vs shuffled (paired) ===")
    w_ad = stats.wilcoxon(ad_dist_shuf, ad_dist_orig, alternative="greater")
    results["ad_distance"] = {
        "mean_orig": float(ad_dist_orig.mean()), "mean_shuf": float(ad_dist_shuf.mean()),
        "mean_increase": float((ad_dist_shuf - ad_dist_orig).mean()),
        "frac_increased": float((ad_dist_shuf > ad_dist_orig).mean()),
        "wilcoxon_shuf_gt_orig_p": float(w_ad.pvalue),
    }
    print(f"  orig {ad_dist_orig.mean():.4f}  ->  shuffled {ad_dist_shuf.mean():.4f}  "
          f"(increased in {results['ad_distance']['frac_increased']:.1%} of windows, "
          f"Wilcoxon p={w_ad.pvalue:.3g})")

    print("\n=== predicted accessibility: original vs shuffled (paired) ===")
    w_pred = stats.wilcoxon(pred_orig, pred_shuf, alternative="greater")
    results["prediction"] = {
        "mean_orig": float(pred_orig.mean()), "mean_shuf": float(pred_shuf.mean()),
        "mean_decrease": float((pred_orig - pred_shuf).mean()),
        "frac_decreased": float((pred_shuf < pred_orig).mean()),
        "wilcoxon_orig_gt_shuf_p": float(w_pred.pvalue),
    }
    print(f"  orig {pred_orig.mean():.4f}  ->  shuffled {pred_shuf.mean():.4f}  "
          f"(decreased in {results['prediction']['frac_decreased']:.1%} of windows, "
          f"Wilcoxon p={w_pred.pvalue:.3g})")

    print("\n=== motif-shell coverage: original vs shuffled (paired, confirms the "
          "perturbation actually destroyed motifs) ===")
    w_cov = stats.wilcoxon(cov_orig, cov_shuf, alternative="greater")
    results["motif_coverage"] = {
        "mean_orig": float(cov_orig.mean()), "mean_shuf": float(cov_shuf.mean()),
        "mean_decrease": float((cov_orig - cov_shuf).mean()),
        "frac_decreased": float((cov_shuf < cov_orig).mean()),
        "wilcoxon_orig_gt_shuf_p": float(w_cov.pvalue),
    }
    print(f"  orig {cov_orig.mean():.4f}  ->  shuffled {cov_shuf.mean():.4f}  "
          f"(decreased in {results['motif_coverage']['frac_decreased']:.1%} of windows, "
          f"Wilcoxon p={w_cov.pvalue:.3g})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        chrom=chrom_all[sample_idx], start=d["start"][sample_idx], end=d["end"][sample_idx],
        ad_distance_orig=ad_dist_orig, ad_distance_shuf=ad_dist_shuf,
        prediction_orig=pred_orig, prediction_shuf=pred_shuf,
        motif_coverage_orig=cov_orig, motif_coverage_shuf=cov_shuf,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
