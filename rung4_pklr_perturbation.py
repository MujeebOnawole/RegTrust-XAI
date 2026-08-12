#!/usr/bin/env python
"""Item 9, rung 4: true single-base mutational perturbation (project_status.md,
"Graded distribution-shift design"), scored against Kircher et al. 2019's
saturation-mutagenesis MPRA of the PKLR promoter in K562 (see "Data
collection" for sourcing/verification).

WHAT THIS TESTS, AND HOW IT DIFFERS FROM EVERY OTHER VALIDATION IN THIS
PROJECT. Every other cross-assay check (item 7, mpra_eval.py) compares the
model's attribution/prediction on INDEPENDENT candidate sequences against
each one's own absolute measured activity -- informative, but each data
point is a different sequence, so the comparison is correlational, not
causal. Here, every data point is the SAME reference sequence with ONE base
changed, and the measured Value column IS the causal effect of that single
edit. This is a genuine in-silico-mutagenesis perturbation-agreement test:
does the model's own predicted_delta = pred(alt) - pred(ref) track the
measured delta, not just "is this sequence active."

ONE LOCUS, NOT A GENERALIZABLE CLAIM. n=1,776 (24h) / 1,794 (48h) variants,
but n_loci=1 (PKLR promoter only -- the one Kircher et al. element tested in
K562). Report this as a locus-level case study ("at the PKLR promoter,
RegTrust-XAI was prospectively challenged against saturation-mutagenesis
measurements"), never as "RegTrust-XAI generalizes to mutational
perturbation" -- see project_status.md's own caution on this, credited to
external review.

WHY ONE FIXED WINDOW, NOT A WINDOW RE-CENTERED PER VARIANT. All variants
sit within a 469bp span, far smaller than WINDOW_BP=2048 -- one window
centered on the locus midpoint covers every variant position with room to
spare, so the reference window is built and scored ONCE, and every variant
is just that same window with a single base substituted at its own
position. This also means AD distance is essentially a LOCUS property here,
not a variant property (see the pushback on the external-feedback
suggestion to stratify the perturbation correlation by AD band within this
one locus -- there is no meaningful within-locus AD spread to stratify by,
since a single base change barely moves a 2048bp window's embedding).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import py2bit
import torch
from scipy import stats

from config import (
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    KIRCHER_PKLR_24H_TSV,
    KIRCHER_PKLR_48H_TSV,
    RUNG4_PREDICTIONS_NPZ,
    RUNG4_RESULTS_JSON,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    WINDOW_BP,
)
from data_module import one_hot_encode
from final_eval import load_ensemble
from trust import l2_normalize_rows, nn_distance
from xai import load_best_checkpoint


def load_variants(tsv_path):
    with open(tsv_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        if len(row["Ref"]) != 1 or len(row["Alt"]) != 1:
            raise SystemExit(
                f"{tsv_path}: non-SNV row found (Ref={row['Ref']!r}, Alt={row['Alt']!r}) -- "
                f"this script only handles single-nucleotide substitutions; verified 0/N such "
                f"rows in the 24h file 2026-08-12, re-check the source file if this fires.")
    return rows


def predict_ensemble_single(ensemble, label_means, label_stds, x, device):
    """Ensemble-mean prediction for ONE window (not batched via DataLoader --
    this script scores one fixed reference window plus ~1,800 single-base
    variants of it, not tens of thousands of independent windows, so the
    DataLoader/worker machinery final_eval.predict_ensemble uses would be
    pure overhead here)."""
    x = x.unsqueeze(0).to(device)
    preds = []
    with torch.no_grad():
        for m, label_mean, label_std in zip(ensemble, label_means, label_stds):
            z = m(x).item()
            preds.append(z * label_std + label_mean)
    return float(np.mean(preds))


def score_locus(tsv_path, tb, ensemble, label_means, label_stds, attribution_model,
                 ref_embed_n, device, tag, max_variants=None):
    variants = load_variants(tsv_path)
    if max_variants is not None:
        variants = variants[:max_variants]
        print(f"  [{tag}] --max-variants override: scoring only the first {len(variants)} variants")
    chrom = "chr" + variants[0]["Chromosome"]
    positions = [int(v["Position"]) for v in variants]
    lo, hi = min(positions), max(positions)
    center = (lo + hi) // 2
    win_start = center - WINDOW_BP // 2
    win_end = win_start + WINDOW_BP
    print(f"  [{tag}] {len(variants)} variants, locus span chr1:{lo}-{hi} ({hi - lo}bp), "
          f"window {chrom}:{win_start}-{win_end}")
    assert win_start <= lo and hi < win_end, (
        f"[{tag}] the fixed {WINDOW_BP}bp window does not cover the full variant span -- "
        f"widen WINDOW_BP or re-derive the window center")

    ref_seq = tb.sequence(chrom, win_start, win_end)
    n_ref_mismatch = 0
    for v in variants:
        offset = int(v["Position"]) - 1 - win_start   # Position is 1-based (verified 2026-08-12)
        if ref_seq[offset].upper() != v["Ref"].upper():
            n_ref_mismatch += 1
    if n_ref_mismatch:
        print(f"  [{tag}] WARNING: {n_ref_mismatch}/{len(variants)} variants' Ref column does "
              f"not match hg38 at their Position -- these rows are still scored (the alt "
              f"sequence is still a real, valid single-base edit of the reference window), but "
              f"this is unexpected given the 200/200 match confirmed during scoping and should "
              f"be investigated, not silently ignored.")

    x_ref = torch.from_numpy(one_hot_encode(ref_seq))
    pred_ref = predict_ensemble_single(ensemble, label_means, label_stds, x_ref, device)

    pred_delta = np.empty(len(variants), dtype=np.float64)
    measured_value = np.empty(len(variants), dtype=np.float64)
    for i, v in enumerate(variants):
        offset = int(v["Position"]) - 1 - win_start
        alt_seq = ref_seq[:offset] + v["Alt"].upper() + ref_seq[offset + 1:]
        x_alt = torch.from_numpy(one_hot_encode(alt_seq))
        pred_alt = predict_ensemble_single(ensemble, label_means, label_stds, x_alt, device)
        pred_delta[i] = pred_alt - pred_ref
        measured_value[i] = float(v["Value"])

    res = stats.spearmanr(pred_delta, measured_value)
    print(f"  [{tag}] corr(predicted_delta, measured_value): n={len(variants)}  "
          f"rho {res.statistic:+.4f}  p {res.pvalue:.3g}")

    with torch.no_grad():
        e_ref = attribution_model.embed(x_ref.unsqueeze(0).to(device)).cpu().numpy()[0]
    e_ref_n = e_ref / (np.linalg.norm(e_ref) + 1e-8)
    ad_distance_ref = 1.0 - float(np.max(ref_embed_n @ e_ref_n))
    print(f"  [{tag}] reference-window AD distance (locus-level, not variant-level): "
          f"{ad_distance_ref:.4f}")

    return {
        "n_variants": len(variants),
        "chrom": chrom, "window_start": win_start, "window_end": win_end,
        "n_ref_mismatch": n_ref_mismatch,
        "pred_ref": pred_ref,
        "corr_pred_delta_vs_measured": {"n": len(variants), "rho": float(res.statistic), "p": float(res.pvalue)},
        "ad_distance_reference_window": ad_distance_ref,
    }, pred_delta, measured_value, [v["Position"] for v in variants], [v["Ref"] for v in variants], [v["Alt"] for v in variants]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pklr-24h", type=Path, default=KIRCHER_PKLR_24H_TSV)
    ap.add_argument("--pklr-48h", type=Path, default=KIRCHER_PKLR_48H_TSV)
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--out", type=Path, default=RUNG4_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=RUNG4_PREDICTIONS_NPZ)
    ap.add_argument("--ref-pool-size", type=int, default=5000,
                     help="training-pool reference sample for the locus's own AD distance")
    ap.add_argument("--max-variants", type=int, default=None,
                     help="cap variants per timepoint for a quick local smoke test; omit for the full run")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.pklr_24h, args.pklr_48h, args.windows_npz, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; see project_status.md 'Data collection'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("\nloading the deployed 5-model ensemble (accuracy convention, same as final_eval.py, "
          "NOT the single attribution checkpoint -- rung 4 tests predicted regulatory effect, "
          "not attribution)...")
    ensemble, label_means, label_stds = load_ensemble(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("\nloading the attribution checkpoint (for model.embed() / AD distance only)...")
    attribution_model, _, _ = load_best_checkpoint(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("\nbuilding training-pool reference sample for the locus's own AD distance...")
    d = np.load(args.windows_npz, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    ref_size = min(args.ref_pool_size, len(pool_idx))
    ref_idx = rng.choice(pool_idx, size=ref_size, replace=False)

    tb = py2bit.open(str(GENOME_2BIT))

    def embed_one(i):
        seq = tb.sequence(str(chrom_all[i]), int(d["start"][i]), int(d["end"][i]))
        x = torch.from_numpy(one_hot_encode(seq)).unsqueeze(0).to(device)
        with torch.no_grad():
            return attribution_model.embed(x).cpu().numpy()[0]

    ref_embed = np.stack([embed_one(i) for i in ref_idx])
    ref_embed_n = l2_normalize_rows(ref_embed)
    print(f"  reference pool: {ref_size} windows")

    results = {"n_loci": 1, "locus": "PKLR promoter (Kircher et al. 2019)", "timepoints": {}}
    all_saves = {}

    print("\n=== scoring 24h (primary timepoint) ===")
    res_24h, delta_24h, meas_24h, pos_24h, ref_24h, alt_24h = score_locus(
        args.pklr_24h, tb, ensemble, label_means, label_stds, attribution_model,
        ref_embed_n, device, "24h", max_variants=args.max_variants)
    results["timepoints"]["24h"] = res_24h
    all_saves["24h"] = (delta_24h, meas_24h, pos_24h, ref_24h, alt_24h)

    print("\n=== scoring 48h (replication check) ===")
    res_48h, delta_48h, meas_48h, pos_48h, ref_48h, alt_48h = score_locus(
        args.pklr_48h, tb, ensemble, label_means, label_stds, attribution_model,
        ref_embed_n, device, "48h", max_variants=args.max_variants)
    results["timepoints"]["48h"] = res_48h
    all_saves["48h"] = (delta_48h, meas_48h, pos_48h, ref_48h, alt_48h)

    tb.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        pos_24h=np.array(pos_24h, dtype=np.int64), ref_24h=np.array(ref_24h), alt_24h=np.array(alt_24h),
        pred_delta_24h=delta_24h, measured_value_24h=meas_24h,
        pos_48h=np.array(pos_48h, dtype=np.int64), ref_48h=np.array(ref_48h), alt_48h=np.array(alt_48h),
        pred_delta_48h=delta_48h, measured_value_48h=meas_48h,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
