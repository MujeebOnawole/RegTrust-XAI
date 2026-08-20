#!/usr/bin/env python
"""Causal, motif-span-exact occlusion coherence -- a supplementary robustness
check of the coherence axis, NOT a replacement for xai.py/results/
xai_results.json (the manuscript's authoritative numbers). Motivated by an
external comparison to this author's own antimicrobial-XAI paper (J
Cheminform, tokenize SMILES -> occlude atom tokens -> aggregate onto BRICS
fragments): does RegTrust's coherence result change if the occlusion unit is
the model's OWN OBSERVABLE MOTIF SPAN (biologically exact) instead of a fixed
32bp bin, and is there measurable context-dependence (a motif's causal effect
changing depending on whether it is masked alone or as part of a cluster) --
the DNA analogue of a substituent's effect depending on scaffold context.

WHAT THIS DOES NOT DO, STATED UP FRONT. It does not build a full hierarchical
Murcko-style tokenization of DNA (no named "regulatory module" ontology, no
learned token vocabulary). That is a real second project, not something to
improvise inside a robustness check. What it tests is the two claims from
that comparison that are cheap, well-defined, and directly checkable against
data already in this project:
  1. CAUSAL vs CORRELATIONAL coherence. The current axis
     (trust.localization_coherence) asks whether a fixed 32bp bin's
     |attribution| percentile-ranks high AND that bin happens to overlap a
     motif hit -- a bin can contain a motif plus unrelated flanking sequence,
     and a motif can straddle two bins (motif_shell.py's own bin-overlap
     convention, see motif_coverage_by_bin). This script instead measures,
     per window, the actual prediction change from masking EXACTLY the bases
     a real motif hit covers (base_pred - masked_pred), matching this
     project's own occlusion_attribution masking convention (uniform 0.25,
     not all-zero) so the two are comparable in the same units.
  2. CONTEXT-SENSITIVITY (additivity). For windows with >=2 motif instances
     close enough to cluster into one "module" (see MODULE_GAP_BP), compares
     the SUM of each instance's own individual knockout effect against the
     JOINT effect of knocking out the whole module at once. If they matched
     exactly, motif effects would be context-independent (each motif
     contributes the same delta whether alone or with neighbors) -- a
     mismatch is direct causal evidence of context-dependent attribution,
     the same phenomenon a scaffold-dependent substituent effect is in
     chemistry.

A CONFOUND THIS SCRIPT DELIBERATELY CONTROLS FOR. A window with a larger
motif shell trivially shows a bigger raw masking effect just because more
sequence was deleted, regardless of whether that sequence was biologically
special. Every real shell-knockout delta is therefore compared against
N_RANDOM_CONTROLS knockouts of RANDOMLY PLACED spans of the SAME total
length (drawn per-instance, so the total masked base count matches exactly)
within the same window -- causal_coherence = |delta_shell| minus the mean
|delta| of those matched random controls. A positive causal_coherence means
the model is more sensitive to the motif-covered bases specifically than to
an equal amount of arbitrary sequence in the same window; a value near zero
means the model would have reacted about as much to any other equally-sized
chunk of that window, i.e. the observed motif isn't doing anything the model
treats as special.

MODULE_GAP_BP=24 is a deliberately named, undefended parameter (not derived
from any specific biological distance), chosen loosely near the scale of a
composite dimer motif (e.g. GATA1::TAL1 spans ~15-20bp) and a helical-turn-
scale cooperative TF-TF spacing -- flag this as a judgment call if the
context-sensitivity result is later reported anywhere, do not present it as
a validated biological threshold.

REUSES, NOT REPLACES, THE VALIDATED PIPELINE. Same attribution checkpoint as
xai.py (load_best_checkpoint), same K562 motif panel (motif_shell.py,
unchanged), same masking value (0.25, matching model.occlusion_attribution),
same consensus cutoff (read from results/xai_results.json, not recomputed),
same calibration pool sample (reproduces xai.py's exact seed/pool draw, the
same discipline sensitivity_thresholds.py already established). Output goes
to its own results/motif_causal_occlusion_results.json -- never overwrites
results/xai_results.json, the manuscript's authoritative file.

COST. Per window this needs roughly 1 (base) + 1 (shell) + N_RANDOM_CONTROLS
(random-matched controls) + n_modules + n_instances forward passes -- on the
order of 10-15 per window with a resolved shell, versus xai.py's fixed 64
per window for its bin-based occlusion attribution. Should be substantially
cheaper than xai.py's own already-completed full-scale run.
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
    CHECKPOINT_DIR,
    ENSEMBLE_SIZE,
    EVAL_PREDICTIONS_NPZ,
    GENOME_2BIT,
    HOLDOUT_CHROMS,
    JASPAR_PFM_PATH,
    RESULTS,
    SEQUENCE_WINDOWS_NPZ,
    SPLIT_SEED,
    TRUST_AGREEMENT_PERCENTILE,
    TRUST_CALIB_MAX_SAMPLES,
    XAI_ERROR_THRESHOLDS,
    XAI_RESULTS_JSON,
)
from data_module import one_hot_encode
from motif_shell import K562_TF_PANEL, load_k562_pssms, motif_hit_positions
from trust import cv_ratio_from_stats, trust_block
from validate_trust_axes import coherence_scenario_a_vs_b
from xai import load_best_checkpoint

MODULE_GAP_BP = 24        # see module docstring -- a chosen parameter, not derived
N_RANDOM_CONTROLS = 5      # matched-length random-span controls per window, for the coverage-extent confound
MASK_VALUE = 0.25         # matches model.occlusion_attribution's own masking convention exactly


def contiguous_spans(covered: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a boolean coverage array -> [(start, end), ...],
    half-open. This IS the finest attribution unit this script uses ("motif
    instance") -- overlapping hits from different panel TFs are already
    merged by motif_hit_positions' own boolean OR, so no separate per-TF
    bookkeeping is needed or invented here."""
    spans = []
    n = len(covered)
    i = 0
    while i < n:
        if covered[i]:
            j = i
            while j < n and covered[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def cluster_spans(spans: list[tuple[int, int]], gap_bp: int):
    """Merge spans separated by <= gap_bp into one "module" span. Returns
    (merged_spans, membership) where membership[k] is the list of original
    spans belonging to merged_spans[k] -- needed for the additivity check
    (sum of member instance deltas vs the module's own joint knockout delta)."""
    if not spans:
        return [], []
    spans = sorted(spans)
    clusters = [[spans[0]]]
    for s in spans[1:]:
        prev_end = clusters[-1][-1][1]
        if s[0] - prev_end <= gap_bp:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    merged = [(c[0][0], c[-1][1]) for c in clusters]
    return merged, clusters


def masked_forward(model, x, spans, device):
    """x: (N_CHANNELS, WINDOW_BP) one-hot tensor, unbatched. spans: list of
    (start, end) half-open regions to set to MASK_VALUE simultaneously (one
    forward pass covers the whole set, matching how a single BRICS-fragment
    knockout in the antimicrobial paper masks one fragment's full atom span
    in one pass, not atom-by-atom)."""
    x_occ = x.clone()
    for s, e in spans:
        x_occ[:, s:e] = MASK_VALUE
    with torch.no_grad():
        return model(x_occ.unsqueeze(0).to(device)).item()


def window_causal_analysis(model, tb, chrom, start, end, pssms, device, rng,
                            n_random_controls=N_RANDOM_CONTROLS, module_gap_bp=MODULE_GAP_BP):
    """One window's full causal-occlusion pipeline. Returns None if no motif
    hit anywhere (same "no shell" convention trust.localization_coherence
    uses -- excluded from trust labelling, never defaulted to a value)."""
    seq = tb.sequence(str(chrom), int(start), int(end))
    covered = motif_hit_positions(seq, pssms)
    if not covered.any():
        return None

    x = torch.from_numpy(one_hot_encode(seq))
    seq_len = x.shape[-1]
    with torch.no_grad():
        base_pred = model(x.unsqueeze(0).to(device)).item()

    instances = contiguous_spans(covered)
    modules, membership = cluster_spans(instances, module_gap_bp)

    shell_pred = masked_forward(model, x, modules, device)   # union(modules) == union(instances)
    delta_shell = base_pred - shell_pred

    rand_deltas = []
    for _ in range(n_random_controls):
        rand_spans = []
        for (s, e) in instances:
            length = e - s
            max_start = seq_len - length
            rs = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            rand_spans.append((rs, rs + length))
        rand_deltas.append(base_pred - masked_forward(model, x, rand_spans, device))
    rand_delta_mean_abs = float(np.mean(np.abs(rand_deltas))) if rand_deltas else 0.0

    causal_coherence = abs(delta_shell) - rand_delta_mean_abs

    additivity_records = []
    for mod_span, members in zip(modules, membership):
        if len(members) < 2:
            continue
        module_delta = base_pred - masked_forward(model, x, [mod_span], device)
        instance_deltas = [base_pred - masked_forward(model, x, [m], device) for m in members]
        additivity_records.append({
            "n_instances": len(members),
            "module_delta": module_delta,
            "sum_instance_deltas": float(np.sum(instance_deltas)),
        })

    return {
        "n_instances": len(instances),
        "n_modules": len(modules),
        "base_pred": base_pred,
        "delta_shell": delta_shell,
        "rand_delta_mean_abs": rand_delta_mean_abs,
        "causal_coherence": causal_coherence,
        "additivity_records": additivity_records,
    }


def run_over_indices(model, tb, chrom_arr, start_arr, end_arr, indices, pssms, device, rng,
                      n_random_controls, module_gap_bp, tag):
    n = len(indices)
    causal = np.full(n, np.nan)
    n_instances = np.full(n, np.nan)
    n_modules = np.full(n, np.nan)
    additivity_records = []
    for pos, i in enumerate(indices):
        res = window_causal_analysis(model, tb, chrom_arr[i], start_arr[i], end_arr[i], pssms,
                                      device, rng, n_random_controls, module_gap_bp)
        if res is None:
            continue
        causal[pos] = res["causal_coherence"]
        n_instances[pos] = res["n_instances"]
        n_modules[pos] = res["n_modules"]
        for rec in res["additivity_records"]:
            rec["window_idx"] = int(i)
            additivity_records.append(rec)
        if (pos + 1) % 5000 == 0:
            print(f"  {tag}: {pos + 1}/{n} windows processed")
    return causal, n_instances, n_modules, additivity_records


def additivity_summary(records):
    if not records:
        return {"n_multi_instance_modules": 0}
    joint = np.array([r["module_delta"] for r in records])
    summed = np.array([r["sum_instance_deltas"] for r in records])
    diff = np.abs(joint) - np.abs(summed)   # >0: superadditive (joint bigger than sum) -> cooperative
                                             # <0: subadditive (joint smaller than sum) -> redundant
    w = stats.wilcoxon(np.abs(joint), np.abs(summed))
    return {
        "n_multi_instance_modules": int(len(records)),
        "mean_abs_joint_delta": float(np.mean(np.abs(joint))),
        "mean_abs_summed_instance_deltas": float(np.mean(np.abs(summed))),
        "mean_signed_diff_joint_minus_summed": float(np.mean(diff)),
        "frac_superadditive": float((diff > 0).mean()),
        "frac_subadditive": float((diff < 0).mean()),
        "wilcoxon_joint_vs_summed_p": float(w.pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-npz", type=Path, default=SEQUENCE_WINDOWS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--eval-predictions", type=Path, default=EVAL_PREDICTIONS_NPZ)
    ap.add_argument("--xai-results", type=Path, default=XAI_RESULTS_JSON)
    ap.add_argument("--out", type=Path, default=RESULTS / "motif_causal_occlusion_results.json")
    ap.add_argument("--pred-out", type=Path, default=RESULTS / "motif_causal_occlusion_predictions.npz")
    ap.add_argument("--thresholds", type=float, nargs="+", default=XAI_ERROR_THRESHOLDS)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--n-random-controls", type=int, default=N_RANDOM_CONTROLS)
    ap.add_argument("--module-gap-bp", type=int, default=MODULE_GAP_BP)
    ap.add_argument("--max-windows", type=int, default=None,
                     help="subsample the test set for a quick smoke test before the full run")
    ap.add_argument("--calib-size", type=int, default=TRUST_CALIB_MAX_SAMPLES,
                     help="pool calibration sample size; override for a quick smoke test "
                          "(same convention as xai.py's own --calib-size)")
    args = ap.parse_args()

    print("preflight: checking required inputs...")
    for p in (args.windows_npz, args.eval_predictions, args.xai_results, GENOME_2BIT, JASPAR_PFM_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run the earlier pipeline steps first (build_features.py, "
                              f"train_cv.py, final_eval.py, xai.py).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading K562 TF panel PSSMs...")
    pssms = load_k562_pssms(JASPAR_PFM_PATH)
    print(f"  {len(pssms)}/{len(K562_TF_PANEL)} panel TFs resolved")

    print("loading attribution checkpoint (same one xai.py used for coherence)...")
    model, label_mean, label_std = load_best_checkpoint(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("loading the already-calibrated consensus cutoff from results/xai_results.json "
          "(not recomputed -- only the coherence side of the taxonomy changes here)...")
    xr = json.loads(Path(args.xai_results).read_text())
    cons_cut = float(xr["trust"]["internal_test"]["cutoffs"]["consensus"])
    print(f"  consensus cutoff (reused): {cons_cut:.4f}")

    print("loading eval_predictions.npz (test set, from final_eval.py)...")
    ep = np.load(args.eval_predictions, allow_pickle=True)
    test_chrom, test_start, test_end = ep["chrom"], ep["start"], ep["end"]
    y_true, y_pred, y_pred_std = ep["y_true"], ep["y_pred"], ep["y_pred_std"]
    pop_std = float(ep["pop_std"])
    err_test = np.abs(y_pred - y_true)
    cvr_test = cv_ratio_from_stats(y_pred_std, pop_std)
    print(f"  {len(y_true)} test windows, pop_std {pop_std:.4f}")

    rng = np.random.default_rng(args.seed)
    test_indices = np.arange(len(test_chrom))
    if args.max_windows is not None and args.max_windows < len(test_indices):
        test_indices = rng.choice(test_indices, size=args.max_windows, replace=False)
        print(f"  --max-windows override: subsampled to {len(test_indices)} test windows")

    tb = py2bit.open(str(GENOME_2BIT))
    print("\ncomputing causal (motif-span-exact) occlusion on the test set...")
    causal_full = np.full(len(test_chrom), np.nan)
    n_inst_full = np.full(len(test_chrom), np.nan)
    n_mod_full = np.full(len(test_chrom), np.nan)
    causal_sub, n_inst_sub, n_mod_sub, additivity_records = run_over_indices(
        model, tb, test_chrom, test_start, test_end, test_indices, pssms, device, rng,
        args.n_random_controls, args.module_gap_bp, "test")
    causal_full[test_indices] = causal_sub
    n_inst_full[test_indices] = n_inst_sub
    n_mod_full[test_indices] = n_mod_sub
    have_shell = ~np.isnan(causal_full) & np.isin(np.arange(len(test_chrom)), test_indices)
    print(f"  test windows with a resolvable motif shell: {have_shell.sum()}/{len(test_indices)} sampled")

    # ---- calibration: reproduce xai.py's EXACT pool sample (same seed, same draw) --------
    print("\nreproducing xai.py's calibration sample "
          f"(seed={SPLIT_SEED}, size={args.calib_size})...")
    calib_rng = np.random.default_rng(SPLIT_SEED)
    d = np.load(args.windows_npz, allow_pickle=True)
    chrom_all = d["chrom"]
    pool_idx = np.where(~np.isin(chrom_all, HOLDOUT_CHROMS))[0]
    calib_size = min(args.calib_size, len(pool_idx))
    calib_idx = calib_rng.choice(pool_idx, size=calib_size, replace=False)
    if args.calib_size == TRUST_CALIB_MAX_SAMPLES:
        print(f"  calibration sample: {calib_size} pool windows (identical to xai.py's own)")
    else:
        print(f"  calibration sample: {calib_size} pool windows (--calib-size override -- "
              f"np.random.Generator.choice(size=N) is not a prefix of a different N's draw, "
              f"so this is NOT a subset of xai.py's own {TRUST_CALIB_MAX_SAMPLES}-window sample, "
              f"only a same-seed sample of a different size -- fine for a smoke test only, "
              f"do not use for a reportable agreement cutoff)")

    print("  computing causal occlusion on the calibration sample...")
    causal_calib, _, _, _ = run_over_indices(
        model, tb, chrom_all, d["start"], d["end"], calib_idx, pssms, device, rng,
        args.n_random_controls, args.module_gap_bp, "calib")
    tb.close()
    causal_calib_valid = causal_calib[~np.isnan(causal_calib)]
    agr_cut = float(np.percentile(causal_calib_valid, TRUST_AGREEMENT_PERCENTILE)) if len(causal_calib_valid) else 0.0
    print(f"  {len(causal_calib_valid)}/{calib_size} calibration windows resolved a motif shell; "
          f"causal-coherence agreement cutoff: {agr_cut:.4f}")

    # ---- apply the SAME trust_block machinery, causal coherence swapped in for bin coherence --
    print("\n=== TRUST TAXONOMY, causal-coherence variant (test set) ===")
    valid = have_shell
    results = {
        "note": ("Supplementary robustness check: is xai.py's bin-overlap coherence result "
                 "sensitive to occlusion-unit granularity? Same checkpoint, same consensus "
                 "cutoff, same test set; only the coherence axis is recomputed as a causal, "
                 "motif-span-exact, random-matched-control-adjusted knockout effect. Compare "
                 "against results/xai_results.json and results/trust_validation_results.json "
                 "(coherence_vs_error, coherence_scenario_a_vs_b_conditional) -- do not treat "
                 "this file as replacing either."),
        "module_gap_bp": args.module_gap_bp,
        "n_random_controls": args.n_random_controls,
        "mask_value": MASK_VALUE,
        "consensus_cutoff_reused": cons_cut,
        "causal_agreement_cutoff": agr_cut,
        "n_test_sampled": int(len(test_indices)),
        "n_test_with_shell": int(valid.sum()),
    }

    scen = np.array([])
    if valid.sum() > 0:
        block, scen = trust_block(cvr_test[valid], causal_full[valid], err_test[valid],
                                   cons_cut, agr_cut, args.thresholds,
                                   "causal coherence (test, shell-covered)")
        results["trust"] = block

        print("\n=== causal_coherence vs |error| (pooled, all scenarios) ===")
        rho_p = stats.spearmanr(causal_full[valid], err_test[valid])
        results["causal_coherence_vs_error_pooled"] = {
            "n": int(valid.sum()), "rho": float(rho_p.statistic), "p": float(rho_p.pvalue)}
        print(f"  n={valid.sum()}  rho {rho_p.statistic:+.4f}  p {rho_p.pvalue:.3g}")

        print("\n=== causal_coherence: Scenario A vs B conditional test "
              "(the correct test, per validate_trust_axes.py's own precedent) ===")
        results["causal_coherence_scenario_a_vs_b_conditional"] = coherence_scenario_a_vs_b(
            scen, err_test[valid])
    else:
        results["trust"] = None

    print("\n=== context-sensitivity / additivity (multi-instance modules only) ===")
    results["additivity"] = additivity_summary(additivity_records)
    a = results["additivity"]
    if a.get("n_multi_instance_modules", 0) > 0:
        print(f"  n={a['n_multi_instance_modules']}  mean|joint| {a['mean_abs_joint_delta']:.5f}  "
              f"mean|summed instances| {a['mean_abs_summed_instance_deltas']:.5f}  "
              f"superadditive {a['frac_superadditive']:.1%}  subadditive {a['frac_subadditive']:.1%}  "
              f"Wilcoxon p {a['wilcoxon_joint_vs_summed_p']:.3g}")
    else:
        print("  no multi-instance modules found in the sampled windows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        chrom=test_chrom, start=test_start, end=test_end,
        y_true=y_true, y_pred=y_pred,
        cv_ratio=cvr_test, causal_coherence=causal_full,
        n_instances=n_inst_full, n_modules=n_mod_full,
        scenario=scen, cons_cut=cons_cut, agr_cut=agr_cut,
        sampled_mask=np.isin(np.arange(len(test_chrom)), test_indices),
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
