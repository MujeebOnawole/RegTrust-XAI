#!/usr/bin/env python
"""Cross-assay attribution-activity correlation for RegTrust-XAI (item 7 in
project_status.md's NEXT UP): for each lentiMPRA element in
mpra_elements.npz, computes element_attribution from the same attribution
checkpoint xai.py selected, then calls trust.cross_assay_attribution_
activity_correlation() per subgroup against that element's independently
measured lentiMPRA log2 activity.

THIS IS AN EXTERNAL VALIDATION CRITERION, NOT AN INFERENCE-TIME TRUST AXIS --
see trust.py's RENAME/RESTRUCTURE note for the full reasoning. It is
computed once, offline, against the lentiMPRA cohort's known measured
activity, to decide whether the (inference-time-only) coherence axis's
attribution is picking up anything cross-assay-verifiable, not as a
per-prediction label.

DOES NOT RE-TRAIN OR RE-SELECT A CHECKPOINT. Reuses xai.py's exact
load_best_checkpoint() (highest val_spearman across train_cv.py's folds), so
the attribution computed here is directly comparable to xai.py's coherence
axis, which used the identical checkpoint.

MODEL_PRED IS THE FULL ENSEMBLE MEAN, not the single attribution checkpoint's
own prediction -- matches final_eval.py's accuracy figures and is the
quantity the partial-correlation design amendment below needs to control
for.

PARTIAL CORRELATION AND OLS DECOMPOSITION, added per external review
2026-08-11 (project_status.md item 7's design amendment): raw
corr(attribution, activity) conflates two things -- attribution genuinely
localizing informative bases, versus attribution and activity both simply
tracking overall predicted regulatory strength (model_pred) for the
element's window. Alongside trust.cross_assay_attribution_activity_
correlation()'s unmodified raw Spearman, this script also reports (a) the
partial Spearman between attribution and activity controlling for
model_pred, computed as a rank-based partial correlation on the SAME top-k%-
by-activity-magnitude subset trust.py's function restricts to, and (b) an
OLS activity ~ model_pred + attribution decomposition (standardized
predictors), checking whether the attribution coefficient remains
meaningful once model_pred is in the model. Neither replaces the raw
correlation -- both are reported alongside it so a reader can see whether
the raw number survives the model_pred control.

COMPUTED SEPARATELY PER SUBGROUP (k562_native / hepg2_designed /
wtc11_designed), never pooled -- per trust.py's own docstring, pooling hides
exactly the distribution-shift comparison the subgroup split exists to
enable.

RUN AS A BATCH JOB: needs the same GPU container as xai.py -- occlusion
attribution (64 forward passes per element window, one window at a time,
same unbatched pattern xai.py already uses) is the slow step here, over
~54k elements. Use --max-elements for a quick local smoke test.
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
    MPRA_ELEMENTS_NPZ,
    MPRA_EVAL_PREDICTIONS_NPZ,
    MPRA_EVAL_RESULTS_JSON,
    PERTURBATION_AGREEMENT_TOP_K_PERCENT,
    SPLIT_SEED,
)
from data_module import one_hot_encode
from final_eval import load_ensemble
from model import occlusion_attribution
from trust import cross_assay_attribution_activity_correlation
from xai import load_best_checkpoint

SUBGROUPS = ["k562_native", "hepg2_designed", "wtc11_designed"]


def element_attribution_and_pred(best_model, ensemble, label_means, label_stds,
                                  tb, chrom, window_start, window_end,
                                  bin_start, bin_end, device):
    """One element's full pipeline: sequence -> one-hot -> occlusion
    attribution on the attribution checkpoint (element_attribution, mean
    |attribution| over the element's own bins, matching trust.py's
    docstring) -> ensemble-mean prediction on the same window (model_pred,
    de-standardized per-model then averaged, same convention final_eval.py's
    predict_ensemble uses)."""
    seq = tb.sequence(str(chrom), int(window_start), int(window_end))
    x = torch.from_numpy(one_hot_encode(seq))

    attr = np.asarray(occlusion_attribution(
        best_model, x, window_stride=BIN_BP, window_width=BIN_BP, device=device))
    elem_attr = float(np.abs(attr[bin_start:bin_end + 1]).mean())

    with torch.no_grad():
        xb = x.to(device).unsqueeze(0)
        preds = [m(xb).item() * ls + lm for m, lm, ls in zip(ensemble, label_means, label_stds)]
    model_pred = float(np.mean(preds))
    return elem_attr, model_pred


def ols_with_se(X, y):
    """Plain OLS via lstsq, with analytic standard errors / t-stats / p-values
    -- no new dependency (statsmodels) needed for a 2-predictor + intercept
    regression. Returns (beta, se, t, p, r_squared)."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float((resid @ resid) / dof) if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return beta, se, t, p, r2


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def cross_assay_block(attribution, activity, model_pred, subgroup_name, top_k_percent):
    """Per-subgroup validation block: raw Spearman (trust.py's function,
    unmodified), partial Spearman controlling for model_pred, and an OLS
    decomposition -- all restricted to the same top-k%-by-|activity|
    subset."""
    n = len(activity)
    n_top = max(2, int(np.ceil(n * top_k_percent / 100.0)))
    top_idx = np.argsort(np.abs(activity))[-n_top:]

    rho_raw = cross_assay_attribution_activity_correlation(attribution, activity, top_k_percent)

    block = {
        "subgroup": subgroup_name,
        "n_total": int(n),
        "n_top": int(len(top_idx)),
        "top_k_percent": top_k_percent,
        "rho_raw": rho_raw,
    }
    if len(top_idx) < 4:
        block["note"] = "too few elements in this subgroup's top-k% for a partial correlation / OLS"
        return block

    attr_t, act_t, pred_t = attribution[top_idx], activity[top_idx], model_pred[top_idx]

    r_attr, r_act, r_pred = (stats.rankdata(v) for v in (attr_t, act_t, pred_t))
    r_ay = float(np.corrcoef(r_attr, r_act)[0, 1])
    r_am = float(np.corrcoef(r_attr, r_pred)[0, 1])
    r_my = float(np.corrcoef(r_pred, r_act)[0, 1])
    denom = np.sqrt((1 - r_am ** 2) * (1 - r_my ** 2))
    rho_partial = float((r_ay - r_am * r_my) / denom) if denom > 1e-8 else None

    X = np.column_stack([np.ones(len(top_idx)), zscore(pred_t), zscore(attr_t)])
    beta, se, t, p, r2 = ols_with_se(X, act_t.astype(np.float64))

    block["rho_partial_controlling_model_pred"] = rho_partial
    block["ols_activity_on_model_pred_and_attribution"] = {
        "intercept": float(beta[0]),
        "model_pred_coef": float(beta[1]), "model_pred_p": float(p[1]),
        "attribution_coef": float(beta[2]), "attribution_p": float(p[2]),
        "r_squared": float(r2),
    }
    rho_raw_s = f"{rho_raw:+.4f}" if rho_raw is not None else "None"
    rho_partial_s = f"{rho_partial:+.4f}" if rho_partial is not None else "None"
    print(f"  {subgroup_name:<16} n={n:>6} n_top={len(top_idx):>5}  "
          f"rho_raw {rho_raw_s}  rho_partial {rho_partial_s}  "
          f"attr_coef {beta[2]:+.4f} (p={p[2]:.3g})  R2 {r2:.4f}")
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements-npz", type=Path, default=MPRA_ELEMENTS_NPZ)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--out", type=Path, default=MPRA_EVAL_RESULTS_JSON)
    ap.add_argument("--pred-out", type=Path, default=MPRA_EVAL_PREDICTIONS_NPZ)
    ap.add_argument("--top-k-percent", type=float, default=PERTURBATION_AGREEMENT_TOP_K_PERCENT)
    ap.add_argument("--max-elements", type=int, default=None,
                     help="subsample for a quick local smoke test; omit for the real full run")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("preflight: checking required inputs...")
    for p in (args.elements_npz, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run build_mpra_features.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading ensemble (for model_pred)...")
    ensemble, label_means, label_stds = load_ensemble(args.checkpoint_dir, ENSEMBLE_SIZE, device)

    print("loading attribution checkpoint (highest val_spearman, same selection as xai.py)...")
    best_model, best_label_mean, best_label_std = load_best_checkpoint(
        args.checkpoint_dir, ENSEMBLE_SIZE, device)

    d = np.load(args.elements_npz, allow_pickle=True)
    n_elements = len(d["chrom"])
    idx = np.arange(n_elements)
    if args.max_elements is not None and args.max_elements < n_elements:
        idx = rng.choice(idx, size=args.max_elements, replace=False)
        idx.sort()
    print(f"  {len(idx)}/{n_elements} elements to score")

    tb = py2bit.open(str(GENOME_2BIT))
    print("\ncomputing per-element attribution + model_pred (occlusion attribution, the slow step)...")
    elem_attr = np.empty(len(idx), dtype=np.float64)
    model_pred = np.empty(len(idx), dtype=np.float64)
    for j, i in enumerate(idx):
        elem_attr[j], model_pred[j] = element_attribution_and_pred(
            best_model, ensemble, label_means, label_stds, tb,
            d["chrom"][i], d["window_start"][i], d["window_end"][i],
            int(d["bin_start"][i]), int(d["bin_end"][i]), device)
        if (j + 1) % 5000 == 0:
            print(f"  {j + 1}/{len(idx)} elements done")
    tb.close()

    activity = d["activity"][idx].astype(np.float64)
    subgroup = d["subgroup"][idx]

    print("\n=== CROSS-ASSAY ATTRIBUTION-ACTIVITY CORRELATION (external validation, not an inference axis) ===")
    results = {
        "attribution_checkpoint_label_mean": best_label_mean,
        "attribution_checkpoint_label_std": best_label_std,
        "n_elements_scored": int(len(idx)),
        "top_k_percent": args.top_k_percent,
        "subgroups": {},
    }
    for sg in SUBGROUPS:
        mask = subgroup == sg
        if mask.sum() == 0:
            continue
        results["subgroups"][sg] = cross_assay_block(
            elem_attr[mask], activity[mask], model_pred[mask], sg, args.top_k_percent)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {args.out}")

    np.savez(
        args.pred_out,
        element_id=d["element_id"][idx], subgroup=subgroup,
        attribution=elem_attr, model_pred=model_pred, activity=activity,
    )
    print(f"Saved -> {args.pred_out}")


if __name__ == "__main__":
    main()
