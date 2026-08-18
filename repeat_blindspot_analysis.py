"""Tests whether repeat-derived sequence content identifies a representational
blind spot in the trust taxonomy, analogous to ProtTrust-XAI's cofactor-chemistry
finding. Requires results/xai_full_dataset_predictions.npz (xai_full_dataset.py)
and data/raw/hg38.2bit already present. Downloads the UCSC hg38 RepeatMasker
track (rmsk.txt.gz, ~150MB) to a local path if not already present there.

Usage:
    python repeat_blindspot_analysis.py --rmsk-path /path/to/rmsk_hg38.txt.gz

If the RepeatMasker file is not yet downloaded, fetch it first:
    curl -o rmsk_hg38.txt.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz

Outputs:
    results/repeat_blindspot_predictions.npz  (per-window repeat_frac, GC, N-frac)
    results/repeat_blindspot_results.json     (controlled logistic regression, holdout + full-dataset)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import py2bit
import statsmodels.formula.api as smf

import config


def compute_repeat_overlap(chrom, start, end, rmsk_path):
    cols = [5, 6, 7, 11]
    names = ["chrom", "start", "end", "repClass"]
    rmsk = pd.read_csv(rmsk_path, sep="\t", header=None, usecols=cols, names=names,
                        compression="infer")
    valid_chroms = set(np.unique(chrom).tolist())
    rmsk = rmsk[rmsk["chrom"].isin(valid_chroms)]

    merged = {}
    for c, sub in rmsk.groupby("chrom"):
        arr = sub[["start", "end"]].to_numpy()
        arr = arr[np.argsort(arr[:, 0])]
        out = []
        cs, ce = arr[0]
        for s, e in arr[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                out.append((cs, ce))
                cs, ce = s, e
        out.append((cs, ce))
        merged[c] = np.array(out)

    def overlap_bp(win_s, win_e, ivals):
        starts = ivals[:, 0]
        ends = ivals[:, 1]
        lo = np.searchsorted(ends, win_s, side="right")
        hi = np.searchsorted(starts, win_e, side="left")
        if lo >= hi:
            return 0
        seg_s = np.maximum(starts[lo:hi], win_s)
        seg_e = np.minimum(ends[lo:hi], win_e)
        return int(np.sum(np.maximum(seg_e - seg_s, 0)))

    n = len(chrom)
    repeat_bp = np.zeros(n, dtype=np.int64)
    by_chrom_idx = {c: np.where(chrom == c)[0] for c in valid_chroms}
    for c, idxs in by_chrom_idx.items():
        ivals = merged.get(c)
        if ivals is None or len(ivals) == 0:
            continue
        for i in idxs:
            repeat_bp[i] = overlap_bp(start[i], end[i], ivals)

    window_len = (end - start).astype(np.int64)
    return repeat_bp, repeat_bp / np.maximum(window_len, 1)


def compute_gc_content(chrom, start, end, twobit_path):
    tb = py2bit.open(twobit_path)
    n = len(chrom)
    gc = np.full(n, np.nan, dtype=np.float64)
    n_frac = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        seq = tb.sequence(str(chrom[i]), int(start[i]), int(end[i]))
        if not seq:
            continue
        seq_u = seq.upper()
        L = len(seq_u)
        gc[i] = (seq_u.count("G") + seq_u.count("C")) / L
        n_frac[i] = seq_u.count("N") / L
    tb.close()
    return gc, n_frac


def run_regression(df, dv, has_holdout_col):
    formula = f"{dv} ~ repeat_derived + ad_distance_z + gc"
    if has_holdout_col:
        formula += " + is_holdout"
    m = smf.logit(formula, data=df).fit(disp=0)
    conf = m.conf_int()
    conf["OR"] = np.exp(m.params)
    conf["OR_lo"] = np.exp(conf[0])
    conf["OR_hi"] = np.exp(conf[1])
    return {
        row: {"OR": conf.loc[row, "OR"], "OR_lo": conf.loc[row, "OR_lo"], "OR_hi": conf.loc[row, "OR_hi"]}
        for row in conf.index
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmsk-path", required=True)
    ap.add_argument("--twobit-path", default="data/raw/hg38.2bit")
    ap.add_argument("--xai-full-npz", default="results/xai_full_dataset_predictions.npz")
    ap.add_argument("--out-npz", default="results/repeat_blindspot_predictions.npz")
    ap.add_argument("--out-json", default="results/repeat_blindspot_results.json")
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.xai_full_npz, allow_pickle=True)
    chrom = d["chrom"]; start = d["start"]; end = d["end"]
    ad_distance = d["ad_distance"]; is_holdout = d["is_holdout"]
    scenario_valid_mask = d["scenario_valid_mask"]
    scenario_vals = d["scenario"]

    print("computing repeat overlap...")
    repeat_bp, repeat_frac = compute_repeat_overlap(chrom, start, end, args.rmsk_path)
    print("elapsed", time.time() - t0)

    print("computing GC content...")
    gc, n_frac = compute_gc_content(chrom, start, end, args.twobit_path)
    print("elapsed", time.time() - t0)

    np.savez(args.out_npz, chrom=chrom, start=start, end=end,
             repeat_bp=repeat_bp, repeat_frac=repeat_frac, gc=gc, n_frac=n_frac)

    n = len(chrom)
    scenario_full = np.full(n, "", dtype="<U1")
    scenario_full[scenario_valid_mask] = scenario_vals

    mask = scenario_valid_mask & (n_frac < 0.5) & ~np.isnan(gc)
    ad_sd = ad_distance[mask].std()

    df = pd.DataFrame({
        "scenario": scenario_full[mask],
        "repeat_derived": (repeat_frac[mask] > 0.5).astype(int),
        "ad_distance_z": (ad_distance[mask] - ad_distance[mask].mean()) / ad_sd,
        "gc": gc[mask],
        "is_holdout": is_holdout[mask].astype(int),
    })
    df["scenario_D"] = (df["scenario"] == "D").astype(int)
    df["scenario_A"] = (df["scenario"] == "A").astype(int)

    results = {"repeat_derived_threshold": 0.5, "ad_distance_sd": float(ad_sd), "n_analyzable": int(mask.sum())}

    for name, sub in [("full_dataset", df), ("holdout_only", df[df.is_holdout == 1]),
                       ("train_pool_only", df[df.is_holdout == 0])]:
        has_ho = sub["is_holdout"].nunique() > 1
        results[name] = {
            "n": int(len(sub)),
            "scenario_D_base_rate": float(sub["scenario_D"].mean()),
            "scenario_A_base_rate": float(sub["scenario_A"].mean()),
            "scenario_D_regression": run_regression(sub, "scenario_D", has_ho),
            "scenario_A_regression": run_regression(sub, "scenario_A", has_ho),
        }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("done, elapsed", time.time() - t0)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
