#!/usr/bin/env python
"""Map lentiMPRA K562 elements onto element-centred accessibility windows, for
the cross-assay validation criterion (trust.cross_assay_attribution_activity_
correlation) -- see notes/data_sources.md for the source data and
trust.py's RENAME/RESTRUCTURE note for why this is a validation cohort, not
part of the deployed inference-time trust taxonomy.

WHAT THIS PRODUCES, PER ELEMENT (not per occlusion bin). Each of the 53,989
lentiMPRA elements is an independent 200bp construct tested in isolation, so
the natural unit here is one element, not one bin of some larger shared
window (see trust.py's parameter-naming fix for why that distinction
matters). For each element that survives filtering:
  - a WINDOW_BP-wide window CENTRED on the element (same construction
    build_features.positive_windows uses for ATAC peaks, reused directly --
    the model needs a fixed-width input regardless of which cohort it is
    scoring), dropped if it would run off a chromosome end;
  - bin_start/bin_end: which of that window's occlusion bins (config.BIN_BP
    wide, same convention as model.occlusion_attribution) the element's own
    span covers -- this is what a future attribution script uses to reduce
    a window's full per-bin attribution array down to the single
    element_attribution scalar trust.py's function actually compares
    against measured activity;
  - the element's own measured log2 activity (column 7 of the bed file);
  - its subgroup: K562-native (in-distribution) vs HepG2-/WTC11-designed
    (tested in K562 but not native to it -- a free, already-collected shift
    stress test, see notes/data_sources.md).

THIS SCRIPT DOES NOT RUN ANY MODEL. No trained checkpoint exists yet (see
project_status.md). It only prepares the coordinate/label table; computing
element_attribution from a trained ensemble is a later step.

RUN AS A BATCH JOB or locally: CPU-only, reads a 2bit genome only to check
chromosome bounds (no sequence extraction needed here, unlike
build_features.py, since sequence is read on demand at train/eval time).
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import py2bit

from config import (
    BIN_BP,
    GENOME_2BIT,
    MPRA_ACTIVITY_BED,
    MPRA_ELEMENT_BP,
    MPRA_ELEMENTS_NPZ,
    PRIMARY_CHROMS,
    WINDOW_BP,
)

N_BINS = WINDOW_BP // BIN_BP


def classify_subgroup(name: str) -> str:
    """K562_peak* is in-distribution (native to the assay's own cell type);
    HepG2_DNasePeakNoPromoter*/WTC11_seq*_F are candidates DESIGNED from the
    other two cell types' accessible regions but tested in this K562 assay
    -- a free, already-collected distribution-shift stress test (see
    notes/data_sources.md), kept as its own reported subgroup rather than
    pooled with the native elements."""
    if name.startswith("K562_"):
        return "k562_native"
    if name.startswith("HepG2_"):
        return "hepg2_designed"
    if name.startswith("WTC11_"):
        return "wtc11_designed"
    return "other"


def load_mpra_elements(path=MPRA_ACTIVITY_BED, chroms=PRIMARY_CHROMS):
    """ENCFF802FUV.bed, confirmed by direct inspection (notes/data_sources.md):
    chrom start end name score strand log2_activity col9 col10 col11
    (no header, tab-separated). Columns 9-11 are NOT used -- their meaning is
    not yet confirmed against a schema document, see notes/data_sources.md
    section 5."""
    chroms = set(chroms)
    elements = []
    dropped_off_primary = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            chrom, start, end, name = parts[0], int(parts[1]), int(parts[2]), parts[3]
            activity = float(parts[6])
            if chrom not in chroms:
                dropped_off_primary += 1
                continue
            elements.append((chrom, start, end, name, activity))
    return elements, dropped_off_primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=MPRA_ELEMENTS_NPZ)
    args = ap.parse_args()

    print("preflight: checking required inputs...")
    for p in (MPRA_ACTIVITY_BED, GENOME_2BIT):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; see notes/data_sources.md.")

    print(f"loading MPRA elements from {MPRA_ACTIVITY_BED} ...")
    elements, dropped_off_primary = load_mpra_elements()
    print(f"  {len(elements)} elements on primary chromosomes "
          f"({dropped_off_primary} dropped: alt/random contigs or chrY)")

    widths = [e - s for _, s, e, _, _ in elements]
    off_nominal = sum(1 for w in widths if w != MPRA_ELEMENT_BP)
    print(f"  element width: {MPRA_ELEMENT_BP}bp nominal, {off_nominal}/{len(elements)} "
          f"rows differ (handled from their own coordinates, not assumed constant)")

    subgroup_counts: dict[str, int] = {}
    for _, _, _, name, _ in elements:
        sg = classify_subgroup(name)
        subgroup_counts[sg] = subgroup_counts.get(sg, 0) + 1
    print(f"  subgroup breakdown: {subgroup_counts}")

    tb = py2bit.open(str(GENOME_2BIT))
    chrom_sizes = {c: s for c, s in tb.chroms().items() if c in PRIMARY_CHROMS}
    tb.close()

    # ---- centre a WINDOW_BP window on each element's own midpoint ----------
    # Same clipping rule build_features.positive_windows uses for ATAC peaks
    # (drop rather than clip if the window would run off a chromosome end,
    # since clipping would silently shrink the window and break the fixed
    # WINDOW_BP shape the model requires); done inline here rather than by
    # calling that function, because this loop also needs to compute each
    # element's bin_start/bin_end in the same pass, which positive_windows
    # has no reason to know about.
    kept = []
    half = WINDOW_BP // 2
    for chrom, e_start, e_end, name, activity in elements:
        summit = (e_start + e_end) // 2
        w_start = summit - half
        w_end = w_start + WINDOW_BP
        size = chrom_sizes.get(chrom)
        if size is None or w_start < 0 or w_end > size:
            continue
        bin_start = max(0, min(N_BINS - 1, (e_start - w_start) // BIN_BP))
        bin_end = max(0, min(N_BINS - 1, (e_end - w_start - 1) // BIN_BP))
        kept.append({
            "chrom": chrom, "window_start": w_start, "window_end": w_end,
            "elem_start": e_start, "elem_end": e_end, "bin_start": bin_start,
            "bin_end": bin_end, "activity": activity, "name": name,
            "subgroup": classify_subgroup(name),
        })
    n_dropped_edge = len(elements) - len(kept)
    if n_dropped_edge:
        print(f"  {n_dropped_edge} elements dropped: centred window would run off a chromosome end")

    bin_spans = [k["bin_end"] - k["bin_start"] + 1 for k in kept]
    print(f"  bin span per element: mean {np.mean(bin_spans):.1f}, "
          f"min {min(bin_spans)}, max {max(bin_spans)} (of {N_BINS} bins/window)")

    out = {
        "chrom": np.array([k["chrom"] for k in kept], dtype=object),
        "window_start": np.array([k["window_start"] for k in kept], dtype=np.int64),
        "window_end": np.array([k["window_end"] for k in kept], dtype=np.int64),
        "elem_start": np.array([k["elem_start"] for k in kept], dtype=np.int64),
        "elem_end": np.array([k["elem_end"] for k in kept], dtype=np.int64),
        "bin_start": np.array([k["bin_start"] for k in kept], dtype=np.int32),
        "bin_end": np.array([k["bin_end"] for k in kept], dtype=np.int32),
        "activity": np.array([k["activity"] for k in kept], dtype=np.float32),
        "name": np.array([k["name"] for k in kept], dtype=object),
        "subgroup": np.array([k["subgroup"] for k in kept], dtype=object),
        "element_id": np.arange(len(kept), dtype=np.int64),
        "window_bp": WINDOW_BP,
        "bin_bp": BIN_BP,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    kept_subgroups: dict[str, int] = {}
    for k in kept:
        kept_subgroups[k["subgroup"]] = kept_subgroups.get(k["subgroup"], 0) + 1
    print(f"\nSaved -> {args.out} ({len(kept)} elements; subgroup breakdown after all "
          f"drops: {kept_subgroups})")


if __name__ == "__main__":
    main()
