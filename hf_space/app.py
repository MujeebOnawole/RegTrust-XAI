"""RegTrust -- public demo of RegTrust-XAI's trust-aware chromatin-
accessibility model. Paste a DNA sequence, get a predicted K562 ATAC-seq
accessibility score PLUS the model's own trust taxonomy (Scenario A-D:
ensemble consensus x attribution coherence with a real K562 TF-motif prior)
and applicability-domain flag, the same three-axis machinery the manuscript
validates against held-out error (see the About tab).

Scope, stated plainly because it matters: this model is trained on ONE cell
type, K562 (a CML-derived, BCR-ABL+ erythroleukemia line), chosen because it
is ENCODE's gold-standard reference for human gene-regulation data, not
because predictions here describe accessibility in an arbitrary tissue. A
prediction is "how accessible would this sequence be in K562 chromatin, and
how much should you trust that specific answer" -- not a general-purpose
accessibility oracle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import Seq2AccessibilityCNN, occlusion_attribution  # noqa: E402
from motif_shell import load_k562_pssms, motif_coverage_by_bin  # noqa: E402
from trust import (  # noqa: E402
    cv_ratio_from_stats,
    l2_normalize_rows,
    localization_coherence,
    nn_distance,
    scenario_labels,
)

ASSETS = Path(__file__).parent / "assets"
DEVICE = torch.device("cpu")
ALPHABET = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}

# dataviz skill: 4-tier ORDERED severity palette (status roles, not arbitrary
# categorical hues) -- A is the best-trust tier, D the worst, so this maps
# directly onto "good / warning / serious / critical", never color alone
# (icon + label ships with every badge). `color` is the accent (chips, plot);
# `banner` is a darker step of the same hue, chosen so WHITE text sits at a
# safe contrast ratio on a solid fill -- same technique the sibling SmeltTrust
# demo uses for its own trust badge (see hf_space's design-reference session).
SCENARIO_STYLE = {
    "A": {"color": "#0ca30c", "banner": "#0a7d0a", "icon": "✓", "label": "High trust",
          "desc": "High ensemble consensus AND attribution concentrates on a real K562 TF motif. "
                  "The model is confident, and its explanation is mechanistically grounded."},
    "B": {"color": "#fab219", "banner": "#b8790c", "icon": "●", "label": "Confident, unexplained",
          "desc": "High ensemble consensus, but attribution does not concentrate on a known motif. "
                  "The model is confident but its reasoning is not independently corroborated here."},
    "C": {"color": "#ec835a", "banner": "#c2410c", "icon": "▲", "label": "Explained, but inconsistent",
          "desc": "Attribution lands on a real motif, but the ensemble disagrees with itself. "
                  "Treat this prediction's magnitude cautiously even though the explanation looks plausible."},
    "D": {"color": "#d03b3b", "banner": "#b91c1c", "icon": "✕", "label": "Low trust",
          "desc": "Low ensemble consensus AND no motif-grounded explanation. This is exactly the "
                  "prediction population the manuscript's own enrichment analysis flags as least reliable."},
    None: {"color": "#898781", "banner": "#5b5a56", "icon": "?", "label": "Unresolved",
           "desc": "No K562-relevant TF motif was found anywhere in this sequence, so the coherence "
                   "axis (and therefore the A-D label) cannot be computed. Consensus and applicability "
                   "domain below are still valid."},
}

ATTR_COLOR = "#2563eb"        # genomics-palette blue -- attribution track bars
MOTIF_COLOR = "#0d9488"       # teal -- motif-hit bin highlight (paired with the hero teal/blue header)


# --------------------------------------------------------------------------------
# load everything once at startup
# --------------------------------------------------------------------------------
def load_ensemble():
    models, label_means, label_stds, val_spearmans = [], [], [], []
    ckpt_paths = sorted((ASSETS / "checkpoints").glob("fold_*.pt"))
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        arch = ckpt["arch"]
        m = Seq2AccessibilityCNN(channels=arch["channels"], kernel=arch["kernel"],
                                  dropout=arch["dropout"]).to(DEVICE)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        label_means.append(float(ckpt["label_mean"]))
        label_stds.append(float(ckpt["label_std"]))
        val_spearmans.append(float(ckpt["val_spearman"]))
    return models, label_means, label_stds, val_spearmans


print("loading ensemble checkpoints...")
ENSEMBLE, LABEL_MEANS, LABEL_STDS, VAL_SPEARMANS = load_ensemble()
ATTR_IDX = int(np.argmax(VAL_SPEARMANS))  # same "highest val_spearman" rule as xai.py
ATTR_MODEL = ENSEMBLE[ATTR_IDX]
print(f"  {len(ENSEMBLE)} models loaded; attribution model = fold index {ATTR_IDX} "
      f"(val_spearman {VAL_SPEARMANS[ATTR_IDX]:+.4f})")

print("loading calibration constants...")
CALIB = json.loads((ASSETS / "calibration.json").read_text())
WINDOW_BP = CALIB["window_bp"]
BIN_BP = CALIB["bin_bp"]

print("loading AD reference-embedding pool...")
REF_EMBED = np.load(ASSETS / "ref_embeddings.npy")

print("loading K562 TF motif panel (JASPAR)...")
JASPAR_PATH = next(ASSETS.glob("JASPAR*.txt"))
PSSMS = load_k562_pssms(JASPAR_PATH)
print(f"  {len(PSSMS)} panel TFs resolved")

EXAMPLES = json.loads((ASSETS / "examples.json").read_text())
print("Ready.")


# --------------------------------------------------------------------------------
# sequence handling
# --------------------------------------------------------------------------------
def clean_and_fit_sequence(raw: str):
    """Strip non-sequence characters, uppercase, and fit to exactly
    WINDOW_BP -- the length every trust-axis cutoff was calibrated at.
    Longer input is centre-cropped; shorter input is centre-padded with N
    (the all-zero one-hot column, "no information here"). Returns
    (sequence, warnings)."""
    warnings = []
    seq = "".join(ch for ch in raw.strip().upper() if not ch.isspace())
    if not seq:
        raise gr.Error("Please paste a DNA sequence.")

    bad = set(seq) - set("ACGTN")
    if bad:
        warnings.append(f"{len(bad)} non-ACGTN character type(s) found ({', '.join(sorted(bad))}) "
                         f"and treated as N (no information).")
        seq = "".join(c if c in "ACGTN" else "N" for c in seq)

    n = len(seq)
    if n > WINDOW_BP:
        start = (n - WINDOW_BP) // 2
        seq = seq[start:start + WINDOW_BP]
        warnings.append(f"Input is {n}bp; centre-cropped to the model's {WINDOW_BP}bp window "
                         f"(every trust cutoff below was calibrated at exactly this length).")
    elif n < WINDOW_BP:
        pad_total = WINDOW_BP - n
        pad_left = pad_total // 2
        seq = ("N" * pad_left) + seq + ("N" * (pad_total - pad_left))
        warnings.append(f"Input is {n}bp; centre-padded with N to the model's {WINDOW_BP}bp window.")
    return seq, warnings


def one_hot_encode(seq: str) -> np.ndarray:
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


# --------------------------------------------------------------------------------
# the analysis pipeline for one sequence
# --------------------------------------------------------------------------------
def analyze(raw_sequence: str):
    seq, warnings = clean_and_fit_sequence(raw_sequence)
    x = torch.from_numpy(one_hot_encode(seq))

    # 1. ensemble consensus
    preds = []
    for m, lm, ls in zip(ENSEMBLE, LABEL_MEANS, LABEL_STDS):
        with torch.no_grad():
            p = m(x.unsqueeze(0)).item()
        preds.append(p * ls + lm)
    preds = np.array(preds)
    pred_mean, pred_std = float(preds.mean()), float(preds.std())
    cv_ratio = cv_ratio_from_stats(pred_std, CALIB["pop_std"])
    consensus_pass = cv_ratio <= CALIB["consensus_cutoff"]

    # 2. coherence (occlusion attribution + K562 motif shell, attribution model only)
    attr = occlusion_attribution(ATTR_MODEL, x, window_stride=BIN_BP, window_width=BIN_BP, device=DEVICE)
    shell, hits = motif_coverage_by_bin(seq, PSSMS, bin_width=BIN_BP, bin_stride=BIN_BP)
    coherence = localization_coherence(attr, shell)
    coherence_pass = (coherence is not None) and (coherence >= CALIB["coherence_cutoff"])

    # 3. applicability domain
    with torch.no_grad():
        embed = ATTR_MODEL.embed(x.unsqueeze(0)).cpu().numpy()
    embed_n = l2_normalize_rows(embed)
    ad_dist = nn_distance(embed_n[0], REF_EMBED)
    ad_in_domain = ad_dist <= CALIB["ad_cutoff"]

    # 4. scenario
    if coherence is None:
        scenario = None
    else:
        scenario = str(scenario_labels(np.array([cv_ratio]), np.array([coherence]),
                                        CALIB["consensus_cutoff"], CALIB["coherence_cutoff"])[0])

    return {
        "seq": seq, "warnings": warnings,
        "pred_mean": pred_mean, "pred_std": pred_std,
        "cv_ratio": cv_ratio, "consensus_pass": consensus_pass,
        "coherence": coherence, "coherence_pass": coherence_pass,
        "ad_dist": ad_dist, "ad_in_domain": ad_in_domain,
        "scenario": scenario, "attr": attr, "hits": hits, "shell": shell,
    }


# --------------------------------------------------------------------------------
# rendering -- everything below renders as one HTML "results" block per run()
# call, built as stat-card grids rather than a plain table, so the result
# reads as a small dashboard, not a debug printout.
# --------------------------------------------------------------------------------
def render_badge(scenario):
    s = SCENARIO_STYLE[scenario]
    label = f"Scenario {scenario}" if scenario else "Scenario unresolved"
    return f"""
<div class="rt-badge-banner" style="background:{s['banner']};">
  <div class="rt-badge-icon">{s['icon']}</div>
  <div class="rt-badge-body">
    <div class="rt-badge-title">{label} &middot; {s['label']}</div>
    <div class="rt-badge-desc">{s['desc']}</div>
  </div>
</div>
"""


def render_hero_stat(res):
    return f"""
<div class="rt-card rt-hero-stat">
  <div class="rt-eyebrow">Predicted K562 accessibility</div>
  <div class="rt-hero-value">{res['pred_mean']:.3f}<span class="rt-hero-unit">&plusmn; {res['pred_std']:.3f}</span></div>
  <div class="rt-hero-caption">Ensemble mean &plusmn; std, raw ATAC-seq fold-change-over-control units
    (same scale as training labels) &mdash; higher means predicted more open/accessible in K562 chromatin.</div>
</div>
"""


def render_axis_card(icon, name, value_str, cutoff_str, passed, note):
    color = "#0ca30c" if passed else "#d03b3b"
    chip_icon = "✓" if passed else "✕"
    return f"""
<div class="rt-card rt-axis-card">
  <div class="rt-axis-top">
    <span class="rt-axis-icon">{icon}</span>
    <span class="rt-axis-name">{name}</span>
  </div>
  <div class="rt-axis-value">{value_str}</div>
  <div class="rt-axis-cutoff">{cutoff_str}</div>
  <div class="rt-chip" style="color:{color};border-color:{color};">{chip_icon} {note}</div>
</div>
"""


def render_axes_grid(res):
    cards = [render_axis_card(
        "⟳", "Consensus", f"{res['cv_ratio']:.3f}",
        f"cutoff &le; {CALIB['consensus_cutoff']:.3f}",
        res["consensus_pass"], "high consensus" if res["consensus_pass"] else "low consensus")]
    if res["coherence"] is None:
        cards.append(render_axis_card(
            "⌁", "Coherence", "&mdash;", "no motif shell resolved", False, "unresolved"))
    else:
        cards.append(render_axis_card(
            "⌁", "Coherence", f"{res['coherence']:.3f}",
            f"cutoff &ge; {CALIB['coherence_cutoff']:.3f}",
            res["coherence_pass"], "motif-grounded" if res["coherence_pass"] else "not motif-grounded"))
    cards.append(render_axis_card(
        "◎", "Applicability domain", f"{res['ad_dist']:.4f}",
        f"cutoff &le; {CALIB['ad_cutoff']:.4f}",
        res["ad_in_domain"], "in-domain" if res["ad_in_domain"] else "out-of-domain (OOD)"))
    return f'<div class="rt-axis-grid">{"".join(cards)}</div>'


def render_results(res):
    warn_html = ""
    if res["warnings"]:
        items = "".join(f"<li>{w}</li>" for w in res["warnings"])
        warn_html = f'<div class="rt-card rt-note"><strong>Input notes</strong><ul>{items}</ul></div>'
    return (
        render_badge(res["scenario"])
        + render_hero_stat(res)
        + render_axes_grid(res)
        + warn_html
    )


def render_attribution_plot(res):
    plt.rcParams["font.family"] = "sans-serif"
    attr = np.array(res["attr"])
    shell = set(res["shell"])
    n_bins = len(attr)
    x = np.arange(n_bins)
    colors = [MOTIF_COLOR if i in shell else ATTR_COLOR for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.bar(x, attr, width=1.0, color=colors, linewidth=0)
    ax.axhline(0, color="#898781", linewidth=0.8)
    ax.set_xlabel(f"position along the {WINDOW_BP}bp window ({BIN_BP}bp bins)", fontsize=11)
    ax.set_ylabel("occlusion attribution", fontsize=11)
    ax.set_title("Per-bin attribution  —  orange bins overlap a K562 TF motif hit",
                 fontsize=12, loc="left", pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9.5)
    fig.tight_layout()
    return fig


def render_motif_table(hits):
    if not hits:
        return [["(no motif hits above threshold)", "", "", ""]]
    return [[name, strand, start, end] for name, strand, start, end in
            sorted(hits, key=lambda h: h[2])]


def run(raw_sequence):
    res = analyze(raw_sequence)
    return (
        render_results(res),
        gr.update(visible=True),
        render_attribution_plot(res),
        render_motif_table(res["hits"]),
    )


# --------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --rt-radius: 14px;
  --rt-border: color-mix(in srgb, var(--body-text-color) 12%, transparent);
  --rt-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 6px 16px rgba(0,0,0,0.05);
}
.gradio-container { font-family: var(--font) !important; max-width: 1220px !important; font-size: 16px !important; }
.gradio-container label span, .gradio-container .prose p, .gradio-container .prose li { font-size: 1rem; }

/* ---- gradient hero header, same technique as the sibling SmeltTrust demo, ---- */
/* ---- re-hued teal/blue for a DNA/genomics identity instead of orange/slag --- */
.rt-hero-header {
  background: linear-gradient(100deg, #042f2e 0%, #0f5f5a 45%, #0d9488 100%);
  border-radius: 18px; padding: 30px 32px; margin-bottom: 20px;
  border: 1px solid #0d9488; box-shadow: 0 12px 40px rgba(13,148,136,0.25);
}
.rt-hero-header h1 {
  color: #ccfbf1 !important; font-size: 2.3rem !important; font-weight: 800 !important;
  letter-spacing: -0.02em; margin: 0 0 6px 0 !important;
  text-shadow: 0 0 24px rgba(45,212,191,0.45);
}
.rt-tagline { font-size: 1.05rem; color: #d1fae5; line-height: 1.5; margin: 0 0 16px 0; max-width: 640px; }
.rt-stat-row { display: flex; gap: 32px; flex-wrap: wrap; padding-top: 14px; border-top: 1px solid rgba(204,251,241,0.18); }
.rt-stat { text-align: left; }
.rt-stat-num { font-size: 1.35rem; font-weight: 800; color: #5eead4; line-height: 1.2; }
.rt-stat-lbl { font-size: 0.78rem; color: #a7f3d0; margin-top: 2px; }

/* ---- generic card -------------------------------------------------------- */
.rt-card {
  border: 1px solid var(--rt-border); border-radius: var(--rt-radius);
  background: var(--background-fill-primary); box-shadow: var(--rt-shadow);
  padding: 18px 20px; margin-bottom: 14px;
}
.rt-eyebrow {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--body-text-color-subdued); margin-bottom: 6px;
}
.rt-section-label {
  font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
  color: #0d9488; margin: 22px 0 10px 2px;
}
.rt-section-label:first-child { margin-top: 4px; }

/* ---- scenario badge: SOLID fill banner (never color alone -- icon+label ship with it) */
.rt-badge-banner {
  display: flex; gap: 16px; align-items: center; color: #fff;
  border-radius: var(--rt-radius); padding: 18px 20px; margin-bottom: 14px;
  box-shadow: var(--rt-shadow);
}
.rt-badge-icon {
  flex: none; width: 44px; height: 44px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem; font-weight: 800; color: inherit; background: rgba(255,255,255,0.18);
}
.rt-badge-title { font-size: 1.16rem; font-weight: 800; margin: 1px 0 4px 0; }
.rt-badge-desc { font-size: 0.92rem; line-height: 1.5; opacity: 0.95; }

/* ---- hero prediction number ---------------------------------------------- */
.rt-hero-value {
  font-size: 2.7rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;
  font-variant-numeric: tabular-nums; color: var(--body-text-color);
}
.rt-hero-unit { font-size: 1.15rem; font-weight: 600; color: var(--body-text-color-subdued); margin-left: 10px; }
.rt-hero-caption { font-size: 0.86rem; color: var(--body-text-color-subdued); margin-top: 8px; line-height: 1.5; }

/* ---- trust-axis card grid -------------------------------------------------- */
.rt-axis-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 14px; }
@media (max-width: 900px) { .rt-axis-grid { grid-template-columns: 1fr; } }
.rt-axis-card { margin-bottom: 0; }
.rt-axis-top { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.rt-axis-icon {
  width: 26px; height: 26px; border-radius: 7px; display: flex; align-items: center; justify-content: center;
  background: color-mix(in srgb, #0d9488 14%, transparent); font-size: 0.95rem;
}
.rt-axis-name { font-size: 0.86rem; font-weight: 700; color: var(--body-text-color); }
.rt-axis-value { font-size: 1.55rem; font-weight: 800; font-variant-numeric: tabular-nums; margin-bottom: 2px; }
.rt-axis-cutoff { font-size: 0.8rem; color: var(--body-text-color-subdued); margin-bottom: 12px; }
.rt-chip {
  display: inline-flex; font-size: 0.78rem; font-weight: 700; padding: 4px 10px;
  border-radius: 999px; border: 1.3px solid; width: fit-content;
}

/* ---- input notes ----------------------------------------------------------- */
.rt-note { border-left: 4px solid #eda100; border-radius: 8px; background: color-mix(in srgb, #eda100 8%, var(--background-fill-primary)); }
.rt-note ul { margin: 6px 0 0 18px; padding: 0; font-size: 0.86rem; color: var(--body-text-color-subdued); }

/* ---- input card, example pills, misc --------------------------------------- */
.rt-input-intro {
  background: color-mix(in srgb, #0d9488 7%, var(--background-fill-primary));
  border: 1px solid color-mix(in srgb, #0d9488 30%, var(--rt-border));
  border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; font-size: 0.88rem;
  color: var(--body-text-color-subdued); line-height: 1.5;
}
.reg-caption { color: var(--body-text-color-subdued); font-size: 0.86rem; line-height: 1.5; }
#seq-box textarea { font-family: var(--font-mono) !important; font-size: 0.92rem !important; letter-spacing: 0.02em; }
#analyze-btn { font-size: 1.02rem !important; height: 48px !important; font-weight: 700 !important; }
#analyze-btn.primary {
  background: linear-gradient(90deg, #0d9488 0%, #0891b2 100%) !important; border: none !important;
  box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
}
.rt-example-row .gr-button { font-size: 0.85rem !important; }
.rt-footer {
  background: var(--background-fill-secondary); border: 1px solid var(--rt-border); border-radius: 10px;
  padding: 14px; margin-top: 22px; text-align: center; color: var(--body-text-color-subdued); font-size: 0.8rem;
}

/* ---- About tab: taxonomy pill + scope callout ------------------------------- */
.rt-scenario-pill {
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; border-radius: 50%; color: #fff; font-weight: 800; font-size: 0.92rem;
}
.rt-scope-card { border-left: 4px solid #0d9488; }
"""

def _about_scenario_row(letter):
    s = SCENARIO_STYLE[letter]
    cons = "High" if letter in ("A", "B") else "Low"
    coh = "High" if letter in ("A", "C") else "Low"
    return f"""
<tr>
  <td style="padding:10px 12px;">
    <span class="rt-scenario-pill" style="background:{s['banner']};">{letter}</span>
  </td>
  <td style="padding:10px 12px;font-weight:600;">{cons}</td>
  <td style="padding:10px 12px;font-weight:600;">{coh}</td>
  <td style="padding:10px 12px;color:var(--body-text-color-subdued);">{s['desc']}</td>
</tr>"""


ABOUT_HTML = f"""
<div class="rt-card">
  <div class="rt-eyebrow">What this is</div>
  <p style="font-size:0.98rem;line-height:1.6;margin:0;">
    <b>RegTrust</b> predicts chromatin accessibility from a raw DNA sequence, in <b>K562</b>
    &mdash; a chronic-myeloid-leukemia-derived cell line and ENCODE's gold-standard reference
    for human gene-regulation data &mdash; and reports whether that specific prediction should
    be trusted, not just what the prediction is.
  </p>
</div>

<div class="rt-section-label">The trust taxonomy</div>
<div class="rt-card">
  <p style="font-size:0.95rem;line-height:1.6;margin:0 0 14px 0;">
    Every prediction gets a <b>Scenario A&ndash;D</b> label from two complementary axes:
    <b>consensus</b> (how much a 5-model ensemble agrees with itself) and <b>coherence</b>
    (whether the model's own attribution lands on a real, curated K562-relevant
    transcription-factor motif &mdash; GATA1, TAL1, KLF1, NFE2, GATA2, RUNX1, MYB, STAT5A/B,
    the erythroid/megakaryocytic/BCR-ABL-signalling program K562's own biology is organised
    around &mdash; not an arbitrary "attribution is concentrated somewhere" statistic).
  </p>
  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
    <thead><tr style="border-bottom:1.5px solid var(--rt-border);">
      <th style="text-align:left;padding:8px 12px;">Scenario</th>
      <th style="text-align:left;padding:8px 12px;">Consensus</th>
      <th style="text-align:left;padding:8px 12px;">Coherence</th>
      <th style="text-align:left;padding:8px 12px;">Meaning</th>
    </tr></thead>
    <tbody>{"".join(_about_scenario_row(l) for l in "ABCD")}</tbody>
  </table>
  </div>
</div>

<div class="rt-card">
  <div class="rt-eyebrow">A third, complementary axis</div>
  <p style="font-size:0.95rem;line-height:1.6;margin:0 0 10px 0;">
    <b>Applicability domain</b> flags whether the input sequence itself looks like anything the
    model was trained on (cosine distance to a 5,000-window training-pool reference in the
    model's own learned embedding space) &mdash; a sequence can get a well-formed A/B/C/D label
    and still be flagged out-of-domain.
  </p>
  <p style="font-size:0.9rem;color:var(--body-text-color-subdued);line-height:1.6;margin:0;">
    All three axes and their cutoffs are validated against real held-out prediction error in the
    underlying manuscript (internal test, n=42,844, chr8/chr9 holdout never seen in training):
    consensus+coherence enrichment separates low- from high-error predictions at every threshold
    tested, and applicability-domain distance correlates with error (&rho; +0.28, p&approx;0).
  </p>
</div>

<div class="rt-card rt-scope-card">
  <div class="rt-eyebrow" style="color:#0d9488;">Scope, stated plainly</div>
  <p style="font-size:0.95rem;line-height:1.6;margin:0;">
    This is a <b>K562-specific</b> model. A prediction here answers "how accessible would this
    sequence be in K562 chromatin, and how much should you trust that specific answer"
    &mdash; not a general cross-tissue accessibility oracle. K562 was chosen because it is the
    single most densely, consistently annotated human cell line in ENCODE, making it the
    natural testbed for validating a trust-aware framework before any claim about generalizing
    across cell types. <b>This tool is for research and methods demonstration; it is not a
    clinical or diagnostic tool.</b>
  </p>
</div>

<div class="rt-card">
  <div class="rt-eyebrow">Model</div>
  <p style="font-size:0.95rem;line-height:1.6;margin:0;">
    Compact 1D CNN (3 convolutional blocks, 2-layer prediction head) trained on ENCODE K562
    ATAC-seq (<code>ENCSR868FGK</code>, 517,790 windows, chromosome-holdout 5-fold ensemble).
    Internal test performance: Spearman &rho; = 0.782 (n=42,844).
  </p>
</div>

<div class="rt-card">
  <div class="rt-eyebrow">Code &amp; manuscript</div>
  <p style="font-size:0.95rem;line-height:1.6;margin:0 0 8px 0;">
    Pipeline code: <a href="https://github.com/MujeebOnawole/RegTrust-XAI" target="_blank"
      rel="noopener noreferrer" style="color:#0d9488;font-weight:600;">github.com/MujeebOnawole/RegTrust-XAI</a>
  </p>
  <p style="font-size:0.95rem;line-height:1.6;margin:0;">
    Manuscript: <i>&ldquo;Trust-Aware Sequence-to-Function Modelling in Regulatory Genomics&rdquo;</i>
  </p>
</div>
"""

THEME = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_text_size="15px",
    block_radius="14px",
    button_large_radius="10px",
)

def _short_label(e):
    if "Scenario A" in e["name"]:
        return "Real test window → A"
    if "Scenario D" in e["name"]:
        return "Real test window → D"
    return "PKLR promoter (K562)"


with gr.Blocks(title="RegTrust", css=CUSTOM_CSS, theme=THEME) as demo:
    gr.HTML(f"""
    <div class="rt-hero-header">
      <h1>🧬 RegTrust</h1>
      <p class="rt-tagline">Trust-aware chromatin-accessibility prediction for K562 &mdash;
      the ENCODE reference cell line for human gene regulation. Paste a DNA sequence and get
      a prediction PLUS a per-prediction reliability label, not just a number.</p>
      <div class="rt-stat-row">
        <div class="rt-stat"><div class="rt-stat-num">517,790</div><div class="rt-stat-lbl">training windows<br>(ENCODE K562 ATAC-seq)</div></div>
        <div class="rt-stat"><div class="rt-stat-num">&rho; 0.78</div><div class="rt-stat-lbl">held-out test<br>accuracy (n=42,844)</div></div>
        <div class="rt-stat"><div class="rt-stat-num">A&ndash;D</div><div class="rt-stat-lbl">per-prediction<br>trust label</div></div>
        <div class="rt-stat"><div class="rt-stat-num">3</div><div class="rt-stat-lbl">complementary trust axes<br>(consensus, coherence, AD)</div></div>
      </div>
    </div>""")

    with gr.Tabs():
        with gr.TabItem("Predict"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=340):
                    gr.HTML(
                        '<div class="rt-input-intro"><b>Paste a DNA sequence (ACGT)</b> and click '
                        '<b>Analyze sequence</b>. Any length works &mdash; it is automatically centred '
                        f'and fit to the model\'s {WINDOW_BP}bp window. Not sure what to try? '
                        'Press an example below.</div>'
                    )
                    seq_box = gr.Textbox(
                        label="DNA sequence", lines=7, max_lines=12,
                        elem_id="seq-box",
                        placeholder="e.g. ACGTACGTACGT... (paste a sequence, or pick an example below)",
                    )
                    run_btn = gr.Button("Analyze sequence", variant="primary", elem_id="analyze-btn")

                    gr.HTML('<div class="rt-section-label">Try an example</div>')
                    with gr.Row(elem_classes="rt-example-row"):
                        example_btns = [gr.Button(_short_label(e), size="sm") for e in EXAMPLES]
                    for e in EXAMPLES:
                        gr.Markdown(f"**{_short_label(e)}** &mdash; {e['description']}", elem_classes="reg-caption")

                with gr.Column(scale=6, min_width=420):
                    gr.HTML('<div class="rt-section-label">2 &middot; Trust-aware prediction</div>')
                    results_html = gr.HTML(
                        '<div class="rt-card reg-caption">Results appear here after you click '
                        '&ldquo;Analyze sequence.&rdquo;</div>'
                    )
                    with gr.Column(visible=False) as detail_group:
                        gr.HTML('<div class="rt-section-label">Attribution</div>')
                        attr_plot = gr.Plot(label=None, show_label=False, container=False)
                        gr.HTML('<div class="rt-section-label">Motif hits</div>')
                        motif_table = gr.Dataframe(
                            headers=["TF", "strand", "start (bp)", "end (bp)"],
                            label=None,
                        )

            run_outputs = [results_html, detail_group, attr_plot, motif_table]
            run_btn.click(run, inputs=[seq_box], outputs=run_outputs)
            for b, e in zip(example_btns, EXAMPLES):
                b.click(lambda seq=e["sequence"]: seq, outputs=[seq_box]).then(
                    run, inputs=[seq_box], outputs=run_outputs)

        with gr.TabItem("About the method"):
            gr.HTML(ABOUT_HTML)

    gr.HTML(
        '<div class="rt-footer">RegTrust &nbsp;|&nbsp; RegTrust-XAI, developed by '
        'Abdulmujeeb T. Onawole, The University of Queensland &nbsp;|&nbsp; '
        'research demo of a trust-aware prediction framework &mdash; not a clinical '
        'or diagnostic tool.</div>'
    )

if __name__ == "__main__":
    demo.launch()
