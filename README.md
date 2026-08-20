# RegTrust-XAI

A trust-aware framework for DNA sequence-to-function prediction in regulatory genomics. A five-model convolutional ensemble predicts K562 chromatin accessibility from DNA sequence, paired with three inference-time reliability signals (ensemble consensus, motif-grounded attribution coherence, and applicability-domain distance) and a five-rung graded distribution-shift evaluation (chromosome holdout, sequence-composition divergence, dinucleotide-shuffle stress test, cross-assay lentiMPRA, and single-nucleotide saturation-mutagenesis perturbation at the PKLR promoter).

Manuscript: *Trust-Aware Sequence-to-Function Modelling in Regulatory Genomics.* (in preparation)

## Data sources

This repository does not include raw or processed data (see `.gitignore`). All data are public and can be re-obtained from their original sources:

- **Training / internal-test data**: ENCODE ATAC-seq, K562, experiment [ENCSR868FGK](https://www.encodeproject.org/experiments/ENCSR868FGK/) (fold-change bigWig `ENCFF019IPA`, replicated peaks `ENCFF057UYP`), assembly GRCh38.
- **Reference genome**: UCSC `hg38.2bit`, [UCSC goldenPath](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/).
- **Cross-assay validation data**: lentiMPRA, K562/HepG2/WTC11 joint library, ENCODE experiment [ENCSR203UFY](https://www.encodeproject.org/experiments/ENCSR203UFY/) (`ENCFF802FUV`, `ENCFF068BWG`). Source paper: Agarwal V, et al. (2025). Massively parallel characterization of transcriptional regulatory elements. *Nature*. https://doi.org/10.1038/s41586-024-08430-9
- **Perturbation-validation data**: saturation-mutagenesis MPRA, PKLR promoter, K562. Source paper: Kircher M, Xiong C, Martin B, Schubach M, Inoue F, Bell RJA, Costello JF, Shendure J, Ahituv N (2019). Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution. *Nature Communications*. https://doi.org/10.1038/s41467-019-11526-w. Processed per-variant tables sourced from the paper's OSF repository, https://doi.org/10.17605/OSF.IO/75B2M.

To reproduce, download these sources into `data/raw/` following the paths referenced in `config.py`, then run the pipeline below.

## Pipeline

Scripts run in this order (see each script's own docstring/header for CLI options):

1. `build_features.py` — construct 2,048 bp training/test windows from ATAC-seq peaks and matched negatives.
2. `build_mpra_features.py` — construct lentiMPRA element windows.
3. `hyper.py` — Optuna hyperparameter search (architecture + optimizer).
4. `train_cv.py` — five-fold chromosome-grouped ensemble training.
5. `final_eval.py` — internal-test (chromosome holdout) evaluation.
6. `xai.py` — occlusion attribution, motif-shell coherence, and the four-scenario trust taxonomy.
7. `mpra_eval.py` — cross-assay attribution-activity correlation against lentiMPRA.
8. `validate_trust_axes.py` — applicability-domain and coherence validation against held-out error.
9. `sequence_composition.py` / `rung1a_composition_divergence.py` — composition-divergence severity axis.
10. `dinuc_shuffle.py` / `rung1b_motif_shuffle.py` — dinucleotide-shuffle stress test.
11. `rung4_pklr_perturbation.py` — PKLR saturation-mutagenesis perturbation-agreement test.
12. `sensitivity_thresholds.py` — consensus/coherence calibration-percentile robustness sweep (reuses `xai.py`'s calibration sample and `trust.trust_block()` unchanged). Applicability-domain reference-pool-size robustness reuses `validate_trust_axes.py --ref-pool-size <N>` directly, no separate script.
13. `xai_full_dataset.py` — extends the trust taxonomy to the full 517,790-window dataset (training pool + held-out test set), design-facing only, reusing the calibrated cutoffs from `xai_results.json` unchanged; includes a holdout-only cross-check against `xai_results.json`'s own published numbers.
14. `repeat_blindspot_analysis.py` — tests whether repeat-derived sequence content (UCSC hg38 RepeatMasker) identifies a representational blind spot in the trust taxonomy, via a confound-controlled logistic regression of Scenario A/D membership on repeat status, applicability-domain distance, and GC content, run on both the chr8/chr9 holdout and the full dataset. Requires a local RepeatMasker download (`--rmsk-path`, not included in this repository).

Supporting modules: `config.py` (paths, hyperparameters, constants), `data_module.py` (sequence loading), `model.py` (CNN architecture + occlusion attribution), `motif_shell.py` (JASPAR K562 motif panel), `trust.py` (consensus/coherence/applicability-domain formulas), `splits.py` (chromosome-grouped cross-validation).

## Results

Summary statistics from each pipeline stage are tracked under `results/*.json`. Per-window prediction arrays (`results/*.npz`) are regenerable from the tracked checkpoints and data but are not stored in this repository.

`results/sensitivity_thresholds.json` and `results/sensitivity_trust_validation_refpool{2500,10000}.json` are the manuscript's own robustness checks against its calibrated cutoffs: sweeping the consensus/coherence calibration percentile, and sweeping the applicability-domain reference-pool size across a 4x range. Both leave the A/A+B-vs-C+D enrichment pattern qualitatively unchanged from the baseline reported in the main results.

`results/xai_full_dataset_results.json` is the full-dataset (design-facing) trust taxonomy run, cross-checked against `results/xai_results.json`'s held-out numbers. `results/repeat_blindspot_results.json` is the repeat-content blind-spot test: repeat-derived windows show *lower* odds of the lowest-trust scenario and *higher* odds of the highest-trust scenario after adjustment for applicability-domain distance and GC content (chr8/chr9 holdout: Scenario D OR 0.78, 95% CI 0.747-0.812; Scenario A OR 1.39, 95% CI 1.298-1.488) — no evidence of a repeat-based blind spot under this operational definition.

## Supplementary causal-occlusion analyses

A side investigation testing whether biologically exact, motif-span occlusion changes the coherence axis's behaviour relative to the fixed 32 bp bins used by `xai.py`. Motivated by a parallel to context-dependent substituent effects in the author's own antimicrobial-XAI attribution work. This does not replace `xai.py`'s bin-overlap coherence definition; it is kept as a separate, design-facing robustness and mechanistic check.

15. `motif_causal_occlusion.py` / `motif_causal_occlusion_full_dataset.py` — causal, motif-span-exact occlusion coherence, and a joint-versus-summed motif-knockout superadditivity test, on the held-out test set and the full 517,790-window dataset respectively.
16. `matched_coherence_comparison.py` — checks whether causal coherence's larger A-versus-B error separation survives when the two coherence definitions' Scenario A populations are matched by size rather than independently calibrated.
17. `ood_causal_coherence_analysis.py` / `ood_causal_coherence_full_dataset_analysis.py` — tests whether either coherence definition (bin-overlap or causal) still discriminates error outside the model's applicability domain.
18. `causal_coherence_tail_diagnosis.py` / `causal_coherence_tail_diagnosis_full_dataset.py` — diagnoses the non-monotonic (U-shaped) relationship between causal-coherence deciles and error.
19. `causal_coherence_confound_controlled.py` / `causal_coherence_confound_controlled_full_dataset.py` / `causal_coherence_confound_controlled_gc_repeat.py` / `causal_coherence_confound_controlled_klf1.py` — sequential confound-control tests of the causal-coherence top-decile residual: target magnitude, applicability-domain distance, and motif-instance count (3-control); adding GC content and repeat-derived content (5-control); adding KLF1 instance density (6-control).
20. `causal_coherence_top_decile_tf_breakdown.py` — per-panel-transcription-factor breakdown of the top-decile residual; identifies KLF1 as a striking but ultimately non-explanatory outlier.

Headline findings, all design-facing (full 517,790-window dataset unless noted) and cross-checked against the held-out chr8/chr9 subset:

- Causal (motif-span-exact) occlusion coherence does not clearly replace bin-overlap coherence as the operational trust axis after confound control — its raw advantage over bin-overlap is mostly explained by shared correlation with target magnitude, applicability-domain distance, and motif-instance count.
- 90.4% of 4,703,837 multi-instance motif modules genome-wide show superadditive joint-knockout effects (91.1% on the held-out test set), indicating that a motif's attributed contribution depends strongly on neighbouring regulatory sequence context rather than being an intrinsic property of the motif alone.
- Both coherence definitions' error discrimination collapses outside the applicability domain, partly reflecting that out-of-domain windows are disproportionately low-consensus (Scenario C/D) to begin with, not only that coherence itself stops carrying information.
- The causal-coherence top-decile U-shape remains only partially explained after six candidate confounds (partial rho settles around +0.05, p≈1e-12, genome-wide); this residual is reported as an open finding, not attributed to any single cause.

Results for these analyses are tracked under `results/motif_causal_occlusion*_results.json`, `results/matched_coherence_comparison_results.json`, `results/ood_causal_coherence*_results.json`, `results/causal_coherence_tail_diagnosis*_results.json`, `results/causal_coherence_confound_controlled*_results.json`, and `results/causal_coherence_top_decile_tf_breakdown_results.json`.
