"""
Central configuration for RegTrust-XAI phase 1: a trust-aware seq2func model
for chromatin accessibility (DNA sequence -> ATAC-seq signal), built on the
same three-axis reliability construction as CancerTrust-XAI and ProtTrust-XAI
(ensemble consensus, attribution coherence, applicability domain). See
project_status.md for the full design rationale and, critically, for what is
NOT yet verified below.

Do not "improve" the trust-axis constants (CV_FOLDS, ENSEMBLE_SIZE,
TRUST_CONSENSUS_PERCENTILE, TRUST_AGREEMENT_PERCENTILE, AD_*_PERCENTILE)
without re-deriving the justification; they are ported unchanged from
CancerTrust-XAI/ProtTrust-XAI, not re-tuned for this domain, following the
"port the quantity, not the instrument, but keep the calibration procedure"
rule in ../TRUST_AWARE_PREDICTION_GUIDE.md.
"""
from pathlib import Path

# --------------------------------------------------------------------------------
# task definition — LOCKED 2026-08-11, see project_status.md "Task" section
# --------------------------------------------------------------------------------
CELL_TYPE = "K562"
ASSAY = "ATAC-seq"
ENCODE_ACCESSION = "ENCSR868FGK"          # K562 ATAC-seq, released 2020-08-05. VERIFIED 2026-08-11
                                            # via ENCODE REST API and downloaded; see notes/data_sources.md.

# External stress test: lentiMPRA, K562 arm of the joint HepG2/K562/WTC11
# library from Agarwal et al. 2025 Nature (DOI 10.1038/s41586-024-08430-9).
# CORRECTED 2026-08-11: the original scoping guess (Zenodo 10558183, GEO
# GSE272169) was wrong — Zenodo 10558183 holds only code/weights/SI tables,
# and GSE272169 is an unrelated series (eczema/mast-cell scRNA-seq). The real
# per-element activity data is deposited directly on ENCODE as a
# functional-characterization-experiment, found by cross-referencing GEO
# series titles against ENCODE accessions. See notes/data_sources.md for the
# full correction and how it was found.
MPRA_SOURCE_DOI = "10.1038/s41586-024-08430-9"
MPRA_ENCODE_ACCESSION = "ENCSR203UFY"     # "MPRA from K562" — VERIFIED, downloaded 2026-08-11

# --------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
RESULTS = Path("results")

ACCESSIBILITY_SIGNAL_BW = DATA_RAW / "encode_k562_atac_fc_ENCFF019IPA.bigWig"                  # 1.6GB, md5-verified
ACCESSIBILITY_PEAKS_BED = DATA_RAW / "encode_k562_atac_replicated_peaks_ENCFF057UYP.bed.gz"    # 258,903 peaks
MPRA_ACTIVITY_BED = DATA_RAW / "lentimpra_k562_ENCFF802FUV.bed.gz"                             # 53,989 elements, has coordinates
MPRA_ACTIVITY_TSV = DATA_RAW / "lentimpra_k562_element_quant_ENCFF068BWG.tsv"                  # 170,927 rows, name-keyed, provenance only
GENOME_2BIT = DATA_RAW / "hg38.2bit"    # UCSC goldenPath hg38.2bit, 835,393,456 bytes, downloaded and
                                          # read-verified 2026-08-11 (py2bit opens it, chrom sizes match
                                          # known GRCh38 lengths, e.g. chr1=248,956,422bp; a test read at
                                          # chr1:10410-10450 returned the expected telomeric repeat).
JASPAR_PFM_PATH = DATA_RAW / "JASPAR2026_CORE_vertebrates_non-redundant_pfms.txt"   # 1,019 motifs,
                                          # verified 2026-08-11, all 10 K562_TF_PANEL entries resolve
                                          # (see motif_shell.py; was hardcoded as a string literal in
                                          # motif_shell.py's own __main__ until this constant was added)

# --------------------------------------------------------------------------------
# window construction (build_features.py)
# --------------------------------------------------------------------------------
PRIMARY_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]   # autosomes + chrX; chrY and all
                                                                   # alt/random contigs excluded, and
                                                                   # K562 is a female-derived line so
                                                                   # chrY has no meaningful signal here.
NEG_TO_POS_RATIO = 1.0   # negative (non-peak) windows sampled per positive (peak-centered) window

SEQUENCE_WINDOWS_NPZ = DATA_PROCESSED / "phase1_sequence_windows.npz"
MPRA_ELEMENTS_NPZ = DATA_PROCESSED / "mpra_elements.npz"   # build_mpra_features.py output

# --------------------------------------------------------------------------------
# sequence representation and model
# --------------------------------------------------------------------------------
WINDOW_BP = 2048           # input window width; ChromBPNet-scale, not Enformer-scale (~200kb) —
                            # deliberately small so this is buildable and trainable on a single GPU,
                            # per the "scoped v1, not the review's outlook" decision in project_status.md
BIN_BP = 32                # output resolution within the window, if predicting a profile rather than
                            # a single scalar; start with a single pooled scalar per window for phase 1
ALPHABET = "ACGT"
N_CHANNELS = 4              # one-hot

CONV_CHANNELS = [64, 96, 128]   # small stack; depth is a HPO target the same way ProtTrust-XAI
                                  # searched num_rgcn_layers rather than fixing it a priori
CONV_KERNEL = 15
POOL_EVERY = 2
DROPOUT = 0.1

# --------------------------------------------------------------------------------
# training (train_cv.py) — FALLBACK DEFAULTS, used only if best_hyperparameters.json
# (hyper.py's output) does not exist yet. Once hyper.py has run, train_cv.py's
# load_hparams() reads the searched lr/weight_decay/batch_size/channels/kernel/
# dropout from that file instead — see hyper.py's own docstring for why this was
# added 2026-08-11 (matching ProtTrust-XAI's build -> hyper -> cv -> final_eval ->
# xai pipeline shape, which this project had initially skipped the hyper step of).
# --------------------------------------------------------------------------------
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5   # epochs without validation-Spearman improvement before stopping
NUM_WORKERS = 4
CHECKPOINT_DIR = Path("checkpoints")
CV_SUMMARY_JSON = RESULTS / "cv_summary.json"
BEST_HYPERPARAMETERS_JSON = Path("best_hyperparameters.json")

# --------------------------------------------------------------------------------
# hyperparameter search (hyper.py) — added 2026-08-11. Searches BOTH the optimizer
# (lr/weight_decay/batch_size, matching ProtTrust-XAI's scope) AND the architecture
# (channel profile/kernel/dropout, matching ProtTrust-XAI's num_layers/node_dim
# search) in one Optuna study, since this model's architecture (CONV_CHANNELS/
# CONV_KERNEL) was never independently justified the way ProtTrust-XAI's RGCN
# depth was benchmarked against GearNet precedent — see model.py's own comment
# that CONV_CHANNELS depth is "a HPO target the same way ProtTrust-XAI searched
# num_rgcn_layers". Runs on ONE chromosome-grouped fold (not the full CV_FOLDS),
# a reduced epoch budget, and a training-pool subsample — a ranking signal across
# trials, not a converged accuracy estimate; train_cv.py's real CV run is what
# produces the reported ensemble.
# --------------------------------------------------------------------------------
HYPER_LR_RANGE = (1e-5, 3e-3)
HYPER_WEIGHT_DECAY_RANGE = (1e-6, 1e-2)
HYPER_BATCH_SIZES = [32, 64, 128]
HYPER_CONV_CHANNEL_PROFILES = [
    "48,64,96", "64,96,128", "64,128,192", "64,96,128,192",
]   # comma-separated strings, not lists -- Optuna's suggest_categorical needs
     # hashable choices, and this doubles as the exact string stored in
     # best_hyperparameters.json for readability
HYPER_KERNELS = [9, 15, 21]
HYPER_DROPOUT_RANGE = (0.0, 0.3)
HYPER_EPOCHS = 8            # reduced per-trial budget (ranking signal, not full convergence)
HYPER_N_TRIALS = 30
HYPER_MAX_TRAIN_WINDOWS = 40000   # subsample of the fold's training windows per trial,
                                     # for tractability -- full-scale pool is ~475k windows

# --------------------------------------------------------------------------------
# internal-test evaluation (final_eval.py)
# --------------------------------------------------------------------------------
EVAL_RESULTS_JSON = RESULTS / "eval_results.json"
EVAL_PREDICTIONS_NPZ = RESULTS / "eval_predictions.npz"   # per-window ensemble mean/std, for xai.py
EVAL_POOL_SAMPLE_SIZE = 5000   # windows sampled from the training pool to estimate pop_std
                                 # (consensus-axis normalization); the full pool is unnecessary for a
                                 # spread estimate and would cost real GPU time for no added precision
EVAL_N_BOOT = 2000

# --------------------------------------------------------------------------------
# trust taxonomy labelling (xai.py)
# --------------------------------------------------------------------------------
XAI_RESULTS_JSON = RESULTS / "xai_results.json"
XAI_PREDICTIONS_NPZ = RESULTS / "xai_predictions.npz"   # per-window scenario labels + both trust axes
XAI_ERROR_THRESHOLDS = [0.1, 0.2, 0.3]   # accessibility-signal-unit error thresholds for enrichment;
                                            # NOT re-derived from label std yet -- reasonable starting
                                            # points given pool label std ~0.86-1.1 in every toy run so
                                            # far, revisit once real full-scale label statistics exist

# --------------------------------------------------------------------------------
# splits — locus/chromosome holdout, NOT random position
# --------------------------------------------------------------------------------
GLOBAL_SEED = 1
SPLIT_SEED = GLOBAL_SEED
CV_FOLDS = 5
ENSEMBLE_SIZE = 5
HOLDOUT_CHROMS = ["chr8", "chr9"]   # internal test; a random split leaks compositionally similar
                                       # sequence across the boundary (Nagai et al. 2026), so splitting
                                       # is at the chromosome level, the DNA analogue of ProtTrust-XAI's
                                       # PIDE20 protein-cluster split and CancerTrust-XAI's lineage split.

# --------------------------------------------------------------------------------
# trust axes: consensus + coherence, percentile cutoffs calibrated on the pool
# --------------------------------------------------------------------------------
TRUST_CONSENSUS_PERCENTILE = 30.0
TRUST_AGREEMENT_PERCENTILE = 70.0
TRUST_CALIB_MAX_SAMPLES = 2000

# --------------------------------------------------------------------------------
# applicability domain (third axis)
# --------------------------------------------------------------------------------
AD_EMBED_PERCENTILE = 95.0
AD_COMBINE = "any_far_is_ood"
AD_REF_POOL_SIZE = 5000   # training-pool reference sample for the nearest-neighbor distance,
                            # same size as EVAL_POOL_SAMPLE_SIZE -- only a single forward pass per
                            # window (no occlusion), so this is cheap relative to xai.py's coherence pass

# --------------------------------------------------------------------------------
# item 8: validating coherence/AD against error (validate_trust_axes.py), not yet
# reported as meaningful until this runs -- see project_status.md NEXT UP item 8
# --------------------------------------------------------------------------------
TRUST_VALIDATION_RESULTS_JSON = RESULTS / "trust_validation_results.json"
TRUST_VALIDATION_PREDICTIONS_NPZ = RESULTS / "trust_validation_predictions.npz"   # per-window
                                                 # AD distance + error, for the quantile stratification
AD_QUANTILE_EDGES = [0.0, 20.0, 40.0, 60.0, 80.0, 95.0, 100.0]   # percentile band edges for AD
                                                 # risk stratification; top band (95-100) matches
                                                 # AD_EMBED_PERCENTILE's own OOD cutoff exactly

# --------------------------------------------------------------------------------
# item 9 rung 1a: composition-divergence stratification (sequence_composition.py) --
# a severity axis independent of the model's own embedding, so it doesn't use AD
# to validate AD (see rung1a_composition_divergence.py's own module docstring)
# --------------------------------------------------------------------------------
COMPOSITION_KMER = 4                     # tetranucleotide frequency; 4**4=256 bins, enough
                                            # resolution without most bins being near-empty at
                                            # WINDOW_BP=2048 (~2045 draws per window)
COMPOSITION_REF_POOL_SIZE = 5000         # matches AD_REF_POOL_SIZE for a comparable reference sample
RUNG1A_RESULTS_JSON = RESULTS / "rung1a_composition_divergence_results.json"
RUNG1A_PREDICTIONS_NPZ = RESULTS / "rung1a_composition_divergence_predictions.npz"

# --------------------------------------------------------------------------------
# item 9 rung 1b: motif-shuffled synthetic controls (dinuc_shuffle.py) -- no real
# labels exist for a synthetic sequence, so this only tests AD-distance and
# model-prediction shift under a review-Table-1-named covariate shift
# ("motif rearrangement"), never error
# --------------------------------------------------------------------------------
RUNG1B_SAMPLE_SIZE = 3000                # real test-set windows to shuffle and re-score;
                                            # smaller than the full 42,844 since this needs TWO
                                            # forward passes per window (original + shuffled) plus
                                            # sequence generation, not just one
RUNG1B_RESULTS_JSON = RESULTS / "rung1b_motif_shuffle_results.json"
RUNG1B_PREDICTIONS_NPZ = RESULTS / "rung1b_motif_shuffle_predictions.npz"

# --------------------------------------------------------------------------------
# item 9 rung 4: true single-base perturbation (Kircher et al. 2019 saturation-
# mutagenesis MPRA, PKLR promoter, K562, GRCh38 -- sourced 2026-08-12, see
# "Data collection" in project_status.md). Position column is 1-based (VCF-style),
# verified 2026-08-12 against hg38.2bit (200/200 real-genome matches at
# Position-1 in 0-based py2bit coordinates, 0/200 matches at the naive 0-based
# reading) -- rung4_pklr_perturbation.py must convert, never read Position
# directly into py2bit.
# --------------------------------------------------------------------------------
KIRCHER_PKLR_24H_TSV = DATA_RAW / "kircher_satmut" / "GRCh38_PKLR-24h.tsv"   # primary timepoint
KIRCHER_PKLR_48H_TSV = DATA_RAW / "kircher_satmut" / "GRCh38_PKLR-48h.tsv"  # replication check
RUNG4_RESULTS_JSON = RESULTS / "rung4_pklr_perturbation_results.json"
RUNG4_PREDICTIONS_NPZ = RESULTS / "rung4_pklr_perturbation_predictions.npz"

# --------------------------------------------------------------------------------
# cross-assay attribution-activity correlation (external validation criterion,
# NOT an inference-time axis, see trust.py's RENAME/RESTRUCTURE note) —
# computed against the lentiMPRA external cohort only (build_mpra_features.py),
# never against the internal chromosome-holdout test, since it needs measured
# lentiMPRA activity, which only exists for that cohort.
# --------------------------------------------------------------------------------
PERTURBATION_AGREEMENT_TOP_K_PERCENT = 10.0   # top-|activity| MPRA elements compared against
                                                 # their own mean occlusion attribution
MPRA_ELEMENT_BP = 200   # nominal element width in data/raw/lentimpra_k562_ENCFF802FUV.bed;
                          # verified 2026-08-11 on the real file: 53,984/53,989 rows are exactly
                          # this width, the rest (201/202/210bp) are handled from their own
                          # coordinates, never assumed to be this constant.
MPRA_EVAL_RESULTS_JSON = RESULTS / "mpra_eval_results.json"   # mpra_eval.py output (item 7)
MPRA_EVAL_PREDICTIONS_NPZ = RESULTS / "mpra_eval_predictions.npz"   # per-element attribution/
                                                 # model_pred dump, for any downstream reuse
