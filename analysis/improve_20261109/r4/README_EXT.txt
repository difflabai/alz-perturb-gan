improve_20261109 round 4 -- EXTERNAL DONORS (GSE268599 MADRC ITG; GSE264648 MTG for an offset check only)
=====================================================================================================
Binding rules: r4/PROTOCOL_EXT.md (+ PROTOCOL.md, r2/PROTOCOL_R2.md, r3/PROTOCOL_R3.md).  External donors are
TRAINING donors only; every score is on SEA-AD held-out inner-fold donors of hvg2000 fold k.  Only the 37 SEA-AD
development donors are ever read.  This file: state after the 2026-11-11 adversarial review and repair.

Python: /mnt/nvme/ai/conda/anaconda3/bin/python3 -B  (scipy 1.16, h5py 3.15, sklearn 1.4.2, pandas, openpyxl;
anndata/scanpy/rpy2 and R are NOT installed -- GSE264648 .rds files are read by the pure-Python RDS reader in
ext_fetch.py).  At most 3 CPU threads per process.  All commands below run from
/ml2/alz/analysis/improve_20261109/r4 unless they say "cd /tmp".


1. FETCH AND SELECT (external store)
------------------------------------
  python3 -B ext_fetch.py meta            # GEO metadata, paper tables, ADNC (NIA-AA 2012) -> ext/meta/
  python3 -B ext_fetch.py terms-eutils    # public-status check via NCBI E-utilities + FTP -> ext/meta/access_terms.json
  python3 -B ext_fetch.py libs            # per ITG library: download, select microglia, write, verify, delete raw
                                          #   (donors whose receipts verify are skipped; to rerun the selection move
                                          #    counts/ cellmeta/ receipts/ donors.json away first, keep counts/genes*.npy)
  python3 -B ext_select_microglia.py validate   # needs the transient valsample/ written by `libs`; delete it after
  python3 -B ext_fetch.py gse264648 --write
  python3 -B test_ext_store.py            # 5 tests
Store: /ml2/alz/runs/improve_20261109/ext/GSE268599/ (counts/{donor}.npz CSR int32 cells x 36,601, same gene axis and
order as SEA-AD rebuilt; {donor}_names.npy; cellmeta/; receipts/; donors.json; validation.json), 270 MB.
Selection version mg-select-v3 (31 donors, 54,015 nuclei; Not AD 7 / 22,619, Low 5 / 11,126, Intermediate 6 / 5,043,
High 13 / 15,227).  v3 = v2 plus an ambient-aware cluster rule (review D1, see caveats).  28 of 31 donors are
byte-identical to v2; MADRC-04 378 -> 413, MADRC-08 663 -> 813, MADRC-23 404 -> 492.

Access terms: both series are public GEO series (E-utilities esummary: GSE268599 pdat 2024/07/22, 146 samples, MTX/TSV,
PRJNA916657; GSE264648 pdat 2024/04/26, 49 samples, RDS, PRJNA1103739), supplementary files on anonymous FTP, no licence
field, GSE268599 samples 'consent: public'.  The GEO disclaimer page wording was NOT re-verified (www.ncbi.nlm.nih.gov
shows scripts a reCAPTCHA); someone with a browser should read https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html.
ext/meta/serrano_pozo2024_MOESM3.xlsx is under the article's CC BY-NC-ND 4.0 licence: keep it INTERNAL.


2. BUILD THE CACHES
-------------------
  cd /tmp && OMP_NUM_THREADS=3 python3 -B /ml2/alz/analysis/improve_20261109/r4/build_cache_ext.py --fold all --overwrite
  cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_cache_ext.py all
caches/hvg2000x/fold{0,1,2} v2 (28 MB): genes, space.npz, type_classifier.pt, heldout/* byte-identical to hvg2000;
SEA-AD train rows identical; then the 31 MADRC donors, AT MOST 1,000 CELLS EACH (metrics.det_subsample with
stable_seed('ext-cap', donor), the evaluate_ext rule; 27,632 of 54,015 cells dropped by the cap; MADRC is now about 70%
of train cells instead of 82%).  External raw_c: slot 0 Braak_index/6 and slot 2 CERAD_index/3 (published), slot 1
Thal IMPUTED as mid(thal_range_from_A)/5 (Thal is unpublished), slot 3 ADNC/3, 4 male, 5 APOE4 dose, 6 age with '90+'
recoded per fold as the mean age of that fold's SEA-AD TRAIN donors aged >= 90 (95.0 / 94.15 / 94.06).
No condition reads raw_c slots 0-2.


3. TRAIN (screen)
-----------------
Code: trainer.py (R4-1..R4-6), generate.py (GR4-1/2), arm_cohort.py (cohort, cohortfix, extperm; mode asmadrc),
arm_pbulk.py, mode_biascal.py, arm_condshrink.py, run_screen.sh (-k).  R4-6 (review F1/D6): with a two-cohort cache the
generator initialisation mu/theta, the age standardisation and the pbulk critic normalisation use the SEA-AD train
donors only and are bitwise equal to the hvg2000 values, so a hvg2000x fit differs from the hvg2000 control only by
the extra donors it trains on.
Tests:
  cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_arm_cohort.py parity   # r4 == round 1 / r3 bitwise
  cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_arm_cohort.py unit
  cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_arm_cohort.py cli
  cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_arm_cohort.py gpu cuda:N   # 300-update smoke
  cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_arm_cohort.py cleanup
  (the copied r3 tests test_core.py etc. still point at smoke_r3: do NOT run their subcommands)
Launch (18 fits, about 18 GPU units; job file documents arms, modes, reading and disk):
  cd /tmp && nohup /ml2/alz/analysis/improve_20261109/r4/run_screen.sh -d cuda:0,cuda:1 -p P -t 3 -g 0.04 -k \
      -j /ml2/alz/runs/improve_20261109/jobs_r4.txt > /ml2/alz/runs/improve_20261109/logs_r4/screen_r4.out 2>&1 &
  -k is REQUIRED for the 3 GB kept-disk limit (prunes snap_*.pt after every mode of a line succeeded; about 133 MB
  per fit kept, about 2.5 GB for the screen including asmadrc).  Keep -g 0.04 (300-update smoke on cache v2:
  0.46-0.51 s/update for every hvg2000x arm, peak CUDA 0.87 GB of the 1.97 GB cap; smoke_r4/test_arm_cohort_gpu.json).
Arms: pbulk (naive pooled), cohortfix,pbulk (critic sees cohort; generator has a fixed train-estimated shrunk per-gene
cohort logit offset), cohortfix,extperm,pbulk (CONTROL: identical except external donors show a fixed permuted
external stage).  The learned-path arm cohort,pbulk was dropped for disk and because the offset is large, additive and
ambient-dominated; it remains implemented.


4. EVALUATE
-----------
Baselines (done 2026-11-11; about 40 min per fold, one process per fold):
  python3 -B evaluate_ext.py run --folds K --out /ml2/alz/runs/improve_20261109/results_r4      # K = 0, 1, 2
  python3 -B evaluate_ext.py reproduce --folds 0,1,2     # SEA-AD-only baselines == results_r3 (21/21 identical, 1e-8)
  python3 -B evaluate_ext.py aggregate
Fits (after the screen):
  python3 -B evaluate_ext.py run --folds K --no-baselines --fits-root r4=/ml2/alz/runs/improve_20261109/fits_r4 \
      --out /ml2/alz/runs/improve_20261109/results_r4                                                 # K = 0, 1, 2
  python3 -B evaluate_ext.py aggregate
  Outputs: results_r4/aggregate_ext.json (comparison_vs_hvg2000, comparison_within_x, missing_references),
  summary_ext.txt, comparison_ext.csv.  GATE: read comparison_vs_hvg2000 only when missing_references is empty, i.e.
  when the round-3 control fits_r3/pbulk/hvg2000/fold{0,1,2}/seed{0,1} are all complete (on 2026-11-11 only
  fold0/seed0 existed, at update 17,501).  PRE-REGISTERED: a stage gain of cohortfix,pbulk is credited to MADRC stage
  information only if it beats cohortfix,extperm,pbulk (comparison_within_x 'vs_extperm_control') by more than 2
  seed-noise SE.  Read the SEA-AD TRAIN-target fit and asmadrc next to it (review F9).
Offset / replication analysis (about 48 min under load; --part3-only about 1 min):
  python3 -B ext_offset.py --n-perm 2000 ; python3 -B test_ext_offset.py
  -> /ml2/alz/runs/improve_20261109/ext/analysis_offset/results.json, offsets.npz
  python3 -B arm_cohort.py measure     # cohortfix offset on cache v2 train donors -> smoke_r4/cohort_offset_measure.json


5. SELECTION AND LABEL CAVEATS
------------------------------
* No author per-cell annotation is open; microglia/PVM are selected by a documented marker pipeline (QC, k-means
  clusters, cluster rule, simulated heterotypic doublets, cell rules).  PVMs are kept, as in SEA-AD's Microglia-PVM.
* v3 cluster rule (review D1): a cluster with median mg >= 1.0, no T/B, mg/pvm >= 0.5 above every NON-astrocyte
  lineage and an astrocyte median < 1.0 is myeloid even if ambient astrocyte RNA breaks the strict margin.  It rescued
  4 clusters (899 nuclei) in MADRC-04/-08/-23; after the cell rules and doublet simulation 273 of them were kept,
  most of the rest were rejected as heterotypic doublets, i.e. these clusters were partly astrocyte-microglia doublets.
* Library quality and yield fall with stage (review D2; ext_offset results.json madrc_qc_stage_trend, Spearman with
  ADNC over 31 donors, 20,000-label permutation p): barcodes rho -0.63 (p 0.0003), QC pass fraction -0.55 (p 0.002),
  kept cells -0.43 (p 0.017), median mito of kept cells +0.61 (p 0.0005).  Selection terms show no significant trend:
  kept fraction of myeloid-cluster nuclei -0.17 (p 0.36), other-lineage-dominates rejection +0.24 (p 0.19; medians
  2.3/3.5/6.7/5.7%), heterotypic doublets +0.15 (p 0.42).  Hence the per-donor cap in the cache.
* Ambient astrocyte RNA is in every MADRC library and its COMPOSITION changes with stage (review D3): SLC1A2 detection
  rho -0.41 (p 0.024; 0.51/0.55/0.42/0.35 by stage), AQP4 -0.36 (p 0.050); none in SEA-AD (AQP4 0.00, SLC1A2 -0.18,
  p 0.29).  An additive cohort offset cannot remove this.  Replication is therefore also reported without the astrocyte
  ambient genes (they do not drive it, see below).  No ambient correction is possible (raw droplets were deleted).
* MADRC 'Not AD' = Braak 0-II, CERAD none, all APOE e3/e3; SEA-AD Not AD = Braak 0,4,4,5: the High-minus-Not-AD
  contrasts are not the same contrast.  APOE4 dose and PVM fraction both rise with stage in both cohorts.
* '90+' ages: 9 of the 32 MADRC donors (all 9 have ITG libraries: MADRC-01, 04, 05, 10, 13, 18, 19, 21, 26).
  The fetch report's "12 of 32" was wrong (review D5/F7).
* Thal is unpublished for MADRC (A score only); raw_c slot 1 is imputed and must not be used as data.
* GSE264648 ages/diagnosis only (no ADNC); used solely for the offset-direction check.


6. CONCLUSIONS THAT SURVIVED REVIEW (numbers from ext_offset.py results.json and evaluate_ext.py aggregate, v3/v2)
--------------------------------------------------------------------------------------------------------------------
Cohort offset (MADRC ITG minus SEA-AD MTG).  Comparable headline: noise-corrected RMS 0.30 of log1p CP10k on the
hvg2000 fold-0 panel (0.297 / 0.299 / 0.309 folds 0-2) = 0.67 of the within-cohort-and-stage donor SD and 2.2x the
noise-corrected SEA-AD High-minus-Not-AD contrast; 656 of 2,000 genes at FDR 5%; 105 genes carry half of it; top genes
astrocyte (GFAP, AQP4, ATP1A2, ALDH1L1, PTGDS) up and neuronal down.  On 6,320 neutral genes 0.086 (0.55 SD).  The
other numbers quoted earlier are ESTIMATOR-SPECIFIC and not comparable (review F8): 0.56-0.59 (cache v2; raw 0.62-0.65)
= arm_cohort shrunk log((C+1)/(N+G)) OLS coefficient used by cohortfix (split-half 0.82-0.83); 0.69-0.72 = evaluate_ext log(P+1e-6) clipped at +-3; neither is noise-corrected.
Depth: binomial thinning cannot test depth for pseudobulk proportions (review F3; it only checks the log1p
nonlinearity: 0.299 vs 0.297).  The depth evidence is a within-cohort regression: the depth gap predicts an offset of
RMS 0.08-0.09 on hvg2000 (0.03 neutral), correlated -0.38 (SEA-AD slope) and +0.35 (MADRC slope) with the observed
offset, i.e. inconsistent; depth is a minor part.  A cell-level ambient filter removes about 14%.  GSE264648's offset
against SEA-AD correlates 0.29 with MADRC's.

Cross-cohort stage replication (High minus Not AD, MADRC vs all 37 SEA-AD donors, Spearman over genes; p one-sided,
MADRC-label / SEA-AD-label permutation, 2,000 draws).  HEADLINE (review F4) = adjusted for age, sex, APOE4, depth and
PVM fraction: hvg2000 folds 0/1/2 0.27 / 0.26 / 0.25 (p 0.037/0.009, 0.032/0.011, 0.043/0.008); neutral panel 0.39
(p 0.095 / 0.008).  Raw (not adjusted): 0.34 / 0.37 / 0.34 (p 0.011-0.025 / 0.006-0.015).  Adjusted without PVM:
0.23-0.25 (p 0.05-0.08 / 0.03-0.04).  Braak slope (adjusted + PVM) 0.22-0.26 (p 0.02-0.04 / 0.01-0.04).  Without the astrocyte
ambient genes the values are unchanged (raw 0.34-0.37; adjusted+PVM 0.25-0.27); without astrocyte, neuronal and the 20%
highest-offset genes they are slightly higher (adjusted+PVM 0.27-0.30, p 0.016-0.029 / 0.002-0.006).  So the
agreement is weak but above both nulls and is not carried by ambient genes.  Split-half reliability on hvg2000 is only
SEA-AD 0.19-0.21 and MADRC 0.30-0.31 (Spearman-Brown), so this is near the noise ceiling.  Per fold (folds 0-8 = 3 repeats over the same
37 donors; SE within a repeat, review F6): MADRC vs fold-train SEA-AD 0.29 +- 0.04; against the fold's held-out
contrast (descriptive) SEA-AD train 0.12 +- 0.05, MADRC 0.21 +- 0.05, combined 0.21 +- 0.06.

Do external donors help BASELINES on SEA-AD held-out donors (evaluate_ext, hvg2000 folds 0-2, pb_rmse)?
  Raw pooling hurts: grand_mean 0.4639 -> 0.4885, ridge 0.4606 -> 0.4756, shrunk 0.4684 -> 0.4891 (0/3 folds lower).
  With a train-fitted cohort offset or a cohort column there is a small gain: ridge_cohort+ext 0.4546 (-0.0061, 3/3),
  ridge+extoff 0.4553 (-0.0053, 3/3), stage_mean_log+extoff 0.4753 (-0.0182, 3/3); the best (0.4546) is 0.009 below
  grand_mean, inside the oracle-floor estimate 0.452 +- 0.014.  Stage metrics: xfit stays near 0 (best 0.041, p 0.08);
  pooled held-out contrast agreement (hoc) rises from about 0.10-0.12 to 0.17-0.20 (p 0.02-0.04, about 10 correlated
  variants, suggestive only).  ext_offset part 3 (direct predictions, folds 0-8, MADRC '90+' age recoded per fold,
  review F5): shrunk+offset 0.4637 vs train mean 0.4630; covariates-only ridge 0.4534 -> 0.4478 with MADRC + offset
  (-0.0056 +- 0.0019, 9/9 folds; unchanged by the age fix) -- the gain is from sex/APOE4/age, not stage.

RECOMMENDATION: use the external donors for VALIDATION of stage direction, not as a route to a lower SEA-AD pseudobulk
error.  Training with them is justified only as the pre-registered screen above (cohortfix vs its extperm control and
the hvg2000 control); never pool raw.  Expect at best a small stage-direction gain and no pb_rmse gain from stage.
