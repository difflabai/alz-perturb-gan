# PCA shrinkage versus generator under-dispersion: protocol

Declared 2026-11-20 (UTC 03:55) before any score in this study exists. CPU
analysis only. No training, no generation by the network, no weight edits, no
change to the default config, the coverage pilot, or the running 17-fit
confirmation queue (whose files and processes this study does not touch).

## Question

Native EMPF fold0/seed0 has held-out SD coverage of 58.3% (target 80%), with
most misses *above* the generated range (real donor SD larger than generated).
The PC coordinates come from a PCA basis fit on the 25 TRAIN donors' cells.
Held-out cells projected onto it lose variance on the high PCs (pilot
diagnosis: real within-donor PC SD on PCs 31-50 is about .995 for TRAIN and
.757 for held-out donors). How much of the SD under-coverage is a property of
the generator, and how much is the scoring basis?

Prior knowledge (not evidence from this study): the review's per-block
recomputation reported baseline held-out SD coverage of 24.6% on PCs 1-10 and
75.0% on PCs 31-50. The 2026-11-14 empirical-challenge stage-blind resampling
score for fold0 replicate 0 is in the committed audit.

## Fixed inputs (held-out development donors of fold 0 only, n = 12)

* Cache `runs/improve_20261109/caches/hvg2000/fold0` (manifest sha256
  `65d5e2f4...`), loaded with the frozen `r5/evaluate.load_real`.
* G0, the selected default: `runs/microglia_empf/verification/final_generated.npz`
  (`cellshape@0.5:biascal:stagehalf`, sha256 `04b8d82c...`).
* G1, the quarter readout: `runs/coverage_empf_20261117/quarter_generated.npz`
  (sha256 `241ce3ef...`).
* E_train, stage-blind empirical resampling: frozen `r5/baselines.generate
  ('resample_blind')` with the empirical-challenge replicate-0 stream
  (`stable_seed('empirical-challenge', 20261114, panel, fold, donor, 0)`). The
  run must reproduce the committed `empirical32/fold0/seed0.json` held-out SD
  and combined interval numbers exactly, or the study stops.
* Two diagnostic resampling references built with the identical depth-qualified
  code path and RNG, differing only in the source donor pool:
  * E_train11: for each target, 11 TRAIN donors (a fixed per-target random
    subset), so pool size matches the next reference;
  * E_heldpool ("oracle"): for each held-out target, the other 11 held-out
    donors. These cells are out-of-sample for the TRAIN basis in the same way
    as the target. **It uses held-out cells and is not a legitimate generator**;
    it estimates what a stage-blind population sampler with no in-sample
    projection inflation would score.
* Real summaries: the frozen 24 deterministic real repeats per donor
  (`observed_summaries(real, 64)` in PC space). Generated draws: M = 32 per
  target, bag pcs of 64 cells, gene moments of 256 cells.

## Views (all scored with the frozen percentile 10/90 bounds and the frozen
interval score, alpha = .2)

(a) **Original basis.** Mean/SD/combined coverage, below/above misses, width
and interval score, overall and per PC block 1-10, 11-30, 31-50. Also the
per-donor log SD ratio log(median generated bag SD / mean real repeat SD) per
block (negative = generator under-dispersed).

(b1) **TRAIN-estimated shrinkage correction.** Per PC rank k, a shrinkage
factor f_k from 25 leave-one-TRAIN-donor-out refits of the frozen space recipe
(gene mean/sd, randomized SVD 50, pc_sd on the other 24 donors' cells):
f_k = mean over left-out donors of their within-donor SD on component k divided
by the mean in-sample within-donor SD of the fitting donors. Only TRAIN data
enter f. SD coordinates of every generated/reference draw are multiplied by f_k
(mean coordinates untouched) and rescored. Note: rescaling *both* real and
generated coordinates by any per-coordinate constant leaves coverage exactly
unchanged; the tests assert this, so a per-coordinate "rescaled basis" can only
matter through a one-sided correction like this one.

(b2) **Out-of-sample within-subspace basis.** For each held-out target, a
rotation/whitening of the 50 TRAIN PCs fit on the pooled real cells of the
other 11 held-out donors. Target real cells (same 24 subsamples) and the
target's generated/reference bags are re-expressed in it and rescored, blocks
by the new variance rank. Neither the target nor generated cells are in-sample
for this rotation. E_heldpool *is* in-sample for it (its donors define the
rotation); it is reported but flagged.

(c) **Gene space (basis-free).** Per gene, within-donor SD of log1p(panel
CP10k). Generated/reference: sqrt(max(sq_log - mean_log^2, 0)) per bag (256
cells, ddof 0). Real: 24 deterministic subsamples of min(n, 256) cells
(`stable_seed('gene-sd-rep', donor, repeat)`), ddof 0, with the second moment
rounded through float16 like the generated archive. Coverage, misses, interval
score and log SD ratio (epsilon .01) per decile of TRAIN-cell mean expression
(space.npz `mean`, train-only), plus gene-mean (mean_log) coverage.

Uncertainty: donor bootstrap over the 12 held-out donors (2,000 resamples,
seed 20261120), percentile 95% CIs for single sources and paired differences.
The 12 donors are the same development donors used in earlier selection; these
are not independent test results.

## Readouts declared now

1. Projection effect on a model-free sampler: E_train minus E_heldpool in (a)
   per block (SD coverage, below, above, log ratio).
2. Surviving generator under-dispersion: for G0 and G1, "above" miss rate and
   log SD ratio in (a), (b1), (b2) and (c). Under-coverage is
   U = max(0, .8 - SD coverage). Report survival U_view / U_(a). Under-
   dispersion is said to *survive* a view if the donor-bootstrap 95% CI of the
   overall SD log ratio (generator vs real) lies entirely below zero in that
   view, and is said to be *basis-explained* if it does not survive in (b2) and
   (c). The generator-versus-oracle paired difference in (a) is reported for SD
   coverage and log ratio.
3. Reference rows always reported: E_train (stage-blind empirical resampling)
   in every view.

No candidate is promoted by this study; it changes no default.

## Mondrian (stage-conditional) conformal analysis

SD coordinates, original basis, base bounds = frozen 10/90 percentiles over the
32 draws, calibration target = real repeat 0, evaluation on all 24 repeats,
CQR signed score s = max(lo - y, y - hi), symmetric expansion by the
ceil((n_cal + 1)(1 - alpha))-th smallest score, SD lower bound clipped at 0.

* Global leave-one-donor-out: calibrate on the other 11 held-out donors
  (rank 10 of 11; finite-sample marginal guarantee >= 10/12 = 83.3%).
* Mondrian leave-one-donor-out by stage: calibrate on the other held-out donors
  of the target's stage. Fold 0 held-out stages: Not AD 1, Low 2, Intermediate
  3, High 6 donors, so n_cal = 0, 1, 2, 5. The required rank exceeds n_cal for
  every stage except High, so those intervals are infinite. This is reported as
  the honest result, not patched; widths/scores are summarized over finite
  cells and the infinite fraction is stated.
* Per-stage coverage of the global calibration is reported with donor counts;
  no per-stage guarantee is claimed and no CI is given for single-donor stages.

Calibration donors are never GAN TRAIN donors; a test donor never calibrates
itself (enforced by code and tests). Applied to G0, G1 and E_train.

## Frozen historical metrics

Profile RMSE, gene mean/detection/log-variance errors and PC/cell energy are
recomputed with the frozen `evaluate.score_generated` for G0, G1 and E_train
and must match the stored pilot/challenge scores; they do not change in this
study (no generated cell changes), so they serve as identity checks.

## Outputs

Code, tests, PROTOCOL, README and small JSON results under
`analysis/pca_shrinkage_20261120/`; arrays under
`runs/research_20261120/pca-shrinkage/` (not in git). Local commit on
`research/pca-shrinkage-20261120`; no push.
