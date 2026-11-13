# NOTES_evaluator (evaluate.py, baselines.py, metrics.py, test_evaluate.py)

## evaluator: interface decisions where PROTOCOL.md is ambiguous

- **Target order / identity.** `generated.npz` targets are re-ordered to the cache
  order (train donors then held-out donors, as in `train/metadata.npz` and
  `heldout/metadata.npz` `donors`). The evaluator refuses a fit whose genes,
  target set, `is_heldout` or `stage` disagree with the cache, or whose arrays
  are not finite. The fit is then listed under `errors` in aggregate.json.
- **Stage** comes from the cache `raw_c[:,3]*3` (ADNC/3), with 0 = Not AD and 3 = High.
- **Completed fit** means `generated.npz` and `fit_receipt.json` both exist. A
  directory without them is listed under `incomplete_fits` and skipped.
- **K** (cells per generated donor) is estimated as median(pb_counts.sum / lib).
  Protocol K = 256. It sets the size of the real-donor subsample used for the
  pseudobulk summary space.
- **Train grand-mean profile** (used to form deviations for the correlations) is
  the donor-weighted mean of train-donor log1p(CP10k) pseudobulk profiles. It
  is not log of pooled counts.
- **Pseudobulk prediction**: model profile = mean over M of the log1p(CP10k)
  of each generated donor's K-cell pseudobulk. The observed profile uses all
  of the donor's cells, as the task specifies. Caveat: a 256-cell generated
  pseudobulk has more count noise for low-expressed genes than an all-cell
  observed one. This adds a floor that is the same for every method, so the
  noise ceilings and the oracle in the test give the scale.
- **Per-gene mean log / nonzero**: gene error e_g = mean over target donors of
  (model - real), where model = mean over M of the generated per-donor value.
  Reported: `rmse_gene` (RMSE of e_g), `rmse_donor_gene` (over donor x gene),
  `bias`, `mae_gene`, the fraction of |e_g| above {0.05, 0.1, 0.25} for mean_log
  and {0.02, 0.05, 0.1} for nonzero, and quartile strata of train mean log
  expression. Depth matching relies on the generator using the target's own
  library sizes (Interface 2). The real side uses the donor's actual cells.
- **Stage contrasts**: High-minus-NotAD = difference of stage means of donor
  log profiles. The ordinal slope is the per-gene OLS slope on stage 0..3
  across donors. The model contrast comes from predicted per-target profiles
  of target set E (held-out or train) averaged within stage. It is compared
  with the real held-out and the real train contrast (2x2: `contrast.high_notad_vs_heldout`,
  `..._vs_train`, `slope_vs_*` under each E). Agreement reports: Spearman,
  Pearson, sign agreement overall and on the top decile of |real effect|,
  top-decile Spearman, OLS slope of model on real (shrinkage), top-decile
  slope, and SD ratio. The real-effect regression slope is attenuated by noise
  in the real contrast (errors-in-variables). Read it next to the noise
  ceiling / split-half and the stage_mean baseline, not as an absolute.
- **Low n**: `real_heldout_low_n` is true when the held-out set has fewer than 2
  Not AD or High donors. In the real folds, folds 0 and 1 each have exactly 1
  held-out Not AD donor, and fold 2 has 2. All 37 development donors include
  only 4 Not AD. The **pooled-across-folds** version (aggregate.json
  `pooled_across_folds`) combines held-out donors of all three folds, each
  predicted by the fit of its own fold, matched by seed. A seed-ensemble pooled
  version and per-baseline versions are also given. Folds have different gene
  panels, so profiles are renormalised to CP10k over the genes present in all
  three folds' caches of that panel before the contrast is formed.
- **Distribution, PC space**: the summary is the bag mean and bag sd (ddof=0) of
  the 50 fold PCs over the stored 64 cells per generated donor (100 dims).
  The real donor summary uses a deterministic 64-cell subsample, matching
  sampling noise. **Difference from the previous calibration metric:** the
  earlier 26-dim summary (sqrt type composition from the type classifier +
  first 20 PC means) cannot be reproduced. Interface 1 has no type classifier,
  and generated.npz stores only 64 cells. Energy score, coverage80 (10-90%
  marginal interval per dim), `spread_ratio` and `within_condition_spread_ratio`
  use the same formulas as `scoring.score_summary_arrays` / `evaluation.evaluate`.
  Added: `within_stage_spread_ratio` = sqrt(mean_t generated variance trace /
  pooled within-stage real variance trace of the evaluated donors), plus a
  `_train_reference` variant against train donors, as in
  `within_stage_spread_ratio_fit_reference`. For the `covars` arm the
  generated spread is conditional on covariates, so a within-stage ratio
  below 1 is expected there.
- **Distribution, pseudobulk space**: log1p(CP10k) pseudobulk restricted to the
  256 genes with the highest train grand-mean profile (all genes when the
  subset is smaller, e.g. risk genes). The real summary uses a deterministic
  K-cell subsample (K = 256 or all cells if fewer).
- **Cell energy distance**: `bench_core.cell_energy` (V-statistic, unchanged).
  Per target: 512 generated cells (from the M x 64 stored) vs up to 512 of the
  target's real cells, deterministic subsamples. Pooled per evaluation set:
  1,500 vs 1,500 cells, as in the benchmark reservoir. **Interpretation:** the
  per-target number includes the target's own donor effect, so a perfect
  stage-population sampler is not near 0 there (test: 0.23 vs oracle 0.04).
  The pooled number compares populations (test: truth 0.03).
- **Gene subsets / panel comparison**: `all`, `risk` (forced risk genes present),
  `shared_<otherpanel>` (native panel normalisation, restricted) and
  `shared_<otherpanel>_renorm` (CP10k renormalised within the shared gene set,
  so hvg2000 and wide numbers are on identical units). Per-cell mean_log /
  nonzero cannot be renormalised from stored generated summaries, so they are
  reported only for native subsets. PC-space metrics are panel-wide and not
  subsettable. Shared subsets are computed for both panels when the other
  panel's same-fold cache exists.
- **Seed stability**: mean pairwise Pearson/Spearman between seeds of model
  High-minus-NotAD and ordinal-slope contrasts (held-out targets and all 37
  targets). **Seed ensemble**: generated donors concatenated along M across
  seeds of the same arm/panel/fold, scored by the same function.
- **Learning curve**: means over records with step > last_step - 5000 of
  critic_gap, D/G adversarial and total losses, penalty, prior regularisation
  and gradient norms.
- **Aggregation**: one row per (arm, panel). Seed-ensemble rows are labelled
  `<arm> [seed-ensemble]` and baselines `baseline:<name>`. Mean over (fold,
  seed) units, with `__se` = SD/sqrt(n_units) and `__fold_se` = SE of fold
  means. Paired comparisons use a "goodness" scale, so a positive difference
  is better: negated for lower-is-better, -|log ratio| for spread ratios,
  -|cov-0.8| for coverage, -|bias| for bias. They also give
  `raw_mean_difference`, the sign count (n_better / n_worse) and no p-values.
  vs base is paired on (fold, seed). vs stage_mean is paired on fold, with the
  baseline shared across seeds. Ensembles are compared with the base ensemble.
- **Incrementality**: a per-fit JSON is reused when evaluator version,
  generated.npz size+mtime, cache manifest sha256 and subset names are
  unchanged. Ensembles are keyed on all seed inputs. Baselines are keyed on
  evaluator+baseline version and manifest sha.
- **Disk**: per fit, a JSON (~100-300 KB) plus an npz with held-out predicted
  profiles (float32, needed for pooled contrasts). No generated counts are written.

## evaluator: baselines (baselines.py)

Every baseline emits arrays in the generated.npz layout for all 37 targets
(M=32, K=256, 64 stored PC cells). Library sizes are drawn from the target's
own observed panel library sizes. The RNG depends only on (panel, fold,
target), giving common random numbers across baselines. Only train-donor
data is fitted.

| baseline | prediction | pseudobulk / contrast | mean_log / nonzero | pb-space distribution | PC-space / cell energy |
|---|---|---|---|---|---|
| grand_mean | log CP10k of pooled train counts | null reference (no stage) | count-model check | partial | not meaningful |
| stage_mean | log(1e4 x mean train-donor proportions per stage) | meaningful | count-model check | partial | not meaningful |
| shrunk_stage_mean | per-gene EB shrinkage of stage deviations toward grand | meaningful | count-model check | partial | not meaningful |
| ridge | stage one-hot + stage/3 + Male + APOE4 + std age, alpha by LOO over train donors | meaningful | count-model check | partial | not meaningful |
| resample | real train donors of the same stage, depth-qualified cells + thinning | meaningful | meaningful | meaningful | meaningful (see caveat) |

Profile-baseline count model: the donor profile is mu_t plus a bootstrapped
train residual (excluding the target itself). Cells are independent-gene
gamma-Poisson with per-gene method-of-moments dispersion estimated within
train donors. Resample depth rule: for each drawn target library L, take a
source cell with library >= L and thin binomially to L. The first version
paired uniformly and thinned only downward. The test showed that version
lowered depth systematically (nonzero RMSE 0.049 vs truth 0.011), so it was
replaced (now 0.013). The selection favours deeper source cells.

Noise ceilings (`baselines/{panel}/fold{k}/noise_ceiling.json`):
- each held-out donor predicted from the other held-out donors of its stage
  (undefined for a lone Not AD donor);
- the same using all other development donors of the fold;
- train-stage mean of log profiles;
- split-half reliability of the real contrast for held-out, train and all
  donors (200 within-stage halvings, plus a Spearman-Brown value);
- agreement of the real train contrast with the real held-out contrast.

## evaluator: important interpretive caveat found by the synthetic test

The 50 PCs are fitted on train cells only, so some PCs can encode
**train-donor identity**. In the simulation, train donor-mean variance on such
PCs was ~0.5, against ~0.05 for held-out donors. Any method that reuses train
donors (resampling, residual bootstrap) then looks over-dispersed in PC
space on held-out targets (spread ratio ~1.5) but not on train targets (~0.96).
A fresh-donor sampler shows the reverse (held-out 1.01, train 0.63). Read
PC-space spread on held-out donors with this in mind. It also argues for
reading the pseudobulk-space distribution metrics alongside the PC-space ones.

## evaluator: what was verified vs not

- The metric logic was verified on synthetic data: `test_evaluate.py` passes
  31/31 checks. The report is in
  `/ml2/alz/runs/improve_20261109/tmp_evaluator_test/test_report.json`.
- The real-cache path was run on the build_cache.py caches. Baselines and
  noise ceilings were computed for every (panel, fold) and written to
  `/ml2/alz/runs/improve_20261109/results/`. hvg2000/fold0 took 3 min
  wall-clock at 6 threads with 0.83 GB RSS. A wide fold takes about 7.5 min,
  and all six (panel, fold) pairs took 31 min. Output is 14 MB.
- Fold-0 real-data observations (hvg2000), descriptive only:
  - The real train-vs-held-out High-NotAD contrast Spearman is -0.06.
  - Split-half Spearman of the real contrast is 0.13 on train donors and 0.12
    on all 37 development donors.
  - Every predictor, including the real-peer noise ceilings, has a held-out
    pseudobulk deviation correlation of about 0. The donor-specific deviation
    dominates.
  - EB shrinkage shrinks 73% of genes fully to the grand mean.
- Checked by reading the trainer's `generate.py`, not by running it: its
  `pcs` are `standardize_counts(counts) @ (W / pc_sd)`, the same projection
  the evaluator applies to real cells. The evaluator has **not** yet scored a
  real `generated.npz`, because no fit existed at the time of writing.

## integration changes (added by the integrator, see NOTES_integration.md)

- baselines.py v4: the M drawn train residuals of the profile baselines are
  re-centred per target on the mean residual of all train donors (removes the
  leave-self-out shift -r_t/(n-1) and the random-repeat shift). Cached v2
  baseline results are recomputed on the next run.
- The pooled-across-folds contrast is not a held-out estimate (each held-out
  donor of fold k trains the other folds' models; stage_mean reaches Spearman
  ~0.93 there by construction). Entries now carry `independent_of_training:
  false` and a warning; use per-fold `*_vs_heldout` metrics for held-out claims.
