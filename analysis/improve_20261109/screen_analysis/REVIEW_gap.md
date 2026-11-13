# REVIEW_gap: adversarial review of the "why generators lose to trivial baselines" analysis

Reviewer scripts (new code written for this review; it does not reuse the analyst's gap_*.py):
- `screen_analysis/review_gap_core.py` has its own cache loader with the development-donor guard from `folds.json` and its own `log1p CP10k`. It reads generated.npz directly. It also draws its own real 256-cell subsamples (4 per donor). Study code is used only to regenerate the baselines' train-target predictions. Those regenerate the stored held-out predictions to within 3e-7. Output: `/ml2/alz/runs/improve_20261109/screen_analysis/review_gap_core.json`.
- `screen_analysis/review_gap_stage.py` scores stage deviations scaled by lambda on train targets (in sample) and on held-out targets. It does this for the models and for the real train stage means. Output: `review_gap_stage.json` and `review_gap_stage.log`.
- `screen_analysis/review_gap_tables.py` writes the fold means to `review_gap_tables.txt`.

All values below are fold means over 3 inner folds whose training sets overlap. "s" means a single seed (the mean of s0 and s1); "ens" means the 2-seed ensemble. I make no significance claims: there are 3 overlapping folds, 2 seeds, 12-13 held-out donors per fold and 1-2 held-out Not AD donors.

Note: the analyst's ANALYSIS_gap.md does not exist. I reviewed the claims in their summary.

## Verdicts per claim

**1. The headline numbers reproduce. CONFIRMED.**
My per-fit held-out pb_rmse equals the stored results/fits value to within 1e-8 for all 24 base/genemean fits. Fold means:

| Prediction | hvg2000 pb_rmse | wide pb_rmse |
|---|---|---|
| base s0 / s1 | 0.690 / 0.687 | 0.484 / 0.481 |
| base ens | 0.655 | 0.460 |
| genemean s0 / s1 | 0.651 / 0.670 | 0.458 / 0.423 |
| genemean ens | 0.583 | 0.385 |
| grand_mean | 0.464 | 0.296 |
| ridge | 0.461 | 0.295 |
| shrunk_stage_mean | 0.468 | 0.299 |

meanlog_rmse_gene on hvg2000 is 0.213/0.215 for base, 0.114 for genemean and 0.094 for the genemean ensemble, against 0.093 for the baselines.

**2. The comparison is like-for-like and Monte Carlo noise is small. QUALIFIED.**
- What holds: models and baselines share one prediction path, the mean over M of `log1p CP10k` of a 256-cell pseudobulk. The ensemble prediction equals the mean of the two seed predictions (by linearity). The MC variance of the prediction is 0.011-0.012 per seed, against a model MSE of about 0.49. Baselines carry 0.007-0.008 of MC variance themselves, so MC noise explains less than 0.005 of the 0.27 MSE gap.
- What is wrong: the analyst attributes the pred-minus-pool gap to the 256-cell depth. The gap is -0.19 for models and -0.13 for baselines, and the analyst says this path "lowers a profile by about 0.12-0.13 log units". It does not.
  - The real depth effect, measured as the real 256-cell pseudobulk minus the full-depth observation for the same donor, is only **-0.011 to -0.013** (hvg2000) and **-0.005** (wide).
  - The pred-minus-pool gap is a between-generated-donor Jensen gap: the log of the mean over 32 generated donors against the mean of their logs. It measures how widely the M generated donors spread, not depth.
  - The noise-free log-of-pooled grand-mean profile scores worse (0.483-0.514 against 0.464) for the same reason. The log of pooled train counts sits above the mean of donor log profiles. The sampling path is not the cause.
  - The baselines' residual bootstrap re-centres each generated donor's profile on the mean donor log, so their prediction lands near Lbar. Lbar is the mean of the train donors' log profiles, and on its own it scores 0.462.
- So the "design caveat" in the analyst's suspected-bugs list is misdescribed. It is still true that the baselines are explicitly calibrated to the scale of the metric's target and the generators are not.

**3. The error splits into a shared per-gene bias and a donor-specific excess. CONFIRMED.**

hvg2000 MSE (shared + donor-specific):

| Prediction | MSE | Shared | Donor-specific |
|---|---|---|---|
| base s | 0.488 | 0.183 | 0.305 |
| genemean s | 0.446 | 0.116 | 0.330 |
| genemean ens | 0.346 | 0.076 | 0.270 |
| grand_mean | 0.220 | 0.022 | 0.197 |

wide: base s 0.238 = 0.116 + 0.122; grand_mean 0.090 = 0.009 + 0.081.

The mean square of the level bias B is 0.166 (base s), 0.097 (genemean s) and 0.057 (genemean ens).

Caveat: "shared" is the error averaged over the 12-13 held-out donors. It therefore also contains the model's stage deviations averaged over the held-out stage mix, not only a stage-free level bias.

The headline's "two reasons of similar size" fits base: of an excess of 0.27, 0.16 is shared and 0.11 donor-specific. For the genemean ensemble the donor-specific part dominates: of an excess of 0.126, 0.054 is shared and 0.073 donor-specific.

**4. The donor-specific excess is the model's stage deviations, which have no held-out value. QUALIFIED. The numbers hold; the interpretation is incomplete.**
- These numbers reproduce:
  - Between-stage variance: 0.096 of the 0.102 prediction variance for base s.
  - Pearson of the model's stage deviations with the held-out real stage deviations: -0.20 to +0.34.
  - Variance ratio of model to real train stage deviations: 1.4 to 5.9.
  - lambda curve, base s: 0.462 at lambda 0, then 0.469, 0.487 and 0.551 at lambda 1.
- Missing context 1, the real stage signal does not transfer either. The **real** train stage-mean deviations correlate **-0.31 to +0.24** with the real held-out stage deviations. The least-squares scale of the real train stage means on held-out donors is **-0.06 to -0.12** in every fold and on both panels. A model with perfect stage means would also show about 0 held-out correlation. The model's ~0 is what the data allow, not a model-specific failure.
- Missing context 2, the model-specific defect shows up in sample. On TRAIN targets the least-squares scale of the model's stage deviations is **0.12-0.34** (hvg2000) and **0.09-0.29** (wide). The real stage means scale at 1.0.
  - At lambda 1, the model's stage deviations make the in-sample fit worse than no stage at all: base s 0.503 against Lbar 0.455; genemean s 0.54 against 0.455.
  - So the stage deviations are about 3-8x too large even on the donors the model was trained on. This in-sample inflation is verified, and it is a better basis for "shrink the stage effects" than the held-out result.
- Minor: the 92-95% between-stage share ignores degrees of freedom. Stages with one held-out donor have no within-stage variance, so per-target MC noise (0.012) is booked as between-stage.
- "About half or more is seed noise": confirmed in the sense that between-stage variance falls 0.093 to 0.066 (base) and 0.128 to 0.071 (genemean) when two seeds are averaged.

**5. The base bias is a reproducible structural defect, about 96% outside the 50-PC span. QUALIFIED.**
- Reproducibility is confirmed:
  - Seed-to-seed Pearson of B is 0.79-0.85; of the per-cell mean-log bias, 0.94-0.96.
  - The train-target correction c correlates 0.92-0.95 with the held-out shared error.
  - B correlates with gene level at -0.49 to -0.53.
  - B by level quartile on hvg2000: Q1-Q3 +0.16 to +0.22, Q4 -0.14 to -0.17.
- Two problems:
  - **The span share is not evidence of a critic blind spot.** The in-span share of the standardised mean-log bias is 3.1-4.1% (hvg2000) and 1.6-3.1% (wide). A gene-permutation null of the same vector gives **3.8-4.2%** and **2.9-3.8%**. The bias sits in the span exactly as much as a random rearrangement of itself would, so "96% outside" is a dimensional fact (50 of 2,000 or 5,000 directions), not a mechanism. The analyst says "near chance", but the chance level they quote (2.5%) uses the raw dimension ratio, and the wide figure (2.6%) is 2.6x that ratio, not "near" it. The causal evidence that gene-level visibility matters is the genemean arm itself (claim 6), not the span fraction.
  - **"It" changes meaning within the claim.** The range compression (Q1-Q3 positive) is in the **pseudobulk** bias B, whose in-span share is 12-14%, close to its own permutation null of 13-16%. The **per-cell mean-log** bias of base is small in Q1-Q3 (about +0.03) and concentrated in Q4 (-0.17). For base, the Q1-Q3 overprediction is almost entirely a pseudobulk-versus-mean-log shape gap of +0.14 to +0.18, not a mean-log bias.

**6. genemean removes the mean-log bias, and its remaining bias is seed-random. QUALIFIED.**
- Confirmed:
  - The mean square of the mean-log bias falls from 0.042 to 0.009 (single seed) and 0.005 (ensemble); baselines are at 0.003.
  - The seed Pearson of the genemean mean-log bias is -0.01 to 0.16.
  - The seed Pearson of the genemean B is 0.09-0.27.
- Over-attribution: the analyst says the residual seed-random bias "is why ensembling helps genemean much more than base". On hvg2000 the ensemble lowers genemean MSE by 0.100. Only 0.040 of that comes from the shared (bias) part; **0.060 comes from the donor-specific part**, which is the seed-noisy stage deviations. For base: 0.017 shared and 0.032 donor-specific. Most of the ensemble gain is averaging away seed-specific stage deviations, not bias.

**7. Most of the post-genemean pseudobulk bias is a count-shape bias in highly expressed genes. QUALIFIED.**
- Reproduced with 4 averaged real 256-cell subsamples on train targets, against one subsample in the analyst's version:
  - Shape-gap mean square on hvg2000: base 0.074-0.085; genemean s 0.058-0.069; genemean ens 0.039.
  - Q4 shape gap for genemean: -0.12 to -0.14.
  - For base, the shape gap is positive in Q1-Q3 (+0.14 to +0.18).
- Caveats:
  - (a) Q4 carries 41-56% of genemean's shared held-out error. The grand_mean baseline's Q4 share is the same (43-55%), so the share is not distinctive. What is distinctive is the sign and size: genemean's mean B in Q4 is -0.10 to -0.17, against -0.01 to -0.02 for grand_mean.
  - (b) "Shape" covers both count dispersion and cell-state composition: anything that separates the log of the mean from the mean of the logs. The analysis cannot tell them apart.
  - (c) "Which a mean-of-log critic input cannot see" is a causal hypothesis that was not tested.

**8. A train-only additive recalibration closes 56-67% of the gap. QUALIFIED; the range is wrong for genemean.**
- The fitting is honest. The correction c is the mean over train targets of pred minus obs; it uses no held-out data. Recalibrated pb_rmse reproduces:

| Prediction | hvg2000 raw → recal | wide raw → recal |
|---|---|---|
| base s | 0.689 → 0.563 | 0.482 → 0.358 |
| base ens | 0.655 → 0.536 | — |
| genemean s | 0.661 → 0.587 | — |
| genemean ens | 0.583 → 0.536 | 0.385 → 0.341 |

- The gap closed against grand_mean is:

| Prediction | hvg2000 | wide |
|---|---|---|
| base s | 55-57% | 66-67% |
| base ens | 62% | 74% |
| genemean s | **37-38%** | **43-44%** |
| genemean ens | **40%** | **49%** |

- The headline's "56-67%", stated next to the genemean ensemble example, holds only for base. After recalibration the shared error is 0.024 against 0.022 for the baselines, and meanlog_rmse_gene is 0.073-0.075 against 0.071-0.072. Both confirmed.
- A recalibrated model is algebraically Lbar plus the model's deviations from its own train-target mean. This "recalibration" is therefore the same as replacing the model's level with the train grand mean.

**9. After recalibration the remaining gap is the stage deviations, and nothing stage-conditioned beats grand_mean. QUALIFIED.**
- The lambda curve is confirmed (hvg2000 0.462/0.466-0.471/0.479-0.497/0.528-0.587; wide 0.295/0.297-0.301/0.305-0.317/0.334-0.375).
- "Best scale is 0" was read off the held-out scores. I fitted the scale by least squares instead: it is -0.06 to +0.16 on held-out donors (noise around 0) and 0.09-0.34 on train donors.
- "No stage-conditioned predictor beats grand_mean" is slightly too strong. Ridge (stage + covariates) beats grand_mean in 3 of 3 folds on both panels, by 0.001-0.004 (hvg2000 0.442/0.472/0.468 against 0.446/0.475/0.471). That margin is negligible and not interpretable with this design, but it is not a loss.
- The real reason the stage terms add nothing on held-out donors is in the data: real train stage means scale at -0.06 to -0.12 on held-out donors.

## Suspected bugs

- "No code bug found": **agree.** I found no defect in evaluate.score_generated, in baselines.generate or in the alignment. My independent loader reproduces every stored number.
- The "design caveat" (that the 256-cell path lowers a profile by 0.12-0.13): **not a real effect as stated.** The pure depth effect is about -0.012 (hvg2000) and -0.005 (wide). The 0.12-0.19 pred-minus-pool gap is the spread between generated donors. The log-of-pooled baseline scores 0.501 because log-of-pooled differs from the mean of logs, not because of the sampling path. The underlying point stands: baselines are explicitly anchored to the mean of train donor logs.

## Additional issues

1. **The generators lose in sample too.** Stored train-target pb_rmse on hvg2000 is base 0.650 and genemean 0.627, against grand_mean 0.458 and stage_mean_log 0.429. The gap is a fit problem, not a generalisation problem. The report frames it as held-out only.
2. **The stage inflation is verified in sample.** The model's stage deviations are 3-8x too large on the training donors themselves (least-squares scale 0.09-0.34). The report's "no held-out value" is the weaker and noisier version of this point.
3. **"Base output bias starts at log pooled train proportions, so the bias develops during training" is not checkable.** No init-time generation exists. The docstring confirms the initialisation, not the drift.
4. **The extrapolation of genemean bias against seed count** (0.017 + 0.080/N) uses one seed pair per fold. The seed-specific estimate ranges from 0.047 to 0.108 across folds. Treat it as a rough hypothesis.
5. **The paired comparison is entirely confounded by stage composition.** The held-out sets carry only 1-2 Not AD donors per fold. Every stage-level statement rests on n of 1-3 donors per stage per fold.

## Corrected headline

Every number reproduces exactly, and the evaluation path is like-for-like. The generators lose to grand_mean on held-out donors and also on train donors (in-sample pb_rmse 0.63-0.65 against 0.46). There are two error sources:

- **A per-gene level bias.** For base it is reproducible across seeds and mostly a pseudobulk-versus-mean-log shape gap in Q1-Q3 plus a mean-log deficit in Q4. genemean removes the mean-log part and leaves a mostly seed-random remainder.
- **Stage deviations that are 3-8x too large even in sample.**

Monte Carlo noise and depth (about 0.01) are negligible. The "0.12-0.13 sampling-path shift" is a between-generated-donor Jensen effect, not depth.

A train-only per-gene recalibration closes 55-74% of the gap for base but only 37-49% for genemean (hvg2000 0.583 → 0.536 for the genemean ensemble). Shrinking the stage deviations to zero recovers the grand mean (0.462 / 0.295). Real train stage means do not transfer to held-out donors either (least-squares scale about -0.1), so with these 37 donors tying grand_mean is the realistic pb_rmse target. Stage signal must be judged elsewhere.
