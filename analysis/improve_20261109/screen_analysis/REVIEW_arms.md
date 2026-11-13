# Review of the arms analysis (arm effects versus base)

Adversarial review of the analyst's structured output for "arm effects versus base". The analyst's file
`screen_analysis/ANALYSIS_arms.md` does not exist; the harness blocked them from writing it. Their claims were taken
from the structured summary and from their scratch outputs `runs/.../screen_analysis/arms_*.{json,md}`.

Every number below was recomputed with new code written from scratch. It reads generated.npz and the
development-donor caches only, and it imports neither evaluate.py, metrics.py nor any arms_*.py.

| script (screen_analysis/) | what it does | output (runs/improve_20261109/screen_analysis/) |
|---|---|---|
| review_arms_common.py | loads caches and generated.npz; implements the metrics | - |
| review_arms_recompute.py | recomputes all 48 fits, 24 seed-ensembles, two M-matched ensemble variants, shared-gene pb_rmse; compares with the evaluator | review_arms_recompute.{json,log}, review_arms_profiles_*.npz |
| review_arms_analysis.py | arm-vs-base effects, ensemble decomposition, seed projection, panels, seeds needed, stage counts, a check of whether covars copies train donors | review_arms_analysis.{json,md} |
| review_arms_extra.py | absolute seed variance D, pb_dev_pearson, M-matched PC metrics, projection sensitivity, depth floor | review_arms_extra.{json,md} |
| review_arms_sens.py | how much the seed-variance estimate must change before the projection reaches the baseline | review_arms_sens.log |

CPU only, 6 threads, no training. Only the 37 development donors are read; the code asserts this on every cache and
every generated.npz.

## Bottom line

Most of the analysis holds. The headline needs one correction and a few qualifications:

* **Confirmed.** genemean is the only arm that robustly beats base, and only on per-gene metrics. It wins 12/12
  within-fold comparisons on held-out meanlog_rmse_gene and nonzero_rmse_gene, on both panels and on train donors.
* **Confirmed.** offset is null and donorpc is worse.
* **Confirmed.** covars and covars,genemean,offset fit train donors better and held-out donors much worse. This review
  adds direct evidence for the memorisation story (section G below).
* **Confirmed.** The single-seed pb_rmse gain of genemean is not robust (8/12 crossed wins on hvg2000, 10/12 on wide).
* **Confirmed.** The seed-ensemble gain comes from seed-to-seed model variance, not Monte-Carlo noise. genemean's seed
  variance is about twice base's.
* **Not supported (correction).** The analyst wrote: "even infinitely many seeds would not bring genemean to the ridge
  baseline". The projection numbers reproduce exactly, but they rest on one seed pair per fold. If the true seed
  variance were only 7-27% larger than that one pair shows, the infinite-seed genemean projection would reach ridge in
  every fold. The fold-to-fold spread of the seed variance is itself a factor of about 1.7. The question is open, not
  answered negatively. The statement does hold for base.
* **Qualified.** The analyst said genemean's PC-space and cell-level cost "largely disappears or reverses" on held-out
  donors with a matched-M two-seed mix. On hvg2000 it is about halved and stays positive in 2 of 3 folds. It reverses
  only on wide. The mixing gain is also partly spread inflation, because genemean's two seeds differ more from each
  other.

## Claim by claim

### 1. The evaluator's metrics reproduce: qualified
* Gene-level metrics: pb_rmse, meanlog_rmse_gene, nonzero_rmse_gene and hn_spearman (vs held-out and vs train) match
  the evaluator to 1.7e-10 or better. This holds for **all 48 fits** on both splits and all 24 ensembles, not only for
  the analyst's sample of 8.
* PC-space metrics: the analyst's agreement to 1.8e-9 came from reusing the evaluator's own deterministic subsample
  seeds. That checks the arithmetic, not robustness. With independent subsamples, single-fit values differ by:

  | metric | max difference | median difference |
  |---|---|---|
  | pc_energy | 0.091 | 0.022 |
  | pc_coverage80 | 0.033 | 0.008 |
  | cell_energy | 0.024 | 0.006 |

  The dominant noise is the 64-cell subsample of each real donor, which the analyst's cell_energy redraw did not
  probe. Paired arm differences are much less affected, because arms share the real subsample. One crossed count still
  moved: wide held-out genemean pc_energy is 6/12 here versus 4/12 for the analyst. PC-metric effects smaller than
  about 0.05 should not be read.

### 2. genemean improves per-gene metrics robustly: confirmed
Held-out meanlog_rmse_gene:

| panel | base | genemean | fold diffs | crossed wins | seed SD base/arm |
|---|---|---|---|---|---|
| hvg2000 | 0.214 | 0.114 | -0.093/-0.110/-0.097 | 12/12 | 0.012/0.016 |
| wide | 0.190 | 0.088 | - | 12/12 | - |

Held-out nonzero_rmse_gene:

| panel | arm minus base | crossed wins |
|---|---|---|
| hvg2000 | -0.020 | 12/12 |
| wide | -0.023 | 12/12 |

Train donors are also 12/12 on both metrics. Caveat: meanlog_rmse_gene is the RMSE over genes of the
**donor-averaged** error. It measures per-gene bias across donors, which is exactly what the genemean critic sees. It
is not per-donor accuracy.

### 3. genemean's single-seed pb_rmse gain is not robust; ensembles improve in every fold: confirmed

| panel | single seed | fold diffs | crossed wins | ensemble fold diffs |
|---|---|---|---|---|
| hvg2000 | 0.689 -> 0.661 | +0.001/-0.067/-0.019 | 8/12 | -0.047/-0.100/-0.068 |
| wide | 0.482 -> 0.441 | - | 10/12 | -0.070/-0.052/-0.104 |

"Every fold" means 3 of 3 overlapping folds.

### 4. genemean costs PC-space and cell-level fit: qualified

Single seed, matching the analyst:

| panel / split | metric | arm minus base | fold diffs | crossed wins |
|---|---|---|---|---|
| hvg2000 held-out | pc_energy | +0.199 | +0.266/+0.081/+0.250 | 0/12 |
| hvg2000 held-out | cell_energy | +0.154 | - | 0/12 |
| train, both panels | pc_energy | +0.35/+0.44 | - | 0/12 |
| train, both panels | cell_energy | +0.21/+0.28 | - | 0/12 |
| wide held-out | pc_energy | +0.022 | - | 6/12 |
| wide held-out | cell_energy | +0.086 | - | 2/12 |

M-matched mix of 16+16 distinct draws, genemean minus base, per fold:

| panel / split | pc_energy | cell_energy | pc_coverage80 |
|---|---|---|---|
| hvg2000 held-out | +0.070/-0.069/+0.094 | +0.091/-0.036/+0.102 | - |
| hvg2000 train | +0.192/+0.052/+0.233 | - | - |
| wide held-out | -0.080/-0.187/-0.278 | -0.033/-0.046/-0.074 | +0.012/+0.160/+0.148 |

* On hvg2000 held-out, the cost is roughly halved and remains in 2 of 3 folds. It has not "largely disappeared".
* The mixture gains on coverage80 track genemean's larger seed-to-seed difference (section B below). Part of the
  "recovery" is a broader mixture, not a better single model.

### 5. genemean weakens agreement with the training High-minus-NotAD contrast: qualified

| panel | hn_spearman_vs_train | fold diffs | crossed wins |
|---|---|---|---|
| hvg2000 | 0.464 -> 0.376 | -0.170/-0.157/+0.065 | 3/12 (indistinguishable by the analyst's own rule) |
| wide | 0.337 -> 0.176 | - | 1/12 (leans worse) |

For stage-only arms, the model contrast on held-out targets is just the model's stage contrast. This metric therefore
measures fit to the *training* contrast, not generalisation.

### 6. covars and the combination overfit donor identity: confirmed, with new supporting evidence

Held-out:

| arm | pb_rmse | pc_energy | cell_energy | pc_coverage80 |
|---|---|---|---|---|
| base | 0.689 | 2.243 | 0.922 | 0.610 |
| covars | 0.927 (0/12) | 3.441 (0/12) | 1.677 (0/12) | 0.285 (0/12) |
| covars,genemean,offset | 0.974 (0/12) | 3.761 | 2.062 | - |

Train donors:

| arm | pc_energy | cell_energy | pb_dev_pearson |
|---|---|---|---|
| base | 2.156 | 0.883 | about 0.06-0.11 |
| covars | 1.302 (12/12 better) | 0.394 (12/12 better) | 0.30-0.54 |

Held-out pb_dev_pearson is about 0 for every arm. covars is *not* better on train pb_rmse (-0.003, 6/12). Its
in-sample gain is in PC, cell and deviation-correlation metrics. The combination's meanlog_rmse_gene is 0.159
held-out and 0.058 train.

**Section G check (new).** For each held-out donor, I found the train donor whose observed deviation profile best
matches the predicted deviation profile. I then counted how often that best match is the train donor nearest in the
covariate vector the arm sees (ADNC/3, Male, APOE4 dose, train-standardised age):

| arm | fraction where best match = nearest-covariate train donor |
|---|---|
| base | 0.00-0.25 (stage alone gives some hits) |
| covars | 0.25-0.85 (median about 0.7) |
| covars,genemean,offset | 0.31-0.58 |

Chance is about 1/24. Correlation with the donor's *own* observed deviation stays at about 0 (-0.06 to +0.06). The
model reproduces a train look-alike, not the held-out donor.

### 7. offset is null; donorpc is worse on gene and pseudobulk metrics: confirmed

| arm | metric (held-out) | arm minus base | crossed wins |
|---|---|---|---|
| offset | pb_rmse | +0.001 | 5/12 |
| offset | meanlog_rmse_gene | +0.002 | 5/12 |
| offset | hn_spearman_vs_heldout | -0.040 | 4/12 |
| donorpc | pb_rmse | +0.104 | 1/12 |
| donorpc | meanlog_rmse_gene | +0.048 | 1/12 |
| donorpc | nonzero_rmse_gene | +0.012 | 0/12 |
| donorpc | pc_energy | -0.020 | 7/12 |
| donorpc | cell_energy | +0.017 | 5/12 |

All other offset metrics fall between 3/12 and 7/12. pbdist_energy was not recomputed; that figure relies on the
evaluator.

### 8. The ensemble gain is seed-model variance, not Monte-Carlo noise: confirmed

D is the mean squared difference between the two seeds' held-out pb predictions. Expressed as a multiple of what
M=32 Monte-Carlo noise alone would give (independent draws), per fold:

| arm | fold values | mean |
|---|---|---|
| base | 9.5/6.4/9.2 | 8.4 |
| genemean | 19.3/12.1/22.2 | 17.9 |
| covars | 82/172/125 | 126 |
| combination | 195/362/175 | 244 |

* Absolute D, genemean / base: 0.43/0.19, 0.28/0.15, 0.48/0.23 on hvg2000, so about 2x.
* The identity MSE_ens = mean MSE_single - D/4 reproduces the M=64 ensemble values exactly.

Two possible artefacts were checked:
* **Common random numbers** (generation seeds depend only on fold and target donor). The per-generated-donor deviation
  correlation across seeds is only +0.02 to +0.25, so Monte-Carlo independence is about right.
* **Matched-M mixes.** Mixing 16+16 shared draws or 16+16 distinct draws gives the same pb_rmse within 0.012 as the
  M=64 ensemble.

The analyst did not rule out one alternative reading. genemean's larger seed variance may reflect non-convergence or
oscillation at 20k updates: its mean critic gap is 13.0 versus 4.5 for base (summary.md lc_critic_gap). In that case
checkpoint averaging might capture part of the gain. This needs no new seeds to test, only the existing learning
curves or a later-iterate generation.

### 9. Infinite seeds would not reach the baselines: qualified (the genemean part is not supported)

The projections reproduce exactly:

| arm | panel | infinite-seed pb_rmse by fold | ridge by fold | seeds to reach ridge |
|---|---|---|---|---|
| genemean | hvg2000 | 0.493/0.466/0.511 | 0.442/0.472/0.468 | only fold 1, at S=26 |
| genemean | wide | 0.319/0.315/0.313 | 0.286/0.302/0.297 | none |
| base | hvg2000 | 0.60-0.65 | - | never plausibly |
| base | wide | 0.42-0.46 | - | never plausibly |

The genemean conclusion rests on D from **one seed pair per fold**, and the projection subtracts D/2 from an MSE of
which D/2 is a large part. The scaling c* of D at which the projection reaches the best baseline is close to 1
everywhere:

| panel | c* by fold |
|---|---|
| hvg2000 | 1.22/0.97/1.17 |
| wide | 1.27/1.07/1.10 |

D itself varies by a factor of 1.7 across folds. Whether genemean ensembles can reach ridge is therefore undetermined
by this design.

meanlog_rmse_gene: the analyst compared against profile baselines only. Per baselines.py, mean_log is "meaningful"
only for `resample` (0.078-0.080 on hvg2000, 0.053-0.055 on wide). Against resample, the seeds needed are:

| panel | analyst (profile baselines) | against resample |
|---|---|---|
| hvg2000 | 2/1/never | 4/2/never |
| wide | 4/13/never | 15/never/never |

The analyst's follow-up line "about 3-4 genemean seeds in 2 of 3 folds" contradicts their own wide number (13). It
holds only for hvg2000.

Depth was checked as an alternative explanation for pb_rmse floors. A perfect self-model of each held-out donor,
scored the evaluator's way (256-cell pseudobulks against the full-depth donor), gives pb_rmse of only 0.043-0.047 on
hvg2000 and 0.026-0.027 on wide. The pb_rmse floors are not a depth artefact.

### 10. The panel gap is a gene-set artefact: confirmed for base; "no panel difference" for genemean is undetermined

All 2,000 hvg2000 genes are in the wide panel in every fold, and the real shared-gene profiles are identical across
caches (max difference 0). Shared-gene renormalised held-out pb_rmse:

| arm | hvg2000 | wide | wide minus hvg2000, by fold | crossed (wide better) | ensembles hvg2000 / wide |
|---|---|---|---|---|---|
| base | 0.689 | 0.663 | -0.037/-0.014/-0.025 | 10/12 | 0.655 / 0.639 |
| genemean | 0.661 | 0.643 | -0.064/+0.072/-0.060 | 7/12 | 0.583 / 0.572 |

For genemean the fold signs are mixed with large swings, so "no difference" really means "cannot tell". The
within-panel nonzero comparison was not recomputed.

### 11. No arm measurably changes held-out High-minus-NotAD agreement: confirmed
* Held-out stage counts (NotAD/Low/Int/High) by fold: [1,2,3,6], [1,2,3,7], [2,1,3,6]. Train NotAD donors: 3/3/2.
* Crossed wins on hn_spearman_vs_heldout are 2-7/12 for every arm.
* With one held-out Not AD donor, the "real" contrast is dominated by that single donor.

### 12. Seeds needed to resolve the genemean pb_rmse effect: qualified
* **Folds treated as fixed.** My estimate is 3.4 seeds per arm, against the analyst's 5. The analyst's variance of the
  seed-paired difference includes a same-seed covariance term. Each seed variance has only 3 degrees of freedom, so the
  90% range of that number is **1.3 to 29**.
* **Folds treated as random.** The analyst says the effect is "not reachable at any seed count". That rests on a fold
  interaction estimated from 3 overlapping folds (2 df). The current effect over its fold SE is -1.39 on hvg2000 and
  -2.30 on wide. This is not evidence of unreachability. It is evidence that 3 folds cannot decide.
* I agree with the design conclusion: more donor splits matter more than more seeds.

## Suspected bugs
* **"No evaluator bug found": agreed.** Independent code reproduces every deterministic metric on all 48 fits and 24
  ensembles to 2e-10. PC metrics agree up to subsampling noise. The depth mismatch in pb_rmse (256-cell generated
  pseudobulks against full-depth real ones) follows the protocol, is shared by baselines, and is small (floor about
  0.045). It is not a bug.
* **summary.md presentation risk: real (presentation, not code).** All-gene pb_rmse on wide (0.482) is not comparable
  with hvg2000 (0.689). The like-for-like shared-gene value is 0.663.
* **Missing report file: real.** ANALYSIS_arms.md does not exist. This review does not create it, since the analyst's
  files are off-limits to it.

## Other issues
1. The crossed-wins counts (x/12) are 4 non-independent comparisons in each of 3 overlapping folds. "12/12" means
   consistent, not significant.
2. Single-fit PC-metric differences below about 0.05 are within real-donor subsample noise.
3. The covars memorisation story was asserted without direct evidence. Section G supplies some.
4. The ensemble's PC "recovery" mixes two models. Coverage gains partly reflect between-seed spread.

## Recommendation (amended)
* Confirm genemean against base only, as the analyst says.
* Include one test that the screen cannot settle: run at least 4-6 seeds per fold, so that the seed variance D (and
  hence the infinite-seed projection against ridge) is estimated from more than one pair.
* Prefer more donor splits (5 folds or repeated splits) over more seeds.
* Before committing seeds, check whether genemean's seed instability is reduced by later-iterate or checkpoint
  averaging. Its critic gap of about 13 suggests the 20k-update endpoint is not settled.
