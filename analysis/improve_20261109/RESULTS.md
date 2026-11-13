# improve_20261109: results of the 48-fit screen

Status 2026-11-13: the selected development default and improvement pass are
recorded in section 8i and `default_config.json`. The initial screen and seed confirmation are complete
(sections 1-7). The latest resumed analyses are in sections 8f and 8h:
extra-fold primary profile confirmation fails, and the round-5 zero-fraction
critic makes the matched control worse. Development study only: 37 development
donors, repeated inner folds, no sealed donors read, not comparable with
benchmark097 or any sealed result. See `TAKEOVER_20261113.txt` for the handoff.

Initial-screen numbers were recomputed by an independent reviewer from the raw
files. Later sections state their own verification scope. Scripts: `screen_analysis/` (`arms_*`,
`failures_*`, `gap_*`, `stage_*`, and the reviewers' `review_*`). Their outputs
and the full structured reports of all nine analysis agents:
`/ml2/alz/runs/improve_20261109/screen_analysis/` (`workflow_outputs.json`).
Screen scores: `/ml2/alz/runs/improve_20261109/results_screen48/summary.md`.

## 1. Bottom line

- No change makes the generator as accurate as a trivial baseline at predicting
  a held-out donor's expression profile. Held-out pseudobulk RMSE on the
  2,000-gene panel: released model (`base`) 0.689; best single model
  (`genemean`) 0.661; `genemean` two-seed ensemble 0.583; train grand mean with
  no stage information 0.464; ridge 0.461.
- One change helps reliably. Letting the critic see per-gene means
  (`genemean`) halves the per-gene mean-expression error on held-out donors
  (0.214 to 0.114) on both panels, in every fold and seed.
- The stage offset does nothing, donor-PC critic features hurt, and
  sex/age/APOE4 conditioning makes the model memorise individual donors.
- The real High-minus-Not-AD contrast does not transfer between donor sets
  with 4 Not AD donors, for any model or baseline. Donor count limits
  stage-effect accuracy; the generator's own fit limits profile accuracy.

Metrics: `pb_rmse` is the RMSE over genes between a held-out donor's observed
pseudobulk log1p(CP10k) profile and the model's mean generated profile.
`meanlog_rmse_gene` is the RMSE over genes of the donor-averaged error in mean
per-cell log expression. `pc_energy` and `cell_energy` are energy distances
between generated and real donors, and cells, in the fold's 50-PC space. Lower
is better for all four. "12/12" counts within-fold comparisons of each arm fit
against each base fit (4 per fold, 3 folds); it shows consistency, not
significance.

## 2. The six requested changes

| Change | What was run | Held-out result | Verdict |
| --- | --- | --- | --- |
| More donors | Metadata inventory (`DATA_EXPANSION.md`) and a power analysis; no training | Going from 4 to 20 Not AD donors raises the projected replication of the stage contrast from about 0.32 to about 0.6; doubling cells per donor raises it only to 0.34 | Most valuable step for stage effects; needs external data and the owner's decision |
| Stage-by-gene offset (`offset`) | 6 fits | `pb_rmse` 0.690 vs 0.689; every metric 3-7/12 | No effect; drop in this form |
| Gene-level critic (`genemean`) | 6 fits per panel | `meanlog_rmse_gene` 0.214 to 0.114 (2,000 genes) and 0.190 to 0.088 (wide), 12/12; `pb_rmse` 0.661 vs 0.689 single seed (8/12, not robust), 0.583 vs 0.655 as two-seed ensembles (3/3 folds) | The only improvement; still behind the baselines on `pb_rmse` |
| Donor-PC critic (`donorpc`) | 6 fits | `pb_rmse` 0.793 (+0.104, 1/12) | Worse; drop in this form |
| Sex/age/APOE4 conditioning (`covars`, and with `genemean`+`offset`) | 6 + 6 fits | `pb_rmse` 0.927 and 0.974 (0/12); coverage of held-out donors 0.28 vs 0.60 | Harmful as built |
| Gene-level cross-fitted metrics | Evaluator with six baselines | All primary metrics reproduce to 1e-8 with independent code | Works; one metric cannot rank arms (section 5) |
| Wider panel (5,028 genes) | `base` and `genemean`, 6 fits each | On the 2,000 shared genes, renormalised: `base` 0.663 vs 0.689; `genemean` 0.643 vs 0.661 (7/12) | No clear accuracy gain; choose on risk-gene coverage and cost (about 2.5x per update) |

Details:

- **`genemean` has costs.** Single-seed `pc_energy` rises by 0.20 and
  `cell_energy` by 0.15 on the 2,000-gene panel (0/12). Mixing two seeds at a
  matched number of generated donors about halves that cost; it reverses only
  on the wide panel. Its seed-to-seed variance is about twice `base`'s, and its
  critic gap stays near 13 logits against 4.5, so it may not have converged.
  On the 2,000-gene panel it keeps only about 25-35% of the strongest real
  stage effects (GPNMB 0.27 vs 1.06 in the training data).
- **Seed ensembling is a real gain.** Seed-to-seed differences in predictions
  are 8x (`base`) to 18x (`genemean`) the Monte Carlo noise, so averaging seeds
  removes model variance, not sampling noise.
- **Why covariate conditioning fails.** 22-24 of the 24-25 training donors have
  a unique (sex, APOE4, age) tuple, so the covariates identify the donor. The
  generator routes 98-99% of its donor-layer variance through them, the latent
  stops mattering, and it reproduces training donors (identification rate 0.85
  vs 0.15 for `base`). A held-out donor gets the profile of the training donor
  with the nearest covariates. No code bug: training and generation standardise
  covariates identically. Reusing the same fits with covariates drawn from a
  random same-stage training donor gives `pb_rmse` 0.657 (`covars`) and 0.598
  (combined arm), so the damage is from point conditioning.
- **Why `offset` does nothing.** The table does move (RMS 0.08-0.17 per stage,
  correlated across seeds), but what it learns correlates only -0.07 to 0.60
  with the real stage log-ratio. It duplicates the existing stage conditioning.
- **Why `donorpc` hurts.** Its whitened 16-feature block takes about 75% of the
  critic's first-layer input variance. This is a scaling problem of this
  implementation, not evidence against the idea.

## 3. What limits accuracy now

- **The generators lose on their own training donors too.** Train-donor
  `pb_rmse` is 0.650 (`base`) and 0.627 (`genemean`) against 0.458 for the
  grand mean. This is a fit problem, not only generalisation.
- **Two error sources of similar size** (held-out squared error, 2,000 genes):
  `base` 0.488 = 0.183 shared by all donors + 0.305 donor-specific; `genemean`
  ensemble 0.346 = 0.076 + 0.270; grand mean 0.220 = 0.022 + 0.197.
  - A per-gene level bias. For `base` it is reproducible across seeds
    (Pearson 0.79-0.85), so it is structural. `genemean` removes most of it.
  - Stage differences that are 3-8x too large, even on training donors, and
    that correlate about 0 with held-out stage differences.
- **A train-only per-gene recalibration**, applied offline, closes 55-57% of
  `base`'s gap to the grand mean (0.689 to 0.563) and 37-40% of the `genemean`
  ensemble's (0.583 to 0.536). Shrinking the predicted stage differences to
  zero recovers the grand mean (0.462).
- **Sampling noise and depth are negligible** (about 2% of squared error and
  about 0.01).
- **Stage signal.** The whole-panel High-minus-Not-AD contrast has
  train-vs-held-out Spearman 0.114 (permutation p 0.18) and split-half
  reliability 0.135 (p 0.15): not distinguishable from no signal. Technical
  noise is only 13-15% of the between-donor variance within a stage, so more
  cells per donor barely help; more Not AD donors do.
- **A narrower signal exists.** Known disease-associated microglia genes move
  in the expected direction more than expression-matched random genes
  (p 0.0002), also after age/sex/APOE4 adjustment (p 0.002). The gene list was
  not frozen in the protocol beforehand, and all 4 Not AD donors are
  APOE4-negative, so this is a lead to pre-register, not a result.

## 4. What did not hold up

Claims made by an analyst and corrected by the reviewer:

- "Even infinitely many seeds would not bring `genemean` to the baselines on
  `pb_rmse`": undetermined; the projection rests on one seed pair per fold.
  The confirmation run tests it directly.
- "96% of the bias lies outside the critic's 50-PC span, so it is a blind
  spot": the in-span share equals a gene-permutation null, so this split is not
  evidence of a blind spot. (The direct evidence is that `genemean` removes the
  bias.)
- "A perfect model would reach only 0.15 held-out contrast Spearman": about
  0.33. The metric still cannot rank arms.
- "No stage-aware predictor beats the grand mean": ridge does, by 0.001-0.004,
  which is negligible.
- "`genemean`'s per-gene error reaches baseline level with 3-4 seeds": against
  real-donor resampling (0.079), the only baseline for which that metric is
  meaningful, it does so in at most 2 of 3 folds.

## 5. Study code: issues found

- No bug that changes a verdict. Two reviewers reproduced every deterministic
  metric on all 48 fits and 24 ensembles.
- `trainer.py` was edited once after training: 18 fits (`offset`, `donorpc`,
  combined arm) failed at sample generation on a CPU/GPU device mismatch in the
  checkpoint comparison (`_same`). The fix moves both tensors to CPU; only
  generation was rerun. All 48 trainings used the pre-edit code, and a fresh
  CPU regeneration matches the stored samples within Monte Carlo error.
- The per-fold held-out contrast Spearman has a null SD of 0.11-0.23 with 1-2
  held-out Not AD donors. Keep it as a descriptive number only.
- PC-space metrics use a 64-cell subsample per real donor; independent redraws
  move a single fit's `pc_energy` by up to 0.09. Do not read differences under
  about 0.05.
- `summary.md` shows the two panels side by side; they are comparable only on
  the shared-gene renormalised subsets.

## 6. Next steps

GPU time only:

1. Confirm `genemean` against `base` with seeds 2-5 on all three folds
   (24 fits, running). Gives six-seed ensembles and a direct test of whether a
   many-seed `genemean` ensemble reaches the baselines.
2. Add train-only recalibrated metrics and covariate-marginalised generation
   to the evaluator, and freeze in `PROTOCOL.md` a disease-associated gene list
   with directions and a pooled cross-fitted donor score before any new screen.
3. New arms, each 3 folds x 2 seeds, all hypotheses: (a) shrink or regularise
   the generator's stage effect, aimed at the 3-8x inflation; (b) `genemean`
   plus a pseudobulk (linear-in-counts) critic block, aimed at the remaining
   level bias; (c) covariates with dropout or coarse bins; (d) late-iterate or
   checkpoint averaging for `genemean`.
4. More donor splits within the 37 donors, since three overlapping folds carry
   the fold-by-arm evidence.

Owner decisions:

1. External Not AD donors. ROSMAP (about 189 donors without pathologic AD) or
   PsychAD need a controlled-access data use certificate through the AD
   Knowledge Portal. Neither publishes Thal phase openly, so mapping to the
   four ADNC levels is inexact.
2. Reserved SEA-AD donors. Releasing the 20 sealed donors after the sealed
   evaluation adds about 2 Not AD donors. Releasing the other 25 adds 3 more
   and breaks written commitments; not recommended for this purpose.
3. Success criterion. With 37 donors the realistic held-out `pb_rmse` target
   is tying the grand mean, not beating it.

## 7. Confirmation run (done 2026-11-10 18:57 UTC)

24 more fits: `base` and `genemean`, seeds 2-5, three folds, 2,000-gene panel;
no failures. With the screen's seeds that is six seeds per arm per fold.
Scores from the round-1 evaluator
(`/ml2/alz/runs/improve_20261109/results/summary.md`; not yet re-derived by an
independent reviewer). Held-out donors, fold-balanced mean ± SE over folds:

| | `pb_rmse` | `meanlog_rmse_gene` | `pc_energy` | 80% coverage |
| --- | ---: | ---: | ---: | ---: |
| `base`, single seed (18 fits) | 0.687 ± 0.017 | 0.214 ± 0.007 | 2.251 | 0.597 |
| `genemean`, single seed (18 fits) | 0.666 ± 0.024 | 0.130 ± 0.012 | 2.443 | 0.598 |
| `base`, six-seed ensemble | 0.629 ± 0.018 | 0.209 ± 0.007 | 2.138 | 0.663 |
| `genemean`, six-seed ensemble | 0.526 ± 0.014 | 0.089 ± 0.010 | 2.031 | 0.772 |
| grand mean (no stage) | 0.464 ± 0.009 | n/a | n/a | n/a |
| ridge | 0.461 ± 0.009 | n/a | n/a | n/a |
| real-donor resampling | 0.506 ± 0.007 | 0.079 ± 0.001 | 2.091 | 0.698 |

- The `genemean` gain holds with six seeds. Per-gene error is lower in every
  fold; the single-seed value is 0.130 rather than the screen's 0.114, so the
  first two seeds were somewhat lucky.
- A six-seed `genemean` ensemble does not reach the baselines on `pb_rmse`:
  0.526 against 0.461-0.464 (worse in 3/3 folds). The earlier open question
  ("could many seeds reach ridge?") is answered for six seeds: no.
- Ensembled, `genemean` no longer costs PC-space fit: `pc_energy` 2.031 against
  2.138 for the `base` ensemble (better in 3/3 folds) and 2.091 for real-donor
  resampling, with 77% coverage of held-out donors. Single seeds still pay the
  cost (2.443 vs 2.251).
- Ensembling helps `genemean` (0.666 to 0.526) far more than `base`
  (0.687 to 0.629), consistent with its larger seed-to-seed variance.

## 8. Round 2: a fix for each failure

Rules, fix list and interfaces: `r2/PROTOCOL_R2.md`. Code: `r2/` (round-1 code
untouched; `base` and `genemean` are bitwise equal to round 1 under the
round-2 trainer). Commands: `r2/README_R2.txt`. Full reports of the twelve
build, review and repair agents:
`/ml2/alz/runs/improve_20261109/smoke_r2/workflow_outputs_r2_build.json`.
Two adversarial reviewers found no blocker; four major and seven minor
findings were all confirmed and repaired.

Every round-2 choice was made after seeing round-1 held-out results on the
same 37 donors, so round-2 results are selection-conditioned development
evidence. Confirmation needs donors used in neither round.

### 8a. Results that needed no new training (round-1 fits, reviewer-recomputed)

Held-out `pb_rmse`, 2,000-gene panel. References: grand mean 0.464; train mean
of observed profiles 0.462; ridge 0.461.

| Fix | Result | Verdict |
| --- | --- | --- |
| Covariates averaged out at generation (`margcov`) | `covars` 0.927 to 0.654; combined arm 0.974 to 0.592; `base` is 0.689 | Removes the memorisation damage; still far from the references |
| Stage-blind generation (`condmix@1`: every generated donor gets a stage drawn at the train frequencies) | `base` 0.687 to 0.618; `genemean` 0.666 to 0.570 | Better, but only by discarding stage information; not a stage fix |
| Half-shrunk stage (`condmix@0.5`) | `base` 0.638; `genemean` 0.591 | Partial gain |
| Train-only per-gene recalibration | `base` 0.682 to 0.558; `genemean` 0.663 to 0.580 | Closes about half the gap |
| Recalibration plus fitted stage shrinkage | 0.463-0.468 for every arm | Ties the references, by shrinking the generator's stage term to 12-49% |
| Seed ensembling on top of that | Gets slightly worse with more seeds (0.463 at one seed to 0.468 at six) | Averaging seeds lets more of the wrong stage term through |

No calibrated generator beats the train mean of observed profiles (0.462)
like-for-like; the best is 0.0016 worse. A calibrated generator is, in effect,
the train mean plus a small shrunken stage term.

Diagnoses that changed the picture:

- **Stage inflation is mostly seed noise.** On the whole panel the real train
  stage deviation is no larger than its own donor-sampling noise. The
  generators' stage deviations are 3-6x larger. For `genemean` 89-95% of that
  is seed-specific and averages out; for `base` 53-66% is, and the rest is a
  reproducible exaggeration.
- **`genemean` does not settle.** Its critic gap plateaus near 13 logits within
  4,000 updates. The gene block carries 93% of the critic's first-layer
  variance and the critic cuts its weights on the released PC features to a
  quarter, which explains the worse PC fit. Over the last few thousand updates
  the generated per-gene means keep moving by about as much as two seeds
  differ. Averaging late states is therefore expected to help `genemean` and
  not `base`.
- **`donorpc` was misdiagnosed in section 2.** Its block had a smaller input
  variance share (0.61) than `genemean`'s (0.84). The damage came from
  whitening, which gives 16 donor-identity directions equal weight.
- **`genemean`'s per-gene gain is mostly a level correction.** After the
  train-only recalibration, `base` and `genemean` have the same per-gene error
  (about 0.07, the same as real-donor resampling).
- **A pseudobulk error remains that `genemean` does not touch**, about 10x the
  real-versus-real level, largest in highly expressed genes. This motivates
  the `pbulk` critic block.
- **The disease-gene direction score cannot be a headline metric on the
  2,000-gene panel**: only 3-5 of the 27 frozen genes are present, all in the
  "up" set. It stays descriptive there.

### 8b. Round-2 screen (finished 2026-11-11 10:43 UTC)

72 fits (twelve arm sets x three folds x two seeds, 20,000 updates) and 258
generation-mode files; no failures. Scored with the round-2 evaluator
(`/ml2/alz/runs/improve_20261109/results_r2/summary.txt`, `aggregate.csv`);
the retrained `genemean` reproduces its round-1 scores exactly. These numbers
are the evaluator's output and have not been through a separate adversarial
analysis. Held-out donors, mean of 6 fits (3 folds x 2 seeds); "vs base" is
the paired difference in `pb_rmse` with the number of folds in which the arm
is better. References: ridge 0.461, grand mean 0.464, resampling 0.506.

| Fix | Arm | `pb_rmse` | vs `base` (0.690) | Late-snapshot average | Verdict |
| --- | --- | ---: | --- | ---: | --- |
| F5 level bias | `pbulk` | 0.612 | better by 0.076 (3/3) | 0.546 | Works; best single change so far |
| F6 `genemean` stability | `genemean` + snapshots | 0.661 | better by 0.028 (2/3) | 0.569 | Snapshot averaging works |
| F6 | `genemean@0.5` | 0.656 | better by 0.032 (2/3) | 0.569 | Same as `genemean` |
| F6 | `genemean@0.25` | 0.672 | better by 0.017 (2/3) | 0.558 | Same as `genemean` |
| F6 | `gmlite@64` | 0.633 | better by 0.056 (3/3) | 0.593 | Better plain, less gain from snapshots |
| F3 stage inflation | `stagedrop@0.5` | 0.670 | better by 0.019 (3/3) | 0.672 | Small gain |
| F1 covariates | `covdrop@0.5` (stage-only generation) | 0.696 | worse by 0.007 (0/3) | 0.682 | No gain; with own covariates 0.95 |
| F4 donor PCs | `donorpc2` | 0.717 | worse by 0.028 (0/3) | 0.707 | Better than `donorpc` (0.793), still worse than `base` |
| F5 | `genemean,pbulk` | 0.835 | worse by 0.146 (0/3) | 0.640 | Two gene-wide blocks together fail |
| F2 stage path | `stageanchor` | 0.836 | worse by 0.147 (1/3) | 0.807 | Fails |
| F2 | `genemean,stageanchor` | 0.841 | worse by 0.152 (0/3) | 0.749 | Fails |
| F1 | `covbin` | 0.886 | worse by 0.197 (0/3) | 0.855 | Fails (memorises again) |

- **Best results.** Single fit: `pbulk` with late-snapshot averaging 0.546.
  Two-seed ensemble of that: 0.511. Generated stage-blind (`condmix@1`), the
  `pbulk` two-seed ensemble reaches 0.493, below real-donor resampling (0.506)
  but still above the grand mean (0.464).
- **Snapshot averaging removes the PC-space cost of the gene-level blocks.**
  Plain `genemean` is worse than `base` on `pc_energy_rep` by 0.21 (0/3
  folds); with snapshot averaging the difference is 0.009.
- **Distribution fit against real-donor resampling.** The `genemean@0.5`
  two-seed stage-blind ensemble beats resampling on `pc_energy_rep` (1.981 vs
  2.081), `cell_energy_rep` (0.792 vs 0.872) and coverage (0.748 vs 0.701),
  and is behind on per-gene mean (0.090 vs 0.079), nonzero fraction (0.020 vs
  0.017) and per-gene variance (0.280 vs 0.193).
- **With the train-only recalibration** every one of these arms sits at
  0.465-0.47 when generated stage-blind: a tie with the grand mean.

Two more 3-fold splits of the 37 donors (inner folds 3-8) have caches built.

### 8c. Round 3 (in progress)

Owner request 2026-11-11: test training length, the obvious recipe options,
particles and architecture, and keep going until the scores are at the top.
Rules, targets and staged screen: `r3/PROTOCOL_R3.md`. Commands:
`r3/README_R3.txt`. Reports of the eleven build, review and repair agents:
`/ml2/alz/runs/improve_20261109/smoke_r3/workflow_outputs_r3_build.json`
(17 review findings, all reproduced and resolved or escalated).

Answered by the build, from existing fits and code, before any new training
(each checked by a reviewer unless noted):

- **Training longer at a constant learning rate does not help.** The recipe as
  used has constant rates, no EMA, and normalised zero-momentum steps: every
  update moves the weights by a fixed amount whatever the gradient. The
  generator is a random walk with no restoring force (the output layer moves
  about 25% of its norm per 1,000 late updates). Late single-state scores
  fluctuate by 0.005-0.04 with no trend; averaging late states gains
  0.03-0.09. The evidence supports averaging more than annealing.
- **More particles are not needed.** All 512 are used uniformly, but they
  collapse to 4-17 distinguishable donor profiles per stage, and the generator
  uses about 2 of its 8 latent directions. Giving each donor a free latent
  leaves the same gap, so particle count and latent size are not the limit.
- **Generator width, depth and cell-noise size are not the limit.** A
  maximum-likelihood probe of the same architecture reproduces every training
  donor to the noise level, and larger versions only memorise cells. One real
  capacity gain: a donor-dependent dispersion (`g.dispz`), which cuts per-gene
  errors 2-3x in the probe. Whether it survives GAN training is untested.
- **The critic penalty stops binding with gene-wide blocks.** The penalty cap
  scales with the square root of the critic's input width, so every 2,150-wide
  critic sits far below its cap and the critic gap rises to 11-20 logits.
  `t.kappa@0.25` restores the released model's absolute cap. A different
  critic module cannot change this.
- **Floor.** An oracle that knows the true population mean would score
  0.452 ± 0.014 on held-out `pb_rmse`; the grand mean is at 0.464. Stage
  carries no measurable pseudobulk information in these donors. Read out
  through 32 generated donors, an ideal stage-blind generator scores 0.474,
  so the round-3 target is 0.479.
- **Why `stageanchor` failed (round 2).** Implementation: the anchored table
  was the shrunken one, near zero, and the learned gains grew to 5-62 in fold
  0. `covbin` failed by memorisation again (all singleton donors identified).
- **Open data.** GEO GSE268599 (MADRC) has 32 donors with published ADNC
  components, 8 of them Not AD, an adjacent temporal region and the same gene
  axis, with no data use agreement.

Stage 1 of the round-3 screen started 2026-11-11 11:53 UTC: 57 fits (71
units), control `pbulk`, one factor per arm: 60,000 updates (with and without
the gene block), annealing to 0.1 and 0.01, generator and critic learning
rates, penalty strength (`t.reg@4`, `t.kappa@0.25`), 1,024 particles,
`g.dispz` (two step sizes), generator width 256, block-wise critic,
cell-level critic (with and without matched penalty), `nostage`,
`fixedstage`, and control seed 1.

**Stage-1 result (finished 2026-11-12 09:34 UTC; 57 fits, no failures;
`/ml2/alz/runs/improve_20261109/results_r3/leaderboard.txt`; evaluator
output, not yet through an adversarial analysis).** Held-out `pb_rmse`, folds
0-2, seed 0 (control and the combination: seeds 0-1). One seed on three folds
resolves about 0.025.

| Arm (on `pbulk`) | Final state | Late-snapshot average (primary) | + half-shrunk stage | Verdict |
| --- | ---: | ---: | ---: | --- |
| Control `pbulk` | 0.612 | 0.542 | 0.502 | |
| `g.dispz` (donor-dependent dispersion) | 0.560 | 0.504 | 0.488 | Helps, beyond seed noise |
| `t.kappa@0.25` (penalty matched to width) | 0.594 | 0.518 | 0.493 | Helps, at the edge of seed noise |
| `g.dispz` + `t.kappa@0.25` (owner-requested) | 0.557 | 0.516 | 0.499 | No better than either alone |
| `g.dispz@0.1` | 0.564 | 0.515 | 0.494 | Like `g.dispz`, worse PC fit |
| `t.glr@0.5` | 0.584 | 0.528 | 0.509 | Within noise |
| `g.h@256` (wider generator) | 0.609 | 0.529 | 0.501 | Within noise |
| + `genemean`, penalty matched | 0.606 | 0.535 | 0.522 | Within noise; no longer fails |
| `t.dlr@0.5` | 0.683 | 0.557 | 0.508 | No gain |
| `t.anneal@0.1` / `@0.01` | 0.604 / 0.615 | 0.558 / 0.563 | 0.539 / 0.553 | No gain |
| `d.cellwhite` (cell-level critic) | 0.822 | 0.591 | 0.564 | Worse |
| `nostage` | 0.862 | 0.594 | n/a | Worse |
| `d.tower` (block-wise critic) | 0.844 | 0.596 | 0.581 | Worse |
| `t.particles@1024` | 0.704 | 0.608 | 0.563 | Worse |
| `fixedstage` | 0.897 | 0.619 | 0.619 | Worse |
| `t.steps@60000` | 0.854 | 0.704 | 0.654 | Worse |
| `t.reg@4` | 0.875 | 0.729 | 0.691 | Worse |

- **Training length.** Along the 60,000-update `pbulk` runs, single-state
  `pb_rmse` is 0.595 at 10,000 updates, 0.645 at 20,000, 0.758 at 30,000,
  0.862 at 40,000 and 0.908 at 50,000. Without the gene block (`base`) the
  curve is flat (0.71, 0.69, 0.70, 0.71, 0.71). Longer training does not help
  the released model and harms the gene-block model.
- **Two-seed ensembles without external donors**, best read-out (snapshot
  average, half-shrunk stage, bias calibration): control 0.478; combination
  0.474. With external donors (section 8d): 0.471 and 0.470. The external
  donors are worth about 0.005 at this read-out, and about 0.026 at the plain
  snapshot-average read-out (0.542 to 0.516).

### 8d. External donors (GEO GSE268599, MADRC)

Rules: `r4/PROTOCOL_EXT.md`. Commands and caveats: `r4/README_EXT.txt`. Reports
of the seven agents (fetch, three uses, two reviewers, repair; 15 findings,
all reproduced, 13 fixed, 2 escalated):
`/ml2/alz/runs/improve_20261109/smoke_r4/workflow_outputs_ext.json`. Store:
`/ml2/alz/runs/improve_20261109/ext/GSE268599/` (290 MB; raw downloads
deleted). All numbers below were recomputed by a reviewer.

- **Data.** 31 donors with inferior temporal gyrus libraries; ADNC computed
  from the published A, B, C scores: 7 Not AD, 5 Low, 6 Intermediate, 13 High.
  54,015 microglia/PVM nuclei selected by a documented marker pipeline (no
  author cell labels are openly available); same 36,601-gene axis as SEA-AD.
  A classifier trained on SEA-AD microglia calls 94% of the selected cells
  microglia.
- **The cohorts differ a lot.** The per-gene offset between MADRC and SEA-AD
  is 0.67 of the donor-to-donor SD and 2.2x the SEA-AD High-minus-Not-AD
  contrast, concentrated in about 100 genes, mostly ambient astrocyte RNA in
  MADRC (GFAP, AQP4). Depth and cell composition explain little of it.
- **Stage contrasts replicate across cohorts, weakly.** High-minus-Not-AD
  over the 2,000 panel genes: Spearman 0.34-0.37 raw, 0.25-0.27 after
  adjusting for age, sex, APOE4, depth and PVM fraction (permutation p
  0.008-0.043). Split-half reliability of the contrast is 0.31 within MADRC
  against 0.21 within SEA-AD. This is the first evidence in this study of a
  stage signal that transfers between donor sets.
- **External donors help simple predictors only after the offset is
  removed.** On SEA-AD held-out donors, pooling them raw makes every baseline
  worse (grand mean 0.464 to 0.489). With a per-gene cohort offset fitted on
  train donors: ridge 0.461 to 0.455 (3/3 folds), stage means 0.494 to 0.475
  (3/3). Held-out agreement of the High-minus-Not-AD contrast rises from about
  0.10-0.12 to 0.18-0.20 (p about 0.03; about ten correlated variants were
  tried, and each fold holds out 1-2 Not AD donors, so this is suggestive).
  The best predictor, 0.455, is inside the floor estimate (0.452 ± 0.014).
- **Caveats.** MADRC "Not AD" donors are Braak 0-II; SEA-AD's are mostly
  Braak IV-V. MADRC library quality and microglia yield fall with stage; the
  caches cap each MADRC donor at 1,000 cells. Ambient RNA composition shifts
  with stage in MADRC. The GEO disclaimer page could not be read by script
  (reCAPTCHA); NCBI's programmatic interfaces list both series as public. One
  supplementary table is CC BY-NC-ND and stays internal.

**Screen result (finished 2026-11-12 02:56 UTC; 18 fits, no failures;
`/ml2/alz/runs/improve_20261109/results_r4/summary_ext.txt`; evaluator output,
not yet through an adversarial analysis).** SEA-AD held-out donors, 3 folds x
2 seeds, `pb_rmse`:

| Arm | Final state | Late-snapshot average | + half-shrunk stage | + bias calibration |
| --- | ---: | ---: | ---: | ---: |
| `pbulk`, external donors pooled naively | 0.586 | 0.516 | 0.501 | 0.479 |
| `cohortfix,pbulk` | 0.818 | 0.684 | 0.672 | 0.558 |
| `cohortfix,extperm,pbulk` (stage labels shuffled) | 0.812 | 0.678 | 0.663 | 0.567 |
| `pbulk` without external donors (round 2, same folds and seeds) | 0.612 | 0.546 | n/a | n/a |

- **Naive pooling helps the generator**, although it hurt the simple
  predictors: 0.516 against 0.546 with snapshot averaging. With half-shrunk
  stage and bias calibration it reaches 0.479, the round-3 target, with
  `pc_energy_rep` 1.869 and `cell_energy_rep` 0.745 (resampling: 2.081 and
  0.872), coverage 0.796 and per-gene variance error 0.192 (resampling 0.193);
  per-gene mean error is behind (0.101 against 0.079). The paired comparison
  with the round-3 control read out identically waits for stage 1.
- **The fixed cohort offset hurts** (0.684 against 0.516), and it scores the
  same with real or shuffled external stage labels (0.684 against 0.678). By
  the pre-registered rule the external donors' stage information gets no
  credit: the gain is from more donors and cells, not from better stage
  effects. The generators' stage metrics stay near zero.

**Combined model with external donors (owner request 2026-11-12; finished
08:30 UTC; 6 fits).** `g.dispz,pbulk,t.kappa@0.25` on `hvg2000x`: `pbulk`
block, critic penalty matched to the input width, donor-dependent dispersion,
MADRC donors pooled naively. SEA-AD held-out donors, 3 folds x 2 seeds,
`pb_rmse`; single-fit rows from `results_r4/summary_ext.txt`, two-seed
ensembles from `smoke_r4/ens/x_ensembles.py` (my script, calling the
evaluator's own scoring functions; not independently reviewed):

| Read-out | `pbulk` + external | Combined + external |
| --- | ---: | ---: |
| Final state, single fit | 0.586 | 0.523 |
| Late-snapshot average, single fit | 0.516 | 0.512 |
| + half-shrunk stage | 0.501 | 0.503 |
| + bias calibration | 0.479 | 0.477 |
| Two-seed ensemble, final state | 0.538 | 0.502 |
| Two-seed ensemble, snapshot average | 0.493 | 0.501 |
| Two-seed ensemble, + half-shrunk stage + bias calibration | 0.471 | 0.470 |

References on the same folds: grand mean 0.464, ridge 0.461, ideal
stage-blind generator through the same read-out 0.474, resampling 0.506.

- The best read-out of either model, as a two-seed ensemble, scores 0.470:
  below the ideal-generator reference and 0.006 above the grand mean.
- The combination's gain is in the single final state (0.523 against 0.586):
  it makes the model good without snapshot averaging. At the best read-out
  the two models are equal.
- Single-fit distribution fit of the combined model at its best read-out,
  against resampling: `pc_energy_rep` 1.818 (2.081), `cell_energy_rep` 0.725
  (0.872), coverage 0.847 (0.701; nominal 0.80), per-gene variance error
  0.195 (0.193), per-gene mean error 0.093 (0.079), nonzero-fraction error
  0.021 (0.017).

Screen design (started 2026-11-11 15:45 UTC; 18 fits, panel `hvg2000x`: same genes,
space and held-out donors as `hvg2000`, MADRC donors added to training):
`pbulk` pooled naively; `cohortfix,pbulk` (fixed, non-learned per-gene cohort
offset); and `cohortfix,extperm,pbulk` (the same with the external donors'
stage labels shuffled, which separates "more donors" from "more stage
information"). Pre-registered reading: credit MADRC stage information only if
`cohortfix` beats the shuffled control by more than 2 seed-noise SE.

### 8e. Adversarial verification of 8c and 8d (2026-11-12) - read this first

Two independent verifiers recomputed the round-3 and external-donor results
from the raw generated files (reports:
`/ml2/alz/runs/improve_20261109/verify_r3/workflow_outputs_verify.json`;
scripts under `verify_r3/` and `verify_ext/`). Every single-fit number in 8c
and 8d reproduced to four decimals. Several readings did not survive, and
they supersede the text above:

- **No generator beats the ideal stage-blind generator like-for-like.** The
  ensemble values 0.470-0.478 pooled 64 generated donors per target and used
  bias calibration; the 0.474 reference used 32 and none. Matched: two-seed
  ensembles at 32 donors score 0.476-0.488; the calibrated ideal generator
  scores 0.458-0.460. The best generator is about 0.012 above the calibrated
  ideal generator and 0.006 above the grand mean (0.464). No read-out beats
  ridge (0.461) or the direct train mean (0.462). The "T1 pass" on the
  leaderboard has zero margin and needs 128 pooled donors.
- **Bias calibration pulls the generator onto the train-mean baseline.** Its
  target is the train mean of log pseudobulk profiles (0.462 as a predictor),
  so its gain is largely mechanical.
- **No model predicts stage effects on held-out donors.** Cross-fitted scores
  are at most 0.047 (z at most 1.6); about 2 of 40 tests reach p 0.05, the
  chance rate. For every arm, stage-blind generation scores as well as or
  better than half-shrunk, which beats full stage conditioning (control:
  0.490 / 0.502 / 0.542). The best model is in effect a stage-blind
  population generator.
- **`g.dispz` is not robust.** Its 0.038 gain at the snapshot-average
  read-out comes from two of three folds (fold 2 reverses), is 2.6 SE by seed
  noise but 1.3 by fold scatter, and shrinks to 0.013 half-shrunk and 0.003
  stage-blind. `t.kappa@0.25` is null at those read-outs. Together they
  interfere (interaction +0.034). Their real effect is a more stable final
  state (seed SD 0.023 against 0.067).
- **Inconclusive, not shown harmful:** `t.glr@0.5`, `g.h@256`, `t.dlr@0.5`
  and both annealing arms. On its declared read-out (final state),
  `t.anneal@0.1` is 0.040 better than the control's final state (2 of 3
  folds, not significant).
- **External donors.** Matched comparison (identical snapshot sets): pooling
  improves the snapshot-average read-out by 0.025 ± 0.012, the half-shrunk
  read-out by 0.002, the bias-calibrated one by 0.007 ± 0.006, and makes the
  stage-blind read-out worse by 0.007 (0 of 6 fits better). The gain is
  entirely a shrinkage of the generator's inflated stage term, which
  half-shrinking already provides; the donor-specific error is unchanged. It
  cannot be credited to "more donors and cells": the shuffled-label control
  existed only for the `cohortfix` arms, and that test had no power (paired
  SE 0.038).
- **Pooling contaminates the generated cells.** Trained with MADRC donors,
  generated SEA-AD cells carry MADRC's ambient astrocyte signal: GFAP +0.88
  and AQP4 +0.65 log units above real held-out donors before bias calibration
  (controls: +0.08, +0.03). Calibration fixes the pseudobulk level but not
  the per-cell distribution (GFAP detection 0.077 against 0.062 real).
  Verifier's recommendation: do not make naive pooling part of the final
  recipe on this evidence.
- **Distribution fit.** The best units pass 3-4 of the 6 metrics against
  real-donor resampling (PC energy, cell energy, coverage, sometimes per-gene
  variance); none passes per-gene mean or nonzero fraction. Coverage above
  0.80 with external donors indicates over-dispersion, not better fit.
- **Untested lead.** Along the 60,000-update runs the 10,000-update state
  (0.595) is better than the 20,000-update final state (0.645); shorter
  training with averaging was never run as an arm.

Where this leaves the model: single-fit held-out `pb_rmse` went from 0.687
(released) to 0.542 (`pbulk`, snapshot average) and to about 0.49 generated
stage-blind; a two-seed ensemble reaches about 0.48 at matched read-out. The
floor for a stage-blind generator read out this way is 0.474. The remaining
measurable gaps are per-gene mean and nonzero-fraction error, and stage
effects, which no model here learns.

### 8f. Extra-fold confirmation (primary profile analysis recovered 2026-11-13)

Pre-declared in `r4/PROTOCOL_CONFIRM.md` before any fit existed: a 2 x 2 of
donor-dependent dispersion and external donors on inner folds 3-8, two seeds,
54 fits, with read-outs and decision rule fixed. Both training halves completed
successfully on 2026-11-12; Claude reached its weekly limit before analysis.
The resumed profile analysis uses `r5/resume_confirmation_profiles.py` and the
unchanged factor-analysis functions from `r4/confirm_analysis.py`. Outputs:
`runs/improve_20261109/results_r5/confirmation_profiles/analysis.{txt,json}`.

All 12 primary 2x2 blocks are present (48 fits: six folds, two seeds, four
cells). At the pre-declared `snapens` read-out, changes in held-out `pb_rmse`:

| Factor | Folds 3-5 | Folds 6-8 | Pooled change | Seed-noise SE | Primary verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Donor-dependent dispersion | -0.0024 | -0.0045 | -0.0035 | 0.0058 | Not confirmed |
| External donors pooled | -0.0044 | -0.0014 | -0.0029 | 0.0055 | Not confirmed |

Neither effect exceeds the pre-declared two-SE threshold. Fold-scatter SEs
are 0.0057 and 0.0052 respectively. These are repeated splits of the same
37 donors, not independent new biological observations.

Secondary read-outs are mixed: external pooling improves calibrated
`biascal@stagehalf` profile error by 0.0062 (seed-noise SE 0.0024), but worsens
`stagehalf` by 0.0119 and `stageblind` by 0.0165. Secondary effects do not
change the primary verdict or resolve the previously verified contamination.
No factor is adopted.

Verification: the resumed profile calculation reproduces 57 existing
evaluator scores to absolute tolerance 1e-12. It reads only existing synthetic
SEA-AD summaries and development caches; no external or sealed donor counts.
It is a profile-only recovery: the full extra-fold distribution analysis and
external-baseline extension are still outstanding, and no independent agent
review was performed during this takeover.

### 8g. Round 5: short training with averaging; the per-gene mean and nonzero gaps

Owner decision 2026-11-12 ("yes do 1 and 2"). Rules: `r5/PROTOCOL_R5.md`;
commands: `r5/README_R5.txt`; reports of the five agents (three tasks, one
adversarial reviewer, one repair; seven findings, all reproduced and fixed):
`/ml2/alz/runs/improve_20261109/smoke_r5/workflow_outputs_r5.json`. Numbers
below were recomputed by the reviewer.

**1. Short training with averaging: no.**

- The round-3 lead (10,000-update state 0.595 against 20,000-update 0.645) was
  single-state noise: single-state `pb_rmse` is flat from 5,000 to 20,000
  updates (10k minus 20k: -0.005 ± 0.040).
- The pre-declared selection picked the EMA at 5,000 updates (decay 0.9995):
  0.506 against 0.543 for the late-snapshot average. Under the pre-declared
  rule it does not replace it: control seed 1 (-0.045, 1.7 SE) and extra folds
  6-8 (-0.021, 1.4 SE) do not clear 2 SE.
- Its gain is shrinkage toward the starting generator, not a better model. The
  read-out still carries 8% of the initial weights; it halves the generated
  between-donor spread (0.53 of real against 0.99); coverage falls from 0.61
  to 0.50; and the gain disappears at the stage-blind read-outs.
- **The untrained generator scores `pb_rmse` 0.499**, better than the trained
  control at its primary read-out (0.542), and better on `cell_energy_rep`
  too. An untrained generator emits the pooled mean profile with little donor
  spread. So `pb_rmse`, and to a degree the energy metrics, reward collapse
  toward the average; none of them can be read without the spread and
  coverage columns next to it. This applies to every round of this study,
  including the stage-blind and calibrated read-outs praised earlier.

**2. Per-gene mean and nonzero gaps: mostly not a modelling problem.**

- Generated counts are over-dispersed with too many zeros at matched mean and
  depth. A plain per-gene negative binomial fitted to real cells reproduces
  the real zero fraction well, so no zero-inflation term is needed; the
  generator's fitted dispersions are off.
- `zcal` (train-only per-gene dispersion calibration, jointly with the mean)
  moves the control from 0.095 to 0.084 on per-gene mean error and from 0.021
  to 0.018 on nonzero fraction (resampling: 0.079, 0.017), at a cost of 0.040
  in `pc_energy_rep`. It is a pull onto train statistics: predicting each
  held-out donor with the train mean of those same statistics scores 0.073
  and 0.016, better than `zcal` and better than resampling.
- After a per-gene constant fitted on train donors, every generator is within
  0.002 (mean) and 0.0005 (nonzero) of resampling. The rest of the error, for
  generators and resampling alike, is donor-to-donor variation. There is
  little left here for a model change to win.
- `pzero` (a critic block with the per-gene zero fraction, penalty matched to
  the width) tests whether training can remove the per-gene constant without
  calibration. Block A (21 fits, with penalty-matched controls) started
  2026-11-12 20:23 UTC and finished 2026-11-13 02:31 UTC, 21/21 successful.
  Its resumed evaluation follows in 8h.

### 8h. Round-5 Block A: zero-fraction critic rejected (2026-11-13)

Resumed from Claude's completed fits, with no new training. Commands are in
`r5/resume_evaluate_pzero.sh`; outputs in
`runs/improve_20261109/results_r5/pzero/` and
`runs/improve_20261109/logs_r5/pzero_compare.txt`. Every comparison uses the
same fold, seed and generation mode. Six pairs: three folds, two seeds.

Primary block comparison at the late-snapshot read-out (`snapens`):

| Held-out metric | Matched control `pbulk,t.kappa@0.19` | Add `pzero` | Change | Seed-noise SE |
| --- | ---: | ---: | ---: | ---: |
| Profile RMSE | 0.5211 | 0.6854 | +0.1643 | 0.0170 |
| Per-gene mean-log error | 0.1023 | 0.1648 | +0.0625 | 0.0030 |
| Per-gene nonzero-fraction error | 0.0220 | 0.0353 | +0.0134 | 0.0005 |
| PC energy | 2.2565 | 2.4933 | +0.2368 | 0.0257 |
| Cell energy | 0.9419 | 1.1898 | +0.2478 | 0.0155 |

Lower is better in all five rows. All six fits are worse on the first three
metrics; all three fold averages worsen on all five. PC spread is effectively
unchanged (1.2245 to 1.2193), while central-80% coverage rises from 0.6021 to
0.6352. Those accompanying columns do not rescue the failed targets and guards.

Calibration does not repair it: at `biascal@stagehalf`, profile RMSE is
0.5253 against 0.4827, mean-log error 0.1152 against 0.0888, and nonzero error
0.0267 against 0.0196. Replacing the pseudobulk block with `pzero` also fails:
`snapens` profile RMSE 0.6858 against 0.5252 for `pbulk,t.kappa@0.25`.
The penalty-only control comparison changes profile RMSE by just -0.0041
(SE 0.0026), so the block's damage is not explained by the penalty change.

Decision: reject this implementation of `pzero`; retain the existing recipe.
The optional 12-fit Block B has not been started. This is evidence about the
tested block/scaling/recipe, not a general rejection of zero-fraction features.

Verification during takeover: recomputed profile RMSE, mean-log error and
nonzero error directly from the saved generation arrays for all 12 fits of
the matched comparison; all 36 values agree with evaluator outputs to 1e-12.
The standard evaluator also scores stage-half, stage-blind and calibrated
read-outs, and seed ensembles at both matched M=32 and pooled M=64. These
takeover results have not had a separate independent-agent review.

### 8i. Committed development default and count-shape pass (2026-11-13)

Owner request: improve the model and commit the best supported configuration
as the default. The fixed test and adoption rule are in
`PROTOCOL_DEFAULT_PASS_20261113.txt`. No new optimizer updates or sealed donor
counts were used. This pass was not reviewed by a separate agent.

**Selected default:** `pbulk`, 20,000 updates, released ParticleGAN 0.9.7,
late-snapshot read-out with half-shrunk stage conditioning and train-only
output-bias calibration (`biascal@stagehalf`). The configuration is in
`default_config.json`; `scripts/microglia.py` uses it for both training and
generation. Raw final-state generation is no longer the served output of
this entry point. Historical research runners and the published report
retain their original behavior and results.

The default was checked on all six additional folds, two seeds each. Against
the same `pbulk` fits at uncalibrated `stagehalf`, means over those 12 pairs:

| Metric | Uncalibrated stagehalf | Selected default |
| --- | ---: | ---: |
| Profile RMSE | 0.50242 | 0.48548 |
| Mean-log gene error | 0.12232 | 0.10623 |
| Nonzero-fraction gene error | 0.02633 | 0.02321 |
| Log-variance gene error | 0.24644 | 0.21067 |
| PC energy | 2.11332 | 2.03264 |
| Cell energy | 0.87325 | 0.81403 |
| PC spread ratio | 1.28735 | 1.26386 |
| Central-80% coverage | 0.65109 | 0.66137 |

The first six errors improve in every additional fold mean; PC spread moves
closer to one in every fold, and coverage moves closer to 0.8 in four of six.
These folds reuse the same 37 development donors. This is a consistency check,
not independent biological confirmation. Complete original/additional-fold
comparisons and paired SEs are in `default_config_evidence.json`.

**New candidate:** reduce spurious within-donor variation in expected counts
without changing each generated donor's expected gene totals or each cell's
expected library. With fixed alpha=0.5, apply the existing joint train-only
bias/dispersion calibration afterwards (`cellshape@0.5:biascal:stagehalf`).
This is a K=256 bag read-out; it does not define iid single-cell generation.

Original folds 0-2, two seeds, six paired fits:

| Metric | Existing zcal reference | Count-shape candidate |
| --- | ---: | ---: |
| Profile RMSE | 0.486234 | 0.486555 |
| Mean-log gene error | 0.083726 | 0.076305 |
| Nonzero-fraction gene error | 0.017744 | 0.016084 |
| Log-variance gene error | 0.184049 | 0.177917 |
| PC energy | 2.061355 | 1.904707 |
| Cell energy | 0.825430 | 0.733089 |
| PC spread ratio | 1.221590 | 1.049159 |
| Central-80% coverage | 0.668501 | 0.701037 |

Both count-shape targets improve beyond two paired seed-noise SE in all three
fold means. It also beats the uncalibrated resampling reference on mean-log
and nonzero errors (0.07903 and 0.01717), but the direct train-statistic
predictor remains better (0.07279 and 0.01604). All distribution guards pass.
The profile guard fails: +0.0003215 with SE 0.000037. Its practical size is
small (about 0.07%), but the pre-declared rule is retained. The candidate is
therefore explicitly opt-in (`--profile count-shape`), without extra-fold
promotion or a claim that it is the validated default.

Verification: expected gene mass and cell libraries conserved in numerical
tests; all six neutral alpha=0 read-outs reproduce the historical zcal
headline metrics exactly; the new default entry point reproduces all 14
arrays of the stored selected read-out bitwise. Twelve entry-point tests
pass. Additional-fold default evaluation: 24 units, zero errors. Full source,
config, environment versions and compact evidence are committed; raw cells,
model checkpoints, generated tensors and machine credentials are excluded.

### 8j. Interval construction and architecture audit (2026-11-13)

The combined coverage metric hides different behaviors: on all 18 default
fits, expression-pattern mean coordinates cover 82.46%, cellular-spread
coordinates 51.14%, and their average 66.80%. Real SDs predominantly miss
below the model's lower bound. These are marginal coordinate ranges, not
clinical patient intervals or simultaneous coverage of whole profiles.

Finite M=32 also makes interpolated sample percentile intervals undercover
in a perfect-model null simulation. A conservative order-statistic diagnostic
raises combined coverage to 73.32% at a cost of about 16.7% more width; it
does not fix the cellular-spread model. Historical scores remain unchanged.

The same real cells show a held-out/train SD ratio of 0.822 in training-selected
PC directions versus 1.001 in fixed random directions with train-only scaling.
This implicates feature-space generalization, rather than establishing a
simple shortage of network capacity. Frozen decoder interventions further
support separating network mean heterogeneity from count dispersion:
halving the former improves PC-SD coverage from 55.6% to 64.6% on six original-fold
fits, whereas multiplying NB shape by four reduces it to 38.2%.

The full [architecture review](INTERVAL_ARCHITECTURE_REVIEW_20261113.txt),
[compact evidence](interval_architecture_evidence_20261113.json) and
[diagnostic figure](interval_architecture_20261113.png) specify a matched
observation-space ablation, a subsequent donor/cell-state decoder experiment,
and the donor encoder needed for real-patient feature vectors. No training
or default change was made. All 18 historical coverage values reproduce;
synthetic finite-M and held-out-exclusion checks pass.

## 9. Limits

Three inner folds whose training sets overlap by about half; two seeds; 37
donors with 4 Not AD (2-3 per training split, 1-2 held out); Not AD donors are
all APOE4-negative and 3 of 4 are male; the 2,000-gene set differs between
folds (1,286 genes shared by all three); one hyperparameter setting per arm,
so a failed arm is a failed implementation, not a refuted idea.
