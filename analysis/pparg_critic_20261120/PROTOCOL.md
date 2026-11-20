# PPARG critic-observation test: protocol (declared before any score)

Declared 2026-11-20, before any variant fit was trained and before any held-out
score of either arm existed. Two native control fits had been launched
(training only, no generation, no scoring) when this file was committed.

## Question

Does showing the EMPF critic a small block of off-PC gene residuals improve
held-out mean recovery for PPARG and the selected genes? The ParticleGAN update
rule stays fixed. We also check whether coverage, interval or profile metrics
regress.

## Premise check (training donors only, recorded before outcomes)

The README's PPARG diagnostic used the frozen 0.9.7 models. It found about 99%
of the Not-AD mean gap outside the 50-PC span. The current default differs in
two ways: it trains with native EMPF, and it uses the `pbulk` arm. That arm
already gives the critic every gene's per-bag pseudobulk log1p CP10k, PPARG
included, for 2,150 inputs in total.

`select_genes.py` reads the raw `snapens` readout of the retained default fit
(fold0/seed0, 20k updates). It uses only the training targets of that readout,
together with the fold-0 TRAINING counts. On these training donors, the
generated Not-AD PPARG mean_log is 1.094 and the observed value is 0.889. That
is a small overshoot, z-gap +0.105, with an off-PC share of 17%. It is not the
0.9.7 undershoot. PPARG ranks 501 of 2,000 genes by off-PC gap score. The
outcome below therefore offers less room to improve PPARG than the README
diagnostic implied. We keep PPARG in the outcome because it was declared before
this check.

## Arms (fold 0, seeds 0 and 1, 10,000 updates each; matched)

- **control**: `pbulk`, the native EMPF default arm. Recipe, hyperparameters,
  streams, `lazy_k=1` and the 2,150-wide critic input are unchanged. The fit is
  trained through `host_pparg.py`, which repeats `empf_comparison_20261117/host.py`'s
  native derivation.
- **variant**: `offpc,pbulk`. It is the control plus the `offpc` block
  (`arm_offpc.py`), which adds 16 critic input columns. Each column is the bag
  mean of a cell's off-PC residual of standardized log expression for one gene.
  The gene set S, in `selected_genes.json`, is PPARG plus the 15 genes with the
  largest stage-RMS off-PC training-donor mean gap. The block's columns are
  standardized on 512 real training reference bags and scaled so its mean
  squared norm equals that of the released 150 features. Nothing else changes:
  the update rule, recipe, optimizers, penalty, generator, prior and streams are
  identical. The critic's first layer is wider, so its initial weights differ
  under the same `critic_init` seed.
- Both arms run 10,000 of the recipe's 20,000-step horizon. That is the matched
  budget the workflow allows on shared, CPU-saturated hardware. Learning rates
  are constant in this recipe because the cosine floor is 1. Snapshots follow
  the trainer rule: 2,500, 5,000 and 7,500 to 9,500 every 500.
- The count-gradient (straight-through vs discrete) ablation is out of scope.

## Readouts (same for both arms)

- **app**: `cellshape@0.5:biascal:stagehalf`, the application default. Its
  snapshot ensemble covers the last 25% of the run plus the final weights, and
  its calibration uses training donors only.
- **raw**: `snapens`, the same states with no calibration or shrinkage.
- Both readouts use the application generation streams
  `SeedSequence([20261114,6,fold,index,0])`, with M=32 generated donors of
  256 cells each.

## Outcomes (12 held-out development donors of fold 0 only)

Each per-donor value comes from generated mean_log averaged over the M draws,
compared with that donor's observed mean_log. Within each arm, per-donor squared
errors are averaged over seeds 0 and 1. CIs come from a paired donor bootstrap:
10,000 resamples of the 12 donors, seed 20261120, percentile 95%.

Primary (app readout):
- **P1**: PPARG held-out mean_log RMSE, variant minus control.
- **P2**: RMSE over the 16-gene block, variant minus control. We also report it
  with PPARG excluded.

The variant **improves** an outcome when the whole 95% CI lies below 0, and
**worsens** it when the whole CI lies above 0. Otherwise the result is
inconclusive. P1 and P2 are reported separately. There is no combined claim.

Secondary non-regression checks use the app readout and frozen metrics
(`analysis/improve_20261109/r5/architecture_score.py`). SD, combined and mean
interval score and coverage get per-donor paired bootstrap CIs, as does the
all-gene mean_log RMSE. Frozen historical metrics are reported per seed, not
bootstrapped: pb_rmse, meanlog/nonzero/logvar gene RMSE, pc_energy_rep,
cell_energy_rep, pc_coverage80_rep and pc_spread_rep. A secondary metric
**regresses** when its CI lies wholly on the worse side.

The raw readout is a diagnostic and is not used for the decision.

References use the same donors and are always reported. Stage-blind empirical
resampling (`baselines.generate('resample_blind')`, M=32, K=256) is the required
reference. Stage-aware `resample` is shown as well.

## Limits declared in advance

This is one fold with two seeds and 12 held-out donors, only one of them Not-AD.
These donors were already used in earlier development selection, and no sealed
or external donors are used. The 10k budget is not the 20k default, so the result
does not qualify a default change. The gene selection used a 20k default fit
trained on the same training donors.
