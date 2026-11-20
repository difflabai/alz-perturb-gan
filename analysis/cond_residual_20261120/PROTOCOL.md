# Condition-aware wide EMPF residual: declared comparison

Written 2026-11-20 about 04:05 UTC, before the probe table, any synthetic
training score or any SEA-AD fit of this study exists. Smoke runs of 60
updates were used only to time the loop; none was scored.

## Question

ParticleGAN 0.10.0 EMPF measures its transport residual for conditioned
continuous inputs wider than 64 features with a pooled Chamfer distance that
ignores the condition, so exchanging two conditions' spreads leaves it at zero
(`../particlegan_review_20261117`). Does a residual that compares rows only
within a condition (1) detect the swap, (2) improve learned per-group spread
on a task with known truth, and (3) improve held-out SD coverage of the
SEA-AD microglia generator? Equivalently: does this blind spot drive the SD
under-coverage of the default?

## Mechanism note (from released source, before any run)

On the wide continuous path (`dim > 64`, not categorical) the released kernel
uses `record.ema_residual` only in (a) `settle_s = clamp(1 - res/0.16)`, which
raises the EBI sharpness and is inactive while res >= 0.16, and (b) the gauge
coefficient `0.05 + 0.10 (1 - res)`. The thermal and optimizer step scales stay
1 on this path; the loss uses Sinkhorn only for widths <= 4. The selected
SEA-AD pilot's `ema_residual` stayed between about 0.77 and 0.94 over 20,000
updates, so (a) never engaged. The adapter therefore can change training only
through the gauge weight unless its residual falls below 0.16. We expect a
small effect; the runs below test it.

## Adapter (one change)

`cond_residual.py`: opt-in `install(penalty)` swaps the released kernel class
for a subclass. On the wide continuous conditioned path it replaces the pooled
Chamfer value with the same statistic restricted to real/fake pairs with an
identical condition row, using the native median scale and `data_scale`. All
other paths, the critic gradient terms, the optimizer and the released package
are unchanged. With one condition value the result is bit-identical to native.

## Part 1. Probe (CPU)

The review fixture (widths 64, 65, 2150). Pass: with the adapter, swapped
spreads score above the correct pairing (correct = 0) at all three widths and
the halved-spread control is positive. Paired-row reorder control: permuting
real, fake and condition rows together leaves each value unchanged (|diff| <
1e-6). Native width 64, unconditioned and single-condition calls are unchanged.
These are unit tests in `test_cond_residual.py`.

## Part 2. Synthetic grouped Gaussian (training)

`synthetic.py`. Four conditions, eight informative coordinates with known
per-group means and SDs (Latin square over SD 0.5/1/1.5/2, so pooled
marginals are identical across coordinates), remaining coordinates
N(0, 0.1^2). Widths 2150 (ALZ width; primary) and 128 (secondary). The loop
mirrors the ALZ host update: EMPF recipe `gan/empf`, batch 32, 512 particles,
z_dim 8, `reg_every=1` (lazy_k 1), continuous input domain, same initial
weights, minibatches, latent/penalty streams and hyperparameters for both
arms. Seeds 0-3 per arm and width, 2,000 updates each. Width 2150 runs on GPU,
width 128 on CPU.

Scores (4096 generated draws per group vs 4096 independent truth draws per
group, informative coordinates): group mean RMSE, group SD RMSE, log-SD RMSE
and bias, 80% central-interval coverage, width, interval score (alpha 0.2).
Reference controls: correct sampler, swapped-spread sampler, half-spread
pooled sampler.

**Synthetic helps** if, at width 2150, the mean paired (adapter - native)
log-SD RMSE is at least 5% lower relative to native, at least 3 of 4 seeds
improve, and the mean interval score is not higher by more than 1%. Seed
bootstrap 95% intervals are reported (4 seeds; crude). Width 128 is reported
the same way and is secondary.

## Part 3. SEA-AD matched fits

Runner `alz_run.py`: the committed native EMPF host derivation (lane
`native`, `pbulk`, hvg2000, recipe total_steps 20000, released
hyperparameters), with a private copy of the derived host files. Equal
execution cap of **5,000 updates** for both lanes (the 20k recipe schedule
truncated identically; a budget-matched comparison, not the 20k default).
Seed 0, fold 0. Fold 1 seed 0 is run as a replication if the budget allows;
both lanes or neither.

Because of the wall budget, the fold-0 pair starts in parallel with Part 2
rather than after it. This departs from running Part 3 only after a positive
Part 2; a "helps" claim for SEA-AD still requires Part 2 to pass. Without
that, Part 3 is read only as a direct test of whether the blind spot drives
SD under-coverage.

Readout: the pilot's `cellshape@0.5:biascal:stagehalf` with its generation
streams (`[20261114, 6, fold, donor index, 0]`), late states at
u >= 0.75 U plus final, 32 generated donors x 256 cells, scored by the frozen
`analysis/improve_20261109/r5/architecture_score.py` on the held-out
development donors.

Primary: held-out SD coverage (distance to 80%) and SD interval score.
Secondary: combined coverage and interval score, widths, `pb_rmse`,
`meanlog_rmse_gene`, `nonzero_rmse_gene`, `logvar_rmse_gene`,
`pc_energy_rep`, `cell_energy_rep`. Paired donor-bootstrap 95% intervals
(2,000 resamples of held-out donors) for the coverage and interval-score
differences. References: stage-blind empirical resampling
(`runs/improve_20261109/results_r5/empirical_challenge_20261114/empirical32/fold{k}/seed0.json`)
and, for fold 0, the frozen 20k native pilot score.

**Blind spot drives SD under-coverage** (supported) only if the adapter raises
held-out SD coverage by at least 2 percentage points with a donor-bootstrap
95% interval excluding zero on every fold run, without a higher SD interval
score. Otherwise we report no evidence for it at this budget.

Also recorded: `ema_residual` trajectories of both lanes (to check the
mechanism note) and, for native fold 0, whether its first 5,000 updates
reproduce the pilot's learning curve (determinism check; exploratory).

## Rules

No metric, threshold, fold or readout is added after the first score; extras
are labelled exploratory. Development donors were reused in earlier
selection; this is development evidence, not a final test. Shared machine:
at most 4 GPU processes, memory fraction 0.04, 3 threads, `taskset -c 3-23`.
Nothing under the main `/ml2/alz` working tree is modified; outputs go to
`/ml2/alz/runs/research_20261120/cond-residual/`.
