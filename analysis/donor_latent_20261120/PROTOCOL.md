# Donor random-effect generator: declared comparison

Declared 2026-11-20 before any generation or score of the fits below exists.
The four training jobs were launched at 03:55 UTC, before this file was
committed. Training writes no evaluation output.

## Question

Raw TRAIN-donor SD coverage of the native EMPF default (fold0/seed0, 27.7%) is far
below its held-out SD coverage (58.3%). One possible reason is that the generator
models stage-level spread rather than donor-to-donor spread. A competing,
already documented explanation is PCA shrinkage: held-out donors projected on the
TRAIN-fitted basis have deflated SDs on PCs 31-50. This study asks a narrow
question. Does adding an explicit continuous donor random effect to the native
generator improve held-out donor-level SD coverage and SD interval score at a matched
training budget? Does it close the gap to stage-blind empirical resampling?

## Arms

Both arms use the same derived native EMPF trainer: ParticleGAN 0.10.0, the released
`empf` preset, `pbulk`, `lazy_k=1` and G/D/prior learning rates .012/.018/.03.
They also share the released streams `SeedSequence([20261109, fold, seed])`, batches of
32 donors x 128 cells, the native bag critic and the EMPF penalty. The host derivation
is `dl_host.py`. It is the same derivation as `empf_comparison_20261117/host.py`
native, with sources checked against the application's pinned source manifest.

* **control** `pbulk`: the native generator.
* **variant** `g.dre,pbulk`: native generator plus `arm_dre.py`. Each generated donor
  (bag) draws `u ~ N(0, I_4)` after the native cell noise. Cell-latent noise is
  multiplied by `exp(A u)`, a donor-level variance effect, and `B u` is added to the
  donor FiLM shift, a donor-level location effect. A and B are initialised to zero,
  so the forward pass equals the native one at update 0. All released initial tensors
  are unchanged. The critic, penalty, prior, recipe and batches are unchanged; the
  native critic already scores one donor bag per row, conditioned on stage.

Budget: **10,000 updates for both arms**. The recipe horizon stays at 20,000 with
constant learning rates, and snapshots follow `snapshot_updates(10000)`. This is half
the default's 20k budget. Results describe a matched 10k comparison and do not
transfer automatically to 20k.

Fits: fold0 and fold1, seed 0, for both arms (4 fits). The fits run concurrently on
the two shared A6000s, with GPU fraction .04, 3 threads and cores 3-23.

## Readout and scoring

* Readout: the application default `cellshape@0.5:biascal:stagehalf` for both arms.
  It uses the six late states of the 10k run, the training-only calibration, M=32
  generated donors per target, K=256 cells and the application's named generation
  streams `[20261114, 6, fold, dev index, 0]`.
* Metrics: the frozen r5 scorer `architecture_score.score`. This gives historical
  pb_rmse, gene mean/detection/log-variance errors, PC/cell energy, and 80% percentile
  intervals for the mean, SD and combined bag summaries, evaluated on 24 real
  64-cell subsamples per donor. `score.py` adds the same interval computations
  per donor, and per PC block (1-10, 11-30, 31-50) for the SD coordinates.
* Empirical references, scored identically on the same folds:
  * `resample_blind`, the stage-blind reference, with the `empirical32` stream
    namespace of `empirical_challenge_20261114`, replicate 0. Its fold0 held-out SD
    score must reproduce the recorded .61856.
  * `resample`, same-stage resampling.
* Uncertainty: a paired donor bootstrap. Within each fold, resample the held-out donors
  with replacement. The statistic is the mean over folds of the per-donor mean
  difference. 10,000 resamples, seed 20261120, 95% percentile CI.

## Endpoints and decision rule (held-out development donors)

Primary:
1. Held-out SD interval score (50 SD coordinates), variant minus control.
2. Held-out SD coverage, distance to the 80% target, per fold.

The variant **improves on the control** only if all of the following hold:
* the bootstrap 95% CI for the SD interval-score difference lies entirely below 0;
* SD coverage is closer to 80% in every scored fold;
* no historical guard fails in any scored fold. The guards, relative to control, are
  pb_rmse +0.5%, and +2% each for meanlog, nonzero and logvar gene RMSE and for PC
  and cell energy.

The variant **beats stage-blind empirical resampling** only if the CI for its SD
interval score minus `resample_blind` lies entirely below 0. If neither condition
holds, the result is reported as negative or inconclusive. Point estimates are
not presented as wins.

Secondary (descriptive, no decision):
* combined and mean coverage/score;
* per-PC-block SD coverage/score, to separate the PCA-shrinkage block (31-50) from
  the leading PCs;
* TRAIN-donor SD coverage/score (the motivating diagnostic);
* generated between-draw spread of the SD summaries vs real within-stage
  between-donor spread;
* the learned random-effect amplitudes ||A||, ||B||.

If fold1 does not finish within the wall budget, fold0 alone is scored with the same
rule and the limitation is stated. No other readouts, budgets, seeds or folds are
substituted after scores exist.

## Limits declared in advance

One seed per fold, two folds, 12 held-out donors per fold. These donors were already
used in earlier development selection, so this is not an independent test. 10k
updates is not the 20k default budget. Empirical resampling reuses observed cells and
is a reference, not a trained model.
