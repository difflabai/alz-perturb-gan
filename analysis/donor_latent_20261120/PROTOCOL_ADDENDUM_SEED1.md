# Addendum: training-seed replicate (declared before any seed-1 output exists)

Declared 2026-11-20, after the seed-0 result in `README.md` was known and before any
seed-1 generation or score exists. The four seed-1 training jobs were launched at
06:21 UTC; training writes no evaluation output.

## Why

An audit pointed out that the seed-0 CI (`variant − control` SD interval score
+.034 [+.014, +.054]) resamples held-out donors only. With one training seed per
fold it cannot separate the random-effect effect from GAN training-trajectory noise.
The variant's extra `u` draw also shifts the shared random stream from update 1, so
the two arms follow different stochastic trajectories.

## What is added

* Fits: the same two arms (`pbulk`, `g.dre,pbulk`), folds 0 and 1, **seed 1**,
  10,000 updates, identical recipe, readout, M, K, generation streams and CUDA
  generation as seed 0. Seed 1 changes the training streams
  `SeedSequence([20261109, fold, 1])` only.
* Scoring: `seed_replicate.py`, which calls the unchanged `score.py` functions on
  the seed-1 archives and writes `completion/seed_replicate.json`.

## Reported quantities (held-out donors, SD interval score over 50 SD coordinates)

1. Seed-1 `variant − control`, paired donor bootstrap as in `PROTOCOL.md`.
2. Native seed-to-seed difference `control(seed1) − control(seed0)` and
   `variant(seed1) − variant(seed0)`, same bootstrap. These are the training-noise
   context for the arm difference.
3. Two-seed `variant − control`: per donor, the mean over seeds of the arm difference,
   same bootstrap. This CI still does not include between-seed variance
   (only two seeds), so it is reported with the per-seed point estimates.

## Interpretation rule (fixed now)

* The seed-0 finding "variant worse than control" is called **replicated** only if
  the seed-1 `variant − control` point estimate is also > 0 in the pooled fold mean
  and the two-seed CI lies entirely above 0.
* If the seed-1 estimate has the opposite sign, or the absolute native seed-to-seed
  difference is at least as large as the seed-0 arm difference (.034), the result is
  reported as **inconclusive with respect to training noise**: the variant did not
  improve on the control, but "worse" is not established.
* The decision in `PROTOCOL.md` (variant advances only if it improves on control) is
  unchanged. The variant can only advance if both seeds individually meet that rule.
