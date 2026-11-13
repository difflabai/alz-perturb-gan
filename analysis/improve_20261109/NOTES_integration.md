# NOTES_integration (component: integrator)

Date: 2026-11-09. Scope: make the cache, trainer/generator and evaluator work
together on the real caches, CPU smoke test, launcher (`run_screen.sh`) and
`README.md`. No 20,000-update fit was launched.

## integration: what was run

All on CPU, `CUDA_VISIBLE_DEVICES=''`, 6 threads per process, study python,
`-B`. The machine was shared (load average 10-22 on 24 cores).

1. `test_cache.py` on the existing caches (all six `{hvg2000,wide}/fold{0,1,2}`
   present): ALL CACHE TESTS PASSED. No cache was rebuilt.
2. Smoke fits into `/ml2/alz/runs/improve_20261109/smoke_fits` (seed 0, fold 0),
   run concurrently (3 processes):
   - `base` hvg2000, 300 updates
   - `covars,genemean,offset` hvg2000, 300 updates
   - `offset` wide, 200 updates
3. `evaluate.py --fits-root .../smoke_fits --out .../smoke_results --folds 0`
   (with baselines and noise ceilings for both panels).
4. Second pass through `run_screen.sh` (`-d cpu -p 3 -u 300`): `base` hvg2000
   fold 0 seed 1, fold 1 seed 0, fold 2 seed 0 (fold 0 seed 0 was skipped as
   complete), then `evaluate.py` over all three folds, which exercises the
   seed-ensemble and pooled-across-folds paths on real fits.
5. All test files (see "Tests" below).

## integration: Interface 2 check (generated.npz)

Verified by script on all six smoke `generated.npz` (and again by the evaluator's
`align_generated`): keys `targets, is_heldout, stage, genes, pb_counts,
mean_log, nonzero, pcs, lib`; `pb_counts/mean_log/nonzero` float32 [37,32,G],
`pcs` float16 [37,32,64,50], `lib` float32 [37,32], `is_heldout` bool, `stage` int64;
T = 37 with 12 / 13 / 12 held-out on folds 0 / 1 / 2 (train/held-out sets equal
`folds.json` `inner_folds[k]`); `stage`
equals `folds.json` `stage_by_donor`; `genes` equals the cache `space.npz`
genes; `pb_counts` nonnegative integers; `lib == pb_counts.sum(-1)/256`;
`nonzero` in [0,1]; all finite; zero-count generated cells 0. The evaluator
estimates K = 256.0 and a generated/target library ratio of 1.000-1.002.

## integration: CPU cost measured in the smoke test

Wall seconds per update from `fit_receipt.json` (3 fits running concurrently,
plus other users' jobs); process CPU seconds from `/usr/bin/time` (user+sys of
the whole process, including start-up and generation):

| arm | panel | updates | wall s/update | process CPU s (user+sys) | generation wall s | peak RSS |
|---|---|---|---|---|---|---|
| base | hvg2000 | 300 | 3.39 | 3503 + 552 = 4055 | 55 | 1.52 GB |
| covars,genemean,offset | hvg2000 | 300 | 3.43 | 3588 + 553 = 4141 | 56 | 1.61 GB |
| offset | wide | 200 | 8.28 | 4327 + 1505 = 5832 | 134 | 1.96 GB |

So roughly 12-13 CPU-seconds per update on hvg2000 and roughly 25-28 on wide
(the process uses ~4 cores on average with 6 threads). A 20,000-update fit on
this loaded CPU is about 19 h (hvg2000) or 46 h (wide) wall time.

## integration: fixes made

1. **Bytecode in read-only directories.** `trainer.py`, `generate.py` and
   `test_trainer.py` now set `sys.dont_write_bytecode = True` before importing
   the released/benchmark modules (`build_cache.py`/`test_cache.py` already
   did). Without it, running the trainer without `-B` writes `__pycache__`
   into `/ml2/codex-storage/.../benchmark` and `core-comparison-20261103`,
   which the hard rules forbid (the cache engineer observed exactly that at
   19:58). `run_screen.sh` also sets `PYTHONDONTWRITEBYTECODE=1` and `-B`.
2. **Pooled-across-folds contrast is leaky (evaluate.py).** On the real
   caches, the pooled High-minus-NotAD agreement of the `stage_mean` baseline
   was Spearman 0.93, slope 0.99, top-decile sign 1.00, while the real
   split-half Spearman is 0.10-0.14 and the per-fold held-out Spearman of the
   same baseline is about 0.1. Cause: every held-out donor of fold k is a
   training donor of the other two folds, so the pooled model contrast (an
   average of train-stage contrasts, each over 2/3 of the 37 donors) and the
   pooled real contrast (all 37 donors) share data. This is a design property,
   not a coding error, and it cannot be removed while pooling across folds.
   The number is kept (it is still a fit-to-all-development-donors quantity),
   but each pooled entry now carries `independent_of_training: false` and a
   `warning`, `aggregate.json` notes carry the warning, and the `summary.md`
   section is titled "LEAKY, not held-out". Held-out contrast claims must use
   the per-fold `hn_*_vs_heldout` metrics.
3. **Profile-baseline residual bootstrap was not centred (baselines.py, v4).**
   With the residual of the target itself excluded, every train target was
   shifted by -r_t/(n-1); the `grand_mean` baseline (no stage information)
   then had a train-target High-minus-NotAD Spearman of -0.64 to -0.80 against
   the real train contrast. With 32 draws cycled over ~25 donors, 7 repeated
   draws also added a random per-target offset. The M drawn residuals are now
   re-centred per target on the mean residual of all train donors
   (`r_jm - mean_m r_jm + mean_j r_j`), so every target gets the same gene-wise
   level correction and keeps its between-donor spread. A first attempt (v3)
   centred on 0; that also removed the systematic fitted-vs-donor-log level
   difference and made the profile baselines worse (grand_mean held-out
   pseudobulk RMSE 0.464 -> 0.500, mean_log RMSE 0.093 -> 0.125), so it was
   replaced. With v4 on hvg2000 fold 0: grand_mean train-target contrast
   Spearman vs train contrast -0.024 (was about -0.7), held-out 0.003 (was
   0.071); pseudobulk/mean_log/nonzero errors equal v2 (0.446 / 0.094 / 0.021).
   The resample baseline is unchanged. `BASELINE_VERSION` is now
   `improve_20261109-baselines-v4`, so cached baseline results (including the
   v2 ones in `/ml2/alz/runs/improve_20261109/results/`) are recomputed on the
   next evaluate run. `test_evaluate.py` passes 31/31 with v4.
4. **Launcher robustness (run_screen.sh).** The script body is a single
   function parsed before it runs (bash otherwise reads a running script
   incrementally, so editing it during a 40-hour screen could change or break
   the running instance; this happened in testing and was caught before it
   took effect). Worker subshells trap INT/TERM so a signal to the whole
   process group still lets them wait for their trainer, which checkpoints
   `recovery.pt` and exits; the launcher then exits 130.

No interface mismatch between the three components was found: the evaluator
read every real `generated.npz` without changes, and the trainer read every
real cache without changes.

## integration: sanity checks on the numbers (held-out donors, final smoke_results, baselines v4)

Means over the three inner folds for baselines; smoke fits are short (300/200
updates), so their numbers say nothing about the arms.

- The stage-mean baseline predicts held-out pseudobulks far better than a
  300-update model: pseudobulk RMSE 0.500 (stage_mean, 3 folds) vs 0.832
  (`base`, 4 fits) and 0.659 (`covars,genemean,offset`, fold 0) on hvg2000;
  0.311 (stage_mean fold 0) vs 0.596 (`offset`, 200 updates) on wide.
- Noise ceilings are plausible: predicting a held-out donor from all other
  same-stage development donors gives RMSE 0.47-0.49 (hvg2000) and 0.31
  (wide), close to the stage-mean baseline; held-out deviation Pearson is
  about 0 for every predictor including real peers (donor-specific deviations
  dominate). Only ridge (with covariates) reaches 0.10 (hvg2000) / 0.07 (wide).
  The grand-mean baseline has a slightly lower RMSE than the stage mean (0.464
  vs 0.500): stage means over 2-3 Not AD train donors are noisy. Split-half
  Spearman of the real contrast is 0.10-0.14 (all 37 donors).
- Resampling baseline: RMSE 0.506, mean_log RMSE 0.079 (best of all
  methods), PC coverage80 0.70, PC spread 1.29 (the over-dispersion on
  held-out donors is the train-donor-PC caveat in NOTES_evaluator.md).
- Correlations of exactly 1 occur only in the `risk` subset on hvg2000, which
  has 4 genes (top decile = 1 gene), so sign agreement there is 0 or 1 by
  construction. Contrast metrics on that subset are not informative.
- Per-fold held-out contrast metrics are very noisy (one Not AD held-out
  donor in folds 0 and 1). Before the v4 baseline fix, the no-stage
  `grand_mean` reached held-out High-minus-NotAD Spearman 0.07 (hvg2000) /
  0.23 (wide) on fold 0; with v4 it is -0.004 +- 0.006 over folds (hvg2000),
  as a null reference should be. I also checked depth-matched sampling
  interacting with expression level as a mechanism and found no support
  (|Spearman with expression level| < 0.09).
- The seed-ensemble (base, fold 0, seeds 0+1) and pooled-across-folds (base
  seed 0, folds 0-2) paths ran on real fits. Seed stability of the held-out
  High-minus-NotAD contrast at 300 updates: Spearman 0.41.
- No identical numbers across arms, no negative variances, spreads, energies
  or RMSEs, no missing (None) headline values in the aggregate rows; `errors`
  and `incomplete_fits` are empty. Learning curves: critic gap 2.2-2.7 for
  `base` at 300 updates, 7.3 for `covars,genemean,offset` (the critic sees
  2,150 features), 0.10 for wide `offset` at 200 updates.
- Generated zero-total-count cells: 0 in every smoke fit on the real caches
  (and 0.0002 for a donorpc test fit); 4.6-5.3% for 100-update base / offset /
  covars test fits on the legacy dev cache. That is an early-training
  property of the dev fixture; the evaluator handles such cells (CP10k
  denominator clamped at 1).

## integration: tests run (all exit 0)

| command | result | wall |
|---|---|---|
| `test_cache.py` | ALL CACHE TESTS PASSED | ~1 min |
| `test_evaluate.py` (baselines v4) | 31/31 checks passed | 366 s |
| `test_trainer.py devcache` | dev caches rewritten | 4 s |
| `test_trainer.py parity` | PASS (bitwise vs released build/advance, 5 updates) | 40 s |
| `test_trainer.py grads` | all arms finite, gradients reach new blocks | 35 s |
| `test_trainer.py arms` | 9 x 100-update fits + generated.npz, incl. real hvg2000/fold0 | 3018 s |
| `test_trainer.py resume` | PASS (--stop-after and SIGINT resume bitwise equal) | 670 s |
| `test_trainer.py timing` | base 2.81, genemean 2.79, donorpc 2.71, offset 2.74, covars+genemean+offset 2.77 s/update (dev cache, G=2000) | 462 s |
| `test_trainer.py cleanup` | removed 215 MB of test fit outputs | 2 s |

`run_screen.sh` was tested on CPU: dry run; a bad arm (exit 1, failure
listed); a 3-fit screen with one completed fit skipped (exit 0); SIGTERM to
the launcher and, after the fix, to the whole process group (trainer wrote
`recovery.pt` at update 10, launcher waited for it and exited 130); rerun
resumed from update 10 and completed (exit 0); rerun skipped the complete fit
(exit 0); rerun with a different `-u` reported a mismatch (exit 1). The
evaluator synthetic test's fake generated files (93 MB) were deleted after
each run, as the evaluator engineer did.

Observation: `.pyc` files dated 19:58 (before this step) exist in the
read-only `faithful097/model/__pycache__`, `benchmark/__pycache__` and
`core-comparison-20261103/__pycache__`. None was written by the integration
runs (checked by mtime). I did not delete them because those directories are
outside the writable area; the owner should decide.

## integration: open issues (not fixed)

- CUDA has never been run (`--device cuda`, `--gpu-mem-fraction`,
  `run_screen.sh -d cuda:N`). The launcher maps `cuda:N` to
  `CUDA_VISIBLE_DEVICES=N` and `--device cuda`.
- A resumed fit must use the same device as the original (trainer checks
  `config.json`), so a CPU-started fit cannot be resumed on GPU.
- 20k-update fits on CPU take about 19 h (hvg2000) / 46 h (wide) each under
  the current load; the screen practically needs GPU authorisation.
- The pooled-across-folds contrast cannot be made held-out; only per-fold
  numbers are clean, and those have 1-2 Not AD held-out donors per fold.
