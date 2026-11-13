# NOTES_trainer (component: trainer and generator)

Files: `trainer.py`, `arms.py`, `generate.py`, `test_trainer.py` (this directory).
Test outputs: `/ml2/alz/runs/improve_20261109/tmp_trainer_devcache/` (`results/*.json`, `test_log.txt`).

## trainer: interpretation choices (ambiguities resolved as faithfully as I could)

1. **Training-stream namespace.** The released adapter derives 11 named streams
   from `SeedSequence([20261103, fold, seed])`. This study uses the same rule with
   root **20261109** (fresh namespace for a new study, seeds not limited to 0..4).
   `build(ctx, seed, stream_root=20261103)` reproduces the released streams; the
   parity test uses it. `--stream-root` exists (hidden) only for that purpose.
2. **Hyperparameters.** t005 is passed explicitly to
   `released_update.resolved_recipe` (generator_lr .012, critic_lr .018,
   particle_lr **.06**, reg_coeff 1, reg_kappa 1, prior_reg .01). Resolved LRs:
   G .012, prior .06, D .018 (checked in the parity test against the released
   recipe configured with the same values).
3. **`--updates` < 20000.** The recipe keeps `total_steps=20000` (released
   recipe); `--updates` is only the endpoint and the `execution_limit` passed to
   `policy.begin_step` (documented as an external cap). The released recipe has
   `lr_floor=1`, so no schedule depends on the endpoint. Short fits are marked
   `full_20k_endpoint: false` in `fit_receipt.json`.
4. **learning_curve.jsonl.** Same rows as released (`step % 250 in (0, 1)`, i.e.
   1, 250, 251, 500, 501, ...; gradient norms and penalty stats are only filled
   on the 251/501/... rows, exactly as released) plus the final step when not on
   that grid; one added field `wall_seconds`.
5. **Checkpoints.** `recovery.pt` every 2000 updates and on SIGINT/SIGTERM or
   exception at an update boundary (status `incomplete`); resume is automatic
   when `recovery.pt` exists and `config.json` matches (device included; only
   `cache` path and `checkpoint_every` may differ). No 5000/10000 diagnostic
   checkpoints (disk). `recovery.pt` is deleted after `final.pt`.
   A rerun on a directory with `final.pt` + `generated.npz` raises
   `FileExistsError`; with `final.pt` but no `generated.npz` it only runs
   generation (nothing is overwritten).
6. **Split validation.** Every train/heldout donor must be one of the 37
   development donors and none of the sealed fold-0 donors; train/heldout sets
   must equal `folds.json` `inner_folds[k]`. The developer flag
   `--allow-nonprotocol-split` (recorded as `protocol_split: false` in
   config.json) relaxes only the second check, for the dev fixture caches.
   The manifest's `train_donors` order must equal `train/metadata.npz donors`.
7. **Final weights.** EMA is off in the released recipe; generation restores the
   policy state (fast weights) from `final.pt`.

## arms

* **offset** — formula in `arms.py` docstring. Choices: pseudocount 1 count per
  gene, "pooled" = all-train-cell pseudobulk proportions (same quantity as the
  generator's output bias, with the same pseudocount), per-gene pooled
  within-stage between-donor variance `v_g` of donor log-ratios, one global
  method-of-moments `tau2`, weight `tau2/(tau2+v_g/n_s)`. On inner fold 0
  (hvg2000) donors per stage are [3,3,6,13] and mean weights
  [0.29,0.29,0.44,0.61]; on the wide panel [0.20,0.20,0.32,0.48]. The table is a
  2-D [4,G] generator parameter, so the released DualNorm optimizer gives it the
  dense polar (orthogonalised) update with the generator LR, like every other
  2-D generator weight; this is a consequence of "joins the generator optimizer
  group", not a separate choice.
* **genemean** — scale `s = sqrt(E||f150||^2 / E||gene-mean block||^2)` over
  256 real train bags of 128 cells (uniform donors, `default_rng(20261109)`,
  independent of training streams); uncentred squared norms as stated in the
  task. hvg2000 fold0: s = 0.824; wide fold0: 0.601; dev cache: 0.852.
* **donorpc** — K = min(16, n_train-1); population SD (ddof=0) of train donor
  scores; component sign fixed so the largest-|loading| gene is positive.
* **covars** — age is standardised with mean/SD (ddof=0) computed from the
  train donors' `raw_c`; on the real caches this equals `space.npz`
  age_mean/age_sd (87.96 / 8.282 on hvg2000 fold0). APOE4 dose is the raw
  0/1/2 count (as `encode_condition`), not rescaled.

## generator

* Target order: train donors (cache order) then heldout donors (cache order).
* Common random numbers: per target, five streams spawned from
  `SeedSequence([20261109, 2, inner_fold, j])`, j = index of the target in
  `folds.json` `development_donors` (stable across panels/arms/seeds/fits).
  Streams: torch global seed (Gamma `rsample` uses the global RNG, as in the
  released sampler), latent (`P.sample`), cell noise, Poisson counts, numpy
  library draw. M=32 generated donors are produced in chunks of 8.
* Zero-total-count generated cells are kept; CP10k denominator clamp_min(1) as
  in `standardize_counts` (their log1p CP10k is 0); rate reported in
  `fit_receipt.json` (`generated_zero_count_cell_rate`).
* `generated.npz` is written with `np.savez_compressed` (~16 MB for 33 x 2000).

## dev fixture

* The legacy fold-0 cache's `raw_c` has Male and APOE4 zeroed (only ADNC and age
  are non-zero), so on the dev caches `covars` only exercises the age slot. The
  covariate path with real Male/APOE values was exercised on the real
  `caches/hvg2000/fold0` cache (100-update smoke fit, protocol split enforced).
* The parity test monkeypatches `contracts.checked_cache` inside the test
  process because the released version SHA-hashes every legacy cache file
  including `evaluation/` (sealed donors). Nothing under `evaluation/` is read.

## CPU cost (important for scheduling)

Profiling the base arm on CPU (6 threads, shared machine, load ~10): ~70% of an
update is `aten::_standard_gamma` + `_standard_gamma_grad` (the Gamma-Poisson
forward/backward of the released CountGenerator over 2 x 32 x 128 x G), which
are essentially single-threaded on CPU. That is the released formulation, so it
was not changed.

Measured on 2026-11-09 with 6 threads (load avg 8-10 on 24 cores) on the dev
cache (G=2000), 30 timed updates after 3 warm-up updates:
base 2.46 s/update (13.7 h per 20k), genemean 2.50 (13.9 h), donorpc 2.50,
offset 2.49, covars+genemean+offset 2.50. CLI 100-update runs: 2.6-2.9 s/update.
On the wide panel (G=5028, 8 timed updates): base 6.68 s/update (37 h per 20k),
genemean 6.79 (38 h). Generation takes about 45-55 s per fit on CPU.
Full 20k fits on CPU are therefore expensive; GPU (`--device cuda`) is the
practical route once a task authorises it (the trainer has not been run on GPU).
A resumed run checks that the device is unchanged.

## Verification status (see tmp_trainer_devcache/results/*.json)

* Parity with released build()/advance() on the legacy fold-0 cache, seed 0,
  released stream root, 5 updates: initial G/D/P tensors, per-update losses,
  gradient norms, penalty stats, parameters and RNG states afterwards are all
  bitwise equal.
* Every arm, plus covars,genemean,offset, ran 100 updates through the CLI and
  produced generated.npz with the right shapes, dtypes and finite values. The
  combination arm also ran on the dev heldout fixture and on the real
  hvg2000/fold0 cache (protocol split, 37 targets, 12 heldout), and donorpc ran
  on hvg2000/fold0 too.
* Resume: both --stop-after 60 and a real SIGINT (stopped at update 5) produce,
  after the resumed run finishes, final models, numpy RNG, learning curve and
  generated.npz bitwise equal to an uninterrupted 100-update run.
* Not tested: CUDA, full 20k fits, wide-panel CLI fits (only timed in-process).
