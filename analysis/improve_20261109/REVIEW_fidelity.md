# Fidelity review: trainer.py, arms.py, generate.py (independent reviewer, 2026-11-09)

I ran everything on CPU with 6 threads, using `python3 -B`. Scratch scripts and outputs are in
`/tmp/claude-1000/-ml2-alz/5f746b05-53aa-4a98-abba-680efd8008a9/scratchpad/`. I edited no code. I checked
that the read-only source dirs gained no `.pyc` files after 23:20.

## Findings

### 1. [major] The offset initialisation cannot tell stage signal from noise (arms.offset_init)
The code matches the docstring formula. However, the sampling variance it assumes for `r_sg`,
`v_g/n_s`, is too small. `r_sg` is a depth-weighted pooled log-ratio, and the train donors' depth CV
is 0.53-0.62. As a result the method-of-moments `tau2` is positive even when there is no stage effect.
I tested this by permuting the stage labels among train donors 50 times, keeping the stage sizes, on
each real cache (`chk3.py`):

```
hvg2000 0 n_s [3,3,6,13] tau2 0.0553 shrunkRMS 0.1188 | permuted: tau2 median 0.0654 [0.0100,0.1652]; frac null tau2>=obs 0.66
hvg2000 1 n_s [3,3,6,12] tau2 0.0000 shrunkRMS 0.0000 | permuted: tau2 median 0.0492; frac null tau2>=obs 1.00
hvg2000 2 n_s [2,4,6,13] tau2 0.0293 shrunkRMS 0.0761 | permuted: tau2 median 0.0500; frac null tau2>=obs 0.72
wide    0 tau2 0.0218 | null median 0.0281 (0.64) ; wide 1 tau2 0.0000 (1.00) ; wide 2 tau2 0.0106 | null 0.0165 (0.68)
```

- With permuted labels, `tau2` should be about 0. Instead its median is 0.05.
- With the real labels, `tau2` is below the null median on every fold.
- On fold 1, `tau2` hits the 1e-6 floor, so the table is effectively zero.

In short, the folds 0 and 2 initial tables have the size you would expect from noise, and fold 1 has
no initial table. Possible fixes:
- Estimate the sampling variance of the pooled `r_sg` itself, for example by a donor bootstrap or a
  depth-weighted variance.
- Or use donor-averaged `r_dg` to match `v_g/n_s`.
- At minimum, document the per-fold `tau2` and the null result.

### 2. [minor] The offset arm is not bitwise deterministic on CPU, so its resume is not bitwise
`self.stage_offset[stage]` has a multithreaded scatter-add backward. I built the same arm 3 times and
ran 6 updates each, in one process (`det.py`):

```
base [0.0, 0.0] ; offset [1.49e-08, 1.49e-08] ; genemean [0.0, 0.0] ; covars [0.0, 0.0]
```

I isolated the cause:
- Index backward varies by up to 1.5e-05 across 20 repeats.
- A `one_hot(stage,4).float() @ stage_offset` backward varies by 0.

For the `covars,genemean,offset` arm, two uninterrupted 10-update CLI fits differ by up to 3e-8 in
generator weights, as does a fit interrupted with `--stop-after 6` and then resumed.
`generated.npz` was still equal in this short test. `test_trainer resume` covers only `base`, and the
notes say resume is "bitwise equal" without this caveat. Possible fixes:
- Replace the indexing with a one-hot matmul. The forward is exactly equal, because each row
  multiplies a single 1.
- Or document the arm as reproducible to float tolerance only.

### 3. [minor] The generation-only path in `run()` skips every config check, including `--updates`
When `final.pt` exists but `generated.npz` does not, `run()` calls `generate.generate_fit` before
comparing `config.json` with the request. To test this, I copied the 300-update smoke fit
`base/hvg2000/fold0/seed0` (config.json, final.pt and fit_receipt.json) into a scratch out-root. I
then ran `trainer.py ... --updates 20000`:

```
[trainer] final.pt present without generated.npz; generation only: ...
[trainer] done: ...        rc=0      fit_receipt: completed 300 300
```

`run_screen.sh` checks `"updates": $UPDATES` only for fits that are already complete. After a trainer
exits with 0 it calls `is_complete` and records the job as `done`. So a fit that was killed during
generation, for example by SIGTERM after `final.pt` was written, is accepted under a different `-u`.
Fix: run the `config.json` diff check, and compare `final.pt['fixed_endpoint']` with `args.updates`,
before the generation-only branch.

## Checked and fine (with evidence)

- **Parity.** I re-ran `test_trainer.parity` for 8 updates instead of 5, with result saving patched
  out. Streams, X, trainpcs, mu, theta, conditions, W and `by` are equal, and so are the initial
  G, D and P. All 8 updates have bitwise-equal losses, penalty stats and gradient norms. Final
  parameters and RNG states are equal. PASS. The test compares against the real released
  `A.build` and `U.advance` after `configure_hyperparameters(T005)`. The only patch is
  `checked_cache`, which keeps it from hashing the sealed evaluation dir.
- **advance() line by line against released_update.advance.** The sequence is the same:
  - begin_step, then the noise and LR asserts.
  - D.train / G.eval.
  - no_grad sample, observe_support, observe_critic_pair.
  - d_loss + pen(D,real,f,c) with PENALTY_OPTIONS {'input_domain':'continuous'}.
  - after_critic_step.
  - D.eval / G.train / requires_grad_(False).
  - A second, separate `ctx.batch(rng)`.
  - reg(P.z) on the full table (512 <= 1024), observe_sampled_rows.
  - constraint_geometry_backward(total, og, (lg,)), after_generator_backward, og.step,
    after_generator_step, requires_grad_(True), finish_step.
  - Everything runs inside serial_autograd.

  The only differences are the documented ones: the features class, the width assert, and extra
  step-0 gradient asserts. Poisson runs on `rate.detach()` with straight-through critic_features, and
  `lib` comes from the real batch.
- **build().** It follows the same order as released_adapter.build: generator_init, critic_init,
  prior_init, optimizers, penalty, loss, prior regulariser, `bind_thermal_source(od.record, loss,
  reg)`, UpdatePolicy streams, the canonical recipe check (EMA 0, 512 particles, z_dim 8), cell and
  count generators, the donor sampler and gamma_global. The LRs are [0.012, 0.06] and [0.018], so the
  particle multiplier is 5.
- **Conditions on the real hvg2000 fold-0 cache.**
  - The base arm's conditions equal `stage_law.mask_adnc(raw_c)`, even though raw_c has nonzero
    Braak/Thal/CERAD/Male/APOE.
  - For covars, slots 0-2 are all zero, Male is in {0,1} and APOE in {0,1,2}.
  - Age uses the train mean 87.96 and SD 8.28, ddof=0. This equals the cache space's age_mean/age_sd.
  - round(3*c[:,3]) equals stage_of(raw_c).
- **Offset head.** Generated proportions sum to 1 within 2.4e-7. The log mu ratio between the
  offset and zero-offset generators equals the centred table row for each donor's own stage, within
  1.3e-6. Train-only inputs: `Context` loads only train counts, and offset_init receives `ctx.by` and
  the train raw_c.
- **genemean.** Real and fake features on the same counts match exactly (max abs diff 0.0). Real
  bags use standardize_counts and fake bags use the straight-through tensor. The scale is 0.824,
  computed from train bags only. donorpc uses the same `mean_cells(F)` path, with mu, V and sd from
  train donor profiles. The gradient tests from `grads.json` are reproduced by the code: the added
  block reaches the generator and the critic's added input columns.
- **Generation.** I recomputed two targets independently in numpy for the `covars,genemean,offset`
  smoke fit: train target H20.33.001 and held-out target H21.33.043. I rebuilt each condition vector
  by hand and used the library pool from the target donor only.
  - pb_counts match exactly and are integer sums.
  - mean_log differs by at most 5.7e-7, using the clamp_min(1) CP10k denominator.
  - nonzero differs by 0.
  - pcs differ by at most 1.7e-3, within float16 precision, using (log1p CP10k - mean)/sd @ W/pc_sd.
  - lib differs by 0.
  - Generated/target library ratio is 0.998 and 1.009.
  - Streams come from SeedSequence([20261109, 2, fold, dev_index]) only, so they do not depend on
    arm or seed. M=32, K=256, chunks of 8.
- **Checkpoint mixing.** restore() refuses a mismatch in any of: flags, seed, fold, panel, cache
  manifest SHA, recomputed arm_state, recipe, or streams. The config.json diff covers device and
  stream_root. generate_fit checks the manifest SHA and flags.
- **Resume.** `covars,genemean,offset` with `--stop-after 6` resumed from update 6, and its
  `generated.npz` equals the uninterrupted run's. For parameters, see finding 2.
