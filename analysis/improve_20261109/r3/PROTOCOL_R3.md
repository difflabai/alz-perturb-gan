# improve_20261109 round 3: training length, schedule, particles, architecture

Owner request (Mikkel, 2026-11-11): "would training longer help? are we
testing the obvious things? keep trying different approaches until you get top
scores" and "also the architecture, is it holding us back? do we need more
particles?". Read `../PROTOCOL.md`, `../r2/PROTOCOL_R2.md` and `../RESULTS.md`
first; every hard rule there still binds unless this file changes it.

## What changes in round 3

The owner has widened the mandate. Still ParticleGAN 0.9.7 and still its own
update machinery (recipe, optimizers, penalty, loss, prior regulariser, policy
calls in the released order). Now ALLOWED, as explicit, named, documented
departures from candidate t005:

- training length (declared total updates);
- ParticleGAN's own recipe options: learning rates, native learning-rate
  annealing (`lr_schedule`, `lr_anneal_start`, `lr_floor`,
  `network_lr_floor`, `network_lr_horizon_cap`), native EMA (`ema_decay`),
  number of particles, latent dimension, batch size, cells per bag;
- generator architecture (width, depth, cell-latent size, head);
- critic architecture and what the critic observes;
- read-outs that do not alter training (shadow EMA copies of the generator,
  snapshots, ensembles over them).

Still FORBIDDEN: replacing the ParticleGAN update with another training method;
loss terms outside the recipe; reading counts of donors other than the 37
development donors; fitting or choosing anything with held-out donors of the
fold being scored; editing round-1 or round-2 code or outputs.

## Round-3 hard rules

- `r2/` is FROZEN (the round-2 screen is training with it). Round-3 code lives
  in `/ml2/alz/analysis/improve_20261109/r3/`, starting as a copy of `r2/`.
  Outputs: `runs/improve_20261109/fits_r3/`, `results_r3/`, `smoke_r3/`,
  `logs_r3/`. Never write into any other run folder.
- With no round-3 flag, `r3/trainer.py` must remain bitwise equal to round 1
  for `base` and `genemean` (initial tensors and first 5 updates, CPU).
- Machine: at most 3 CPU threads per process. GPUs are shared with the round-2
  screen: test fits of at most 300 updates and generation only, one process at
  a time per task (integration and repair: two), `--gpu-mem-fraction 0.04`.
  Never kill or signal processes you did not start. Disk is 99% full (about
  140 GB free): keep scratch small and delete it.
- No `.md` files from subagents; document in module docstrings and return
  notes in structured output. Python:
  `/mnt/nvme/ai/conda/anaconda3/bin/python3 -B`. No git operations.
- Selection hygiene: three rounds now reuse the same 37 donors. Every round-3
  result is selection-conditioned development evidence. A final recipe needs
  confirmation on donors used in no round (the owner's sealed donors), which
  only the owner can release.

## Where things stand (held-out donors, 2,000-gene panel; lower is better)

- `pb_rmse` references: ridge 0.461, grand mean 0.464, real-donor resampling
  0.506. Round 1: `base` 0.687 (six-seed ensemble 0.629); `genemean` 0.666
  (six-seed ensemble 0.526).
- Round 2, seed 0 only (interim): `genemean@0.5` 0.623, `gmlite@64` 0.626,
  `pbulk` 0.645, `genemean` 0.651, `genemean@0.25` 0.664, `stagedrop@0.5`
  0.667; late-snapshot ensembles of ONE fit: `genemean@0.25` 0.549, `pbulk`
  0.549, `genemean` 0.562, `genemean@0.5` 0.566. Worse than `base`:
  `covdrop@0.5` 0.713, `donorpc2` 0.718, `stageanchor` 0.868,
  `genemean,stageanchor` 0.879, `covbin` 0.887, `genemean,pbulk` 0.955.
  Tables: `runs/improve_20261109/results_r2_interim/` (seed 0) and, once it
  exists, `results_r2/` (both seeds).
- The released recipe as used here trains with CONSTANT learning rates
  (`lr_floor = network_lr_floor = 1`) and `ema_decay = 0`, with normalised
  zero-momentum optimizers. Round-2 diagnosis: the `genemean` generator never
  settles (late per-gene means drift by about one seed-to-seed difference) and
  averaging late states helps a lot. Annealing and EMA are therefore the first
  obvious things to test.
- Generator (`pg_model.CountGenerator`): donor MLP (z 8 + condition 7 -> 128 ->
  256 FiLM scale/shift), cell path (8-dim noise -> 128), head (128 -> 128 ->
  genes), softmax x library, Gamma-Poisson. Critic: 150 bag features (+ blocks)
  -> 128 -> 128 -> 1. 512 particles for 24-25 training donors.
- Stage: no method recovers held-out stage contrasts with 2-3 Not AD training
  donors; stage conditioning currently adds error; stage-blind generation
  scores better. This is a data limit; round 3 does not try to fix it by
  modelling.

## Targets ("top scores")

| ID | Target | Reference |
| --- | --- | --- |
| T1 | Held-out `pb_rmse` of one trained generator with a generation procedure fixed in advance (snapshots, EMA and a pre-declared stage-shrink mode allowed; no per-gene recalibration) within 0.005 of the grand mean | 0.464 |
| T2 | Distribution fit at least as good as real-donor resampling on every one of: `pc_energy_rep`, `cell_energy_rep`, `pc_coverage80_rep` (closer to 0.80), `meanlog_rmse_gene`, `nonzero_rmse_gene`, `logvar_rmse_gene` | resample row of the evaluator |
| T3 | T1 and T2 from a single seed, and again as a seed ensemble, on folds 0-2 and on the extra folds 3-8 | |

A stage-only predictor cannot go far below the grand mean on `pb_rmse`,
because most of that error is donor-to-donor variation the model has no
information about; the evaluator task must estimate that floor. T1 is about
reaching the floor; T2 is where a generator can be better than a mean.

## Work items

| ID | Question | Work | Task |
| --- | --- | --- | --- |
| R1 | Would training longer help? Is the constant learning rate the problem? | Training-option flags for total updates, native annealing, native and shadow EMA, learning-rate multipliers; dense snapshots so one long run gives the whole score-versus-updates curve | core |
| R2 | Do we need more particles, a bigger latent, bigger batches? | Recipe-option flags for particles, latent dimension, donors per batch, cells per bag; analysis of particle usage in existing fits | core, capacity |
| R3 | Is the generator architecture holding us back? | Capacity diagnosis on existing fits; generator width, depth, cell-latent and head options | capacity |
| R4 | Is the critic architecture holding us back? | Block-wise critic so gene blocks cannot starve the PC features; critic width and depth | critic |
| R5 | Should the critic see cells, not only bag summaries? | A cell-level set critic (per-cell embedding, pooled) alongside the released features | cellcritic |
| R6 | What is the right stage handling given the data? | Failure analysis of `stageanchor` and `covbin`; stage-blind training; a fixed, non-learned, shrunken stage effect; a pre-declared default stage shrink at generation | stage |
| R7 | What is the floor, and are we there? | Evaluator r3: floor estimate, targets table with pass or fail, score-versus-updates curves, EMA and snapshot units, leaderboard | evaluator |
| R8 | More donors without a data use agreement? | Inventory of open-access human microglia single-nucleus datasets with per-donor neuropathology; feasibility only, no large downloads | data |

## Additions after the study lead's review (2026-11-11 05:35 UTC)

Added after the task prompts were written. Where this section and a task
prompt differ, this section wins. Tasks that had already finished when it was
added could not see it; the integration task closes any gap they left.

1. **Critic-penalty strength is an obvious knob that was missing.** With
   `genemean` the critic gap sits near 13 logits against 4.5 for `base`, so
   the critic may simply be too strong for the wider input. Add flags
   `t.reg@M` (multiplier on t005 `reg_coeff`), `t.kappa@M` (`reg_kappa`) and
   `t.priorreg@M` (`prior_reg`), through the recipe's own fields. Owner: core
   if still open, otherwise integration. The screen gets `t.reg@4` and
   `t.dlr@0.5` on the chosen base observation.
2. **Train-only bias calibration baked into the generator (`biascal`).**
   Round 2 showed that a per-gene additive recalibration fitted on train
   donors closes about half of the gap, but it was applied to predictions
   only. Build it as a generation mode that changes the generator itself: a
   per-gene shift of the output-logit bias chosen so that the generated
   pseudobulk mean over the fold's TRAIN target donors matches their observed
   mean (one or two fixed-point iterations at fixed common random numbers;
   train donors only; no retraining; works on round-1, round-2 and round-3
   fits; composable with `ema@D`, `snapens` and the stage modes, for example
   `biascal+snapens`). Because it changes generated cells, every distribution
   metric must be re-scored under it. Owner: the stage task
   (`r3/mode_biascal.py`, `r3/test_mode_biascal.py`); integration builds it if
   it is missing. Run it on the best round-2 fits now and report.
3. **Target T1 is split.** T1a: as written above (no calibration). T1b: the
   same with `biascal` applied. Both are reported; neither may use held-out
   donors. Units generated stage-blind are flagged and listed separately: a
   stage-conditioned generator that reaches T1 only by ignoring stage has not
   reached it as a stage model.
4. **Primary read-outs, declared before any round-3 score exists.** For arms
   trained with constant learning rates: `snapens`. For annealed arms: the
   final state. For native-EMA arms: the EMA generator. Stage-conditional
   arms are read with `stagehalf` (and separately stage-blind); stage-blind
   arms as trained. Shadow-EMA read-outs (`ema@D`) are secondary and their
   best decay, if used later, is chosen on seed 0 and confirmed on seed 1.
   These choices were informed by round-1 and round-2 held-out results on the
   same donors.
5. **One change at a time, against a control trained in round 3.** The chosen
   base observation is trained in round 3 at 20,000 updates with no other
   flag, so that every round-3 arm is compared with a control that has the
   same snapshots and shadow EMAs. Each factor (training length, annealing,
   EMA, learning rates, penalty strength, particles, latent size, generator
   architecture, critic architecture, cell-level critic, stage handling) gets
   at least one arm that differs from the control in that factor only.
   Combinations come after, and each names the single-factor arms it combines.
6. **Staged screen with a selection rule fixed in advance.**
   - Stage 1 (`jobs_r3.txt`): seed 0 only, folds 0-2, at most 80 units (one
     unit = one 20,000-update fit; a 60,000-update fit costs 3).
   - Stage 2 (`jobs_r3_stage2.txt`, written after stage 1 is scored): seed 1
     for the control and for the five arms with the lowest stage-1 primary
     read-out `pb_rmse` among those that pass at least as many T2 metrics as
     the control, plus at most three combinations of single factors that each
     beat the control in at least 2 of 3 folds.
   - Stage 3: the best two stage-2 arms and the control on the extra folds
     3-8, two seeds.
   - Final confirmation needs the owner's sealed donors; not part of round 3.
7. **Disk.** At most 25 GB for the whole round-3 screen. Mode files hold the
   same arrays as `generated.npz`; if the budget would be exceeded, write
   fewer modes per fit rather than fewer fits.
8. **Deferred to round 4, noted so it is not forgotten:** the count-gradient
   estimator (Poisson forward with straight-through rate features; frozen-model
   probes in the repository README found straight-through and discrete
   gradients disagreeing in sign for some genes).

## Decisions after the build review (2026-11-11 11:55 UTC, before any round-3 score)

1. **T1 reference corrected.** The evaluator task showed, and a reviewer
   confirmed, that an ideal stage-blind generator read out through 32
   generated donors per target scores 0.474 (`resample_blind`), above the
   original threshold of 0.469: the finite read-out adds about 0.011. T1 is
   therefore "within 0.005 of `resample_blind` through the same read-out"
   (0.479 on folds 0-2). The grand mean (0.464), the direct train mean (0.462)
   and the oracle floor (0.452 ± 0.014) stay in every table.
2. **Control seed 1 is in stage 1** (three extra fits; review finding E5), so
   the control's seed noise is measured in the same run. One seed on three
   folds resolves differences of about 0.025; smaller stage-1 differences are
   not evidence.
3. **Stage-2 selection uses the interval, not the point estimate**: an arm
   goes to stage 2 only if its primary read-out beats the control's primary
   read-out by more than 2 SE of the paired difference (the leaderboard's
   `dCtrlP` interval excludes 0) or if it is needed as a declared null
   (particles, generator width). At most five arms.
4. **What the build already answered, without new training** (full reports:
   `runs/improve_20261109/smoke_r3/workflow_outputs_r3_build.json`):
   - Constant learning rates with normalised zero-momentum steps make the
     generator a random walk with no restoring force; more updates at a
     constant rate show no trend (late single-state scores fluctuate by
     0.005-0.04); averaging late states gains 0.03-0.09.
   - All 512 particles are used uniformly; they collapse to 4-17
     distinguishable donor profiles per stage; a free latent per donor leaves
     the same gap. Particle count and latent size are not the limit.
   - Generator width, depth and cell-noise size do not help a
     maximum-likelihood probe of the same architecture. A donor-dependent
     dispersion does (`g.dispz`).
   - Every gene-wide critic block makes the critic penalty stop binding (the
     penalty cap scales with the square root of the input width), and the
     critic gap rises to 11-20 logits. `t.kappa@0.25` restores the released
     model's absolute cap.
   - Stage carries no measurable pseudobulk information in these 37 donors
     (within-stage RMS 0.468 against total 0.466).
   - Open data: GEO GSE268599 (MADRC; 32 donors with published ADNC
     components, 8 Not AD; inferior temporal gyrus for 31; same gene axis)
     would roughly triple the Not AD donors without a data use agreement.

## Owner-requested combination run (2026-11-12 00:40 UTC)

Mikkel, after seeing the fold-0 interim scores: "do everything that helps
combined and see how it does. we are looking for the best model". This
departs from the staged rule above (combinations only after stage 1 is fully
scored, and only of factors that beat the control by more than 2 SE): the
combination was launched while stage 1 had 33 of 57 fits done, on the strength
of fold-0, single-seed leads for `t.kappa@0.25` and `g.dispz`. It is recorded
here so the selection history stays complete.

`runs/improve_20261109/jobs_r3_combo.txt`, 18 fits, folds 0-2:

- A. `g.dispz,pbulk,t.kappa@0.25`, seeds 0-3 (for seed ensembles).
- B. `g.dispz,genemean,pbulk,t.kappa@0.19`, seed 0: both gene-level blocks,
  with the penalty matched to the 4,150-wide critic input
  (sqrt(150/4150) = 0.19). Round 2's `genemean,pbulk` failed with the penalty
  unmatched.
- C. `g.dispz,pbulk,t.glr@0.5,t.kappa@0.25`, seed 0.

Read-outs fixed in advance: `snapens` (primary), then `stagehalf`,
`biascal@stagehalf`, `ema@0.9998`, and seed ensembles of each. Stage-blind
read-outs are reported separately. The single-factor arms of stage 1 remain
the evidence for which factor does what.

## Interfaces

- Arm strings as in round 2. Training-option flags are ordinary flags with the
  prefix `t.` (training and recipe), `g.` (generator architecture) or `d.`
  (critic architecture), for example `genemean@0.5,t.anneal@0.1,t.steps@40000`.
  Each `t.`/`g.`/`d.` flag is defined by the core task or registered by a
  plug-in file `r3/arm_<name>.py`. Fit folders:
  `fits_r3/{arm}/{panel}/fold{k}/seed{s}/`.
- Snapshots: generator and prior state every 1,000 updates over the last 25%
  of training, plus every 5,000 updates throughout. Shadow EMA generator
  copies at decays 0.999, 0.9995 and 0.9998 are always kept (read-out only)
  and saved with the final state. Generation modes: `snapens`, `snap@U`,
  `ema@d`, and the round-2 modes.
- `config.json` records every departure from t005 under `departures`.
- Evaluation: `r3/evaluate.py` reads round-1, round-2 and round-3 fit roots and
  writes `results_r3/`, including `leaderboard.txt` and `targets.json`.
