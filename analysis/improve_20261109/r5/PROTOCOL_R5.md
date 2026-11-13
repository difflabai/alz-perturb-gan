# improve_20261109 round 5: short training with averaging; per-gene mean and nonzero gaps

Owner decision (Mikkel, 2026-11-12): "yes do 1 and 2", in reply to the two
next steps proposed after the adversarial verification (`../RESULTS.md` 8e):

1. Short training plus averaging. Along the 60,000-update runs the
   10,000-update state scored better than the 20,000-update final state
   (0.595 against 0.645, one seed), and this was never tested.
2. The per-gene mean and nonzero-fraction gaps: the two distribution metrics on
   which every generator is still behind real-donor resampling
   (`meanlog_rmse_gene` about 0.10 against 0.079; `nonzero_rmse_gene` about
   0.022 against 0.017).

Read `../PROTOCOL.md`, `../r3/PROTOCOL_R3.md`, `../r4/PROTOCOL_CONFIRM.md` and
`../RESULTS.md` (8e first) before anything else. All hard rules still bind.

## Rules

- `r3/` and `r4/` are FROZEN: the confirmation screens are training with them.
  Round-5 code lives in `/ml2/alz/analysis/improve_20261109/r5/` (already
  created as a copy of `r3/`). Outputs only under
  `runs/improve_20261109/{fits_r5,results_r5,smoke_r5,logs_r5}`. Existing fit
  folders are read-only: generation modes for older fits are written with
  `--out` under `smoke_r5/`.
- No external donors (8e: pooling contaminates the generated cells). Only the
  37 SEA-AD development donors. Sealed donors stay sealed.
- Still ParticleGAN 0.9.7 with its own update machinery; departures from t005
  only as named flags recorded in `config.json`.
- With no round-5 flag, `r5/trainer.py` stays bitwise equal to round 1 (`base`)
  and r3 (`pbulk`).
- Every calibration is fitted on the fold's TRAIN donors only.
- Comparisons are like-for-like: the same number of generated donors per
  target (M = 32 unless stated), the same calibration applied to the
  reference, the same folds and seeds, paired by fold and seed. Report the
  seed-noise SE and the fold scatter (t on the per-fold differences), as the
  8e verifier required.
- Machine: at most 3 CPU threads per process; GPUs are shared with the
  confirmation screens: generation and test fits of at most 300 updates, one
  process at a time per task, `--gpu-mem-fraction 0.04`. Disk is 99% full
  (about 75 GB free): delete mode files once scored, keep under 3 GB per task.
- No `.md` files from subagents; no git operations; do not kill processes you
  did not start. Python: `/mnt/nvme/ai/conda/anaconda3/bin/python3 -B`.

## Work item 1: when to stop, and what to average (no new training)

With constant learning rates the declared horizon does not affect the
trajectory (verified bitwise), so the state of a 20,000-update fit at update U
is the final state of a U-update fit. Every round-3 fit has snapshots at 5,000,
10,000 and 15,000-19,000 updates, each carrying shadow EMAs at decays 0.999,
0.9995 and 0.9998. The question is answered by generating from those states.

- Candidate read-outs, fixed here: `snap@U` and `snapema@U:D` for U in
  {5000, 10000, 15000} and D in {0.999, 0.9995}; references `snapens`
  (15,000-19,000 plus final), `ema@0.9995` and the final state.
- Selection: choose the best candidate by held-out `pb_rmse` on folds 0-2,
  seed 0, of the control `pbulk`. Confirm on seed 1 of the control, on the
  combination `g.dispz,pbulk,t.kappa@0.25` (seeds 0-3) and, when the
  confirmation screen has finished, on folds 3-8 (`pbulk`, seeds 0-1). A
  read-out replaces `snapens` as the default only if it is better on the
  confirmation sets by more than 2 seed-noise SE and in both extra-fold
  repeats.
- Also report every candidate at the stage-blind read-out and with bias
  calibration, and the six distribution metrics.

## Work item 2: the count-shape gap

- Diagnose first, from existing fits: how much of the per-gene mean-log and
  nonzero-fraction error is explained by the mean (it disappears once the
  pseudobulk mean is calibrated) and how much is shape (dispersion, excess
  zeros, cell-to-cell heterogeneity); which genes; whether generated cells are
  under- or over-dispersed relative to real cells at matched depth.
- Fix A, no training: `zcal`, a train-only per-gene calibration of the
  generator's dispersion (and, if the diagnosis supports it, a per-gene
  zero-inflation term) so that the generated nonzero fraction over the fold's
  train targets matches the observed one, composable with `biascal`.
- Fix B, training: `pzero`, a critic block with the per-gene fraction of cells
  with zero counts in the bag (straight-through for generated bags), with the
  critic penalty matched to the input width (`t.kappa`), alone and with
  `pbulk`.
- Targets: `meanlog_rmse_gene` and `nonzero_rmse_gene` at least as good as
  real-donor resampling at matched M, without making `pb_rmse`,
  `pc_energy_rep` or `cell_energy_rep` worse than the control beyond 2
  seed-noise SE. A calibration is reported next to the same calibration
  applied to the resampling reference.
