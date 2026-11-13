# improve_20261109: verification and extra-fold confirmation

Written 2026-11-12 10:20 UTC by the study lead, BEFORE any fit on inner folds
3-8 exists for the arms below. Read `../PROTOCOL.md`, `../r3/PROTOCOL_R3.md`,
`PROTOCOL_EXT.md` and `../RESULTS.md` first; all hard rules there still bind.

## Why

Rounds 1-3 and the external-donor screen all used inner folds 0-2 of the same
37 SEA-AD development donors, and every design choice was made after seeing
held-out scores on those folds. Two changes survived stage 1 (donor-dependent
dispersion `g.dispz`; external MADRC donors pooled into training). This step
(a) re-derives the round-3 and external-donor numbers adversarially, and
(b) tests the two changes on inner folds 3-8 (two more stratified 3-fold
splits of the same 37 donors, `../r2/folds_r2.json`), which were never used to
choose anything. The donors are the same, so this is a robustness check, not
independent confirmation; that needs the owner's sealed donors.

## Confirmation design (2 x 2, pre-declared)

Arms, inner folds 3-8, seeds 0 and 1, 20,000 updates, released t005 recipe:

| | no external donors (`hvg2000`) | external donors pooled (`hvg2000x`) |
| --- | --- | --- |
| no dispersion change | `pbulk` | `pbulk` |
| donor-dependent dispersion | `g.dispz,pbulk` | `g.dispz,pbulk` |

Also `g.dispz,pbulk` on `hvg2000x` folds 0-2 (seeds 0-1), which completes the
same 2 x 2 on the original folds. 54 fits.

- Primary read-out: `snapens` (late-snapshot average), held-out `pb_rmse`.
- Secondary read-outs: `stagehalf`; `biascal@stagehalf`; two-seed ensembles of
  each; the six distribution metrics against real-donor resampling.
- Primary analysis: the main effect of each factor on `pb_rmse` at `snapens`
  over folds 3-8, as the mean paired difference (same fold and seed), with the
  seed-noise SE of the evaluator. A factor is confirmed if the difference is
  favourable in both repeats (folds 3-5 and 6-8) and exceeds 2 SE.
- Reference rows on the same folds: grand mean, ridge, resampling, the ideal
  stage-blind generator (`resample_blind`), and for `hvg2000x` the baselines
  with external donors.
- No other arm, read-out or fold may be added to this analysis afterwards.
  Anything else is reported as exploratory.

## Rules for this step

- Code: `r4/` may be extended (it is not in use by any running job when this
  step starts) but existing behaviour must not change: bitwise parity with
  round 1 (`base`) and r3 (`pbulk`), and identical caches for folds 0-2.
  Verification tasks write nothing but their own scripts and results under
  `runs/improve_20261109/verify_r3/` and `verify_ext/`.
- Only the 37 development donors and the two external series; sealed donors
  stay sealed. At most 3 CPU threads per process; no GPU for verification
  tasks; the cache task may run GPU test fits of at most 300 updates.
- No `.md` files from subagents; no git operations.
