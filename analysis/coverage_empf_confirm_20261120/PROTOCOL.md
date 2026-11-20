# Native EMPF across folds and seeds: confirmation of the default and the coverage readout

Written 2026-11-20 03:55 UTC, before any fit named below exists. Owner request
the same day: "continue but pull and fix anything you need to". The README and
`../coverage_empf_20261117/README.md` both state the outstanding step: the
owner-selected native EMPF default and the quarter-contraction readout rest on
one fold and one seed (fold 0, seed 0), and "additional native folds/seeds are
needed before changing the public default". This pass supplies them.

## Fits (training, GPU)

Native EMPF exactly as the committed default (`configs/microglia_default.json`,
`scripts/microglia.py train`): ParticleGAN 0.10.0, released native preset,
`pbulk`, 20,000 updates, `lazy_k=1`, released hyperparameters. New fits:

- folds 0-2: fold 0 seed 1; folds 1-2 seeds 0-1 (5 fits);
- folds 3-8: seeds 0-1 (12 fits).

17 new fits. The pilot (fold 0, seed 0) is reported but is NOT part of the
confirmation set, because the readout candidate was chosen on it. Fits go to
the CLI's own output root (`runs/microglia_empf/fits/`); the retained pilot and
every older fit folder stay untouched. GPU memory fraction 0.04, three threads
per process; the GPUs are shared with other jobs.

## Read-outs (no further training)

For every fit, with the pilot's code paths and generation streams:

- A. default: `cellshape@0.5:biascal:stagehalf`;
- B. candidate: `cellshape@0.25:biascal:stagehalf` (bias/nonzero calibration
  refitted on that fit's TRAIN donors only);
- A+cal and B+cal: the pilot's SD-only interval calibration, two fixed 6/6
  (or 6/7) held-out donor halves per fold, cross-fitted, donor order from
  stable_seed('interval-calibration-split', 20261113, donor ID).

Same 37 development donors; no sealed or external donor counts; no empirical
expression-cell resampling in any generator.

## Primary question 1: does the quarter-contraction readout confirm?

Paired B minus A per fit, over the 17 confirmation fits. B confirms if, on the
mean over the 17 fits, all ten pilot gates pass with the pilot's thresholds:

1-2. SD coverage and combined coverage both closer to 80%;
3-4. SD and combined proper interval scores both lower;
5. profile RMSE (`pb_rmse`) higher by at most 0.5%;
6-10. `meanlog_rmse_gene`, `nonzero_rmse_gene`, `logvar_rmse_gene`,
   `pc_energy_rep`, `cell_energy_rep` each higher by at most 2%;

AND the two proper interval scores are lower in each of the three repeats
(folds 0-2 without the pilot, 3-5, 6-8). Report the paired SE over fits and
the number of fits that improve, for every metric.

## Primary question 2: does the SD calibration confirm?

For A and for B separately, on the mean over the 17 fits and in each repeat:
SD and combined coverage both closer to 80%, and SD and combined proper scores
both lower than without calibration. Report widths. Calibration widens ranges
around a trained network's output; it does not change generated cells.

## Secondary: the owner-selected EMPF default against the previous default

For each fold and seed where both exist, native EMPF readout A against the
previous BISPF count-shape default (`pbulk`, same readout family) on the same
held-out donors, paired: profile RMSE, the five gene/energy errors, coverage
and proper interval scores. This is descriptive: the owner selected EMPF; this
pass does not reverse an owner selection. Also report stage-blind empirical
resampling as the reference that still wins interval scores in the pilot.

## Reading rules

Lower-is-better metrics are read only next to donor spread and coverage
(RESULTS.md 8g: collapse toward the pooled mean lowers RMSE and energy). No
read-out, threshold, fold or metric may be added after the first new score
exists; anything else is labelled exploratory. A pass is development evidence
on reused donors, not an independent final test; that needs the sealed donors.

## Rules

New code only under `analysis/coverage_empf_confirm_20261120/`; new outputs
only under `runs/coverage_empf_confirm_20261120/` and the CLI's own fit root.
Existing source, protocols, fits and evidence files are read-only. CPU work at
most three threads per process. No pushes.
