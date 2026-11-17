# Native EMPF variability and coverage pass

Owner request, 2026-11-17: improve variability and combined coverage. Empirical
resampling is a reference that reuses observed cells, not the trained-network
deliverable. This pass compares changes to the selected native EMPF network;
it does not require a mixed empirical/GAN generator or replace failed old gates.

Keep the completed 20,000-update native EMPF fit, released hyperparameters,
`lazy_k=1`, six late states, stage-half handling, M32/K256 and the existing
generation streams. No new training or model-weight edits in this bounded pass.
Default config, historical source manifests, scores and receipts stay unchanged.
Use only the same 37 development donors. No sealed/external donors. CPU only,
one serial worker, three threads/cores, new outputs below 500MB.

Baseline diagnosis (already observed, not candidate evidence): mean-coordinate
coverage 80.10%, SD coverage 58.31%, combined 69.20%. SD misses are 8.40% below
and 33.29% above. TRAIN real/generated mean PC SD is .9041/.6912; held-out
.7634/.6858. The old fixed cell-mean contraction may remove too much variation
for native EMPF. Rank endpoints alone improve SD/combined coverage to
64.54/75.16%; they are a finite-draw diagnostic, not better generated cells.

## Fixed count-generator candidate

One candidate only: `cellshape@0.25:biascal:stagehalf`. Halve the inherited
contraction, preserving expected bag gene totals and per-cell library sizes.
Refit the existing joint bias/nonzero calibration on the 25 GAN TRAIN donors
only. This changes the network's generated count distribution; it does not
resample observed expression cells. Same frozen weights and random streams.
No held-out strength grid, alpha rescue, altered update rule or weight tuning.
Check TRAIN and held-out mean/SD/combined coverage, below/above misses, width,
proper interval score, and all frozen gene/profile/energy metrics.

Candidate advances only if held SD and combined coverage are both closer to
80%, both proper interval scores decrease, profile RMSE rises at most .5%
against the selected native baseline, and mean/detection/log-variance and
PC/cell energy errors rise at most 2%. This is a one-fold/seed development
screen, not a qualification or permission gate. A failed candidate is retained
as failed; it cannot replace the default from a coverage-only win.

## Separate targeted interval calibration

Fit nonnegative conformal-style interval corrections for SD coordinates only;
leave mean bounds exactly unchanged. Base bounds are the existing M32 10/90
percentiles. Fixed donor order: stable_seed('interval-calibration-split',
20261113, donor ID), matching the earlier all-coordinate CQR diagnostic.
Split the 12 GAN-held-out donors into two fixed halves of six, calibrate on
one half, evaluate on the other, then swap. Six donors means six calibration
examples per coordinate; neither 50 coordinates nor 24 cell subsamples inflate
the donor sample size. Fit from real summary repeat 0 only. Evaluate all 24
fixed subsamples in the disjoint test donors, matching historical scoring.

For each SD coordinate use score max(0, low-y, y-high), rank
ceil((n_cal+1)*.8), and expand bounds symmetrically by that order statistic.
Clip SD lower bounds at zero, its known support. No strength/quantile search.
Report both swap directions, assembled cross-fitted per-donor scores, widths
and coverage. A changed test donor's counts must not change its correction.
Calibration donors must never belong to that GAN's TRAIN set. A saved correction
cannot be evaluated on its own calibration donors without an explicit error.

Apply this one fixed procedure to the unchanged baseline and fixed candidate,
reporting every outcome, regardless of candidate success. A calibration result
is useful only if both SD and combined coverage approach 80% and both proper
scores improve. These are calibrated ranges around a trained network's output,
not a repaired cell generator or a guarantee of joint patient-level coverage.
All donors have been used in prior development selection; calibration/test
separation here does not create a new independent final test. No theoretical
coverage guarantee is claimed for this reused cohort and small calibration set.

Method references: [Romano, Patterson and Candès, 2019](https://arxiv.org/abs/1905.03222)
and [Shafer and Vovk, 2008](https://jmlr.org/papers/v9/shafer08a.html).

Pin application/model ancestry plus new execution sources before candidate
scoring. Preserve weights and raw metrics. Run leakage/support/mass-preservation
tests, inspect actual CLI output and record results in a local commit. A default
change requires subsequent evidence; this pass supplies an experimental readout
and interval-calibration workflow if it helps. No push is planned in this pass.
