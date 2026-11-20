# PCA shrinkage versus generator under-dispersion

Native EMPF fold0/seed0 covers only 58.3% of held-out donor SD coordinates
(target 80%). The coordinates come from a 50-PC basis fit on the 25 TRAIN
donors' cells, and held-out cells lose variance when projected onto it. This
CPU study asks how much of the SD under-coverage belongs to the generator and
how much to that basis. It rescores the existing pilot archives. It does no
training or generation by the network and changes no default. The
[protocol](PROTOCOL.md) was committed (e300bb7, 03:53 UTC) before any score
existed. Post-hoc corrections and an audit-driven repair of view (b1) are in
the [addendum](PROTOCOL_ADDENDUM.md).

**Short answer (verdict: mixed).** The basis inflates in-sample variance on
PCs 11-50, but most of the generator's SD under-coverage is not explained by
it. G0's shortfall sits on PCs 1-10, where TRAIN-estimated projection
shrinkage is small (factor .96).

* A model-free contrast estimates the projection effect. Stage-blind
  resampling from TRAIN cells (in-sample for the basis) is **over**-dispersed
  against held-out donors: log SD ratio +.101 overall. The same resampling
  from the *other held-out donors' cells* (an out-of-sample "oracle") is
  calibrated: -.008 [-.030, .016]. The paired difference is +.109
  [.100, .117] overall, +.124 on PCs 11-30 and +.191 on PCs 31-50. On PCs
  1-10 it is **-.086 [-.105, -.070]**: significant and opposite in sign, not
  absent. This contrast mixes basis in-sample-ness with population
  differences between the 25 TRAIN donors and the 11 other held-out donors
  (stage mix, depth). The PC 1-10 sign is a reminder of that confound. The
  independent TRAIN-only refit factors (b1) support an inflation of similar
  size on PCs 11-50. The paired difference in SD *coverage* is not
  significant (-6.7 points [-15.3, +3.0]), so the effect is established for
  the log SD ratio only.
* Against that oracle, in the same basis, the default generator G0 is
  under-dispersed on PCs 1-10 (log ratio -.474 [-.504, -.446]) and PCs 11-30
  (-.118). On PCs 31-50 it is slightly **over**-dispersed against the oracle
  (+.071 [.052, .088]), while it looks close to held-out SD in the raw view
  (+.030). This is consistent with G0 not reproducing TRAIN's in-sample
  inflation there.
* In the out-of-sample whitened basis (b2), 61% of G0's SD under-coverage
  remains: 66.7% coverage and log ratio -.090 [-.105, -.076]. This is
  **weak** evidence on its own, because b2 does not remove the projection
  effect it was meant to control: E_train's log ratio is +.101 in (a) and
  +.104 in (b2). The "not basis-explained" conclusion rests mainly on the
  oracle contrast on PCs 1-10 and on the b1 factor there (.96, log -.04),
  which is an order of magnitude smaller than G0's -.47 deficit.
* In gene space (per-gene within-donor SD of log CP10k), G0 has no SD
  under-coverage: 82.4% [80.4, 84.5], with every expression decile at
  76-86%. The deficit is therefore in coordinated top-PC variation rather
  than in marginal per-gene spread. We did not measure how much of per-gene
  SD is count-sampling noise. Four held-out donors have fewer than 256 cells
  (see Limits).
* The quarter readout G1 has an overall log ratio near 0 only because errors
  offset: -.319 [-.347, -.293] against the oracle on PCs 1-10 and +.152
  [.130, .174] on PCs 31-50.

Global leave-one-donor-out conformal calibration over the 12 held-out donors
brings G0 to 82.0% SD coverage, and its SD interval score improves from .806
to .586. The matched comparison, calibrated E_train with the same procedure,
scores .617: paired G0 minus E_train -.031 [-.062, +.006], indistinguishable
(exploratory). Stage-conditional (Mondrian) calibration cannot be done for 3
of the 4 stages in this fold, because their intervals are unbounded.

## Views and numbers (12 held-out donors; donor-bootstrap 95% CIs)

Log SD ratio = mean over donors and coordinates of log(median generated bag SD /
mean real SD); negative means the source is under-dispersed. Below and above
are miss rates: the real value lies below or above the generated 10/90 range.

### (a) Original TRAIN-fit basis

| Source | SD coverage | below / above | SD IS ↓ | log SD ratio: all | PCs 1-10 | 11-30 | 31-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 default (`cellshape@0.5`) | 58.3% [54.0, 62.0] | 8.4 / 33.3 | .806 | -.121 [-.142, -.099] | -.424 | -.121 | +.030 |
| G1 quarter (`cellshape@0.25`) | 62.2% [55.2, 68.7] | 20.2 / 17.6 | .668 | -.019 [-.039, .001] | -.269 | -.024 | +.111 |
| E_train, stage-blind resampling (reference) | 69.8% [63.1, 76.0] | 27.6 / 2.5 | .612 | +.101 [.080, .122] | -.037 | +.121 | +.150 |
| E_train11 (pool-size control) | 69.3% | 28.0 / 2.7 | .644 | +.099 | -.047 | +.118 | +.153 |
| E_heldpool, oracle (uses held-out cells) | 76.5% [70.9, 81.4] | 10.7 / 12.8 | .474 | -.008 [-.030, .016] | +.050 | -.003 | -.041 |

Per-block SD coverage for G0 is 24.6% / 58.5% / 75.0% on PCs 1-10 / 11-30 /
31-50; for G1 it is 44.8% / 71.7% / 61.4%. Pool size has no effect: E_train
minus E_train11 is +.002 [-.007, .012] in log ratio. Even the oracle reaches
only 76.5%, so the attainable stage-blind coverage in this basis is below 80%.
Paired G0 minus oracle: SD coverage -18.2 points [-22.4, -13.7], log ratio
-.114 [-.123, -.104].

### (b1) TRAIN-only shrinkage correction (one-sided)

The factors come from 25 leave-one-TRAIN-donor-out refits of the frozen
`build_cache.fit_space` recipe. The first run (aa99dda) deviated from that
recipe by sampling the SVD subsample with replacement; this was repaired
and b1 recomputed (see [addendum](PROTOCOL_ADDENDUM.md)). With the repaired
recipe, a refit on all 25 TRAIN donors reproduces the frozen `space.npz`:
mean and SD are identical, pc_sd differs by at most 2.4e-6 relative, and
every one of the 50 PCs has |cos| ≥ .9999994 with its frozen counterpart.

The out-of-sample/in-sample within-donor SD factor averages **.957 / .849 /
.764** on PCs 1-10 / 11-30 / 31-50 (log -.04 / -.16 / -.27). The first run
gave .95 / .86 / .82.

| Source after correction | SD coverage | below / above | SD IS ↓ | log SD ratio |
| --- | ---: | ---: | ---: | ---: |
| G0 | 20.9% [16.8, 25.8] | 0.2 / 79.0 | 1.332 | -.305 [-.326, -.283] |
| G1 | 47.3% [41.2, 53.0] | 0.7 / 52.0 | .836 | -.202 [-.222, -.182] |
| E_train | 82.9% [78.6, 86.7] | 3.2 / 13.9 | .483 | -.082 [-.103, -.062] |

Multiplying generated SD coordinates by the factor fixes the in-sample
reference: E_train goes from 69.8% to 82.9%. On PCs 31-50 the factor
over-corrects E_train (-.12 log afterwards), so the late-PC factors are
larger than the oracle contrast implies. The correction makes G0 much worse
(20.9%). G0 therefore does not carry TRAIN's in-sample inflation, the
correction does not apply to it, and for G0 this view is an upper bound on
under-dispersion, not an estimate. Rescaling real and generated coordinates
by the same per-coordinate constant cannot change coverage (tested).

The first run reported that late PCs are unstable across refits of the
recipe (mean alignment .53 on PCs 31-50). That came from the recipe
mismatch: with the frozen recipe the full-TRAIN refit is identical. This
study did not measure PC stability under leave-one-donor-out refits, so that
claim is withdrawn.

### (b2) Out-of-sample basis: leave-one-held-out-donor-out whitening of the 50 PCs

| Source | SD coverage | below / above | SD IS ↓ | log SD ratio | rank 1-10 | 11-30 | 31-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 66.7% [62.8, 70.0] | 7.8 / 25.5 | .786 | -.090 [-.105, -.076] | -.379 | -.006 | -.031 |
| G1 | 68.2% [63.8, 72.5] | 18.6 / 13.2 | .726 | .000 [-.016, .016] | -.255 | +.079 | +.050 |
| E_train | 68.2% [60.6, 75.0] | 30.2 / 1.6 | .784 | +.104 | -.032 | +.151 | +.125 |
| E_heldpool (in-sample for this rotation) | 74.6% | 8.6 / 16.8 | .576 | -.026 | +.062 | -.023 | -.074 |

TRAIN cells stay over-dispersed after rotation (+.104, against +.101 in the
original basis). Any rotation inside the TRAIN-fit 50-dimensional subspace
keeps the inflation of their total variance. So b2 does not control the
projection effect, and G0's survival in b2 is weak evidence by itself.

### (c) Gene space (basis-free; 2,000 genes; deciles of TRAIN mean expression)

| Source | SD coverage | below / above | SD IS ↓ | top-decile SD coverage | gene-mean coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 82.4% [80.4, 84.5] | 5.7 / 11.9 | .931 | 76.1% | 81.9% |
| G1 | 82.1% [79.9, 84.2] | 5.7 / 12.3 | .929 | 74.3% | 81.9% |
| E_train | 80.7% [78.4, 82.8] | 4.9 / 14.4 | .899 | 78.1% | 81.5% |
| E_heldpool | 78.3% [74.6, 81.4] | 7.1 / 14.6 | .938 | 72.9% | 78.8% |

The coverage gradient across deciles is shared by all sources, the oracle
included, so it reflects the metric rather than the generator. Gene-space
interval scores are in log CP10k units and are not comparable with PC units.

### Survival of SD under-coverage (U = max(0, 80% - SD coverage))

| | (a) original | (b2) out-of-sample basis | (c) gene space | (b1) one-sided correction |
| --- | ---: | ---: | ---: | ---: |
| G0 U, share of (a) remaining | .217 (100%) | .133 (61%) | 0 (0%) | .591 (273%) |
| G0 under-dispersion survives (log-ratio CI < 0) | yes | yes | no | yes |
| G1 U, share of (a) remaining | .178 (100%) | .118 (66%) | 0 (0%) | .327 (184%) |

The declared rule calls under-dispersion "basis-explained" only if it fails to
survive in both (b2) and (c). For G0 it survives in (b2), so by the rule it is
**not** basis-explained. Because b2 leaves E_train's inflation unchanged, the
stronger support is the oracle contrast. On PCs 1-10, G0 covers 25% of SD
coordinates against the oracle's 74%, and its log ratio against the oracle
is -.47. The TRAIN-estimated shrinkage there is only .96 (log -.04).

## Conformal calibration (SD coordinates, original basis)

Base bounds: the frozen 10/90 percentiles. Calibration uses real repeat 0 of
the other held-out donors; evaluation uses all 24 repeats. The CQR signed
score expands bounds symmetrically, and SD lower bounds are clipped at 0.

| Source | Calibration | SD coverage | SD width | SD IS ↓ | Combined coverage | Combined IS ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0 | none | 58.3% | .222 | .806 | 69.2% | .945 |
| G0 | global LOO (n_cal 11, rank 10) | 82.0% [78.6, 85.0] | .448 | .586 | 81.1% | .835 |
| G1 | global LOO | 84.9% [80.2, 88.8] | .470 | .581 | 82.8% | .832 |
| E_train | none (reference) | 69.8% | .413 | .612 | 78.2% | .832 |
| E_train | global LOO | 84.3% | .522 | .617 | 85.5% | .835 |

Leave-one-donor-out calibration with 11 calibration donors has a finite-sample
marginal guarantee of at least 10/12 = 83.3% per coordinate. Calibration uses
real repeat 0 and evaluation uses 24 overlapping subsample repeats.

Paired donor-bootstrap differences in SD interval score (exploratory, not
declared in the protocol):

| Comparison | Difference [95% CI] |
| --- | ---: |
| G0 calibrated minus E_train calibrated (**matched**) | -.031 [-.062, +.006] |
| G0 calibrated minus E_train raw (unmatched: only G0 uses held-out calibration data) | -.026 [-.077, +.026] |
| G1 calibrated minus G0 calibrated | -.005 [-.019, +.007] |
| G0 calibrated minus G0 raw | -.220 [-.330, -.121] |

Calibrated G0 is indistinguishable from calibrated stage-blind resampling.
The earlier 6/6 split calibration gave G0 85.55% coverage and an SD IS of
.624. Next to .586 here, that is an unpaired point comparison without
uncertainty and was not declared. These are calibrated ranges around the
generator's output; they do not repair the top-PC under-dispersion.

**Mondrian by stage.** Fold 0's held-out donors are Not AD 1, Low 2,
Intermediate 3 and High 6. Leave-one-out within stage leaves 0, 1, 2 and 5
calibration donors. The required rank ceil((n+1)·.8) exceeds n for the first
three stages, so those intervals are **unbounded**: 6 of 12 donors, with 100%
coverage and infinite width and score. Only High is calibratable, at rank 5 of 5
(guarantee 83.3%). It gives G0 83.7% SD coverage, width .527 and IS .651. With
global calibration, per-stage G0 coverage is 77.7 / 85.4 / 85.8 / 79.7%
(Not AD / Low / Intermediate / High). These are point estimates from 1, 2, 3
and 6 donors and carry no per-stage guarantee. Stage-conditional calibration
needs donors pooled across folds; one fold cannot support it.

## Limits

* One fold and one seed (fold 0/seed 0). All 12 held-out donors were used in
  earlier development selection, so this is not an independent test.
  Percentile donor-bootstrap CIs over 12 donors tend to under-cover. They
  also exclude variation from generator seed, resampling replicate (E_train
  replicate 0 only) and fold.
* E_heldpool uses held-out cells. It is a diagnostic oracle, not a
  generator, and is in-sample for the (b2) rotation. It differs from TRAIN
  in population as well as basis status, so the E_train-minus-oracle contrast
  is not a pure projection effect. The E_train11 control rules out pool size
  only (+.002 [-.007, .012]).
* Generated archives store cells only as 64-cell TRAIN-basis projections and
  256-cell gene moments. A full gene-space leave-donor-out PCA cannot be
  applied to generated cells without regeneration, so (b2) is a rotation
  inside the TRAIN 50-PC subspace.
* Gene space: SD comes from float16 second moments in the archives, and the
  real side is rounded the same way. Low-expression ratios use epsilon .01;
  their CIs are wide. Four of the 12 held-out donors have fewer than 256
  kept cells (158, 184, 209 and 241). For them the real per-gene SD uses all
  cells, the 24 repeats are identical, and the cell count does not match the
  256-cell generated bags. The share of per-gene SD due to count noise was
  not measured.
* The b1 factors on PCs 31-50 over-correct the in-sample reference and are
  approximate.
* Paired conformal comparisons were added after the protocol and are
  exploratory. The guarantee is marginal and per coordinate.
* Frozen historical metrics (profile RMSE, gene errors, PC/cell energy) were
  recomputed for G0, G1 and E_train. They match the stored scores exactly,
  since no generated cells changed.

## Reproduce

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
taskset -c 3-23 $PY -B -m pytest -q -p no:cacheprovider analysis/pca_shrinkage_20261120/test_shrinkage.py
taskset -c 3-23 $PY -B analysis/pca_shrinkage_20261120/run.py   # ~11-26 min, CPU, 3 threads
```

`shrinkage.py` is the reusable NumPy API. It provides per-block interval
tables, log SD ratios, donor bootstrap, whitening, gene-moment SDs and
leave-one-donor-out global or Mondrian conformal calibration with eligibility
and self-exclusion guards. `references.py` is the stage-blind resampler with
a configurable donor pool; with the TRAIN pool, the tests check it
bit-for-bit against the frozen baseline. The run checks input hashes and
reproduces the pilot's raw held-out interval numbers and the committed
empirical-challenge fold0/replicate0 score exactly, or it stops. Arrays are in
`runs/research_20261120/pca-shrinkage/arrays.npz`.
