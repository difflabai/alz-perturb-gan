# PCA shrinkage versus generator under-dispersion

Native EMPF fold0/seed0 covers only 58.3% of held-out donor SD coordinates
(target 80%). The coordinates come from a 50-PC basis fit on the 25 TRAIN
donors' cells, and held-out cells lose variance when projected onto it. This
CPU study asks how much of the SD under-coverage belongs to the generator and
how much to that basis. It rescores the existing pilot archives. It does no
training or generation by the network and changes no default. The
[protocol](PROTOCOL.md) was committed before any score existed.

**Short answer.** The basis matters for PCs 11-50, but most of the generator's
SD under-coverage survives. That shortfall sits on PCs 1-10, where projection
shrinkage is small.

* A model-free check isolates the projection effect. Stage-blind resampling
  from TRAIN cells (in-sample for the basis) is **over**-dispersed against
  held-out donors: log SD ratio +.10 overall and +.15 on PCs 31-50. The same
  resampling from the *other held-out donors' cells* (an out-of-sample
  "oracle") is calibrated, at -.01 [-.03, .02]. On PCs 11-50, the difference
  (+.12 to +.19 log SD) is pure projection inflation.
* Against that oracle, in the same basis, the default generator G0 is
  under-dispersed on PCs 1-10 (log ratio -.47 [-.50, -.45]) and PCs 11-30
  (-.12). On PCs 31-50 it matches held-out SD. It does not reproduce TRAIN's
  in-sample inflation there.
* In an out-of-sample basis (rotation fit leave-one-donor-out on the *other*
  held-out donors), 61% of G0's SD under-coverage survives: 66.7% coverage,
  versus 58.3% originally. Its under-dispersion survives (log ratio -.090
  [-.105, -.076]) and is concentrated on the top components (-.38).
* In gene space (per-gene within-donor SD of log CP10k), G0 has **no**
  under-coverage: 82.4% [80.4, 84.5]. Every expression decile has 76-86%
  coverage. Per-gene SD is dominated by count-sampling noise, which the
  calibrated count readout reproduces. The deficit is in *coordinated*
  multi-gene variation, which the top PCs capture, not in marginal per-gene
  spread.
* The quarter readout G1 has an overall log ratio near 0 only because it
  over-disperses PCs 31-50 (+.15 against the oracle). It still
  under-disperses PCs 1-10 (-.32).

Leave-one-donor-out conformal calibration over all 12 held-out donors brings
G0 to 82.0% SD coverage. Its SD interval score improves from .806 to .586.
Stage-conditional (Mondrian) calibration cannot be done honestly for 3 of the
4 stages in this fold: with 1, 2 and 3 held-out donors, those intervals are
unbounded.

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

From 25 leave-one-TRAIN-donor-out refits of the cache recipe, the
out-of-sample/in-sample within-donor SD factor averages **.95 / .86 / .82** on
PCs 1-10 / 11-30 / 31-50. Multiplying generated SD coordinates by it fixes the
in-sample reference: E_train goes from 69.8% to 86.5%, log ratio -.05. It makes
G0 worse: 29.9%, log ratio -.27. G0 therefore does not carry TRAIN's in-sample
inflation, so this correction does not apply to the generator. It bounds
G0's under-dispersion from above; it does not estimate it. Rescaling real and
generated coordinates by the same per-coordinate constant cannot change
coverage (tested). A "rescaled basis" matters only through such a one-sided
correction.

The refit reproduces the frozen top PCs (the ten best principal cosines are
at least .96). Late PCs are not reproducible: the smallest principal cosine is .02.
The mean norm of each frozen PC's projection onto the refit span is .98, .77
and .53 for PCs 1-10, 11-30 and 31-50. The PCs 31-50 directions are
themselves unstable fitting noise, which is another reason not to read
generator quality from them.

### (b2) Out-of-sample basis: leave-one-held-out-donor-out whitening of the 50 PCs

| Source | SD coverage | below / above | SD IS ↓ | log SD ratio | rank 1-10 | 11-30 | 31-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 66.7% [62.8, 70.0] | 7.8 / 25.5 | .786 | -.090 [-.105, -.076] | -.379 | -.006 | -.031 |
| G1 | 68.2% [63.8, 72.5] | 18.6 / 13.2 | .726 | .000 [-.016, .016] | -.255 | +.079 | +.050 |
| E_train | 68.2% [60.6, 75.0] | 30.2 / 1.6 | .784 | +.104 | -.032 | +.151 | +.125 |
| E_heldpool (in-sample for this rotation) | 74.6% | 8.6 / 16.8 | .576 | -.026 | +.062 | -.023 | -.074 |

TRAIN cells stay over-dispersed after rotation. Any rotation inside the
TRAIN-fit 50-dimensional subspace keeps the inflation of their total variance.

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
| G0 U, share of (a) remaining | .217 (100%) | .133 (61%) | 0 (0%) | .501 (231%) |
| G0 under-dispersion survives (log-ratio CI < 0) | yes | yes | no | yes |
| G1 U, share of (a) remaining | .178 (100%) | .118 (66%) | 0 (0%) | .241 (135%) |

The declared rule calls under-dispersion "basis-explained" only if it fails to
survive in both (b2) and (c). For G0 it survives (b2), so it is **not**
basis-explained. It is confined to coordinated top-PC variation. A deficit of
that size on PCs 1-10 (25% coverage against the oracle's 74%) cannot come
from projection, because the shrinkage factor there is .95.

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
marginal guarantee of at least 10/12 = 83.3%. It is tighter than the earlier
6/6 split, which took the maximum of six scores and guaranteed 6/7 = 85.7%;
that split gave G0 85.55% coverage and an SD IS of .624. After calibration,
G0's SD interval score is numerically below raw stage-blind resampling's. The
exploratory, undeclared paired difference is -.026 [-.077, +.026], so the two
are indistinguishable. G1 minus G0 after calibration is -.005 [-.019, +.007].
Calibration's gain over raw G0 is -.220 [-.330, -.121]. The combined scores
are also indistinguishable (.835 against .832). These are calibrated ranges around the
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
  earlier development selection; this is not an independent test. Bootstrap
  CIs resample 12 donors and are optimistic about between-fold variation.
* E_heldpool uses held-out cells. It is a diagnostic oracle, not a generator.
  Its pool (11 donors) is smaller than TRAIN's (25), but the E_train11
  control shows that pool size does not matter.
* Generated archives store cells only as 64-cell TRAIN-basis projections and
  256-cell gene moments. A full gene-space leave-donor-out PCA cannot be
  applied to generated cells without regeneration. (b2) is therefore a
  rotation inside the TRAIN 50-PC subspace.
* Gene-space SD comes from float16 second moments in the archives. The real
  side is rounded the same way. Low-expression ratios use epsilon .01; their
  CIs are wide.
* Frozen historical metrics (profile RMSE, gene errors, PC/cell energy) were
  recomputed for G0, G1 and E_train. They match the stored scores exactly,
  since no generated cells changed.

## Reproduce

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
taskset -c 3-23 $PY -B -m pytest -q -p no:cacheprovider analysis/pca_shrinkage_20261120/test_shrinkage.py
taskset -c 3-23 $PY -B analysis/pca_shrinkage_20261120/run.py   # ~11 min, CPU, 3 threads
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
