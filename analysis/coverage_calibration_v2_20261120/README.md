# EMPF coverage pilot: block, stage and donor-CI rescore, and calibration v2

This is a CPU-only re-analysis of the
[2026-11-17 EMPF coverage pilot](../coverage_empf_20261117/README.md). It adds
readouts and a revised SD calibrator; it does not generate cells, train, or
change the default. The pilot's pinned files (`calibration.py`, `run.py`,
`PROTOCOL.md`, its test file and its inputs) are hash-checked before and after
and are not edited. Everything here concerns one native EMPF fit (fold 0,
seed 0) and its 12 held-out development donors, which were already used in
development selection. The donor-bootstrap intervals below resample only these
12 donors. They do not include fold, seed or training variability.

## Reproduction of the pilot

[`rescore_pilot.py`](rescore_pilot.py) reloads the fold-0 cache and the two
retained generated archives, checks their SHA-256 against the committed
[pilot comparison](../coverage_empf_20261117/completion/comparison.json), and
recomputes the pilot's raw and split-half-calibrated aggregates with the
pinned calibrator. All 90 compared values (held-out raw, held-out calibrated
and TRAIN raw; mean/SD/combined; five metrics) agree to within 2.2e-16. For
example, the SD interval score is .806254 (current) and .667691 (quarter).

## Corrections to the pilot README wording

1. **"Coverage targets 80%."** With six calibration donors and alpha=.2 the
   conformal rank is ceil(7 × .8) = 6 = n. Each SD correction is therefore the
   maximum score over the six donors. The finite-sample nominal level is 6/7 =
   85.7%, not 80%. The observed calibrated SD coverage (85.55% current, 86.65%
   quarter) matches that level and should not be read as an overshoot of an 80%
   target. `calibration_v2.fit_sd` records `nominal_coverage = rank/(n+1)` and
   warns when `rank == n`.
2. **"The quarter readout plus calibration has lower proper scores."** For the
   SD score this is supported: 2.1% lower, 95% donor-bootstrap CI [0.5, 4.0],
   with 9 of 12 donors lower. For the combined score the difference (0.8%,
   [-0.1, 1.8], 9 of 12 donors) is within donor uncertainty. Under the
   leave-one-donor-out calibration below, neither difference remains (SD 0.3%
   [-1.8, 2.8]; combined 0.2% [-0.9, 1.3]).

## The equal-weight headline combines opposite block effects

Held-out donors, raw M32 10/90 percentile bounds. Interval scores: lower is better.

| SD coordinates | Coverage, current | Coverage, quarter | Below-miss, current → quarter | Score, current | Score, quarter |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCs 1–10 | 24.6% | 44.8% | 0.2% → 2.2% | 2.142 | 1.476 |
| PCs 11–30 | 58.5% | 71.7% | 3.2% → 12.4% | .597 | .431 |
| PCs 31–50 | 75.0% | 61.4% | 17.7% → 37.0% | .347 | .500 |
| All 50 (pilot headline) | 58.3% | 62.2% | 8.4% → 20.2% | .806 | .668 |
| Variance-weighted | 42.8% | 55.2% | 4.4% → 11.8% | 1.337 | .938 |

The variance-weighted row weights each PC's coordinates by that PC's TRAIN-cell
variance (`pc_sd²`, normalized). The pilot's equal-weight row gives the 20
late PCs 40% of the weight.

Paired donor-bootstrap contrasts, quarter vs current, raw bounds (B = 20,000,
seed 20261120, percentile 95% CI; positive score change = quarter lower):

| Contrast | Estimate [95% CI] | Donors with lower quarter value |
| --- | ---: | ---: |
| SD score, all PCs | 17.2% [13.9, 21.6] | 12/12 |
| Combined score, all PCs | 7.4% [5.9, 9.1] | 12/12 |
| SD score, PCs 1–10 | 31.1% [26.9, 37.0] | 12/12 |
| SD score, PCs 31–50 | −43.9% [−63.2, −22.1] | 2/12 |
| SD coverage, all PCs (points) | +3.9 [−1.3, +9.8] | — |
| SD coverage, PCs 1–10 (points) | +20.2 [+15.6, +24.9] | — |
| SD coverage, PCs 31–50 (points) | −13.6 [−22.5, −2.9] | — |

The 17.2% SD gain is robust across these 12 donors. It comes from a large gain on the top
PCs together with a clear loss on PCs 31–50, where the quarter readout's
intervals now sit above the observed SDs. The overall SD coverage change is
within donor uncertainty. The pilot's split-half calibration also worsens the
current model's PCs 31–50 SD score (.347 → .394; −13.5% [−22.9, −5.2]).
[`rescore.json`](rescore.json) has every group × block × readout and all
contrasts.

## The held-out deficit on late PCs is PCA shrinkage

[`pca_shrinkage.py`](pca_shrinkage.py) refits the cache's PCA recipe. With all
25 TRAIN donors, the refit reproduces the cache basis: identical gene
scaling, PC SDs within 2.4e-6 relative, and loading cosines ≥ .999999. It
then (a) leaves each TRAIN donor out and projects that donor, and (b) adds
each held-out donor in turn and projects that donor in-sample. Values are the
mean within-donor 64-cell bag SD per block, using the 24 fixed subsamples
behind the interval targets:

| Real donors | PCs 1–10 | PCs 11–30 | PCs 31–50 |
| --- | ---: | ---: | ---: |
| TRAIN, in-sample (cache basis) | .746 | .892 | .995 |
| TRAIN, each left out of a refit | .716 | .756 | .762 |
| Held-out, out-of-sample (cache basis) | .802 | .751 | .757 |
| Held-out, each added to a refit | .808 | .877 | .987 |
| Generated, current, held-out targets | .516 | .672 | .784 |
| Generated, quarter, held-out targets | .598 | .743 | .852 |

Moving the same donors in or out of the PCA fit moves their late-PC SD between about
.76 and .99. The held-out reduction on PCs 11–50 is therefore an out-of-sample
PCA effect, not a property of these donors. Generated cells are new cells, so
they behave like out-of-sample cells. On PCs 31–50 the current readout is
close to the out-of-sample real level (.784 vs .757–.762), and the quarter
readout exceeds it (.852). This matches the quarter readout's increase in below-misses there.
On PCs 1–10 both readouts are under-dispersed under every reference (.52–.60
vs .72–.81). The same effect explains why raw TRAIN SD coverage (27.7% current,
49.9% quarter) is lower than held-out coverage (58.3%, 62.2%): TRAIN targets
on later PCs carry in-sample PCA variance that no generated cell has. This
is a fold-0 diagnostic. Refit PCs are not matched one-to-one with cache PCs,
so only block means are compared.

## Calibration v2

[`calibration_v2.py`](calibration_v2.py) keeps the pilot's nonnegative,
SD-only, donor-level design. Mean bounds are exactly unchanged and SD lower
bounds are clipped at zero. It adds:

- the recorded nominal level `rank/(n+1)` and a `RankAtMaximumWarning` when
  `rank == n`;
- rejection of evaluation donors that were in GAN training (the pilot's
  `apply_sd` checked only calibration donors against the training set);
- optional separate lower and upper corrections, each at alpha/2. The joint nominal level then comes from a union bound;
- `loo_sd`: leave-one-donor-out calibration over every held-out donor. The
  generator never saw these donors and its bounds are not refit, so CV+ and
  jackknife+ reduce to split conformal with the other n − 1 donors. With 12
  held-out donors: n = 11, rank 10, nominal 10/12 = 83.3%;
- `pooled_loo_sd`: the same across several fits (folds/seeds), excluding every
  row that shares the evaluation donor's identity. Rows from different fits
  are only approximately exchangeable.

Held-out results, all PCs (the pilot's split-half rows reproduced for comparison):

| Readout | SD coverage | SD score | Combined coverage | Combined score |
| --- | ---: | ---: | ---: | ---: |
| Current + split-half (n=6, nominal 85.7%) | 85.55% | .623801 | 82.83% | .854009 |
| Quarter + split-half | 86.65% | .610538 | 83.68% | .846772 |
| Current + leave-one-out (n=11, nominal 83.3%) | 83.74% | .582626 | 81.92% | .833421 |
| Quarter + leave-one-out | 85.20% | .580913 | 82.95% | .831960 |
| Current + leave-one-out, lower/upper | 88.59% | .511278 | 84.35% | .797747 |
| Quarter + leave-one-out, lower/upper | 89.87% | .511806 | 85.29% | .797406 |

Leave-one-out calibration gives narrower SD ranges and lower scores than the
split halves, and its coverage is near its nominal level. With 11 donors at
alpha/2 = .1, the lower/upper variant is again at rank = n, so each side
takes the maximum. It scores lower on these donors because the misses are lopsided:
the current readout's SD misses are 33% above and 8% below. That variant was
chosen after the pilot's diagnosis and is evaluated on the same 12 donors, so
it is a candidate to test prospectively, not a confirmed improvement. Under
leave-one-out calibration the quarter readout still scores better on PCs 1–10
(SD 10.6% [9.4, 11.7]) and worse on PCs 31–50 (−29.3% [−36.9, −21.0]).

## Stages

There are 12 held-out donors: 1 Not AD, 2 Low, 3 Intermediate and 6 High.
TRAIN has 3, 3, 6 and 13. The pilot's fixed halves are unstratified, and one
calibration half contains no Not-AD donor. Raw held-out SD coverage, current →
quarter: Not AD 51.8% → 76.5% (1 donor), Low 65.7% → 68.4% (2), Intermediate
58.7% → 63.2% (3), High 56.7% → 57.2% (6). With one to three donors per stage,
stage-conditional coverage is not estimable here. Leave-one-out calibration
at least uses every other donor, including the other stages, for each
correction.

## Proposal for the multi-fold confirmation

These readouts can be computed for the running confirmation with no new
training. Before its scores exist, its owner should consider adding:

- SD and combined scores per PC block, with PCs 31–50 non-regression reported
  next to the equal-weight number;
- paired donor-bootstrap CIs for each gate delta;
- leave-one-donor-out or pooled calibration in place of n=6 halves.

This PR does not edit the confirmation's protocol or pinned sources.

## Reproduction

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
$PY -B analysis/coverage_calibration_v2_20261120/rescore_pilot.py --data-root /ml2/alz   # ~10 s
$PY -B analysis/coverage_calibration_v2_20261120/pca_shrinkage.py --data-root /ml2/alz   # ~5 min, 3 threads
$PY -B -m pytest -p no:cacheprovider -q tests/test_microglia_intervals_v2.py tests/test_microglia_intervals.py
```

`--data-root` is the checkout that holds the local `runs/` tree (fold-0 cache
and the two generated archives); it is only read. Outputs:
[`rescore.json`](rescore.json) and [`pca_shrinkage.json`](pca_shrinkage.json).
The rescore also confirms end to end that perturbing an evaluation donor's
observed bags leaves that donor's own calibrated bounds unchanged. It checks the split-half,
leave-one-out and lower/upper schemes. The pilot's `run.py` assert
could not fail, because it perturbed rows that were never passed to the
calibrator. The tests cover rank = n at n = 6, exchangeable-donor
simulations reaching `rank/(n+1)`, the invariance check, and training-donor
rejection.
