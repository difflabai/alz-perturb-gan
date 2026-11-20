# Native EMPF across nine folds and two seeds: confirmation result

Pre-declared in [PROTOCOL.md](PROTOCOL.md) (commit `d578e83`, before any fit
existed), with the read-out choices locked in [DECLARATIONS.md](DECLARATIONS.md)
before the first official score. 17 new native EMPF fits (folds 0-8, seeds 0-1,
excluding the fold 0 / seed 0 pilot) trained 20,000 updates each through the
committed CLI, with no failures. Every fit was scored with the pilot's own
read-out, scorer and SD calibration; `score.py` reproduces the pilot exactly
(788 values, maximum difference 0) and an independent reviewer recomputed one
new fit to 1e-17. Full output: [completion/analysis.txt](completion/analysis.txt)
and [completion/analysis.json](completion/analysis.json).

Held-out development donors; means over the 17 fits; "better" counts fits.
Coverage targets 80%; interval scores are proper scores, lower is better.

## Question 1: quarter-contraction readout (B) against the default (A) - PASS

All ten pilot gates pass on the mean, and both interval scores improve in each
of the three repeats (folds 0-2 without the pilot, 3-5, 6-8).

| | A `cellshape@0.5` | B `cellshape@0.25` | better |
| --- | ---: | ---: | ---: |
| SD coverage | 60.2% | 63.5% | 15/17 |
| combined coverage | 70.5% | 72.5% | 15/17 |
| SD interval score | .720 | .640 | 15/17 |
| combined interval score | .937 | .898 | 16/17 |
| profile RMSE | .4895 | .4895 | 5/17 |
| per-gene mean error | .0802 | .0811 | 1/17 |
| cell energy | .721 | .730 | 1/17 |
| PC energy | 1.874 | 1.862 | 12/17 |
| PC spread ratio (1 = real) | 1.08 | 1.14 | 2/17 |

The costs are inside the pre-declared limits but consistent across fits: B
slightly over-disperses in PC space (spread ratio further above 1), and its SD
misses move from mostly above the interval (28.5% to 15.7%) to more often
below it (11.3% to 20.7%).

## Question 2: SD-only interval calibration - CONFIRMED for A and B

| | A raw | A calibrated | B raw | B calibrated |
| --- | ---: | ---: | ---: | ---: |
| SD coverage | 60.2% | 87.5% | 63.5% | 87.7% |
| combined coverage | 70.5% | 84.1% | 72.5% | 84.6% |
| SD interval score | .720 | .605 (17/17) | .640 | .625 (12/17) |
| combined interval score | .937 | .879 (17/17) | .898 | .891 (12/17) |

Calibration overshoots 80%: with six or seven calibration donors the conformal
rank is the largest score, so the nominal level is about 86-88%, not 80%.
It widens ranges around the network's output and does not change generated
cells. On top of B the gain is small; calibrated A has the best proper scores.

## Secondary, descriptive: EMPF against the previous default

The owner-selected EMPF default (readout A) does **not** reproduce its pilot
advantage across folds. Against the previous BISPF count-shape default on the
same 17 fold/seed pairs and held-out donors:

| | previous default | EMPF A | EMPF better |
| --- | ---: | ---: | ---: |
| profile RMSE | .4853 | .4895 | 6/17 |
| SD interval score | .710 | .720 | 6/17 |
| combined interval score | .923 | .937 | 7/17 |
| per-gene log-variance error | .1895 | .1925 | 4/17 |

The profile RMSE and both interval-score differences are within the previous
default's own stream-to-stream spread (re-scoring the same fits with a
different random stream), so the two are best described as indistinguishable,
with EMPF slightly behind. On the pilot fold/seed EMPF was ahead on all three
(profile RMSE -.0074, SD score -.0195, combined -.0064); across the 17 other
fold/seed pairs it is not.

Against stage-blind empirical resampling (which reuses observed cells): raw
A and B still lose the combined interval score (.937 and .898 against .876).
Calibrated A beats it on the SD score (.605 against .665, 15/17) and ties it
on the combined score (.879 against .876, 8/17), at 84% combined coverage
against resampling's 75%.

## Limits

All 37 donors were used in earlier development; this is development evidence,
not an independent test (that needs the sealed donors). Seeds of a fold share
generation streams and folds 3-8 share one empirical stream per fold, so paired
SEs are descriptive. No default was changed by this pass.
