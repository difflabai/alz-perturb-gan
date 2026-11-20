# Condition-aware wide EMPF residual: results

Verdict: **negative.** Declared comparison: [PROTOCOL.md](PROTOCOL.md).

**Answer.** A condition-aware residual fixes the probe's blind spot, but it
did not improve learned spread. On the synthetic task (4 seeds), per-group SD
error got worse. On SEA-AD, in a single-seed pair per fold, the adapter run
had lower held-out SD coverage than the native run on both folds; with one
training seed this is not an established adapter effect, but no fold showed a
gain. We found no evidence that the blind spot drives the default's SD
under-coverage, although this test had little power: the adapter can reach
training only through a gauge-weight change of about 0.006-0.009. In this
host the residual is only a scalar control signal. Gradients never see it, and
on the wide path it can change only the gauge weight. The comparison was
declared in [PROTOCOL.md](PROTOCOL.md) (commit `84d0372`) before any score
existed.

## 1. The probe: the adapter detects the swap

[`cond_residual.py`](cond_residual.py) provides an opt-in `install(penalty)`.
On EMPF's wide continuous conditioned path, it computes the released Chamfer
residual using only real/fake pairs that share a condition row. The scale
(median) is the native one, and nothing else changes. The table shows
`ema_residual` after one penalty call ([probe.py](probe.py),
[results/probe.json](results/probe.json)).

| Width | Arm | Correct | Swapped spreads | Both halved | Correct, rows reordered | Swapped, rows reordered |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | native | 0 | 0.931 | 0.618 | 0 | 0.931 |
| 64 | adapter | 0 | 0.931 | 0.618 | 0 | 0.931 |
| 65 | native | 0 | **0** | 0.586 | 0 | 0 |
| 65 | adapter | 0 | **1.000** | 0.729 | 0 | 1.000 |
| 2150 | native | 0 | **0** | 0.586 | 0 | 0 |
| 2150 | adapter | 0 | **1.000** | 0.729 | 0 | 1.000 |

The "correct" case uses fake identical to real, so its residual of 0 is
exact by construction. The 1.000-vs-0 separation is therefore a best case; it
does not measure how well the statistic separates correct from swapped
pairings at realistic finite sample sizes, where the correct residual is above
zero. In the reorder control, real, fake and condition rows are permuted
together, and every value stays the same (largest difference 6.4e-8). At width 64 the adapter does not route the
call, so the output matches native. A single condition, an unconditioned call
or width 64 gives output bit-identical to native. All 20 tests pass
([test_cond_residual.py](test_cond_residual.py)).

## Why a small training effect was expected

On the wide path the released kernel reads `ema_residual` in two places only:

- the EBI sharpening `settle_s = clamp(1 - res/0.16)`, which is inactive while
  res ≥ 0.16;
- the gauge weight `0.05 + 0.10(1 - res)`.

On this path the thermal and optimizer step scales stay at 1. In every run
below, both arms kept `ema_residual` between 0.60 and 0.96, so `settle_s`
never engaged. The adapter's residual is higher because a within-condition
nearest neighbour is farther away. That lowers the gauge weight, for example
from 0.067 to 0.062 on SEA-AD fold 0 and from 0.068 to 0.060 on fold 1. This
is the adapter's only path into training. The arms share every random stream
and minibatch. They diverge only through this perturbation, so one-seed
differences mix any systematic effect with trajectory divergence.

## 2. Synthetic grouped Gaussian: no improvement

The task has four conditions and 8 informative coordinates. Their per-group
SDs form a Latin square over 0.5/1/1.5/2, so the pooled marginals are equal.
The remaining coordinates are N(0, 0.1²). The update mirrors the ALZ host:
EMPF, batch 32, 512 particles, `lazy_k=1`, 2,000 updates, seeds 0-3, and
identical streams in both arms ([synthetic.py](synthetic.py)). The table
reports means over 4 seeds of informative-coordinate scores against
independent truth draws. Brackets give the seed-bootstrap 95% interval of the
paired difference ([results/synthetic.json](results/synthetic.json)).

| Width | Metric | Native | Adapter | Adapter - native |
| ---: | --- | ---: | ---: | ---: |
| 2150 | log-SD RMSE ↓ | 0.154 | 0.194 | +0.041 [-0.006, 0.085]; 1/4 seeds better |
| 2150 | group SD RMSE ↓ | 0.173 | 0.222 | +0.049 [0.008, 0.090] |
| 2150 | log-SD bias | -0.028 | -0.063 | -0.035 (narrower) |
| 2150 | group mean RMSE ↓ | 0.367 | 0.319 | -0.048 [-0.156, 0.023] |
| 2150 | 80% coverage | 76.2% | 74.8% | -1.3 pts |
| 2150 | interval score ↓ | 4.705 | 4.707 | +0.002 [-0.145, 0.100] |
| 128 | log-SD RMSE ↓ | 0.202 | 0.222 | +0.020 [-0.021, 0.061]; 2/4 seeds better |
| 128 | 80% coverage | 76.4% | 77.3% | +0.9 pts |
| 128 | interval score ↓ | 4.574 | 4.614 | +0.040 [-0.017, 0.105] |

The reference samplers show that the scores can detect a spread error. At
width 2150, the correct sampler scores log-SD RMSE 0.012 and coverage 79.8%.
The swapped-spread sampler scores 0.816 and 73.1%, with interval score 5.88
against 4.41. **The declared "helps" gate failed.** It required log-SD RMSE at
least 5% lower in at least 3 of 4 seeds. Instead, log-SD RMSE was 26.5% higher
and only 1 of 4 seeds improved. The seed-to-seed spread within native
(log-SD RMSE 0.11-0.18 at width 2150) is as large as the arm difference.

## 3. SEA-AD matched fits: no SD coverage gain on either fold

Both arms used the committed native EMPF host (`pbulk`, hvg2000, seed 0) with
the 20k recipe truncated identically at **5,000 updates**. The readout was the
pilot's `cellshape@0.5:biascal:stagehalf` (late states 3750-4750 plus final),
scored by the frozen `architecture_score.py` on the held-out development
donors ([alz_run.py](alz_run.py), [results/alz.json](results/alz.json)).
Brackets give the paired 95% interval from 2,000 bootstrap resamples of
held-out donors. That interval does not include training-seed variability.

| Fold (held-out donors) | Metric | Native 5k | Adapter 5k | Adapter - native | Stage-blind empirical |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 (12) | SD coverage | 58.9% | 55.6% | -3.3 pts [-5.3, -1.7] | 69.8% |
| 0 | SD interval score ↓ | .8289 | .8624 | +.0334 [.0154, .0559] | .6119 |
| 0 | SD width | .2158 | .2073 | -.0084 | .4132 |
| 0 | combined coverage | 70.2% | 66.0% | -4.2 pts [-5.3, -3.1] | 78.2% |
| 0 | combined interval score ↓ | .9522 | .9794 | +.0272 [.0056, .0494] | .8322 |
| 1 (13) | SD coverage | 70.8% | 62.9% | -7.9 pts [-11.4, -4.2] | 65.2% |
| 1 | SD interval score ↓ | .5565 | .5489 | -.0077 [-.0323, .0156] | .6335 |
| 1 | SD width | .2650 | .2114 | -.0536 | .3999 |
| 1 | combined coverage | 74.2% | 71.4% | -2.8 pts [-5.4, -0.2] | 74.8% |
| 1 | combined interval score ↓ | .8280 | .8105 | -.0175 [-.0503, .0137] | .8358 |

Relative changes in the historical errors (adapter vs native; positive means
worse):

- Fold 0: profile RMSE +1.0%, mean-log gene -4.2%, detection -2.9%, log-variance -4.1%, PC energy +2.2%, cell energy +2.6%.
- Fold 1: profile RMSE +2.2%, mean-log gene +15.5%, detection +7.6%, log-variance +5.6%, PC energy +1.4%, cell energy +5.9%.

The declared support rule required a gain of at least 2 points in SD coverage,
with an interval above zero, on every fold. **It failed on both folds.** In
this single-seed pair, the adapter run had narrower generated SD ranges and
lower SD coverage on both folds. The donor-bootstrap intervals cover donor
sampling only, not training-seed variation, and the synthetic task shows
seed-to-seed spread as large as the arm difference, so the drop is not
attributed to the adapter. What holds is that SD coverage did not rise.

Observed SDs mostly fall above the generated interval (readout-level
under-dispersion): on fold 0, 31.9% of SD coordinates lie above and 9.2%
below for native 5k, and 33.3% above and 8.4% below for the 20k pilot. This
was not decomposed. Possible contributors include the generator itself, the
readout's `cellshape@0.5` within-donor mean shrinkage (applied after
generation, which directly reduces within-donor spread), and the PCA summary
used for the bag statistics.

For context, the frozen 20k native pilot on fold 0 had SD coverage 58.3% and
SD interval score .8063. Native at 5k gave 58.9%, so the under-coverage
is already present early in training. The native 5k arm reproduced the
pilot's first 5,000 updates exactly (40 logged rows, max difference 0;
[results/determinism_fold0.json](results/determinism_fold0.json)).
Exploratory observation: on fold 1, native 5k beat stage-blind empirical
resampling on SD coverage and SD interval score, but not on fold 0.

## Limits

- The study used one training seed per fold, two development folds, and 5,000 rather than 20,000 updates.
- The adapter perturbs a chaotic trajectory, and the donor-bootstrap intervals omit seed variation. The SEA-AD differences are therefore not a measured systematic effect of the adapter. Our conclusion is narrower: correcting the residual did not raise SD coverage on any fold or on the synthetic task.
- Synthetic devices differed from the protocol: it said width 2150 runs on GPU, but only seeds 0 and 2 did; seeds 1 and 3 at width 2150 and all width-128 runs ran on CPU. Device is recorded in each run JSON and matches within every pair. CPU thread counts were not recorded in these run JSONs, so thread matching within pairs cannot be audited; from the session log, seed 2 at width 2150 used 3 threads (native) and 1 thread (adapter), and two width-128 native runs were restarted single-threaded after stalling on an overloaded CPU. `synthetic.py` now records `threads` for any future run.
- The SEA-AD and synthetic intervals are percentile bootstraps over 12-13 donors or 4 seed pairs, with no multiplicity control across metrics, folds and widths; they are likely anti-conservative, and the 4-seed intervals are crude.
- The protocol header says "about 04:05 UTC"; the protocol commit is 04:02:34 UTC (see the erratum in PROTOCOL.md). After the protocol commit, `alz_run.py` gained a provenance-check normalisation for the output root and `synthetic.py` an `SYN_THREADS` variable and thread recording; neither changes training. `analyze.py` was written afterwards and implements the declared gates.
- One planned GPU run failed with an out-of-memory error on shared GPU 0 and was rerun on GPU 1.
- Leakage check: fits and calibration use train donors only; scoring uses held-out donors only. The development donors were already used in earlier selection, so these results are development evidence, not a final test.
- We did not test a stronger adapter that feeds the conditional discrepancy into the critic or generator objective. With the released kernel, the residual cannot reach gradients except through the gauge and EBI weights. The released restore path was not re-verified for the adapter. Its kernel holds no extra checkpointed state.

## Reproduce

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
export PYTHONPATH=/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages
cd analysis/cond_residual_20261120
$PY -B -m pytest -q -p no:cacheprovider test_cond_residual.py
$PY -B probe.py --json probe.json
$PY -B synthetic.py train --arm adapter --seed 0 --width 2150 --updates 2000 --device cuda --out run.json
CUDA_VISIBLE_DEVICES=1 $PY -B alz_run.py train --lane adapter --fold 0 --updates 5000
$PY -B alz_run.py generate --lane adapter --fold 0
$PY -B analyze.py alz --folds 0,1
```

Checkpoints, generated archives and logs are in
`/ml2/alz/runs/research_20261120/cond-residual/`, outside git. The ALZ runner
writes its derived host files there. It checks them against the shared native
derivation, apart from the output root, and never writes the shared runtime
directory.

## Next steps

- Do not adopt the adapter as a quality fix. It is suitable only as a diagnostic or observer-correctness fix.
- If the blind spot matters, it has to reach gradients: a per-condition discrepancy term in the critic or generator objective, or a gauge/EBI schedule driven by conditional fit, tested with several seeds per arm.
- Decompose the SD under-coverage: generator spread before readout, the effect of the `cellshape@0.5` within-donor shrinkage, and PCA-summary shrinkage. Also check fit-ensemble spread from the 17 confirmation fits and a donor-level random-effect latent.
