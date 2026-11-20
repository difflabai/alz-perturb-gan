# EMPF coverage improvement pilot

This pass improves the trained network's count readout and separately tests
SD-only interval calibration. It uses no empirical expression-cell resampling.
The frozen native EMPF model still has 20,000 completed updates, `lazy_k=1`,
and the same released hyperparameters. The 37 development donors and previous
results remain unchanged.

## Results

| Method | SD coverage | Combined coverage | SD interval score ↓ | Combined score ↓ | SD width | Combined width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current native EMPF | 58.31% | 69.20% | .806254 | .945235 | .222067 | .486709 |
| Quarter contraction | 62.21% | 71.46% | .667691 | .875349 | .271340 | .519325 |
| Current + SD calibration | 85.55% | 82.83% | .623801 | .854009 | .509562 | .630456 |
| Quarter + SD calibration | 86.65% | 83.68% | .610538 | .846772 | .506357 | .636833 |

Coverage targets 80%. The quarter-contraction count generator improves SD
proper score **17.2%** and combined proper score **7.4%**. Unlike a change to
interval endpoints, this changes the network-generated count distribution.
It halves the inherited cell-mean contraction from .5 to .25, then refits the
existing bias/nonzero calibration using the 25 GAN-training donors only.
Expected bag gene totals and each cell's expected library size are preserved
before joint calibration. It passed all ten fixed pilot checks.

Raw profile RMSE improves .465124 to .464805 (0.07%). Detection error improves
0.05% and PC energy 2.69%. Mean-log gene error worsens 0.81%, log-variance error
1.65%, and cell energy 0.22%; these remain within the fixed pilot guards.
Mean-coordinate coverage is 80.10% before and 80.71% after the count change.

SD calibration is a different result: it expands only variability-summary
ranges, leaving expression-mean bounds and all generated cells unchanged.
The wider ranges are explicit above, and proper interval scores improve
despite their width cost. The current network plus calibration is slightly
closer to the 80% coverage target; the quarter readout plus calibration has
lower proper scores. The six-donor calibration sets limit quantile resolution.

## Donor separation and limits

The native GAN fit uses 25 donors. Its 12 held-out development donors are
split into two fixed halves: calibrate on six, evaluate on the other six,
then swap. Each donor supplies one real calibration bag per coordinate;
24 real-cell subsamples are used only for evaluation. Calibrators reject
GAN-training donors, duplicate donor identities, insufficient calibration
sample size, and evaluation on their own calibration donors. SD lower bounds
respect zero support. Mean bounds are exactly unchanged by calibration.

These donors were already used in development selection. The two swaps do
not supply independent final-test data, multiple independent cohorts or
patient-level simultaneous coverage guarantees. Only one native fold/seed
was tested. Confirm the fixed quarter readout on additional native EMPF
folds/seeds before a research-qualified default change. The public default
continues to use the previously selected .5 readout; these results are a
separate experimental improvement with reusable calibration code.

Empirical resampling remains a useful external reference to the network.
It reuses observed training-donor cells and is not a trained neural network.
There is no empirical/GAN mixture in this pass. Comparisons with that reference
do not substitute for demonstrating learned distributional fidelity.

## Reproduction and evidence

Use the recorded research environment, frozen selected native fit, development
caches, and baseline NPZ/receipt pinned in [source_manifest.json](source_manifest.json).
Outputs are under `runs/coverage_empf_20261117/`. Generation deliberately rejects
an existing candidate NPZ; inspect retained outputs before attempting a rerun.

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
$PY -B analysis/coverage_empf_20261117/run.py --phase generate
$PY -B analysis/coverage_empf_20261117/run.py --phase analyze
$PY -B analysis/coverage_empf_20261117/audit.py
```

The NumPy-only [calibration API](calibration.py) exposes `fit_sd` and `apply_sd`
for `[donor,100]` arrays: 50 PC means followed by 50 PC SDs. Supply one observed
bag per calibration donor, the actual GAN-training donor IDs, and disjoint
evaluation IDs. A correction belongs to its recorded model, count readout and
TRAIN-fitted PCA basis. Saved artifacts in [completion/calibration](completion/calibration)
include those bindings and donor partitions; they are not generic corrections
for another model or patient representation.

The [frozen protocol](PROTOCOL.md) and [prelaunch checks](prelaunch.json) were
committed before candidate generation/scoring. All 50 automated tests passed.
The [direct numerical audit](completion/audit.json) reloaded pinned observed
cells and generated arrays, recomputed 210 metric values and 13 decisions,
checked all four calibration artifacts and exact unchanged mean bounds, and
reproduced the original native historical scores. This was a same-agent audit,
not an independent-agent or external-cohort review.

See the [complete comparison](completion/comparison.json),
[generation receipt](completion/quarter_generated.receipt.json), and
[completion manifest](completion.json). All work ran on CPU, three threads/cores,
with zero model-training updates and about 25MB of new local outputs. Raw data,
weights and generated binary archives remain local.

Method background: [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222)
and [A Tutorial on Conformal Prediction](https://jmlr.org/papers/v9/shafer08a.html).
