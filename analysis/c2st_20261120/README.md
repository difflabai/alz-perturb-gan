# Donor-held-out classifier two-sample test (C2ST)

This is a learned two-sample metric for the microglia generators. Small
classifiers try to tell generated cells or donor bags apart from cells of
held-out development donors. Every item of a donor stays in one fold, so a
classifier never scores a donor it saw during training. The statistic is the
pooled out-of-fold ROC AUC; 0.5 means the classifier cannot tell the two
apart. The [protocol](PROTOCOL.md) and [plan](plan.json) were committed
before any real-data score (commit `4a373e1`). [Amendment 1](PROTOCOL.md)
records one input fix.

Scope: one fold (fold 0) and one fit per generator. Only the 12 held-out
development donors are scored (Not AD 1, Low 2, Intermediate 3, High 6), and
they were already used for model selection. The CIs come from a donor
bootstrap of out-of-fold scores, so classifier-training variability is not
included. No default, config or runtime source changed.

## Results (fold 0, pooled out-of-fold AUC [95% donor-bootstrap CI])

MLP, the primary classifier:

| generator | cell_pc | bag_gene | bag_pcsum | bag_inspan | bag_offspan |
| --- | --- | --- | --- | --- | --- |
| **empf05** (default native EMPF, 0.5 contraction) | **.907** [.863, .940] | **.965** [.902, .999] | 1.000 [.999, 1] | .875 [.733, .982] | .969 [.920, 1] |
| empf025 (quarter contraction) | .905 [.861, .938] | .962 [.898, 1] | .999 [.998, 1] | .877 [.728, .985] | .966 [.913, 1] |
| prev097 (0.9.7 BISPF default, fresh generation) | .941 [.901, .966] | .921 [.764, 1] | 1.000 | .871 [.687, .998] | .993 [.979, 1] |
| resample_stage (stage-matched empirical) | .832 [.762, .882] | .900 [.793, .990] | .989 [.966, 1] | .858 [.757, .953] | .924 [.825, .999] |
| resample_blind (stage-blind empirical) | .855 [.796, .896] | .951 [.868, 1] | 1.000 | .892 [.798, .975] | .930 [.826, 1] |
| *post hoc:* fold-disjoint real donors | .740 [.658, .802] | .583 [.418, .728] | .991 [.983, .999] | .317 [.199, .424] | .785 [.603, .930] |

Linear (logistic) classifier, the secondary:

| generator | cell_pc | bag_gene | bag_pcsum | bag_inspan | bag_offspan |
| --- | --- | --- | --- | --- | --- |
| empf05 | .515 [.422, .598] | .743 [.554, .920] | .997 | .614 [.394, .809] | .850 [.670, .990] |
| empf025 | .497 [.413, .571] | .744 [.557, .918] | .996 | .604 | .854 |
| prev097 | .580 [.471, .679] | .812 [.644, .949] | .997 | .587 | .846 |
| resample_stage | .535 | .833 [.712, .935] | .983 | .677 | .862 |
| resample_blind | .568 | .882 [.771, .993] | 1.000 | .708 | .878 |

Levels:

- `cell_pc`: one cell as 50 TRAIN-fitted PCs.
- `bag_gene`: one bag is the mean log1p(CP10k) of 256 cells over 2000 genes.
- `bag_pcsum`: the PC mean and SD of 64 cells, the space used by the interval
  metrics.
- `bag_inspan` / `bag_offspan`: `bag_gene` split into its component inside the
  50-PC span and the residual outside it.

### Paired differences against empf05 (MLP, AUC(other) − AUC(empf05))

| other | cell_pc | bag_gene | bag_offspan |
| --- | --- | --- | --- |
| empf025 | −.001 [−.014, .010] | −.003 [−.008, .001] | −.003 [−.010, .001] |
| prev097 | **+.034 [.011, .059]** | −.043 [−.147, .007] | +.024 [.000, .060] |
| resample_stage | **−.075 [−.110, −.049]** | −.064 [−.129, −.007] | −.046 [−.101, −.000] |
| resample_blind | **−.052 [−.085, −.023]** | −.014 [−.043, .004] | −.040 [−.099, .001] |
| fold-disjoint real (post hoc) | −.167 [−.224, −.118] | −.382 [−.516, −.256] | −.185 [−.333, −.063] |

### Calibration (real vs real, 20 disjoint stage-stratified 6/6 donor halves)

| level | MLP halves mean (sd) / q95 | linear halves mean (sd) | within-donor permutation null, MLP mean / max |
| --- | --- | --- | --- |
| cell_pc | **.535** (.091) / .691 | **.542** (.096) | .500 / .519 |
| bag_gene | .572 (.147) / .761 | **.528** (.161) | .507 / .570 |
| bag_pcsum | .558 (.149) / .781 | .573 (.141) | .510 / .573 |
| bag_inspan | .575 (.154) / .855 | .590 (.193) | .511 / .554 |
| bag_offspan | .608 (.184) / .814 | .603 (.177) | .502 / .546 |

The declared calibration rule requires a mean halves-null AUC within 0.5 ± 0.05.
**It passes at the cell level and for the linear bag_gene classifier. It fails
for the MLP at every bag level and for the linear classifier at the three
other bag levels.** The halves null is very noisy with 12 donors: its SD is
.15–.19, and a 20-split mean has a standard error near .035–.04.

In a synthetic run with exchangeable donors, the same design gave means of
.525 and .478 (SD .17–.24). The real-data offsets are therefore not clearly a
leak, but they are not demonstrated to be noise either. The within-donor
permutation null is centred at .50–.51. All detection thresholds use the
larger of the halves-null q95 and the permutation-null maximum.

## What the declared decisions say

1. **Detection.** Every generator, including both empirical references, is
   detected by the MLP at every level. At `cell_pc` the linear classifier is
   near chance for every generator, at .50–.58. The cell-level discrepancy is
   therefore not a mean shift. It is in the shape of the distribution
   (higher moments and dependence between PCs).
2. **Quarter vs 0.5 readout.** Not resolved at any MLP level: the paired CIs
   all include 0. The only resolved result is the secondary linear `cell_pc`,
   where the quarter readout is lower (−.019 [−.035, −.004]). The quarter
   readout's coverage gain does not show up as lower C2ST detectability.
3. **Previous 0.9.7 default.** At the cell level, prev097 is *more*
   detectable than EMPF by +.034 [.011, .059]. That is resolved in EMPF's
   favour, and the linear classifier agrees (+.064 [.017, .103]). Bag levels
   are unresolved. prev097 is a fresh generation from the frozen fits_r3
   fold0/seed0 model with the archived r5 code; the receipted 2026-11-14
   archives were not retained.
4. **Empirical reference.** At `cell_pc`, both empirical resampling
   references are *less* detectable than EMPF, by −.052 for stage-blind and
   −.075 for stage-matched; both CIs exclude 0. EMPF does not match empirical
   resampling at the cell level. At bag levels most differences are
   unresolved.
5. **Blind-spot link, declared expectation not met.** PPARG ranks 535th of
   2000 by univariate within-donor AUC for empf05, so it is not in the top 1%.
   Only 22% of its real-minus-generated standardized gap is outside the
   PC span. The out-of-span C2ST (`bag_offspan`) is detected at .969, but
   empirical resampling is detected there too (.92–.93). Descriptively, the
   MLP's permutation importance puts PPARG at rank **15** for empf05 and **7**
   for empf025, against 34 for prev097 and 412 for resample_blind. So the
   classifier does use PPARG for the GAN fits and not for resampling.

   Held-out PPARG bag means by stage (real vs empf05):
   - Not AD (1 donor): .74 vs .89
   - Low: 1.52 vs 1.08
   - Intermediate: 1.26 vs 1.27
   - High: 1.25 vs 1.04

   The earlier finding that Not AD PPARG is under-generated, with 99% of the
   gap off-span, came from 0.9.7 fits and pooled fitting cells. On these
   held-out donors and EMPF it is not reproduced. The remaining PPARG gap is
   at Low/High and mostly inside the PC span.

## Interpretation and caveats

- **Bag-level C2STs on 12 donors mostly do not isolate generator error.**
  Empirical resampling copies real training-donor cells, yet it is detected
  about as strongly as the GANs (bag_gene .90–.95, bag_pcsum ≈ 1). The top
  univariate genes are shared by EMPF and resampling, all with held-out donors
  higher: AL008633.1, TRERF1, ATP2A3, LIPG, MIR181A2HG, GALNT8. These reflect
  a difference between training and held-out donors (or depth/cohort), not a
  GAN-specific defect. Resampling also reuses the same 25 source donors in
  every fold, which a classifier can learn to recognise.
- **Post hoc fold-disjoint reference.** Class-1 items are drawn from real
  training donors assigned only to the target's own fold, with no depth
  thinning and stage-blind sources. This removes the cross-fold reuse of
  source donors. It drops bag_gene to .58 [.42, .73], against .965 for EMPF
  (paired −.38 [−.52, −.26]). That is the strongest evidence that EMPF's
  gene-level bag profiles are distinguishable beyond donor-novelty structure.
- This reference is not a null:
  - Its `bag_inspan` and linear `cell_pc` AUCs fall well *below* .5. A finite
    25-donor population partitioned across folds makes the left-out group
    anti-correlated with the training groups. The PCA is also centred on
    exactly those donors.
  - `bag_pcsum` stays at .99 even between real training and held-out donors.
    The PC-SD summaries separate TRAIN-projected donors from held-out donors,
    which is consistent with out-of-sample PCA projection shrinkage. This
    caveat also applies to the PC-space interval metrics.
- **Cell level, post hoc ([posthoc_cell.json](results/posthoc_cell.json)).**
  A logistic classifier on [z, z²] of the 50 PCs reaches .716 for empf05,
  against .907 for the MLP. Second moments explain only part of the
  discrimination; the rest is higher-order structure.
  - Within target donors, EMPF's 32 generated donors pooled have about the
    same per-PC spread as one real donor: geometric-mean SD ratio 1.005, with
    PC3 at .68 and PC1 at 1.16.
  - Pooled resampled donors are 1.27× wider, as a population of different
    donors should be.
  - This fits the separately proposed hypothesis that EMPF under-represents
    donor-to-donor variability. It is a post hoc observation, not a test.
- The permutation-importance ranks for `cell_pc` are PC2, PC1, PC4 and PC3 for
  empf05. Univariate shifts are small (|AUC − .5| ≤ .08).
- Per-stage cell AUCs for empf05 are .90–.93 at every stage. The Not AD value
  rests on one donor.

## Reproduce / reuse

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
cd analysis/c2st_20261120
$PY -B -m pytest -q test_c2st.py                         # 20 synthetic tests (CPU)
# generic: score any generated.npz (repository contract) against held-out donors
CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B cli.py --device cuda score \
  --cache runs/improve_20261109/caches/hvg2000/fold0 \
  --generated mine=/path/generated.npz --levels cell_pc bag_gene --out c2st.json
# declared study (phases can run in parallel by --levels / --generators)
W=runs/research_20261120/c2st/work; CA=runs/improve_20261109/caches/hvg2000/fold0
$PY -B make_empirical.py --cache $CA --out-dir runs/research_20261120/c2st/empirical
$PY -B cli.py --device cuda study --phase generators --cache $CA --work $W
$PY -B cli.py --device cuda study --phase null --cache $CA --work $W
$PY -B cli.py study --phase merge --work $W --out results/c2st_fold0.json
$PY -B posthoc.py --cache $CA --out results/posthoc_cell.json
$PY -B reference_disjoint.py --cache $CA --work $W --out results/posthoc_disjoint.json
```

- `c2st.py` holds the reusable core, NumPy plus optional torch with no
  repository paths:
  - `auc`, `grouped_folds`, `run_c2st` (MLP or linear; all fold × seed
    networks are trained in one batched pass)
  - `donor_bootstrap`, `paired_bootstrap_diff`
  - `halves_null`, `permutation_null`
  - `permutation_importance`, `univariate_auc`
- `features.py` builds the five item levels from a RealFold and an aligned
  generated archive. Real items are identical for every generator.

Resources: all runs used the shared A6000s at a 0.04–0.05 memory fraction,
3 threads and `taskset -c 3-23`, with no more than 4 concurrent processes.
Generator and null phases took about 15–35 minutes each; the host load
average was about 130–200. The prev097 regeneration took 95 minutes on CPU.

Outputs live under `runs/research_20261120/c2st/` (not in git): generated
archives, empirical references, out-of-fold scores and per-phase JSON. Git
holds the merged [results](results/c2st_fold0.json) and the post hoc JSONs.
