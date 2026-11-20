# Donor-held-out classifier two-sample test (C2ST)

This is a learned two-sample metric for the microglia generators. Small
classifiers try to tell generated cells or donor bags apart from cells of
held-out development donors. Every item of a donor stays in one fold, so a
classifier never scores a donor it saw during training. The statistic is the
pooled out-of-fold ROC AUC; 0.5 means the classifier cannot tell the two
apart.

**Verdict: mixed.** At the cell level, the default native EMPF fit is
clearly detectable. It is less detectable than the previous 0.9.7 default
and more detectable than empirical resampling, and both differences hold
up when the classifier is refit. The bag-level C2STs do not isolate
generator error on 12 donors: their real-vs-real calibration fails, and
the bag items are built asymmetrically. The expected link to the PPARG
blind spot was not found.

Scope: one fold (fold 0) and one fit per generator. Only the 12 held-out
development donors are scored (Not AD 1, Low 2, Intermediate 3, High 6), and
they were already used for model selection. No default, config or runtime
source changed.

## Protocol

- [PROTOCOL.md](PROTOCOL.md) and [plan.json](plan.json) were committed
  before any real-data score (commit `4a373e1`).
- [Amendment 1](PROTOCOL.md#amendment-1-recorded-before-any-empirical-reference-or-prev097-score)
  regenerated the empirical references, because the declared files held
  scores and no cells.
- [Amendment 2](PROTOCOL.md#amendment-2-audit-repairs-declared-2026-11-20-before-any-repair-score)
  (commit `0cf9fde`) declared the audit repairs and their decision rules
  before any repair score existed. The repairs are classifier-refit
  sensitivity, full second-order cell classifiers, and PPARG importance
  with repeats. Results: [results/repairs.json](results/repairs.json),
  code: [repairs.py](repairs.py).
- Classifiers: an MLP (primary) and a linear logistic classifier
  (secondary). Folds are grouped 6-fold, stage-stratified, over the 12
  donors.
- Uncertainty: 95% donor-bootstrap CIs (2000 resamples) of the out-of-fold
  scores, and paired donor bootstraps against `empf05`.
- Nulls: 20 real-vs-real halves splits and 20 within-donor label
  permutations.
- Generators:
  - `empf05`: the default native EMPF fold0/seed0 (0.5 readout).
  - `empf025`: the quarter-contraction readout of the same fit.
  - `prev097`: the previous 0.9.7 BISPF default, as a fresh generation.
  - `resample_stage` and `resample_blind`: stage-matched and stage-blind
    empirical resampling.
- Levels:
  - `cell_pc`: 50 training-fitted PCs per cell.
  - `bag_gene`: mean log1p(CP10k) of 256 cells over 2000 genes.
  - `bag_pcsum`: the PC mean and SD of 64 cells.
  - `bag_inspan` / `bag_offspan`: `bag_gene` split into its part inside the
    50-PC span and the residual outside it.

## Results (fold 0, pooled out-of-fold AUC [95% donor-bootstrap CI])

MLP, the primary classifier:

| generator | cell_pc | bag_gene | bag_pcsum | bag_inspan | bag_offspan |
| --- | --- | --- | --- | --- | --- |
| **empf05** (default) | **.907** [.863, .940] | **.965** [.902, .999] | 1.000 [.999, 1] | .875 [.733, .982] | .969 [.920, 1] |
| empf025 (quarter readout) | .905 [.861, .938] | .962 [.898, 1] | .999 [.998, 1] | .877 [.728, .985] | .966 [.913, 1] |
| prev097 (0.9.7 default, fresh generation) | .941 [.901, .966] | .921 [.764, 1] | 1.000 | .871 [.687, .998] | .993 [.979, 1] |
| *empirical:* resample_stage | .832 [.762, .882] | .900 [.793, .990] | .989 [.966, 1] | .858 [.757, .953] | .924 [.825, .999] |
| *empirical:* resample_blind (stage-blind) | .855 [.796, .896] | .951 [.868, 1] | 1.000 | .892 [.798, .975] | .930 [.826, 1] |
| *post hoc:* fold-disjoint real training donors | .740 [.658, .802] | .583 [.418, .728] | .991 [.983, .999] | .317 [.199, .424] | .785 [.603, .930] |
| real-vs-real halves null, mean (SD) / q95 | .535 (.091) / .691 | .572 (.147) / .761 | .558 (.149) / .781 | .575 (.154) / .855 | .608 (.184) / .814 |
| within-donor permutation null, mean / max | .500 / .519 | .507 / .570 | .510 / .573 | .511 / .554 | .502 / .546 |

Linear (logistic) classifier, the secondary:

| generator | cell_pc | bag_gene | bag_pcsum | bag_inspan | bag_offspan |
| --- | --- | --- | --- | --- | --- |
| empf05 | .515 [.422, .598] | .743 [.554, .920] | .997 | .614 [.394, .809] | .850 [.670, .990] |
| empf025 | .497 [.413, .571] | .744 [.557, .918] | .996 | .604 | .854 |
| prev097 | .580 [.471, .679] | .812 [.644, .949] | .997 | .587 | .846 |
| resample_stage | .535 | .833 [.712, .935] | .983 | .677 | .862 |
| resample_blind | .568 | .882 [.771, .993] | 1.000 | .708 | .878 |
| fold-disjoint real (post hoc) | .383 [.317, .437] | .533 [.382, .700] | .991 | .282 | .768 |
| halves null mean (SD) | .542 (.096) | .528 (.161) | .573 (.141) | .590 (.193) | .603 (.177) |

### Paired differences against empf05, with classifier-refit sensitivity

AUC(other) − AUC(empf05). The "declared" column shows the declared run
with its donor-bootstrap CI. The other columns come from Amendment 2, which
refits the classifier 5 times (the declared run plus 4 refits with new CV
fold assignments and new network seeds). The refit-augmented CI resamples
donors and also picks a random refit in each draw. A difference is
**robust** only if that CI excludes 0 and all 5 refit estimates have the
same sign.

| level / classifier | other | declared | 5 refit estimates | refit-augmented CI | robust |
| --- | --- | --- | --- | --- | --- |
| cell_pc / MLP | empf025 | −.001 [−.014, .010] | −.001 … −.011 | [−.020, .009] | no |
| cell_pc / MLP | prev097 | **+.034** [.011, .059] | +.027 … +.034 | [.007, .056] | **yes** |
| cell_pc / MLP | resample_stage | **−.075** [−.110, −.049] | −.067 … −.081 | [−.112, −.047] | **yes** |
| cell_pc / MLP | resample_blind | **−.052** [−.085, −.023] | −.050 … −.056 | [−.089, −.024] | **yes** |
| cell_pc / linear | empf025 | −.019 [−.035, −.004] | −.015 … −.024 | [−.037, −.003] | **yes** |
| cell_pc / linear | prev097 | +.064 [.017, .103] | +.041 … +.064 | [.002, .097] | yes (barely) |
| cell_pc / linear | resample_stage | +.019 [−.029, .068] | +.011 … +.024 | [−.029, .066] | no |
| cell_pc / linear | resample_blind | +.052 [.009, .102] | +.042 … +.063 | [.011, .105] | yes |
| bag_gene / MLP | empf025 | −.003 [−.008, .001] | −.003 … +.007 | [−.007, .013] | no |
| bag_gene / MLP | prev097 | −.043 [−.147, .007] | −.034 … −.056 | [−.177, .023] | no |
| bag_gene / MLP | resample_stage | −.064 [−.129, −.007] | −.042 … −.064 | [−.120, .002] | no |
| bag_gene / MLP | resample_blind | −.014 [−.043, .004] | −.027 … +.007 | [−.057, .025] | no |
| bag_gene / linear | empf025 | +.001 [−.027, .028] | −.026 … +.014 | [−.047, .037] | no |
| bag_gene / linear | prev097 | +.069 [−.025, .199] | −.045 … +.069 | [−.091, .144] | no |
| bag_gene / linear | resample_stage | +.090 [−.054, .262] | +.051 … +.104 | [−.057, .262] | no |
| bag_gene / linear | resample_blind | +.139 [.022, .284] | +.106 … +.154 | [.022, .299] | yes |

The empf05 MLP AUC itself is stable across refits: .900–.915 at `cell_pc`
and .954–.981 at `bag_gene`. The linear `bag_gene` AUC is not stable
(.743–.841).

The other declared paired comparisons are all in
[results/c2st_fold0.json](results/c2st_fold0.json) under `comparisons`. They
were not refit. Among them, at `bag_offspan` the MLP difference
resample_stage − empf05 is −.046 [−.101, −.000], and the linear difference
resample_blind − empf05 is +.028 [.006, .052].

### Second-order cell classifiers (Amendment 2, item 2)

All three models below are fit within the declared folds, with
training-fold standardization:

- `[z, z²]`: logistic regression on each PC and its square, with no
  cross-PC terms.
- `[z, zzᵀ]`: logistic regression on each PC plus every pairwise product
  (1325 features).
- QDA: Gaussian class models with Ledoit-Wolf-shrunk covariance.

| generator | `[z, z²]` | `[z, zzᵀ]` | QDA | MLP | MLP − `[z, zzᵀ]` | MLP − QDA |
| --- | --- | --- | --- | --- | --- | --- |
| empf05 | .716 [.643, .770] | .857 [.813, .892] | .853 [.787, .903] | .907 | +.050 [.039, .060] | +.053 [.033, .082] |
| empf025 | .716 | .864 [.819, .900] | .861 [.802, .904] | .905 | +.041 [.034, .048] | +.045 [.028, .064] |
| prev097 | .757 | .912 [.880, .937] | .911 [.869, .939] | .941 | +.029 [.013, .041] | +.030 [.015, .046] |
| resample_stage | .822 | .796 [.748, .833] | .779 [.729, .822] | .832 | +.036 [.012, .054] | +.052 [.023, .078] |
| resample_blind | .846 | .816 [.770, .851] | .784 [.735, .822] | .855 | +.038 [.020, .051] | +.071 [.043, .095] |

The linear classifier gives .515 and `[z, z²]` gives .716, but a model that
includes cross-PC covariance reaches .853–.857. **Most of the cell-level
separation of EMPF is therefore second-moment structure: cross-PC
covariance, which a per-PC variance model cannot see.** The MLP gains a
further +.05 [.04, .06] beyond full second order. By the declared rule this
counts as discrimination beyond second moments, but it is the smaller part.
Under the second-order models EMPF is still more detectable than empirical
resampling (.857 vs .796–.816). Those differences were not paired-tested.

### PPARG and the blind-spot link

- **Declared expectation not met.** For empf05, PPARG ranks 535 of 2000 by
  univariate within-donor AUC (.392), so it is not in the top 1%. Only 22% of
  its real-minus-generated standardized gap lies outside the PC span. The
  out-of-span C2ST (`bag_offspan`) detects empf05 at .969, but it also
  detects empirical resampling at .92–.93.
- **Permutation importance (Amendment 2, item 3).** Each of the 5 refits
  used 50 permutation repeats. For empf05, PPARG's AUC drop averages .0008,
  with a 95% range over repeats of [−.00002, .0017]. Its rank among all
  genes moves between 12 and 818 across refits. In 2 of 5 refits it falls
  below the all-gene 95th percentile.
  - empf025 behaves similarly: ranks 9 to 637, mean drop .0009.
  - prev097: ranks 15 to 478, mean drop .0003.
  - resample_blind: ranks 337 to 1949, mean drop −.0002.
  - The declared reliance criterion fails for every generator, so **the
    earlier statement that "the MLP relies on PPARG" is withdrawn.** The
    drops are tiny, about 0.1% of AUC, and unstable under refitting.
- Held-out PPARG bag means by stage (real vs empf05) are descriptive only:
  - Not AD (1 donor): .74 vs .89
  - Low: 1.52 vs 1.08
  - Intermediate: 1.26 vs 1.27
  - High: 1.25 vs 1.04

## What the results support

1. **Cell-level detection of the default.** empf05 is detected at the cell
   level: MLP .907 [.863, .940], against a halves-null q95 of .691 and a
   permutation-null maximum of .519.
   - The declared nulls compare held-out donors with held-out donors, so
     they do not include the floor between training and held-out donors.
   - The post hoc fold-disjoint reference uses real training donors as
     "generated", so no generator is involved. It already reaches .740
     [.658, .802] at the cell level. It is the more relevant comparator for
     the .907, but it is post hoc and not a null.
   - empf05 exceeds it by +.167 [.118, .224].
2. **Mostly covariance.** The cell-level discrepancy is not a mean shift,
   since the linear classifier scores .515. Most of it is captured by a
   second-order model with cross-PC covariance (.857), and the MLP adds
   about +.05 beyond that.
3. **EMPF vs 0.9.7.** EMPF is less detectable than the 0.9.7 default at the
   cell level. The difference is +.034 for the MLP and +.064 for the linear
   classifier, and both hold under refitting. The MLP refit range is .027
   to .034.
4. **EMPF vs empirical resampling.** At the cell level, the MLP detects EMPF
   more readily than either empirical reference. The differences are −.052
   (stage-blind) and −.075 (stage-matched), and both hold under refitting.
   By declared decision 3, EMPF does not match empirical resampling there.
   The direction depends on the classifier:
   - The linear classifier detects stage-blind resampling *more* readily
     than EMPF, by +.052, which is also robust. Resampling carries a mean
     shift that EMPF does not.
   - The resampling references reuse the same 25 training donors in every
     fold. That inflates their AUC and works against this direction.
5. **Quarter vs 0.5 readout.** No MLP level resolves the two readouts. The
   secondary linear classifier does resolve a small difference at
   `cell_pc`: −.019 [−.035, −.004], robust under refitting, with the
   quarter readout less detectable.

## Withdrawn or revised claims

These changes follow the audit.

- *"Higher-order distribution shape."* Withdrawn. The earlier `[z, z²]`
  model had no cross-PC terms and used global standardization. With
  training-fold standardization and covariance terms, second-order models
  explain most of the gap (see above).
- *"Bag-level detection reflects a difference between training and
  held-out donors."* Withdrawn. The fold-disjoint real-donor reference
  reaches only .583 [.418, .728] at `bag_gene`. Real training donors are
  therefore not strongly separable from held-out donors in gene space when
  no source donor is reused across folds. Resampling's high bag AUC
  (.90–.95) more likely comes from two features of the construction:
  - the same source donors are reused across folds, which the classifier
    can learn;
  - depth thinning.

  The shared top univariate genes (AL008633.1, TRERF1, ATP2A3, LIPG,
  MIR181A2HG), all higher in held-out bags, are an observation, not a
  mechanism.
- *"Bag-level resampling is detected about as strongly as EMPF."* Revised.
  The MLP `bag_gene` difference resample_stage − empf05 is −.064
  [−.129, −.007] in the declared run, but it is not robust to refitting
  (refit-augmented CI [−.120, .002]). The linear classifier detects
  stage-blind resampling *more* readily than EMPF at `bag_gene`: +.139
  [.022, .284], robust.
- *"The MLP relies on PPARG for the GAN fits."* Withdrawn (see above).
- *"`bag_pcsum` shows a projection asymmetry between training and held-out
  donors."* This is now stated as a **hypothesis**. The observation is that
  real training donors are separated from held-out donors at .991, with no
  generator involved. The disjoint reference used in-sample training PCs
  and no depth thinning, so differences in depth or library size are
  confounded with in-sample PCA shrinkage. A leave-donor-out PCA reference
  with matched depth is needed to attribute it.

## Limitations

- **Scope.** One fold, one fit per generator, and 12 held-out development
  donors that were already used for model selection. Not AD is a single
  donor. No multiplicity correction was applied across roughly 100 AUCs and
  comparisons.
- **Calibration.** The declared real-vs-real calibration rule (mean within
  .5 ± .05) fails at every MLP bag level and at three linear bag levels.
  - Its pass/fail labels are themselves noisy: the standard error of a
    20-split mean is about .035, and each q95 comes from 20 values.
  - The halves null does not include the floor between training and
    held-out donors. The fold-disjoint reference suggests that floor is
    about .74 at the cell level and .99 at `bag_pcsum`.
- **Bag-construction asymmetry, not fixed here.** Each real bag is a
  within-donor bootstrap of one donor, so the 32 bags are near-replicates.
  Each generated bag is a distinct generated donor. Per-item variance
  therefore differs by design, and bag-level AUCs partly measure that. At
  the cell level, each real set is one donor while each generated set pools
  32 generated donors. Bag-level AUCs should not be read as generator
  error.
- **Empirical references are not a null.** They reuse the same 25 source
  donors in every fold. The fold-disjoint reference is not a null either:
  it falls below .5 at `bag_inspan` and linear `cell_pc`, because a finite
  donor population is split across folds and the PCA is centred on those
  donors.
- **Uncertainty.** The declared CIs leave out classifier-refit variance.
  Amendment 2 adds it for `cell_pc` and `bag_gene` only.
- **Data.** Generated cells are stored only as PCs, so there is no
  gene-space cell level, and PPARG cannot be tested per cell. prev097 is a
  fresh generation; the receipted 2026-11-14 archives were not kept.
- **Post hoc observations.** The per-PC SD ratio comes from
  [results/posthoc_cell.json](results/posthoc_cell.json). Pooled over the 32
  generated donors of a target, EMPF has about the spread of one real
  donor (geometric mean 1.005), while pooled resampling is 1.27. This is
  consistent with too little between-donor variability, but it is
  confounded with the cell-level asymmetry above and is not a test.
- **Runtimes.** The host was heavily loaded (load average 120–200), so
  runtimes are inflated.

## Next steps

1. Re-run the `cell_pc` and `bag_gene` C2STs on the 17 fits of the
   multi-fold confirmation queue once it finishes (`cli.py score` accepts
   any generated.npz). Pooling folds lowers the 12-donor noise floor.
2. Fix the bag-construction asymmetry. Either draw real bags so their
   between-bag variance matches the generated bags, or compare at the
   donor-set level. Then rebuild a calibrated bag-level null with more
   donors.
3. Add a leave-donor-out PCA, depth-matched training-donor reference. This
   would separate projection shrinkage from generator error in `bag_pcsum`
   and in the interval metrics.
4. Test the donor-variability hypothesis directly. Compare the spread
   between generated donors with the spread between real donors, and score
   a donor random-effect variant with the same C2ST. The covariance
   finding above suggests looking at cross-PC covariance between donors in
   particular.
5. Store a gene-space subsample of generated cells, so PPARG and other
   genes can be attributed at the cell level.

## Reproduce / reuse

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
cd analysis/c2st_20261120
$PY -B -m pytest -q test_c2st.py                         # 20 synthetic tests (CPU)
# generic: score any generated.npz (repository contract) against held-out donors
CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B cli.py --device cuda score \
  --cache runs/improve_20261109/caches/hvg2000/fold0 \
  --generated mine=/path/generated.npz --levels cell_pc bag_gene --out c2st.json
# declared study
W=runs/research_20261120/c2st/work; CA=runs/improve_20261109/caches/hvg2000/fold0
$PY -B make_empirical.py --cache $CA --out-dir runs/research_20261120/c2st/empirical
$PY -B cli.py --device cuda study --phase generators --cache $CA --work $W
$PY -B cli.py --device cuda study --phase null --cache $CA --work $W
$PY -B cli.py study --phase merge --work $W --out results/c2st_fold0.json
# post hoc and Amendment 2 repairs
$PY -B posthoc.py --cache $CA --out results/posthoc_cell.json
$PY -B reference_disjoint.py --cache $CA --work $W --out results/posthoc_disjoint.json
$PY -B repairs.py refit --cache $CA --work $W --levels cell_pc   # and --levels bag_gene
$PY -B repairs.py secondorder --cache $CA --work $W
$PY -B repairs.py pparg --cache $CA --work $W
$PY -B repairs.py merge --work $W --out results/repairs.json
```

- `c2st.py` holds the reusable core, NumPy plus optional torch with no
  repository paths:
  - `auc`, `grouped_folds`, `run_c2st` (MLP or linear; all fold × seed
    networks are trained in one batched pass)
  - `donor_bootstrap`, `paired_bootstrap_diff`
  - `halves_null`, `permutation_null`
  - `permutation_importance`, `univariate_auc`
- `features.py` builds the item levels. Real items are identical for every
  generator.
- Resources:
  - Shared A6000s at a 0.04–0.05 memory fraction, 3 threads,
    `taskset -c 3-23`, and no more than 4 concurrent processes.
  - The repairs took about 55 minutes of wall time with 4 processes.
- Outputs under `runs/research_20261120/c2st/` are not in git: archives,
  empirical references, out-of-fold scores, and refit and repair scores.
  Git holds the merged JSON results.
