# Donor-held-out classifier two-sample test (C2ST) — declared protocol

Declared 2026-11-20, before any C2ST score on real or generated microglia
data exists. Only synthetic data (unit tests, timing) were run before this
commit. Code: `c2st.py` (core), `features.py` (item builders), `cli.py`
(CLI + declared study), `plan.json` (fixed inputs), `test_c2st.py`.

## Question

Can a small classifier, trained on other donors, tell generated microglia
from real cells/bags of held-out development donors? Where does it look?
This adds a learned two-sample metric next to the frozen energy, profile and
interval metrics. It does not replace them and changes no default.

## Data and separation

- Cache `runs/improve_20261109/caches/hvg2000/fold0` (manifest sha256
  `65d5e2f4…`), 2000-gene panel, TRAIN-fitted PCA50.
- Only the 12 fold-0 held-out development donors enter any item set
  (stages: Not AD 1, Low 2, Intermediate 3, High 6). GAN-training donors are
  never scored. No sealed/external donors.
- Generated items for held-out target donor *d* are those conditioned on
  *d*; real items for *d* are *d*'s own cells.

## Inputs (plan.json, hashes checked at run time)

| name | role | archive |
| --- | --- | --- |
| `empf05` | **primary**: default native EMPF fold0/seed0, `cellshape@0.5:biascal:stagehalf` | `runs/microglia_empf/verification/final_generated.npz` |
| `empf025` | quarter-contraction readout, same fit | `runs/coverage_empf_20261117/quarter_generated.npz` |
| `prev097` | previous 0.9.7 BISPF default (fits_r3 pbulk fold0/seed0, same readout) | fresh generation with the archived r5 `mode_cellshape.py` (the receipted 2026-11-14 archives were not retained), `runs/research_20261120/c2st/prev097_cellshape05.npz` |
| `resample_blind` | **reference**: stage-blind empirical resampling (all training donors, depth-thinned to the target library distribution) | `runs/improve_20261109/results_r3/baselines/hvg2000/fold0/resample_blind.npz` |
| `resample_stage` | reference: stage-matched empirical resampling | `…/fold0/resample.npz` |

`prev097` was still generating when this protocol was declared; its hash is
bound by its generation receipt (`"sha256": "receipt"` in plan.json), and it
is scored with the identical code/settings once it exists.

All archives use the repository generated.npz contract: 32 generated donors
per target, 256 cells per generated donor, 64 of them stored as PCs.

## Item levels

- `cell_pc` (primary cell level): one cell = 50 PCs. Per held-out donor,
  n = min(cells, 256) real cells and n generated cells for that target
  (fixed seeds; identical real items for every generator). Real PCs pass
  through float16 like the stored generated PCs. Generated cells exist only
  in PC space, so there is no gene-space cell level.
- `bag_gene` (primary donor-bag level): one bag = mean per-cell
  log1p(panel CP10k) over 256 cells, 2000 genes. Real: 32 bags per donor,
  cells resampled with replacement from that donor. Generated: the 32
  generated donors of that target.
- `bag_pcsum`: per-bag PC mean and SD over 64 cells (interval-metric space).
- `bag_inspan` / `bag_offspan`: `bag_gene` standardized with the TRAIN PCA
  scaling, split into its projection onto the 50 PC directions and the
  orthogonal residual. The residual is invisible to anything computed from
  the 50 PCs (the critic's input). This is the PPARG blind-spot test.

## Classifier and cross-fitting (fixed)

- Grouped 6-fold CV over the 12 held-out donors, stratified by stage (seed
  20261120); every item of a donor (real and generated) is in the same fold.
  Pooled out-of-fold ROC AUC is the statistic; 0.5 = indistinguishable.
- `mlp` (primary): cell 50→128→64→1, 40 epochs, batch 256; bag d→64→1,
  150 epochs, batch 64, dropout .2, weight decay 1e-2; AdamW lr 1e-3,
  class-balanced loss, features standardized with training-fold statistics,
  3-seed ensemble per fold. `linear` (secondary): logistic regression with
  the same trainer (no hidden layer, lr 1e-2, weight decay 1e-2).
- No hyperparameter tuning on real data. Same folds/items/seeds for all
  generators (matched controls).

## Uncertainty and calibration

- 95% CI: donor bootstrap of out-of-fold scores (2000 resamples, no refit;
  it omits classifier-training variability). Per-stage AUC on that stage's
  donors (Not AD has one donor; descriptive only).
- Paired differences vs `empf05`: joint donor bootstrap of AUC(other) −
  AUC(empf05).
- Real-vs-real null: 20 random stage-stratified splits of the 12 held-out
  donors into disjoint 6/6 halves, label by half, folds = one donor of each
  half, same classifier. **Calibration passes if the mean null AUC is within
  0.5 ± 0.05** for that level/classifier. Its 95th percentile is reported.
- Within-donor label-permutation null (primary generator's items, 20
  permutations, same folds, each classifier).

## Declared decisions

1. **Detection**: a generator is detected at a level if the bootstrap CI
   lower bound > 0.5 and its AUC exceeds both the halves-null 95th
   percentile and the maximum permutation-null AUC.
2. **Readout comparison**: quarter (`empf025`) is less distinguishable than
   `empf05` at a level only if the paired 95% CI of AUC(empf025) − AUC(empf05)
   is entirely below 0. Otherwise "not resolved". Same rule for `prev097`
   and for the empirical references.
3. **Reference**: report empirical resampling AUCs alongside; a generator
   "matches empirical resampling" at a level only if the paired CI of
   AUC(resample_blind) − AUC(empf05) includes 0.
4. **Blind-spot link** (pre-stated expectation, not a gate): if the PPARG
   mechanism persists in EMPF, `bag_offspan` is detected, and PPARG ranks
   in the top 1% (≤ 20 of 2000) of genes by univariate within-donor AUC
   distance from 0.5 for `empf05`. Reported either way, with the signed
   off-span share of PPARG's real-minus-generated standardized mean gap.

Primary endpoints: `empf05` MLP AUC at `cell_pc` and `bag_gene`. Everything
else is secondary/descriptive; no multiplicity correction. One fold/seed of
one fit; 12 reused development donors (already used for model selection).

## Attribution (descriptive)

For `cell_pc` and `bag_gene` MLPs of every generator: per-feature
within-donor univariate AUC and out-of-fold permutation importance (AUC drop,
2 repeats), top 25 each, plus focus features (PPARG, CLEC5A, and the PPARG
lipid/DAM neighbours present in the panel) with ranks and per-stage means.

## Resources

Shared RTX A6000s, ≤ 0.05 per-process memory fraction, 3 CPU threads,
`taskset -c 3-23`, at most 4 concurrent GPU processes (levels split across
processes). Large outputs under `runs/research_20261120/c2st/`.
