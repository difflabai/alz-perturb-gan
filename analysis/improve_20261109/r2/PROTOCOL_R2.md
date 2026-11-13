# improve_20261109 round 2: fix each failure found in the screen

Owner request (Mikkel, 2026-11-10): "for each of the failures and mitigations
attempt to fix them". Read `../PROTOCOL.md` (still binding, including every
hard rule) and `../RESULTS.md` (what failed and why) first. This file adds the
round-2 rules, the fix list and the interfaces.

## Round-2 hard rules (in addition to `../PROTOCOL.md`)

- Round-1 code is FROZEN: never edit `../trainer.py`, `../arms.py`,
  `../generate.py`, `../evaluate.py`, `../baselines.py`, `../metrics.py`,
  `../build_cache.py`, `../run_screen.sh`, `../folds.json`, the tests, or any
  `.md` outside `r2/`. A confirmation run is using them on the GPUs right now.
  All round-2 code lives in `/ml2/alz/analysis/improve_20261109/r2/`; import
  round-1 modules read-only or copy them.
- Never modify or delete anything under `/ml2/alz/runs/improve_20261109/fits/`,
  `caches/{hvg2000,wide}/fold{0,1,2}/`, `results/`, `results_screen48/` or
  `screen_analysis/`. Round-2 outputs go to `runs/improve_20261109/fits_r2/`,
  `results_r2/`, `posthoc_r2/`, `smoke_r2/` and new cache folders
  `caches/hvg2000/fold{3..8}/`.
- The released ParticleGAN 0.9.7 update rule stays unchanged: same optimizer,
  penalty, loss, prior regulariser and policy calls, in the same order, as
  `../trainer.py::advance`. A fix may change only the application: generator
  head, conditioning (including condition dropout), what the critic observes,
  generation-time procedure, post-hoc calibration, evaluation. No added loss
  terms, no EMA of weights, no optimizer or learning-rate changes. If a fix
  cannot be done inside these limits, say so instead of bending them.
- Round-1 arm names keep their exact semantics in round 2. `base` and
  `genemean` in `r2/trainer.py` must stay bitwise equal to round 1 (initial
  tensors and the first 5 updates on CPU), and round-1 `final.pt` files must
  load in `r2/generate.py`.
- Every transform is fitted on the inner fold's TRAIN donors only. Anything
  tuned (shrinkage factors, dropout rates, scales) is either fixed in advance
  and documented, or chosen by cross-fitting inside the train donors. Never
  use held-out donors to choose anything.
- Machine use: at most 3 CPU threads per process
  (`OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3`). The GPUs are
  shared with a confirmation run until about 19:00 UTC, and the owner has said
  sharing is fine (2026-11-10): any task may use a GPU for short test fits of
  at most 300 updates and for generation, one process at a time per task
  (the integration and repair tasks: two), always with
  `--gpu-mem-fraction 0.06`. Never kill or signal processes you did not start.
- Subagents cannot write `.md` report files in this environment. Put design
  documentation in module docstrings and return notes in your structured
  output. Python: `/mnt/nvme/ai/conda/anaconda3/bin/python3 -B`.

## Failures and their fixes

| ID | Failure (RESULTS.md) | Fix to attempt | Owner task |
| --- | --- | --- | --- |
| F1 | `covars` memorises donors: nearly unique (sex, APOE4, age) tuples act as a donor key | `covdrop` (covariate dropout in the condition seen by generator and critic) and `covbin` (coarse covariates: sex, APOE4 carrier, age tertile); generation mode `margcov` (covariates marginalised over same-stage train donors) | cov |
| F2 | `offset` learns non-stage structure; stage path is duplicated | An anchored stage path (for example a learnable gain per stage on a fixed train stage log-ratio direction) | stage |
| F3 | Predicted stage differences are 3-8x too large, even on train donors | Stage-condition dropout with shrunk generation, and/or generation-time condition shrinkage; post-hoc stage-deviation shrinkage | stage, posthoc |
| F4 | `donorpc` block takes about 75% of critic first-layer input variance | `donorpc2`: unwhitened or rescaled block with a documented, bounded variance share | critic |
| F5 | Remaining per-gene level bias (pseudobulk-vs-mean-log shape gap) | `pbulk`: a per-bag pseudobulk (linear in counts) critic block, composable with `genemean`; post-hoc train-only per-gene recalibration | critic, posthoc |
| F6 | `genemean`: critic gap near 13, twice the seed variance, worse PC/cell fit | `genemean@s` block-scale variants; late-snapshot ensembles at generation; diagnosis of which it is | gm |
| F7 | No generator matches the grand mean on held-out pseudobulk | Post-hoc calibration of predictions (recalibration, stage shrinkage, seed ensembling) evaluated honestly against the same baselines | posthoc |
| F8 | Evaluator: per-fold contrast Spearman cannot rank arms; PC metrics depend on one 64-cell subsample; panels shown side by side; only one meaningful meanlog baseline | Evaluator r2: repeat-averaged PC metrics, frozen DAM direction score, pooled cross-fitted donor score, like-for-like tables, recalibrated metrics, second-moment metric, generation-mode variants | evaluator |
| F9 | Three overlapping folds carry all fold-by-arm evidence | Two more repeated stratified 3-fold splits of the 37 donors (inner folds 3-8) with caches | splits |

## Interfaces

### Arm strings

Comma-joined sorted flags as in round 1. A flag may carry one parameter:
`name@value` (for example `genemean@0.5`, `covdrop@0.5`). Round-1 flags without
a parameter mean exactly what they meant in round 1. Fit folders use the arm
string with `@` kept: `fits_r2/{arm}/{panel}/fold{k}/seed{s}/`.

### Arm plug-ins (written by the core task, used by every arm task)

Each new arm lives in its own file `r2/arm_<name>.py` and registers itself with
the registry in `r2/arms.py`. The core task defines and documents the hook API
in the `r2/arms.py` docstring. It must cover: (a) building the per-donor
condition matrix; (b) a per-batch condition transform during training (for
dropout), driven by its own seeded stream so that arms without it keep the
round-1 random streams bit for bit; (c) wrapping or replacing the generator
class; (d) one or more critic feature blocks appended after the released 150
features, several of which can be combined (`genemean` + `pbulk`); (e)
generation-time hooks: the condition used for a target donor and named
generation modes; (f) checkpointed state for the exact consistency check.
Arm tasks edit ONLY their own `arm_<name>.py` and `test_arm_<name>.py`.

### Fit outputs (extends Interface 2)

- Same files as round 1, plus generator and prior snapshots `snap_{u}.pt` at
  updates 16000, 17000, 18000 and 19000 (small: generator and prior state
  only; scaled proportionally for short smoke fits).
- `generated.npz`: as round 1 plus `sq_log` float16 [T,M,G], the mean over the
  K cells of squared per-cell log1p(panel CP10k).
- Generation modes write `generated_{mode}.npz` with the same layout:
  `snapens` (the M generated donors split evenly over the four snapshots and
  the final state), `margcov` (F1), and any mode an arm registers (for example
  `condshrink@0.5`). Modes never retrain.

### Folds

`r2/folds_r2.json`: same schema as `../folds.json`, inner folds 0-2 identical
to round 1, plus inner folds 3-5 and 6-8 (two more stratified 3-fold splits of
the same 37 development donors, seeds documented). `r2/trainer.py` validates
splits against this file.

### Frozen stage-signal gene list (F8)

Literature disease-associated-microglia set, fixed here before any round-2
fit. These donors have already been examined with similar lists, so within
this study the score is descriptive; it is frozen for use on new data.

- Expected higher with pathology: APOE, TREM2, TYROBP, SPP1, GPNMB, LPL, CD9,
  LGALS3, CST7, ITGAX, CLEC7A, AXL, CSF1, CD63, CTSD, CTSB, FTH1, LILRB4,
  MITF, MYO1E.
- Expected lower with pathology: P2RY12, TMEM119, CX3CR1, P2RY13, SELPLG,
  CSF1R, SALL1.

Score: over the listed genes present on the panel, the signed agreement of the
model's High-minus-NotAD mean-log contrast with the expected direction, and
its correlation with the held-out donors' contrast, each compared with an
expression-matched random-gene null. Report how many listed genes are present.

### Evaluation

`r2/evaluate.py` reads fits from one or more roots (round-1 `fits/` and
`fits_r2/`), treats each generation mode as its own unit (`arm` + `#mode`),
and writes to `results_r2/`. Primary metrics: `pb_rmse`, `meanlog_rmse_gene`,
`nonzero_rmse_gene`, and their train-only recalibrated versions; PC and cell
metrics averaged over repeated real-cell subsamples; the DAM score and pooled
cross-fitted donor score as the stage metrics. A change counts as a fix only
if it beats round-1 `base` on held-out donors AND is compared honestly with
the grand-mean, ridge and resample baselines.
