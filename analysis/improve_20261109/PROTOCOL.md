# improve_20261109: development study for gene-level accuracy

Owner request (Mikkel, 2026-11-09): act on the review of the stage-conditioned
ParticleGAN microglia model. This is a NEW exploratory development study. It is
separate from benchmark097 and from every sealed/audited result in this repo;
its numbers are not comparable with those and it makes no SOTA or held-out claim.

## Why

Review findings on fold-0 training donors (CPU-only, frozen models):

1. The real High-minus-NotAD contrast is barely reproducible across donor halves
   (split-half Spearman ~0.10 over 2,000 genes; only 3 Not AD training donors).
2. Critic gap stays ~4-6 logits through 20k updates in all five seeds.
3. 90-96% of per-stage squared mean-expression error is outside the 50-PC span
   the critic sees.
4. Per-stage mean profiles correlate 0.63-0.88 with real; top-decile effects are
   shrunk to slope ~0.36; seed-to-seed effect correlation ~0.50.
5. Selection used PC-space metrics on four calibration donors.

## Hard rules

- Donors: ONLY the 37 development donors in `folds.json`
  (`development_donors`). Never read counts or expression for the 20
  `sealed_evaluation_donors_fold0`, nor for the 25 SEA-AD donors outside the
  57-donor fitting universe. Donor-level *metadata* for other donors may be
  read only by the data-expansion inventory task.
- Do not modify, move or delete anything outside
  `/ml2/alz/analysis/improve_20261109/` (tracked code/docs for this study) and
  `/ml2/alz/runs/improve_20261109/` (untracked caches, fits, results).
  Everything else in `/ml2/alz` and `/ml2/codex-storage` is read-only.
- No git commits, branches, stashes, checkouts or resets. No pushes.
- All new training starts from scratch (fresh generator, critic, particle
  prior, optimizer state). Existing checkpoints are read-only.
- Keep the released ParticleGAN 0.9.7 update rule
  (`analysis/faithful097/model/released_update.py::advance`, recipe
  `bispf_optimized`, 512 particles, z_dim 8, batch 32 donors x 128 cells,
  20,000 updates, EMA off, hyperparameters of candidate t005:
  generator_lr .012, critic_lr .018, particle_lr .06, reg_coeff 1, reg_kappa 1,
  prior_reg .01). Arms change the *application* (generator head, what the
  critic observes, conditioning, gene panel), never the optimizer/penalty/
  policy calls. Every difference from the released adapter must be listed in
  the code docstring.
- The machine is shared. CPU: at most 6 threads per process unless told
  otherwise. GPU: only when a task explicitly says so; a fit needs <1 GB.
- Disk is 98% full (about 170 GB free). Keep outputs small; no raw generated
  count tensors on disk.
- Python: `/mnt/nvme/ai/conda/anaconda3/bin/python3` (torch 2.14, numpy 1.26,
  sklearn 1.4). ParticleGAN 0.9.7 is pinned at
  `/ml2/alz/artifacts/native-releases/particlegan-0.9.7/site-packages`
  (see `analysis/faithful097/model/release_env.py`).

## Key existing sources (read-only)

- Released adapter/update: `/ml2/alz/analysis/faithful097/model/`
- Generator, standardisation, straight-through count features:
  `/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/pg_model.py`
- Data loading, condition encoding, space fitting, metrics:
  `/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/bench_core.py`
- Critic features and critic, stage law, contracts, usage:
  `/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/`
- Existing fold-0 cache (33 optimization donors):
  `.../core-comparison-20261103/caches/fold0/` (`space.npz`,
  `train/counts_hvg.npz`, `train/metadata.npz`)
- Raw per-donor counts, 36,601 genes:
  `/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt/`
  (`counts/{donor}.npz`, `{donor}_names.npy`, `mg_meta.json`,
  `candidate_cohort.json`)
- Reference fits of the released model: `/ml2/alz/runs/CAL_GPU0_V1_lf097_t005_f0_pg_s{0,1}`,
  `/ml2/alz/runs/CAL_GPU0_V1_cf097_t005_f0_pg_s{2,3,4}` (`fit/final.pt`,
  `fit/learning_curve.jsonl`, `completed.json`)

## Design

Three inner folds over the 37 development donors (`folds.json`,
`inner_folds[k]`: `train_donors`, `heldout_donors`). All transforms (gene
selection, scaling, PCA, type classifier if any, library pools, offsets,
covariate standardisation) are fitted on that inner fold's train donors only.

### Interface 1: cache (written by `build_cache.py`, read by trainer/evaluator)

`/ml2/alz/runs/improve_20261109/caches/{panel}/fold{k}/`

- `manifest.json`: panel, inner_fold, train_donors, heldout_donors, n_genes,
  forced_genes present/missing, SHA256 of files, recipe notes.
- `space.npz`: `genes` (str[G]), `hvg` (int index into the 36,601-gene axis),
  `mean`[G], `sd`[G], `W`[G,50], `pc_sd`[50], `age_mean`, `age_sd` - same
  recipe and semantics as the existing fold cache `space.npz`
  (`bench_core.py`), fitted on train donors only.
- `train/counts.npz` (scipy CSR, cells x G, float32 integer-valued),
  `train/metadata.npz` with `cell`, `donor` (per cell), `type`, `donors`
  (ordered donor ids), `raw_c` [n_donors,7] exactly as
  `bench_core.encode_condition` (Braak/6, Thal/5, CERAD/3, ADNC/3, Male,
  APOE4 dose, raw age), `genes`.
- `heldout/counts.npz`, `heldout/metadata.npz`: same layout, same gene axis,
  projected with the train-fitted space. Zero-panel-count cells are dropped
  with the same rule as the existing cache and counted in the manifest.

Panels: `hvg2000` (existing recipe, 2,000 genes) and `wide` (5,000 HVGs by
the same rule plus forced inclusion of AD/microglia risk genes present on the
axis: APOE, TREM2, INPP5D, ACSL1, SORL1, SPI1, DGAT1, DGAT2, ABCA1, BIN1,
CD33, TYROBP, NLRP3, PLCG2, ABI3, MS4A6A, MS4A4A, CD2AP, PICALM, CLU, CR1,
ABCA7, GRN, TMEM119, P2RY12, CX3CR1, CSF1R, SPP1, GPNMB, LPL, CD74, HLA-DRA).

### Interface 2: fit output (written by trainer, read by evaluator)

`/ml2/alz/runs/improve_20261109/fits/{arm}/{panel}/fold{k}/seed{s}/`

- `config.json` (arm flags, panel, fold, seed, hyperparameters, updates,
  device), `final.pt`, `learning_curve.jsonl` (same fields as the released
  adapter, every 250 updates), `fit_receipt.json` (completed updates, wall
  seconds, device, peak memory).
- `generated.npz`, produced after training with fixed common random numbers
  (generation seed depends only on fold and target donor index, not on arm or
  training seed). For every target donor t among the fold's train + heldout
  donors (T = 37): M = 32 generated donors x K = 256 cells, conditioned as the
  arm conditions (stage only, or stage + covariates of the target donor), with
  library sizes sampled with replacement from the TARGET donor's own observed
  panel library sizes (depth-matched). Arrays:
  - `targets` (str[T]), `is_heldout` (bool[T]), `stage` (int[T]), `genes` (str[G])
  - `pb_counts` float32 [T,M,G]: counts summed over the K cells
  - `mean_log` float32 [T,M,G]: mean per-cell log1p(panel CP10k)
  - `nonzero` float32 [T,M,G]: fraction of the K cells with count > 0
  - `pcs` float16 [T,M,64,50]: fold-space PC coordinates of the first 64 cells
  - `lib` float32 [T,M]: mean generated library size
- Arm names are comma-joined sorted flags, `base` when none:
  - `offset`: learnable stage x gene additive logit offset on the generator
    output, initialised from shrunken per-stage train pseudobulk log-ratios.
  - `genemean`: critic additionally sees the per-bag mean of standardised
    log expression for every panel gene (documented scaling).
  - `donorpc`: critic additionally sees per-bag means projected on principal
    components of train-donor pseudobulk profiles (alternative to `genemean`).
  - `covars`: condition vector also carries Male, APOE4 dose and age
    standardised on train donors (pathology slots other than ADNC stay 0).

### Interface 3: evaluation (written by `evaluate.py`)

`/ml2/alz/runs/improve_20261109/results/` - per-fit JSON plus one aggregate
table (JSON + CSV). Primary metrics are computed on HELD-OUT inner-fold
donors, with the same quantities on train donors reported as fit quality:

- Pseudobulk prediction: per held-out donor, error between its observed
  pseudobulk log1p(CP10k) profile and the model's mean generated pseudobulk
  profile (RMSE over genes; Pearson of deviations from the train grand mean).
- Per-gene mean per-cell log expression and nonzero-fraction errors
  (depth-matched).
- Stage-contrast agreement (High minus Not AD and pathology-ordered slope):
  Spearman, sign agreement on the top decile, regression slope - against the
  held-out contrast and against the train contrast.
- Donor-level distribution: energy score, spread ratio and 80% coverage of
  held-out donor summaries; cell energy distance in fold PC space.
- Seed stability and seed-ensemble versions of the above.

Baselines evaluated with exactly the same code path: train grand-mean
pseudobulk (no stage), train stage-mean pseudobulk, shrunken stage mean,
stage + covariate ridge regression on train pseudobulks, and real-donor
resampling (train donors of the same stage, with depth-matched thinning or a
documented alternative). Report noise ceilings (held-out donor vs other
held-out donors of the same stage).

A model change counts as an improvement only if it beats the `base` arm AND
is compared honestly against the stage-mean baseline on held-out donors.
