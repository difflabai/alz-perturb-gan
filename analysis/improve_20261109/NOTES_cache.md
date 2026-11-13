# cache

Component files: `build_cache.py`, `test_cache.py`, this note; outputs in
`/ml2/alz/runs/improve_20261109/caches/`.

## Commands

```
cd /ml2/alz/analysis/improve_20261109
CUDA_VISIBLE_DEVICES='' /mnt/nvme/ai/conda/anaconda3/bin/python3 -B build_cache.py --fidelity      # legacy check
CUDA_VISIBLE_DEVICES='' /mnt/nvme/ai/conda/anaconda3/bin/python3 -B build_cache.py [--overwrite]    # 2 panels x 3 folds
CUDA_VISIBLE_DEVICES='' /mnt/nvme/ai/conda/anaconda3/bin/python3 -B test_cache.py
```

The build takes about 30 s and peaks at about 2.9 GB RSS. It is deterministic: a rebuild of
hvg2000/fold0 gave byte-identical files, including the same SHA256 for every file.

## Recipe source

The recipe matches `bench_core.fit_space` and the legacy preparation
`core-comparison-20261103/prepare_folds.py` + `fold_data.py`, which wrote `caches/fold0`.
That includes the receipt and SHA checks, the HVG rule, the stable argsort, dropping train
cells with zero panel counts, `std+1e-6`, `randomized_svd` on `rng(0).choice(min(20000,n))`
with `n_iter=5, random_state=0`, `pc_sd` and the per-donor age scaler. Per-donor rule from
`bench_core.load_dataset`: the cell count must equal `candidate_cohort.json` and be at least 100.
All 37 development donors pass this rule.

## Fidelity against legacy fold-0 cache (33 optimization donors, hvg2000)

Result file: `caches/fidelity_legacy_fold0.json`. This run used the study python
(numpy 1.26.4, sklearn 1.4.2) with 1 BLAS thread.

- Exact matches: the HVG set and order (2000), the gene names, the cell set and order
  (14,762), donor, type, counts (all entries equal; legacy dtype int32, ours float32), `mean`
  and `sd` (bitwise), and `age_mean`/`age_sd`. No zero-panel cells were dropped.
- `W` is not bitwise equal. After sign alignment there are 0 sign flips, the max abs diff is
  4.8e-5, the min column |cosine| is 0.99999980, and the min principal cosine of the subspace
  is 0.9999990. `pc_sd` has a max abs diff of 3.6e-6 (rel 2.0e-6). The train PCs differ from
  legacy `real_pcs.npy` by at most 3.5e-3 (in unit-variance PC units).
- Cause: float32 BLAS rounding inside `randomized_svd`/matmul, not a recipe difference. I ran
  the same script under the legacy runtime
  (`benchmark/current_base_20261029/runtime/bin/python`: numpy 2.4.6, sklearn 1.9.1) with 1
  thread, which the legacy preparation also used. That run reproduces legacy `W`, `pc_sd` and
  `real_pcs` **bitwise** (0.0 diff; `fidelity_legacy_fold0_legacy_runtime_1thread.json`). The
  legacy runtime with 6 threads gives a 6.8e-5 W diff, and the study python with 6 threads
  gives 9.4e-5. Library versions and thread count are the only sources of the difference.
- Decision: to keep the cache bit-reproducible, `build_cache.py` defaults to 1 BLAS/torch
  thread (`CACHE_THREADS` env, 6 at most). The study python is used as the protocol requires,
  so `W` matches legacy only up to the tolerance above.

## Interpretation choices

- **Heldout zero-panel cells**: the legacy cache dropped zero-panel cells only in train and
  *retained* them in evaluation (it had 0 such cells). PROTOCOL Interface 1 says heldout
  zero-panel cells are dropped and counted, so both splits are dropped. For all 6 caches the
  count is 0 in both splits, so the two choices give the same data.
- **raw_c**: this is the full `bench_core.encode_condition` vector, as the protocol requires.
  The legacy cache stored stage-only rows `[0,0,0,ADNC/3,0,0,age]`. Slots 3 and 6 agree with
  legacy.
- **wide panel**: the top 5000 by the same ranking come first. Then the forced genes that are
  present and not already in the panel are appended in the listed PROTOCOL order, so
  hvg2000 = wide[:2000] for each fold. All 32 forced genes are on the axis and none have zero
  train counts. CR1, GPNMB, LPL (and SPP1 in folds 0 and 2) are already in the top 5000, so
  G = 5028 / 5029 / 5028. Per-gene rank, score and train totals are in the manifest
  (`forced_genes.detail`).
- **space.npz** adds `genes`, as Interface 1 requires. Because of that extra key,
  `bench_core.load_space` (which passes every key to `Space`) would fail on it; consumers
  should drop `genes` before building a `Space`.
- **Extra files** not in Interface 1: `type_classifier.pt` (bench_core.TypeClassifier fitted on
  train PCs, as in the legacy cache) and `hvg` inside metadata.npz.
- **Metadata files**: `mg_meta.json` and `candidate_cohort.json` are single shared JSON files,
  so they are parsed whole. Only rows keyed by the cell names in development donors'
  `_names.npy` and the development donors' cohort entries are accessed. No count or name file of
  any other donor is opened. `check_donor_allowed` raises before any file access, and the
  counts directory is never globbed.

## Outputs

Total size is 48,541,315 bytes: hvg2000 13.7 MB, wide 34.8 MB, plus 4 small fidelity JSONs.
Cells: fold0 11,486 train / 5,546 heldout; fold1 10,697 / 6,335; fold2 11,881 / 5,151.

## Observation (not this component)

At 19:58 on 2026-11-09, before my first run, `.pyc` files for pg_model, stage_law, critic,
usage and contracts were written into the read-only `benchmark/__pycache__` and
`core-comparison-20261103/__pycache__`. My code imports only bench_core with bytecode writing
disabled. Some other process likely imported those modules without `-B`.
