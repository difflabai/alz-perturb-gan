# Independent review: leakage and statistics (build_cache, evaluate, baselines, metrics)

Reviewer: independent subagent, 2026-11-09. CPU only, 6 threads. I opened count files only for the 37 development donors, and only through the study caches. No code files were edited.

## Checked and fine (demonstrated)

- **Caches contain only development donors.** In all 6 caches, the per-cell `donor` and `donors` in train and heldout metadata equal `folds.json` `inner_folds[k]` exactly. All 37 are development donors, and none is one of the 20 sealed donors. `manifest.donor_guard.count_files_read_by_this_process` equals the 37 development donors.
- **Transforms are fitted on train donors only.** I recomputed `mean`, `sd` and `pc_sd` from `train/counts.npz` alone, together with `age_mean`/`age_sd` from train `raw_c`. All match `space.npz` with a max difference of 0.0 in every cache. HVG selection, forced-gene dropping, the type classifier, the baseline dispersion, the ridge alpha (LOO over train donors), the shrinkage and the residual pool all use `~is_heldout` only.
- **The hvg2000 recipe matches the legacy cache.** `fidelity_legacy_fold0.json` shows HVG set and order, genes, cells, counts, `mean` and `sd` bitwise equal. `W` has a minimum column cosine of 0.9999998 and age is identical. The legacy `bench_core.fit_space` code is the same algorithm.
- **The build_cache CLI exposes no donor argument.** `check_donor_allowed` runs before every count file is opened. `trainer.validate_split` rejects non-development or sealed donors even with `--allow-nonprotocol-split`.
- **Metrics recomputed by hand agree with the evaluator.** I used my own numpy and scipy code with my own cache loader on base/hvg2000/fold0/seed0 and on the stage_mean baseline for hvg2000 fold 0. Every value agrees to float precision: pb RMSE, deviation Pearson against the train grand mean, High-minus-NotAD Spearman and slope, ordinal-slope Spearman, pb energy score, PC 80% coverage and PC energy score.
  - Fit: RMSE 0.834438, devP -0.024988, hn Spearman 0.180686, slope 0.068985, pb energy 14.7525, PC coverage80 0.54.
  - stage_mean: RMSE 0.485178, hn Spearman -0.051182, pb energy 7.19563, PC coverage80 0.63667.
  - Held-out contrasts use held-out donors only.
- **Seed-ensemble predictions pool correctly.** The ensemble `pred_profile_heldout` equals the mean of the two seed predictions (max difference 2.4e-7).
- **The pooled contrast pairs each donor with the model that did not train on it.** Each fold contributes exactly its own held-out donors (37 unique). My recomputed Spearman of 0.27336 equals the aggregate value. The leakage through the other folds' training sets is real and is already flagged.
- **Paired comparisons pair correctly.** vs_base pairs on (fold, seed). vs_stage_mean pairs on fold.
- **Baselines go through the same scoring path.** They use `score_generated` and draw libraries from the target's own `libs`. The resampling pool excludes the target and contains train donors only. The noise ceiling matches its documented definition.

## Findings

1. **Major: profile-only baselines are credited with PC and cell metrics that their own `MEANINGFUL` record marks as not meaningful.**
   - `summary.md` lists pc_energy, pc_coverage80, pc_spread and cell_energy for grand_mean, stage_mean, shrunk and ridge with no marking.
   - The paired base vs stage_mean comparison reports pc_energy as -0.520 (0/4), cell_energy as n_worse=4 and pc_coverage80 as n_worse=4. `baselines.MEANINGFUL` declares `pc_space/cell_energy: NOT meaningful` for these baselines.
   - Fix: mask these metrics, or set them to None, for the profile baselines in rows, paired comparisons and the summary.
2. **Major: the non-renormalised `shared_<panel>` subset is not like-for-like across panels.**
   - The same 2000 genes and the same stage_mean baseline on fold 0 give these held-out pb RMSEs:
     - hvg2000 `all`: 0.4852.
     - wide `shared_hvg2000`: 0.3822. CP10k uses the 5,028-gene denominator.
     - wide `shared_hvg2000_renorm`: 0.4875.
   - grand_mean shows the same pattern: 0.4460, 0.3501 and 0.4461.
   - mean_log and nonzero exist only in the non-renormalised form, because per-cell values cannot be renormalised from generated.npz. Their per-cell metrics are therefore never like-for-like across panels (0.0964 against 0.0838).
   - The `_renorm` pseudobulk variant is the valid comparison. hvg2000 is a strict subset of wide in all folds, so the renorm subset equals the hvg2000 panel.
   - Fix: label the non-renorm shared subset as within-panel only, or drop it. For per-cell metrics, add per-cell CP10k over the shared genes to Interface 2.
3. **Major: the goodness scale for `abs1` metrics returns None for non-positive values, so it silently drops pairs where a slope is negative.**
   - In `goodness`: `return -abs(np.log(v)) if v > 0 else None`.
   - In smoke_results, `hn_slope_vs_heldout` for base vs stage_mean has n=1 of 4 units. The dropped units are base seed1 fold 0 (-0.123), fold 2 (-0.001) and stage_mean fold 0 (-0.043). A wrong-sign slope, the worst outcome, is excluded instead of being penalised.
   - Fix: use a goodness that is defined for v ≤ 0, for example `-abs(v-1)`, or keep the pair and count it as worse.
4. **Minor: SEs and means treat non-independent units as independent.**
   - `summary.md` reports "± SE over (fold, seed) units". Seeds in the same fold share held-out donors and training data, so the base pb_rmse SE is 0.0105 against a fold SE of 0.0146.
   - Row means are not fold-balanced. base has 4 units, with fold 0 counted twice, against 3 baseline folds.
   - vs_stage_mean reuses the fold-0 baseline twice in sign counts (n=4).
   - The inner folds also share 2/3 of their training donors, which the notes do not state.
   - Fix: report the fold SE (seed-averaged within fold) in the summary, and state the dependence.
5. **Minor: the designated `stage_mean` comparator is weaker than simple alternatives.**
   - It is log1p of the mean train proportion, which is a log-of-mean and is biased against donor log profiles.
   - Fold 0 profile RMSE before generation: stage_mean 0.507, train stage mean of log profiles 0.477, grand_mean 0.483, ridge 0.440. After generation: stage_mean 0.485 against grand_mean 0.446.
   - "Beats stage_mean" is therefore a low bar. Fix: also require beating ridge, shrunk_stage_mean or a mean-of-log stage mean.
6. **Minor: the "noise ceiling" is not a ceiling.**
   - On fold 0, `heldout_peers_same_stage` RMSE is 0.532 and `all_dev_peers` is 0.473. The no-stage grand_mean scores 0.446 and ridge 0.442.
   - The ceiling uses full-depth observed profiles while predictions are 256-cell generated pseudobulks. Treat it as a peer-variability reference.
7. **Minor: depth mismatch in the real summaries for small donors.** 9 of the 37 donors have fewer than 256 cells (149-245). `real_pb_subsample` then uses all their cells, while generated donors always use K=256, so the real distribution_pb summaries for these donors are noisier.
8. **Minor: `evaluate.py` has no development-donor guard.** `load_real` reads any cache under `--caches-root`, and grep finds no sealed or development check in evaluate.py or baselines.py.
9. **Minor: build_cache reads the whole shared per-cell metadata file.**
   - `build_cache._cell_meta` parses all of `mg_meta.json` (82 donors: 37 development, 20 sealed, 25 other). It includes per-cell `Number of UMIs` and `Supertype` for sealed donors.
   - Only development rows are used, and the manifest discloses it. PROTOCOL limits reading other donors' metadata to the inventory task.
