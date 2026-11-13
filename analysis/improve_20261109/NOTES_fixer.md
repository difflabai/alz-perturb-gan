# NOTES_fixer (component: fixer)

Date: 2026-11-09/10. Scope: verify and fix the findings of the two independent
reviews (`REVIEW_fidelity.md`, `REVIEW_leakage_stats.md`) of the integrated
harness, re-run every test and a fresh CPU smoke test. No 20,000-update fit was
launched. All runs: CPU, `CUDA_VISIBLE_DEVICES=''`, at most 6 threads per
process, study python with `-B`.

## fixer: per-finding decisions

### F1 [major, fidelity] offset_init shrinkage could not tell stage signal from noise — FIXED (arms.py)

Verified. The pooled stage log-ratio `r_sg` is depth-weighted, so its
sampling variance is larger than `v_g/n_s`, and the global method-of-moments
`tau2` was positive under permuted stage labels.

Fix (`arms.offset_init`, version `offset_init-v2-donor-averaged`, formula in
the arms.py docstring): `r_sg` is now the donor-averaged log-ratio of stage s
minus the all-train-donor average, `m_g = mean_d r_dg`. Its sampling variance
under no stage effect is exactly `v_g (1/n_s - 1/n)` for exchangeable donors.
That term replaces `v_g/n_s` in `tau2` and in the weights. The table now holds
stage contrasts only (`sum_s n_s r_sg = 0`); the gene level stays in the
released output bias. Everything is still fitted on train donors only.

Permutation check (stage labels permuted among train donors 50 times, stage
sizes kept; script and output in the session scratchpad `fixer/perm.py`,
`fixer/perm_v2.json`):

| cache | n_s | tau2 (real labels) | shrunk RMS per stage | null tau2 median [5%, 95%] | null at floor | null >= observed |
|---|---|---|---|---|---|---|
| hvg2000 fold0 | 3,3,6,13 | 0.00313 | 0.011, 0.009, 0.012, 0.024 | 1e-6 [1e-6, 0.045] | 0.56 | 0.38 |
| hvg2000 fold1 | 3,3,6,12 | 1e-6 (floor) | ~3e-6 | 1e-6 [1e-6, 0.037] | 0.60 | 1.00 |
| hvg2000 fold2 | 2,4,6,13 | 1e-6 (floor) | ~5e-6 | 1e-6 [1e-6, 0.055] | 0.60 | 1.00 |
| wide fold0 | 3,3,6,13 | 1e-6 (floor) | ~5e-6 | 1e-6 [1e-6, 0.028] | 0.60 | 1.00 |
| wide fold1 | 3,3,6,12 | 1e-6 (floor) | ~5e-6 | 1e-6 [1e-6, 0.022] | 0.62 | 1.00 |
| wide fold2 | 2,4,6,13 | 1e-6 (floor) | ~6e-6 | 1e-6 [1e-6, 0.034] | 0.58 | 1.00 |

The null is now calibrated: its median sits at the floor and about 60% of
permutations hit it. With the real labels the global estimator finds no stage
signal that exceeds donor noise. `tau2` is at the floor in 5 of 6 caches and,
on hvg2000 fold 0, is not distinguishable from the null (38% of permutations
are at least as large). **Consequence for the study:** the `offset` arm now
starts from an essentially zero table (RMS <= 0.024 logits) and has to learn
any stage offset in training. This is the honest result of shrinkage with 2-3
Not AD train donors. It agrees with the real split-half Spearman of about 0.1.
The earlier non-zero tables (RMS ~0.08-0.12) were mostly noise. One global
`tau2` over all genes can hide a few genes with real effects. A per-gene or
mixture prior would be a design change and was not made.

### F2 [minor, fidelity] offset arm not bitwise deterministic on multithreaded CPU — FIXED (arms.py, test_trainer.py)

Verified by reading the code: the scatter-add backward of `stage_offset[stage]`.
The arm now applies `one_hot(stage, 4) @ stage_offset`. The forward is bitwise
equal (each output is 1*x plus exact zeros) and the matmul backward is
deterministic. New test `test_trainer.py resume_offset` covers the
`covars,genemean,offset` arm:
(a) the same arm built 3 times in one process, 6 updates each, must have a max
parameter difference of exactly 0;
(b) a 20-update CLI fit interrupted with `--stop-after 12` and resumed must give
a final model, numpy RNG, learning curve and `generated.npz` bitwise equal to
an uninterrupted fit;
(c) checks the generation-only path (F3).
Results are under "tests" below.

### F3 [minor, fidelity] generation-only path skipped config and --updates checks — FIXED (trainer.py, run_screen.sh)

Verified by reading `run()`. Now the `config.json` diff check runs before the
generation-only branch. That branch also requires `final.pt` to have
`fixed_endpoint == completed_updates == --updates`, `status == completed`, and
the same seed and flags. If `final.pt` exists without `config.json`, the run is
refused. As a second line of defence, `run_screen.sh` also checks
`"updates": $UPDATES` in `config.json` after a fit returns 0, and marks a
mismatch as failed. Test: `resume_offset` (c) copies `config.json`, `final.pt`
and `fit_receipt.json` of a 20-update fit into a fresh root. With
`--updates 30` it must fail without writing `generated.npz`. With
`--updates 20` it must generate a file bitwise equal to the original.

### F4 [major, leakage-stats] profile baselines scored on PC/cell metrics marked NOT meaningful — FIXED (evaluate.py)

Verified in code (`aggregate` never read `meaningful_metrics`).
`evaluate.mask_not_meaningful` maps each `baselines.MEANINGFUL` key to its
headline metrics. In aggregate rows and in every paired comparison, it sets
to None each metric whose entry starts with "NOT". Each row lists the
blanked metrics in `masked_not_meaningful`. The per-baseline JSONs still hold
the raw numbers, together with `meaningful_metrics`. `resample` is not
masked, because its PC metrics are meaningful.

### F5 [major, leakage-stats] non-renormalised shared subset not like-for-like across panels — FIXED by relabelling (evaluate.py)

Verified in code. The `shared_<other>` subset is renamed
`shared_<other>_withinpanel`. Each subset block now carries
`cross_panel_comparable`: "all metrics" for `_renorm`, and "nonzero only (CP10k
uses this panel's denominator)" for `_withinpanel`. Nonzero fractions do not
depend on the CP10k denominator, so they compare across panels. mean_log,
pseudobulk, contrast and distribution metrics of `_withinpanel` do not. The
aggregate notes, summary.md and README say to use `_renorm` for cross-panel
claims.
Not done: adding per-cell CP10k over the shared genes to `generated.npz`. That
would extend PROTOCOL Interface 2, which this role must not reinterpret. So
there is no like-for-like cross-panel per-cell mean_log metric. Since hvg2000 is
a strict subset of wide in every fold, `shared_hvg2000_renorm` on wide is the
like-for-like pseudobulk comparison with the hvg2000 panel.

### F6 [major, leakage-stats] abs1 goodness dropped wrong-sign slopes — FIXED (evaluate.py)

Verified in code. Contrast regression slopes (`hn_slope_vs_*`) now use the
direction `slope1`, with goodness `-|v - 1|`. This is defined for every real
value, so a negative slope counts as worse. Spread ratios keep `-|log v|`; they
are positive by construction, so None is returned only for an exactly
collapsed spread of 0. New unit check in test_evaluate.

### F7 [minor, leakage-stats] SEs treated (fold, seed) units as independent; means not fold-balanced — FIXED (evaluate.py)

Row values are now fold-balanced means: seeds are averaged within a fold, then
the mean is taken over folds. `__se` is the SE over fold means and
`__n_folds` counts them. `__unit_mean`, `__unit_se` and `__n` keep the old
per-unit numbers for reference. Paired comparisons first average the
seed-paired goodness within each fold, then report the mean difference and
sign counts over folds. A baseline fitted once per fold is no longer counted
twice. The notes state that the folds are not independent either: from
`folds.json`, any two inner-fold training sets share 12-13 of their 24-25
donors, about half (not 2/3, as the review said).

### F8 [minor, leakage-stats] stage_mean comparator is weak — FIXED by adding comparators (baselines.py, evaluate.py)

Verified from the definition: it is the log of the mean proportion. A new
baseline, `stage_mean_log`, takes the mean of train-donor log1p CP10k profiles
within each stage, with no shrinkage. Paired blocks now exist against
`stage_mean`, `stage_mean_log`, `shrunk_stage_mean` and `ridge`
(`vs_<name>_baseline`). `stage_mean` is kept because PROTOCOL names it. The
notes, summary.md and README say that beating it alone is a low bar.
`BASELINE_VERSION` is now v5.

### F9 [minor, leakage-stats] "noise ceiling" is not a ceiling — FIXED by documentation (baselines.py, evaluate.py, README)

The definition is unchanged, as are the file and key name `noise_ceiling`,
which keeps the protocol wording and the output layout. Its JSON now carries a
`role` field stating that it is a peer-variability reference, not a ceiling.
The aggregate notes, summary.md and README say the same.

### F10 [minor, leakage-stats] real pseudobulk summaries of donors with < 256 cells were shallower — FIXED (evaluate.py)

Verified: `det_subsample` returns all n < K cells. In `real_pb_subsample`,
donors with n < K now get K cells drawn with replacement, using a
deterministic per-donor seed. Their real K-cell pseudobulk then has the same
expected depth as a generated K-cell donor. Each result lists those donors under
`real_pb_summary_cells`. Interface 2 (K = 256 for every target) is unchanged.
`EVALUATOR_VERSION` is now v2, so everything is rescored.

### F11 [minor, leakage-stats] evaluator had no development-donor guard — FIXED (evaluate.py)

`evaluate.check_development_donors` runs in `load_real` before any count file
is opened. It checks the manifest donor lists and each split's metadata
`donors` and per-cell `donor`, and raises PermissionError for a donor outside
the 37 development donors or in `sealed_evaluation_donors_fold0`. Baselines
get data only through `load_real`. New unit check in test_evaluate.

### F12 [minor, leakage-stats] build_cache parsed the whole shared mg_meta.json — FIXED as far as possible (build_cache.py); caches rebuilt

`_cell_meta` now streams `mg_meta.json` and `candidate_cohort.json` with
`ijson` (`use_float=True`). It keeps only the rows whose key is a cell name
from a development donor's own `_names.npy`, and only the development
entries of the cohort file. All other rows are discarded as they are parsed
and are never stored or inspected. Missing development rows raise an error.
The bytes of a shared file necessarily still pass through the parser, so a
fully separate file would need someone to read the whole file once. The
manifest `donor_guard.metadata_note` says this.
All 6 caches were rebuilt with `--overwrite`. Every data file
(`space.npz`, `type_classifier.pt`, `train/*`, `heldout/*`) has the same SHA256
as before. Only the manifest changed: `donor_guard`, `source_pins` (the new
build_cache.py hash), wall time and RSS. So the manifest SHA changed, and any
fit made against the old caches would be refused (none exists except the old
smoke fits, which were deleted). `test_cache.py`: ALL CACHE TESTS PASSED.

## fixer: tests run after the fixes (all exit 0; logs in `/ml2/alz/runs/improve_20261109/smoke_logs_v2/`)

| command | result | wall |
|---|---|---|
| `build_cache.py --overwrite` (all 6) | data files byte-identical to before (SHA256), manifests updated | 33 s |
| `test_cache.py` | ALL CACHE TESTS PASSED | ~1 min |
| `test_evaluate.py` | 34/34 checks passed (31 old + 3 new: slope goodness, sealed-donor guard, masking/per-fold paired/reference baselines) | 574 s |
| `test_trainer.py devcache` | ok | 4 s |
| `test_trainer.py parity` | PASS (base still bitwise equal to released build/advance) | 44 s |
| `test_trainer.py grads` | all arms pass | 36 s |
| `test_trainer.py resume_offset` (new) | PASS: in-process max param diff [0.0, 0.0]; stop-after 12 + resume bitwise equal (models, RNG, curve, generated.npz); generation-only with `--updates 30` refused (rc 1, no generated.npz), with `--updates 20` bitwise equal | 443 s |
| `test_trainer.py resume` (base) | PASS | 1029 s |
| `test_trainer.py arms` | PASS, 9 x 100-update fits incl. real hvg2000/fold0 | 3027 s |
| `test_trainer.py timing` | 2.63-2.67 s/update for all arms (dev cache, G=2000) | 444 s |

Note on `arms`: the first attempt failed because stale 100-update fits from the
earlier integration run were still in `tmp_trainer_devcache/fits`. The offset
fit there had the old `arm_info` (v1 offset init), and the trainer refused to
mix it with the new config, as intended. I deleted the directory and the rerun
built every fit from scratch. In `resume_offset` (c), the refusal of a
different `--updates` came from the `config.json` diff, which now runs first.
The separate `final.pt` endpoint check is a second guard and was not exercised
on its own.

## fixer: post-review smoke test

`run_screen.sh -d cpu -p 3 -u 300 -o smoke_fits_v2 -l smoke_logs_v2` with
`base` hvg2000, `covars,genemean,offset` hvg2000 and `offset` wide (fold 0,
seed 0, 300 updates each, concurrent): 3 completed, 0 failed. The fits took
3.6-3.9 s/update on hvg2000 and 8.9 s/update on wide while the evaluator and
trainer tests ran at the same time.

Each `generated.npz` was checked:
- shapes [37,32,G] and pcs [37,32,64,50];
- 12 held-out targets;
- all values finite and pb counts integer;
- lib equal to pb sum/256;
- 0 zero-count cells.

Critic gap at step 300: base 2.62, covars,genemean,offset 9.78, offset/wide
1.40. Offset init in config: hvg2000 fold0 tau2 0.00313, wide fold0 tau2 1e-6.

The smoke fits were then evaluated with
`evaluate.py --fits-root smoke_fits_v2 --out smoke_results_v2 --folds 0`
(baselines v5, both panels). This took 14 min with 1.0 GB RSS and gave 0 errors.

Sanity checks:
- base hvg2000 held-out pb RMSE is 0.8344, the same as the reviewers' value for
  the pre-fix smoke fit, so the base arm was unchanged by the fixes.
- Baseline values reproduce the review (stage_mean 0.4852, grand_mean 0.4460,
  ridge 0.4419; wide `shared_hvg2000_renorm` stage_mean 0.4875).
- The new stage_mean_log gives 0.4802.
- Profile baselines show `-` for PC and cell metrics; resample keeps them.
- Paired blocks against the four reference baselines are present, with n
  counted over folds.
- Wrong-sign `hn_slope_vs_heldout` values are now paired. For example,
  offset/wide -0.046 against stage_mean -0.047 is n=1, where it was previously
  dropped.
- The 9 donors with fewer than 256 cells are listed under
  `real_pb_summary_cells`.
- At 300 updates every arm is far worse than every baseline on pseudobulk
  RMSE. That is expected so early in training and says nothing about the arms.

Removed: `/ml2/alz/runs/improve_20261109/smoke_fits`, `smoke_results`,
`tmp_evaluator_test`, `tmp_evaluator_test.log` and `tmp_trainer_devcache`.

Not removed, because they are not mine:
- `results/` holds a baselines-v4/evaluator-v1 run from the integrator, which
  the next evaluate run recomputes.
- `smoke_logs/` and `logs/` are older logs.

## fixer: not changed / open

- No per-cell shared-gene CP10k in Interface 2 (F5). Doing that is a protocol change.
- `tau2` is one global prior. Per-gene shrinkage would be a design choice for the owner.
- Other component notes (`NOTES_trainer.md`, `NOTES_integration.md`) are left as written, including the
  "resume bitwise equal" claim (now true for the offset arm as well, per `resume_offset`) and "noise
  ceiling" wording (see F9).
- CUDA is still untested. No 20,000-update fit was run.
