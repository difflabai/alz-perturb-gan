# Query tool and EMPF recomputation of the headline biology

This pass turns the current experimental default into a query tool. That default is native EMPF on ParticleGAN 0.10.0, fold 0/seed 0. The pass also recomputes the README's headline biology on that model. Nothing else changed: the frozen 0.9.7 report, `docs/`, the default config and every pinned source are as they were.

**Verdict: mixed.** The tool works and its outputs say what they can and cannot support. The biology recomputation is mostly negative for the single EMPF fit:
- The CLEC5A direction holds.
- The default readout halves stage contrasts by construction.
- Across-gene agreement is weak.
- The Low-stage peak is not recovered.
- The recorded EMPF gains over the previous default are within generation noise.
- Stage-blind empirical resampling still beats the model on most held-out distribution metrics.

## Protocol and amendments

The [protocol](PROTOCOL.md) was committed (75e0fdf) before any score existed. It fixes the model, readout, replicate streams, metrics, references, thresholds and the comparisons below. It has two amendments:

1. **Amendment 1 (post hoc, b19c85f).** An own-stage (`snapens`) readout sensitivity. It was declared after part C showed the halving and before that readout was generated. It is reported beside the declared result, never instead of it.
2. **Amendment 2 (post hoc, after an independent audit).** Declared before the recount. It does three things:
   - Replaces the PC blind-spot share with a bounded definition.
   - Changes the tool's Monte Carlo interval to a Student-t interval with a readout note.
   - Records that Part A has no donor-bootstrap CIs.

**Protocol deviation.** The protocol required that the adapter reproduce the scored CPU archive bitwise for replicate 0. That CPU check was stopped unfinished after 90 minutes on an oversubscribed host. A weaker check replaced it: in one process on GPU, with the same models, plans and replicate-0 streams, the adapter's generation loop matches the frozen `generate.generate_arrays` bitwise for 3 targets and all six archive arrays ([adapter validation](results/adapter_validation.json)). As a result, we have **not verified** that the GPU replicates 1–8 are exchangeable with the recorded CPU archive. We assume it, and every comparison of the recorded archive against the replicate spread depends on that assumption.

## What we found

- **The CLEC5A direction holds on EMPF, and the default readout halves its size by design.**
  - Measure: within-subtype High-minus-Not-AD contrast in mean log1p(panel CP10k).
  - Default `cellshape@0.5:biascal:stagehalf` readout: **+0.235** (Monte Carlo SD 0.044, 8/8 replicates positive).
  - Why it is halved: `stagehalf` gives half of each target's generated donors the training stage mixture, which shrinks every stage contrast by about one half.
  - Post-hoc own-stage readout: **+0.492** (SD 0.041, 8/8 positive).
  - Which readout matches better depends on the observed reference, so neither readout "matches the data" unconditionally:
    - Against the same 33 donors as the frozen report: observed **+0.470** (donor-bootstrap 95% CI [0.20, 0.76]). Own-stage is the closer match. This reference includes 8 of EMPF's 12 held-out donors.
    - Against the 25 donors this fit was trained on: observed **+0.317** [0.055, 0.619]. Own-stage overshoots and stagehalf is the closer match.
  - Both observed CIs rest on 3 Not AD donors and are anticonservative.
  - For comparison, 0.9.7 gave +0.454 (five-model mean) against an observed +0.489.
- **Neither model recovers the observed Low-stage peak.** The observed CLEC5A within-subtype profile peaks at Low, both for the 33 donors and for the 25 training donors. The EMPF profile peaks at Intermediate under both readouts, and so did 0.9.7. The model's Low-to-Intermediate gaps are small (about 0.02–0.04) and carry no uncertainty, so this is descriptive.
- **Across genes, within-subtype agreement is weaker than in 0.9.7. The comparison is not fully matched.**
  - Same 33 donors: EMPF reaches a descriptive Spearman of 0.241 with 59.3% direction agreement. The published 0.9.7 five-model mean is 0.502 / 65.9%.
  - Why that comparison is unfair to EMPF: the reference is fully in-sample for 0.9.7, which trained on all 33 donors. For EMPF it contains 8 held-out donors, and all 3 of its Not AD donors are EMPF training donors.
  - Matched in-sample comparison: against EMPF's own 25 training donors it reaches **0.308 / 63.2%**. That is below the 0.9.7 single-model mean (0.395) but inside the 0.9.7 single-model range (0.241, 0.364, 0.402, 0.445, 0.521; per-model values are post hoc).
  - Agreement falls as held-out donors are added: 0.308 for the 25 training donors, 0.241 for the 33, 0.156 for all 37.
  - The comparison also mixes readouts (0.9.7 used full stage conditioning) and panels (1613 shared genes).
- **The recorded EMPF improvement over the previous default is within generation Monte Carlo noise.**
  - Eight GPU replicates of the same model give a profile RMSE of 0.4697 ± 0.0038 (SD) and a held-out SD interval score of 0.822 ± 0.012.
  - None of the recorded EMPF-minus-previous differences in profile RMSE, gene errors, SD/combined interval score or energy exceeds 2 Monte Carlo SDs.
  - The recorded CPU archive scored better than 7 of 8 replicates on profile RMSE (z ≈ −1.19) and on SD interval score (z ≈ −1.27), but better than only 3 of 8 on combined interval score (z ≈ −0.32). Exchangeability with the CPU archive is unverified (see the protocol deviation).
  - Combined coverage is the one difference beyond 2 SD, and it is lower than the previous default in all 8 replicates.
  - Stage-blind empirical resampling beats every replicate on profile RMSE, both interval scores, PC energy and coverage.
- **Gene-level donor intervals: the pilot's calibration does not help, and empirical resampling scores better.**
  - Raw stage-level 80% intervals already over-cover held-out donors, at 86.8% for gene bag means and 87.0% for SDs.
  - The pilot's nonnegative rule can only widen intervals. It raised the SD interval score by **8.3%** (95% CI 6.1 to 10.7%) and worsened it for 12 of 12 donors.
  - The model's raw intervals score **5.2%** worse (CI 3.3 to 7.2%) than stage-blind empirical intervals.

## Use the tool

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
$PY -B scripts/microglia_query.py --genes CLEC5A PPARG APOE --stages "Not AD" High
$PY -B scripts/microglia_query.py --genes CLEC5A --stages "Not AD" Low --out q.json
# A fits ensemble: one bundle per trained fit
$PY -B scripts/microglia_query.py --genes CLEC5A --stages 0 3 --bundle fitA.npz fitB.npz
```

The tool needs NumPy only, with no torch, weights or raw cells. It reads a query bundle (`.npz` plus a `.json` sidecar of about 3 MB). The default bundle is `runs/research_20261120/query-tool/bundle_fold0_seed0.npz`, kept outside git. To rebuild it, generate the replicates with GPU or CPU, then build:

```bash
export PYTHONPATH=artifacts/native-releases/particlegan-0.10.0/site-packages
O=runs/research_20261120/query-tool
for r in 1 2 3 4 5 6 7 8; do CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B \
  analysis/query_tool_20261120/generate_cells.py --replicate $r --out $O/rep$r.npz --device cuda --gpu-mem-fraction .06; done
$PY -B analysis/query_tool_20261120/build_bundle.py --replicates $O/rep{1..8}.npz \
  --eval-out analysis/query_tool_20261120/results/interval_evaluation.json --bundle $O/bundle_fold0_seed0.npz
```

`generate_cells.py` checks the selected weights and metadata against `selected_model.json`. It imports the derived native host from a private, hash-verified snapshot (`native_host.py`) and never calls `host.activate`, which rewrites shared files (the recorded activation race). Each replicate takes about 2-3 minutes on a shared A6000 at 6% memory. To add a fit to an ensemble, build one bundle from that fit's replicates. This first version hardcodes the selected fold0/seed0 fit and cache, so another fit needs `FIT` and `CACHE` parameterised. No confirmation-queue fit was read in this pass.

### Output schema `microglia-query/1`

```text
{schema, query:{genes, stage_a, stage_b, contrast:"stage_b minus stage_a"},
 model:{ensemble_size, fits:[{bundle, fit, readout, readout_note, fold, final_weights_sha256, replicates,
        generation_devices, panel_genes, bundle_sha256}], units, stage_mean_definition,
        within_subtype_definition, interval_definition,
        calibration:{rule, alpha, deployed_calibration_donors, deployed_rank, crossfit_halves,
                     heldout_crossfit_evaluation:{raw, calibrated, calibration_useful, note}},
        flag_definitions, status},
 scope:[...],
 genes:[{gene, in_panel, in_panel_by_fit,
   flags:{not_in_panel, not_in_panel_of_some_fits, poorly_fit, pc_blind_spot, low_expression, single_fit},
   generated:{stage_means:{<stage>:SUMMARY}, contrast:SUMMARY,
              within_subtype:{stage_means:{<stage>:SUMMARY}, contrast:SUMMARY|null}},
   per_fit:[{fit, flags, heldout_mean_log_rmse, stage_blind_empirical_rmse, poorly_fit_rmse_cut,
             pc_outside_share_by_stage, pc_standardized_gap_by_stage, pc_captured_fraction,
             observed_detection, crossfit_heldout_coverage:{mean|sd:{raw, calibrated}},
             donor_intervals_80:{<stage>:{mean|sd:{raw:[lo,hi], calibrated:[lo,hi]}}},
             observed_reference:{donors, pooled:OBS|null, within_subtype:OBS|null, note},
             stage_blind_empirical_reference:{stage_mean, contrast:0.0, donor_interval_80:{mean, sd}}}]}]}
SUMMARY = {estimate, per_fit, mc_replicate_sd, mc_standard_error, between_fit_sd (null for one fit),
           standard_error_total, mc_interval95, mc_interval95_t_df, mc_interval95_note,
           replicate_range, fraction_replicates_positive, replicates, uncertainty_scope}
OBS = {stage_a_mean, stage_b_mean, contrast, ci95_donor_bootstrap, bootstrap_valid_draws}
```

A gene outside the panel returns only `{gene, in_panel:false, flags:{not_in_panel:true}, note}`. Units throughout are the mean per-cell log1p(panel CP10k). Donor SD intervals use the same units over a K=256-cell bag. Generated stage means weight every generated target donor of the stage equally. That covers 37 development-donor conditions × 32 generated donors × 8 replicates. `mc_interval95` is a Student-t interval for the generator's mean (df = replicates − 1 for one fit, fits − 1 for several). It covers generation Monte Carlo, plus between-fit spread when there are several bundles. It is **not** an inferential CI for the biological effect: with one fit it is far narrower than the scientific uncertainty (CLEC5A within-subtype [0.198, 0.272] against an observed donor CI of [0.20, 0.76]), and `uncertainty_scope` says that fit-to-fit variability is unmeasured. When a bundle uses the `stagehalf` readout (the default bundle does), the tool writes a `readout_note`, adds it to `scope` and prints it to stderr: generated stage contrasts are about halved by construction. Before Amendment 2 this field was `approx_ci95` with z = 1.96. The observed reference covers the fold's 37 development donors. It is data, not model output, and its CI comes from a stage-stratified percentile donor bootstrap. With only 3 Not AD donors in most references, that CI has discrete support and is anticonservative; the bootstrap does not re-enforce the ≥3-distinct-donor rule within each resample. Calibrated lower bounds are clipped at 0, since both coordinates are nonnegative. Clipping does not change coverage. The held-out evaluation applied it to SD only.

Examples: [Not AD vs High](results/example_query_NotAD_High.json) for CLEC5A, PPARG, GPNMB and the absent APOE and TREM2, and [Not AD vs Low](results/example_query_NotAD_Low.json).

## Results

### A. Generation Monte Carlo of the frozen metrics (default readout, 8 replicates, held-out donors)

| Metric (held-out) | Replicate mean ± SD | Range | Recorded EMPF archive (z vs replicates) | Previous default | Stage-blind empirical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Profile RMSE ↓ | .4697 ± .0038 | .4642–.4750 | .4651 (−1.19) | .4725 | **.4577** |
| Mean-log gene error ↓ | .0692 ± .0018 | .0659–.0712 | .0697 (+0.33) | .0681 | .0776 |
| Log-variance gene error ↓ | .1880 ± .0049 | .1788–.1934 | .1900 | .1843 | .1908 |
| SD coverage (80%) | 57.7 ± 0.7% | 57.0–58.8% | 58.3% | 59.5% | 71.1% |
| SD interval score ↓ | .8216 ± .0121 | .8042–.8427 | .8063 (−1.27) | .8258 | **.6186** |
| Combined coverage | 69.1 ± 0.7% | 68.0–70.1% | 69.2% | 70.8% | 78.7% |
| Combined interval score ↓ | .9488 ± .0110 | .9383–.9669 | .9452 (−0.32) | .9516 | **.8448** |
| PC energy ↓ | 1.871 ± .016 | 1.852–1.895 | 1.861 | 1.875 | **1.763** |
| Cell energy ↓ | .6154 ± .0119 | .6024–.6324 | .6040 | .6150 | .6297 |
| High-vs-Not-AD Spearman vs TRAIN ↑ | .415 ± .126 | .230–.570 | .515 | .523 | −.004 |

The previous-default and empirical columns are single recorded archives with no Monte Carlo spread of their own. The 2-SD rule ignores the previous default's own generation variance, so it is conservative about calling differences real. **This part has no donor-bootstrap CIs.** The frozen `architecture_score` returns only pooled held-out metrics, and reimplementing it per donor was out of scope (Amendment 2). The model-versus-empirical and model-versus-previous gaps therefore carry no donor-level uncertainty here. The stage-contrast agreement metric varies strongly between replicates (SD .126), so a single generation should not be used to rank models on it. [Full scores](results/monte_carlo_scores.json).

The post-hoc own-stage readout scores worse on held-out metrics:

| Metric | Default readout | Own-stage readout |
| --- | ---: | ---: |
| Profile RMSE ↓ | .4697 ± .0038 | .5019 ± .0053 |
| SD interval score ↓ | .822 | .848 |
| Combined interval score ↓ | .949 ± .011 | 1.040 ± .021 |

The half shrink that lowers held-out profile error is the same step that halves the modelled stage contrasts. [Scores](results/monte_carlo_scores_unshrunk_posthoc.json).

### B. Gene-level donor intervals

Twelve held-out donors × 24 bags × 2,000 genes, cross-fitted 6/6 with the coverage pilot's halves. In every split, the calibration donors are neither evaluation donors nor training donors.

| Intervals | Mean coverage | Mean score ↓ | SD coverage | SD score ↓ | Combined coverage | Combined score ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model, raw | 86.8% | .5251 | 87.0% | .9125 | 86.9% | .7188 |
| Model, SD calibrated (pilot rule) | 86.8% | .5251 | 91.3% | .9884 | 89.0% | .7568 |
| Model, mean+SD calibrated | 91.2% | .5954 | 91.3% | .9884 | 91.3% | .7919 |
| Stage-blind empirical, raw | 84.8% | .5023 | 84.5% | .8639 | 84.6% | **.6831** |

Donor-bootstrap 95% CIs over the 12 donors:

| Comparison | Change in interval score | 95% CI | Donors affected |
| --- | ---: | --- | --- |
| SD calibration (pilot rule) vs raw, SD score | +8.3% | 6.1 to 10.7% | worse for 12 of 12 |
| Mean+SD calibration vs raw, combined score | +10.2% | 8.3 to 12.0% | not computed |
| Model raw vs empirical raw, combined score | +5.2% (model worse) | 3.3 to 7.2% | not computed |

These CIs do not propagate the uncertainty of the rank-6-of-6 calibration fit.

Neither calibration variant is useful by the declared rule, which requires coverage to move closer to 80% and the score to fall. The two analyses calibrate different things. The pilot calibrated PC-summary SDs, which under-covered. Stage-level gene intervals pool across target covariates and replicates, and they already over-cover. These gene-level intervals reach about 87% coverage, while the frozen metric's held-out SD and combined coverage is 58–69%. The two coverages are not interchangeable.

The bundle still ships a correction refit on all 12 held-out donors (rank 11). Its median is zero, and its own coverage cannot be evaluated on those donors. [Evaluation](results/interval_evaluation.json).

### C. Headline biology, side by side

| | 0.9.7 (frozen report) | EMPF default readout | EMPF, own-stage readout (post hoc) |
| --- | ---: | ---: | ---: |
| Models × Monte Carlo replicates | 5 × 8 | 1 × 8 | 1 × 8 |
| Panel / training donors | 0.9.7 panel / 33 | fold0 panel / 25 | fold0 panel / 25 |
| CLEC5A within-subtype contrast, model (MC SD) | +0.454 (fits .17–.61) | +0.235 (.044) | +0.492 (.041) |
| CLEC5A observed, same 33 donors [donor CI] | +0.489 | +0.470 [0.20, 0.76] | +0.470 [0.20, 0.76] |
| CLEC5A observed, EMPF's 25 training donors [donor CI] | – | +0.317 [0.055, 0.619] | +0.317 [0.055, 0.619] |
| Replicates with the observed sign | 40/40 | 8/8 | 8/8 |
| CLEC5A pooled contrast, model / observed (33) | +0.388 / +0.478 | +0.187 / +0.460 | +0.392 / +0.460 |
| CLEC5A model profile peak (observed: Low) | Intermediate | Intermediate | Intermediate |
| Within-subtype Spearman / direction, same 33 donors | .502 / 65.9% (in-sample) | .241 / 59.3% (8 held-out donors included) | .224 / 58.8% |
| Same, against the model's own training donors (matched in-sample) | .502 / 65.9% (its 33) | .308 / 63.2% | .283 / 62.4% |
| 0.9.7 single-model Spearman (post hoc) | .241 / .364 / .402 / .445 / .521 (mean .395) | – | – |
| Model-to-observed effect L2 ratio (33) | .69 | .41 | .89 |

The 0.9.7 column was recomputed from the frozen stage-replicate summaries. It reproduces the published values exactly: CLEC5A +0.454 and +0.489, Spearman .502, 65.9% direction agreement, and the report's CLEC5A stage profiles. The observed contrast on the EMPF panel differs slightly from +0.489 because CP10k uses each panel's total.

CLEC5A stage profiles (Not AD, Low, Intermediate, High):

| Source | Not AD | Low | Intermediate | High |
| --- | ---: | ---: | ---: | ---: |
| 0.9.7 model | .374 | .772 | .871 | .828 |
| EMPF default | .494 | .792 | .811 | .729 |
| EMPF own-stage | .247 | .840 | .884 | .739 |
| Observed, 33 donors | .237 | 1.038 | .685 | .707 |
| Observed, 25 training donors | .295 | .996 | .719 | .613 |

**Observed references and their CIs.** The observed CIs come from a stage-stratified percentile donor bootstrap with 2,000 resamples. The 33-donor and 25-donor references each have only 3 Not AD donors, all of them EMPF training donors. So the CIs have discrete support and are anticonservative. The ≥3-distinct-donor rule is not re-enforced within each resample.

Stage-blind empirical resampling predicts zero stage contrast for every gene by construction, so on contrasts it is a reference rather than a competitor. The 12 held-out donors include only one Not AD donor, which is too few for a held-out contrast under the ≥3-donor rule, so that reference is reported as not estimable. With all 37 development donors, the observed contrast is +0.430 [0.14, 0.73], and the model contrast for the three eligible subtypes is +0.234. [Default readout](results/biology_097_vs_empf.json) · [Own-stage readout](results/biology_097_vs_empf_unshrunk_posthoc.json).

**Generated subtypes are classifier predictions.** The fold's frozen classifier was checked on real cells and on generated cells:

| Classifier check | Value |
| --- | ---: |
| Donor-held-out accuracy, real cells | 73.3% |
| Balanced accuracy, real cells | 60.7% |
| Held-out recall, Micro-PVM_2 (contrast group) | 0.84 |
| Held-out recall, Micro-PVM_2_3-SEAAD (contrast group) | 0.59 |
| Mean confidence, real cells | 0.79 |
| Mean confidence, generated EMPF cells | 0.61 |

Within-subtype synthetic contrasts therefore carry misclassification bias that this pass did not quantify. [Classifier check](results/classifier_check.json).

### D. Gene flags in the bundle

| Flag (of 2,000 panel genes) | Count | How the count arises |
| --- | ---: | --- |
| `poorly_fit` | 201 | 200 from the fixed 90th-percentile cut on held-out donor mean-log RMSE (.318), which flags 10% of genes by construction. 1 from the 1.5× stage-blind empirical RMSE rule. The count says nothing about absolute fit quality. |
| `pc_blind_spot` | 398 | Amendment 2 rule (below). |
| `low_expression` | 63 | Observed mean detection below 1%. |

The model's per-gene held-out RMSE beats the stage-blind empirical mean for only 817 of 2,000 genes (median .0771 vs .0762). CLEC5A and PPARG are both flagged `poorly_fit`. Both vary strongly from donor to donor, and their model RMSE is about equal to the empirical RMSE.

**PC blind-spot amendment.** The original implementation reported the signed ratio `outside_j / z_j`. That ratio is unbounded and can be negative. Amendment 2 replaces it with the bounded share `outside_j² / (outside_j² + inside_j²)`, where `inside = W Wᵀ z`, and keeps the declared thresholds: share > 0.5 in any stage where |z_j| > 0.05.

The recount is again **398**, and the set is the same gene for gene. The reason is algebraic: at the 0.5 threshold, both rules reduce to |outside_j| > |inside_j|. The audit's concern was therefore correct about the *reported share values*, which were not shares, but it did not change any flag. The bundle now reports bounded shares in [0, 1]:

| Gene | Bounded share by stage (Not AD, Low, Intermediate, High) | Standardized gap | Static PC-captured fraction | Flagged? |
| --- | --- | --- | ---: | --- |
| CLEC5A | .00, .05, .11, .84 | .12 / .15 / .008 / .014 | 8.3% | No: its High share is large, but the High gap (.014) is below the 0.05 cut |
| PPARG | .00, .02, .20, .05 | .10 / .08 / .04 / .03 | 4.3% | No |

The earlier README offered a causal explanation for PPARG: that the joint bias calibration hides its training gaps. That explanation was never tested, because no pre-calibration gaps were computed, so it is withdrawn. The bundle rebuild left every other array and the whole interval evaluation unchanged, which was checked.

## Limits

- **One fit only** (fold 0/seed 0, 25 GAN-training donors). Fit-to-fit variability is unmeasured. For 0.9.7, single-model CLEC5A contrasts ranged from .17 to .61. The 17-fit confirmation queue was deliberately not read. Every model-side uncertainty here is generation Monte Carlo only, so 8/8 sign agreement says little about robustness.
- **No held-out stage contrast.** The across-gene comparisons use correlated genes and observational stage labels. They are descriptive training-association checks, not held-out prediction accuracy.
- **References are unmatched.** The 33-donor reference is in-sample for 0.9.7 but partly held-out for EMPF. The panels differ (1613 shared genes, panel-specific CP10k). The readouts differ (0.9.7 used full stage conditioning).
- **The own-stage readout is post hoc.** It explains the attenuation; it is not a readout selection. It worsens held-out distribution metrics.
- **The protocol-required CPU bitwise check did not finish.** Replicates 1–8 are GPU draws, and their exchangeability with the CPU archive is assumed. Each replicate also refits the training-only joint calibration on its own streams, so the Monte Carlo spread includes that calibration's variation.
- **Part A has no donor-bootstrap CIs**, and its comparators have no Monte Carlo spread of their own.
- **The bootstrap CIs are approximate.** Observed-reference CIs rest on 3 Not AD donors and are anticonservative. The Part B CIs do not include calibration-fit uncertainty.
- **`generate_cells.py` hardcodes the fold0/seed0 fit and cache.** Another fit needs them parameterised.
- **No result here changes** the default, the frozen report or the population-qualification status.

## Next steps

1. After the confirmation queue finishes, parameterise the fit and cache in `generate_cells.py`. Then build one bundle per completed fit and pass them together to `--bundle`. That gives between-fit uncertainty and lets the biology be recomputed across folds and seeds. Folds whose held-out set has ≥3 Not AD donors would also allow a held-out contrast.
2. Decide explicitly which readout serves biology queries: stagehalf for held-out profile accuracy, own-stage for contrast magnitude. The tool can take a bundle for each.
3. Replace the widening-only calibration for gene-level intervals with one that can also narrow, then re-evaluate it against stage-blind empirical resampling.
4. Test whether pre-calibration gaps change the PC blind-spot flags.
