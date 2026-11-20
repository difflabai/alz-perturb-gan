# Query tool and EMPF recomputation of the headline biology

This pass makes the current experimental default (native EMPF on ParticleGAN 0.10.0, fold 0/seed 0) usable as a query tool. It also recomputes the README's headline biology on that model. The frozen 0.9.7 report, `docs/`, the default config and every pinned source are unchanged. The [protocol](PROTOCOL.md) was committed before any score existed. Its single amendment, a post-hoc sensitivity, was committed before that sensitivity was generated.

## What we found

- **The CLEC5A direction holds on EMPF, but the default readout halves its size.** With the default `cellshape@0.5:biascal:stagehalf` readout, the within-subtype High-minus-Not-AD contrast for CLEC5A is **+0.235**. The Monte Carlo SD is 0.044, and all 8 of 8 replicates are positive. On the EMPF panel, the observed contrast for the same 33 donors is **+0.470**, with a donor-bootstrap 95% CI of [0.20, 0.76]. The `stagehalf` plan gives half of each target's generated donors the training stage mixture, so it shrinks every stage contrast by about one half by design. In the post-hoc readout where every donor keeps its own stage, the contrast is **+0.492** (SD 0.041, 8/8 positive). For comparison, 0.9.7 gave +0.454 (five-model mean) against an observed +0.489.
- **Neither model recovers the observed Low-stage peak.** The observed CLEC5A within-subtype profile peaks at Low in every reference. The EMPF profile peaks at Intermediate under both readouts, as the 0.9.7 profile did.
- **Across genes, agreement is lower than in the published 0.9.7 figure.** Against the same 33 donors, EMPF reaches a descriptive within-subtype Spearman of **0.241** with **59.3%** direction agreement. The post-hoc readout gives 0.224 and 58.8%. The published 0.9.7 five-model mean gives 0.502 and 65.9%. Individual 0.9.7 models range from 0.24 to 0.52 (mean 0.39), so the single EMPF fit sits at the bottom of that single-model range. The 0.9.7 models trained on all 33 reference donors. EMPF trained on 25 of them; against those 25 its agreement is 0.308 and 63.2%.
- **The recorded EMPF improvement over the previous default is within generation Monte Carlo noise.** Eight replicates of the same EMPF model give a profile RMSE of 0.4697 ± 0.0038 (SD). The recorded EMPF archive scored 0.4651 and the previous default 0.4725. The held-out SD interval score is 0.822 ± 0.012, against 0.806 recorded for EMPF and 0.826 for the previous default. None of the recorded EMPF-minus-previous differences in profile RMSE, gene errors, SD/combined interval score or energy exceeds 2 Monte Carlo SDs. The one difference that does exceed 2 SDs is combined coverage, which is lower than the previous default's in all 8 replicates. The scored EMPF archive was a favourable draw. Across all 8 replicates, stage-blind empirical resampling stays better on profile RMSE, both interval scores, PC energy and coverage.
- **Gene-level donor intervals: the pilot's calibration does not help, and empirical resampling scores better.** Raw stage-level 80% intervals for gene bag means and SDs already over-cover the held-out donors (86.8% / 87.0%). The pilot's nonnegative rule can only widen intervals. It raised the SD interval score **+8.3%** (95% CI 6.1 to 10.7%) and worsened it for all 12 donors. Calibrating the means as well raised the combined score +10.2%. The model's raw gene intervals score **5.2%** (CI 3.3 to 7.2%) worse than stage-blind empirical intervals. The tool therefore reports raw and calibrated intervals side by side, together with this evaluation.

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
 model:{ensemble_size, fits:[{bundle, fit, readout, fold, final_weights_sha256, replicates,
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
           standard_error_total, approx_ci95, replicate_range, fraction_replicates_positive,
           replicates, uncertainty_scope}
OBS = {stage_a_mean, stage_b_mean, contrast, ci95_donor_bootstrap, bootstrap_valid_draws}
```

A gene outside the panel returns only `{gene, in_panel:false, flags:{not_in_panel:true}, note}`. Units throughout are the mean per-cell log1p(panel CP10k). Donor SD intervals use the same units over a K=256-cell bag. Generated stage means weight every generated target donor of the stage equally. That covers 37 development-donor conditions × 32 generated donors × 8 replicates. `approx_ci95` covers generation Monte Carlo, plus between-fit spread when there are several bundles. With one fit, `uncertainty_scope` says that fit-to-fit variability is unmeasured. The observed reference covers the fold's 37 development donors. It is data, not model output, and its CI comes from a stage-stratified donor bootstrap. Calibrated lower bounds are clipped at 0, since both coordinates are nonnegative. Clipping does not change coverage. The held-out evaluation applied it to SD only.

Examples: [Not AD vs High](results/example_query_NotAD_High.json) for CLEC5A, PPARG, GPNMB and the absent APOE and TREM2, and [Not AD vs Low](results/example_query_NotAD_Low.json).

## Results

### A. Generation Monte Carlo of the frozen metrics (default readout, 8 replicates)

| Metric (held-out) | Replicate mean ± SD | Range | Recorded EMPF archive | Previous default | Stage-blind empirical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Profile RMSE ↓ | .4697 ± .0038 | .4642–.4750 | .4651 | .4725 | .4577 |
| Mean-log gene error ↓ | .0692 ± .0018 | .0659–.0712 | .0697 | .0681 | .0776 |
| Log-variance gene error ↓ | .1880 ± .0049 | .1788–.1934 | .1900 | .1843 | .1908 |
| SD coverage (80%) | 57.7 ± 0.7% | 57.0–58.8% | 58.3% | 59.5% | 71.1% |
| SD interval score ↓ | .8216 ± .0121 | .8042–.8427 | .8063 | .8258 | .6186 |
| Combined coverage | 69.1 ± 0.7% | 68.0–70.1% | 69.2% | 70.8% | 78.7% |
| Combined interval score ↓ | .9488 ± .0110 | .9383–.9669 | .9452 | .9516 | .8448 |
| PC energy ↓ | 1.871 ± .016 | 1.852–1.895 | 1.861 | 1.875 | 1.763 |
| Cell energy ↓ | .6154 ± .0119 | .6024–.6324 | .6040 | .6150 | .6297 |
| High-vs-Not-AD Spearman vs TRAIN ↑ | .415 ± .126 | .230–.570 | .515 | .523 | −.004 |

The previous-default and empirical values are single recorded archives without their own Monte Carlo spread, so this is descriptive. Replicate 0, the recorded archive, was generated on CPU. Replicates 1-8 were generated on a GPU with the same model and readout. The recorded value falls inside the replicate range for every metric above. The stage-contrast agreement metric varies strongly between replicates (SD .126), so a single generation should not be used to rank models on it. [Full scores](results/monte_carlo_scores.json).

The post-hoc readout without stage-half mixing scores worse on held-out distribution metrics. Profile RMSE is .5019 ± .0053, the combined interval score 1.040 ± .021, and SD coverage 55.3%. The half shrink that lowers held-out profile error is the same step that halves the modelled stage contrasts. [Scores](results/monte_carlo_scores_unshrunk_posthoc.json).

### B. Gene-level donor intervals (12 held-out donors × 24 bags × 2,000 genes, cross-fitted 6/6 with the pilot's halves)

| Intervals | Mean coverage | Mean score ↓ | SD coverage | SD score ↓ | Combined coverage | Combined score ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model, raw | 86.8% | .5251 | 87.0% | .9125 | 86.9% | .7188 |
| Model, SD calibrated (pilot rule) | 86.8% | .5251 | 91.3% | .9884 | 89.0% | .7568 |
| Model, mean+SD calibrated | 91.2% | .5954 | 91.3% | .9884 | 91.3% | .7919 |
| Stage-blind empirical, raw | 84.8% | .5023 | 84.5% | .8639 | 84.6% | .6831 |

Neither calibration variant is useful by the declared rule, which requires coverage to move closer to 80% and the score to fall. The pilot calibrated PC-summary SDs, which under-covered. Stage-level gene intervals pool across target covariates and replicates, and they already over-cover. The bundle still ships a correction refit on all 12 held-out donors (rank 11). Its median is zero, and its own coverage cannot be evaluated on those donors. [Evaluation](results/interval_evaluation.json).

### C. Headline biology, side by side

| | 0.9.7 (frozen report) | EMPF default readout | EMPF, own-stage readout (post hoc) |
| --- | ---: | ---: | ---: |
| Models × Monte Carlo replicates | 5 × 8 | 1 × 8 | 1 × 8 |
| Panel / training donors | 0.9.7 panel / 33 | fold0 panel / 25 | fold0 panel / 25 |
| CLEC5A within-subtype contrast, model | +0.454 | +0.235 (SD .044) | +0.492 (SD .041) |
| CLEC5A observed, same 33 donors | +0.489 | +0.470 [0.20, 0.76] | +0.470 [0.20, 0.76] |
| Replicates with the observed sign | 40/40 | 8/8 | 8/8 |
| CLEC5A pooled contrast, model / observed | +0.388 / +0.478 | +0.187 / +0.460 | +0.392 / +0.460 |
| CLEC5A model profile peak (observed: Low) | Intermediate | Intermediate | Intermediate |
| Within-subtype Spearman / direction, all genes | .502 / 65.9% | .241 / 59.3% | .224 / 58.8% |
| Same, against the model's own training donors | (same 33) | .308 / 63.2% | .283 / 62.4% |
| Model-to-observed effect L2 ratio | .69 | .41 | .89 |

The 0.9.7 column was recomputed from the frozen stage-replicate summaries. It reproduces the published values exactly: CLEC5A +0.454 and +0.489, Spearman .502, 65.9% direction agreement, and the report's CLEC5A stage profiles. The observed contrast on the EMPF panel differs slightly from +0.489 because CP10k uses each panel's total. The model profiles are 0.9.7 [.374, .772, .871, .828], EMPF default [.494, .792, .811, .729] and EMPF own-stage [.247, .840, .884, .739]. The observed profile for the same 33 donors on the EMPF panel is [.237, 1.038, .685, .707].

The 12 held-out donors contain one Not AD donor, so they cannot support a held-out contrast under the ≥3-donor rule; that reference is reported as not estimable. With all 37 development donors the observed contrast is +0.430 [0.14, 0.73]. Three subtypes are eligible there, and the model contrast for those subtypes is +0.234. [Default readout](results/biology_097_vs_empf.json) · [Own-stage readout](results/biology_097_vs_empf_unshrunk_posthoc.json).

Generated subtypes are classifier predictions. The fold's frozen classifier has 73.3% donor-held-out accuracy (balanced 60.7%) and a mean confidence of 0.79 on real cells. On generated EMPF cells its mean confidence is only 0.61, so generated cells look less subtype-like than real ones. [Classifier check](results/classifier_check.json).

### D. Gene flags in the bundle

Of 2,000 panel genes, 201 are flagged `poorly_fit`. The rule is a held-out donor mean-log RMSE above the panel's 90th percentile (.318), or above 1.5× the stage-blind empirical RMSE. Another 398 are flagged `pc_blind_spot` and 63 `low_expression`. The model's per-gene held-out RMSE beats the stage-blind empirical mean for only 817 of 2,000 genes (median .0771 vs .0762). CLEC5A and PPARG are both flagged `poorly_fit`. Both have large donor-to-donor variation, and their model RMSE is about equal to the empirical RMSE.

PPARG is **not** flagged as a PC blind spot under the declared training-gap rule. The joint `biascal` calibration fits per-gene output bias on the training donors, which shrinks the training stage gaps the rule measures. Its static PC-captured fraction is only 4.3%, which the tool reports separately.

## Limits

- One fit (fold 0/seed 0, 25 GAN-training donors). Fit-to-fit variability is unmeasured, and the 17-fit confirmation queue was deliberately not read. Every contrast uncertainty here is generation Monte Carlo only, not biological or training uncertainty.
- The across-gene comparisons use correlated genes and observational stage labels. They are descriptive training-association checks, not held-out prediction accuracy. The held-out donors cannot support a stage contrast.
- The own-stage readout is post hoc. It is reported to explain the attenuation, not to select a readout. It worsens held-out distribution metrics.
- The adapter's bitwise CPU reproduction of the scored archive (replicate 0, first two targets) is recorded in `results/adapter_validation.json` when it completes. Replicates 1-8 ran on GPU, so they are exchangeable draws, not a bitwise reproduction.
- No result here changes the default, the frozen report or the population-qualification status.
