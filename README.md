# ALZ: a generative model of human microglia

[**Read the evidence report →**](https://difflabai.github.io/alz-perturb-gan/)

We trained five ParticleGAN 0.9.7 models of human microglial gene-expression distributions at four Alzheimer’s neuropathology stages. This repository contains the research code, experiment history, aggregate results, and a shareable report examining what the trained models actually support.

**Dataset: SEA-AD middle temporal gyrus.** This is an independent proof of concept. Prima Mente’s challenge atlas is planned for Spring 2027 and was not used for training or analysis here. These are not challenge benchmark results.

The current model conditions on **pathology stage**, not gene edits or compounds. It is an observational proof of concept for generating cell populations and prioritizing experiments. It does not predict a treatment response, and we have not demonstrated state-of-the-art performance.

## What the model says

Training used 14,762 SEA-AD middle temporal gyrus cells from 33 donors, a 2,000-gene panel, and five independent 20,000-update fits. All five fits started from scratch using the released ParticleGAN 0.9.7 formulation, with generator EMA disabled. We sampled 12.8 million unique synthetic cells from the frozen models; this adds no biological donors.

The resulting analysis supports **CLEC5A as a candidate for a human microglial experiment**:

- Within two donor-supported subtypes, its observed High-minus-Not-AD contrast is **+0.489** and the five-model mean is **+0.454** on the mean per-cell log1p(panel CP10k) scale.
- All five models and all 40 Monte Carlo contrasts agree on the positive direction. The observed association remains positive after donor-covariate adjustment and stage-specific donor-deletion checks.
- The primary external ROSMAP pathology association agrees in direction, with published global FDR **0.00628**.
- The model misses the observed Low-stage peak: its average profile peaks at Intermediate. A consistent endpoint contrast is not an accurate full stage profile.

[Published mouse knockout experiments](https://pubmed.ncbi.nlm.nih.gov/39443966/) motivate testing **reduction** of CLEC5A. That direction does not come from our stage-conditioned generator. The proposed human-cell test measures amyloid uptake and degradation, viability, inflammation, and synapse loss together. CLEC5A is an existing biological lead, not a newly discovered target.

## How well it fits

<details>
<summary>Technical calibration sanity check</summary>

Four calibration donors were used during hyperparameter selection. These are selection results, not an independent final test. Means across five fits and two generation replicates per fit:

| Metric | Generator | Matched resampling | Interpretation |
| --- | ---: | ---: | --- |
| Donor energy score ↓ | 2.0069 | 2.0946 | Generator 4.2% lower; better in 4/5 fits |
| Cell energy distance ↓ | 0.2842 | 0.2349 | Generator 21.0% higher; worse in 5/5 fits |
| Central 80% donor-summary coverage | 64.7% | 68.4% | Both below nominal 80% |

Resampling reuses observed cells; it is an empirical reference, not a trained neural network or a model-architecture benchmark.

</details>

Across the 2,000 correlated genes, the within-subtype training-association comparison has descriptive Spearman correlation **0.502** and direction agreement **65.9%**. It is not held-out gene-prediction accuracy. The report includes failed/sensitive signals, sex/APOE imbalance, incomplete marker coverage, and a comparison with the simpler real-data-plus-literature workflow that still needs to be done.

## Inspect the evidence

- [Interactive report and all-gene explorer](https://difflabai.github.io/alz-perturb-gan/): four-stage curves, five-seed effects, external cohort checks, donor sensitivity, and a concrete experiment proposal.
- [All 2,000 gene results, CSV](docs/evidence/genes.csv), [portable aggregate data](docs/evidence/report-data.json), and [SHA256 provenance manifest](docs/evidence/manifest.json).
- [Frozen inference plan](analysis/primamente_inference_hypothesis_plan_20261108.json), [model quality review](analysis/primamente_model_capability_review_20261108.json), [donor sensitivity](analysis/primamente_donor_covariate_sensitivity_20261108.json), and [selected CLEC5A numerical audit](analysis/primamente_selected_CLEC5A_numerical_audit_20261108.json).
- [Registration suggestion and its evidential basis](docs/prima-mente-suggestion.md).

The screen retains 614 stable exploratory signals, 107 with primary ROSMAP support, and 58 that additionally pass the donor sensitivity filter. These counts are not significant discoveries or an estimate of predictive precision. Only three training donors are Not AD; none carry APOE4. Important genes including APOE, TREM2, and INPP5D are absent from the measured panel.

## Repository map

| Path | Contents |
| --- | --- |
| `docs/` | Static report, downloadable aggregate evidence, figures, and earlier project archive |
| `scripts/report/` | Portable report builder, original stage-profile exporter, and explorer verification |
| `analysis/` | Versioned research scripts, plans, audits, results, and experiment/runtime snapshots |
| `analysis/faithful097/model/` | Released ParticleGAN 0.9.7 integration used in the research lineage |
| `src/alz_perturb_gan/`, `tests/`, `results/smoke_results.json` | Earlier synthetic perturbation/decision-model experiment, preserved as legacy work |

The original perturbation smoke experiment used PG 0.8.0, synthetic data, and a different objective. Its [archived README](docs/archive/early-perturbation-smoke.md) explains that experiment. **The root `pyproject.toml` still installs that legacy package; it is not the environment recipe for the current human-microglia model.** The [earlier literature-only suggestions](docs/archive/early-literature-suggestions.md) are also archived rather than presented as outputs of the current model.

All versioned research history is preserved. Raw cell data, local environments, model checkpoints, and machine credentials are not included. Original research runners require their recorded data/environment setup; not every historical script is a portable command-line tool. The report itself is portable.

## Rebuild the report

Use Python 3.10 or newer:

```bash
python3 -m pip install -r scripts/report/requirements.txt
python3 scripts/report/build_report.py
python3 scripts/report/build_report.py --check
python3 -m http.server 8765 --directory docs
```

Open `http://localhost:8765`. No GPU, raw cells, checkpoints, or external requests are needed to build or browse the report. See [report build instructions](scripts/report/README.md) for the explorer checks and provenance details. GitHub Pages serves the static `docs/` directory on `main`.

## Sources and scope

Human training data: [SEA-AD](https://brain-map.org/consortia/sea-ad). Primary external association: [Mathys et al., Cell 2023](https://doi.org/10.1016/j.cell.2023.08.039), using the [pinned ROSMAP author repository](https://github.com/mathyslab7/ROSMAP_snRNAseq_PFC/tree/28388f829be6affa1578e56fa09c4be8cd273817). Full biological references and interpretation limits are linked in the report.

Prepared while exploring the [Prima Mente Alzheimer’s Translation Challenge](https://www.primamente.com/challenges/). No affiliation or clinical claims. Repository code and report are MIT licensed; third-party data retain their own terms.
