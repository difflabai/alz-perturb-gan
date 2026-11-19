# PsychAD cross-disorder microglia benchmark

Can the model recover the Alzheimer's-associated microglial shift and
tell it apart from diffuse Lewy body disease in previously unseen donors?
This is an opt-in test for interpreting model-learned features across
related diseases.

## Protocol

Use the public PsychAD MSSM microglia slice from
[CELLxGENE](https://cellxgene.cziscience.com/collections/84ce6837-548d-4a1f-919f-0bc0d9a3952f).
The benchmark input has raw counts and `donor_id` and `diagnosis`
columns. Use the Census release `2025-11-17` for the recorded run.
Keep microglial cells (CL:0000129) from dataset
`37a17b78-4864-4a42-b67b-31c00962795a` and the following source labels.

| Census diagnosis | Analysis label |
|---|---|
| normal | control |
| dementia \|\| Alzheimer disease | Alzheimer |
| Lewy body dementia | Lewy |
| dementia \|\| Lewy body dementia | Lewy |

The extract is capped at 128 nuclei per donor using seed 0.
Sum expression per donor and apply log(1 + counts per 10,000).
For seeds 0–2, take a stratified 75/25 split of donors. Select the
512 highest-variance training genes and fit a 12-component PCA to
standardized training profiles. Apply the same transform to held-out
donors.

Fit EMPF for 300 CPU steps per arm. Also fit EMPF with permuted
training-donor diagnoses and compare with empirical training-donor
resampling and an additive mean shift from the training controls.
Draw 256 profiles per arm and diagnosis. For each disease, compare
the direction of its model-vs-control mean shift with both held-out
disease-vs-control shifts. Specificity sums the two differences
between matched and mismatched cosine similarity. Score each group's
full donor distribution with energy distance and average across the
three groups. The screen requires specificity above 0.2 and both
metrics better than every baseline on each seed.

To build the input from its pinned public release, install
`cellxgene-census==1.18.0` in an isolated environment and run the
[fetch script](../scripts/fetch_psychad.py). It writes the H5AD and a
receipt with its source version, selection and SHA-256.

```sh
python paper_watch/scripts/fetch_psychad.py paper_watch/data
```

From the repository root, in the paper-watch environment:

```sh
PSYCHAD_H5AD=/path/to/psychad-mssm-microglia.h5ad \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m paper_watch.alztest.psychad
```

## Scope

This compares donor-level pseudobulk profiles in a shared representation
within one public cohort. Evaluation uses known diagnosis labels and
tests the generator's conditional distributions. The three seeds reuse
parts of the cohort, so they measure split and fit sensitivity.

## Public-cohort result

The extract contains 64,816 nuclei from 619 donors. Each seed evaluates
65 control, 82 Alzheimer, and 9 Lewy donors. The held-out Lewy group is
the smallest, so cohort and donor sensitivity matter.

| Arm | Mean specificity ↑ | Mean energy distance ↓ |
|---|---:|---:|
| EMPF (300 updates per fit) | 0.061 | 2.880 |
| EMPF with shuffled donor labels | 0.108 | 3.056 |
| Mean shift | **0.354** | 1.763 |
| Empirical training-donor sampling | 0.349 | **1.728** |

EMPF missed the predefined screen on all three splits. Both simple
baselines outperformed EMPF on both metrics in every split. The
fixed-budget test therefore gives us an application benchmark for
future disease-specific models, with separate measures for contrast
direction and distribution fit. The score uses known diagnostic labels
and applies to this donor cohort and training budget.

[Per-seed metrics and data receipt](psychad_cross_disorder_results.json)
