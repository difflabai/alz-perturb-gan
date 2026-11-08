# A model-grounded suggestion for Prima Mente

This replaces the earlier literature-only draft. The [full evidence report](https://difflabai.github.io/alz-perturb-gan/) examines the trained model, all 2,000 genes, donor sensitivity, external cohort support, and model errors. The [original draft](archive/early-literature-suggestions.md) remains available as historical work.

## Suggested optional registration answer

> I’d like to test whether reducing CLEC5A in human stem-cell-derived microglia improves amyloid-β clearance without increasing synapse loss. Our analysis of human microglial gene expression, supported by a trained generative model, prioritizes CLEC5A as a pathology-associated candidate. Published mouse knockout results motivate testing whether the benefit translates to human cells, including across APOE3 and APOE4 backgrounds.

## What supports that answer

Five frozen, independently initialized, 20,000-update ParticleGAN 0.9.7 models generate gene-expression count distributions conditional on Alzheimer’s neuropathology stage. They do **not** condition on a perturbation. The training data comprise 14,762 human SEA-AD cells from 33 donors and a 2,000-gene panel. We examined 12.8 million unique generated cells using matched sampling across stages.

CLEC5A’s within-subtype High-minus-Not-AD contrast is positive in the observed data (+0.489), the five-model mean (+0.454), all five model fits, and all 40 Monte Carlo contrasts. Its observed coefficient remains positive after donor-covariate adjustment (+0.572). The primary published ROSMAP pathology association agrees, with global FDR 0.00628. These are observational associations and sensitivity checks, not an intervention-effect estimate.

[CLEC5A-deficiency experiments in mice and cells](https://pubmed.ncbi.nlm.nih.gov/39443966/) supply the reason to test **reduction**. The proposal is a human-cell translation experiment on an existing lead, not a claim that our model discovered the target or predicted a therapeutic response.

## The most useful lab design

Compare CLEC5A reduction with non-targeting controls across human microglial backgrounds, ideally including APOE3 and APOE4. Confirm target engagement and perturbation specificity. Measure amyloid uptake **and degradation**, viability, inflammation, and synaptosome engulfment or neuron co-culture synapse density together. Expression and lipid-droplet readouts can help interpret the response but should not substitute for function. Set quantitative success and failure thresholds with the experimental team in advance.

## Why the report includes negative evidence

The model reproduces the CLEC5A endpoint direction but misses the observed Low-stage peak. Several other apparent signals reverse or weaken after donor-covariate checks. Only three training donors are Not AD, and none carry APOE4. On four calibration donors used during selection, the model beats resampling on mean donor energy score but loses on cell distribution distance. We have not established that the generator improves target selection over observed expression plus literature alone.

These limits make a focused, falsifiable functional experiment more defensible than a broad therapeutic claim. The form has not been submitted by this project.
