> **LEARNING EXPERIMENT.** This repository is an exploratory learning experiment. It is not a validated scientific or medical tool, not medical advice, and not affiliated with Prima Mente or the Alzheimer's Disease Data Initiative.

# alz-perturb-gan

A small, swappable experiment in scoring cell perturbations before the [Prima Mente Alzheimer's Translation Challenge](https://www.primamente.com/challenges/) data exists.

In spring 2027 that challenge plans to release an atlas of about 150 million neurons, astrocytes, and microglia under combinatorial chemical and genetic perturbations, with multimodal readouts. Stage one scores models on a set of evals, including general benchmarks and phenocopy (finding perturbations that reproduce a target cell state). Stage two asks shortlisted entries for one therapeutic hypothesis. Registration opened in October 2026. This repo is practice for that shape of problem. It does not produce a therapeutic hypothesis, and it must not be read as one.

Until that atlas exists, the public stand-in is the Arc Institute Virtual Cell Challenge set: h5ad, on the order of 300,000 H1 cells and 300 CRISPRi knockdowns. The code trains and scores against a `PerturbationSet`, so that file can replace the synthetic simulator without a change to the models. See [How to fetch the Virtual Cell data](#how-to-fetch-the-virtual-cell-data).

## The idea

The model under test is a small decision model that returns a typed yes/no judgment, in the sense of a single probability rather than a paragraph:

- question: does perturbation P move the control cell state toward target T?
- answer: a **noul**, a probability in (0, 1)
- ranking: sort candidate perturbations by that probability

Training uses a GAN signal as well as real labels.

1. A generator maps `(control state, perturbation embedding, particle latent)` to a perturbed expression profile. The latent comes from a learnable particle table.
2. A critic scores real versus generated outcomes under the same condition.
3. The decision head is trained on two labels. On a real cell, the label is 1 when the observed outcome is closer to the target than the control is. On a generated outcome, the same geometric label is used, multiplied by the critic's detached probability that the sample looks real. When the critic rejects a sample, that sample barely teaches the decision head.

The generator is also fit with a mean-squared error term so a short CPU run has a regression signal, not only an adversarial one. Weights are `rec_weight=1` and `generator_adv_weight=0.1` for the generator, and `decision_adv_weight=1` for the critic-gated decision term.

That loop is plain PyTorch. Optimizers, the relativistic-pairing loss, the critic penalty, and the learning-rate schedule come from ParticleGAN 0.8.0's recipe API: `recipe.make_optimizers`, `recipe.make_critic_penalty`, `recipe.make_loss`, and `scale_learning_rates`. The package pin is `particlegan==0.8.0`.

## Baselines

Simple baselines are the bar in this field. Ahlmann-Eltze, Huber, and Anders, [Nature Methods 2025](https://doi.org/10.1038/s41592-025-02772-6), found that deep perturbation models did not beat deliberately simple ones. This repo fits the same four and scores every other model against them:

| Baseline | What it predicts |
| --- | --- |
| `no_change` | The control profile. |
| `mean_effect` | Control plus the mean training delta. With one shared control, that is the mean training expression. |
| `additive` | For a combination of A and B, `y_A + y_B - y_control`. Missing members fall back to the mean delta. Eval combination profiles are never read. |
| `linear` | `Y ≈ G W P^T + b`. `G` is the leading left singular vectors of the centered training matrix (one row per gene). `P` is the perturbation embedding. `W` solves the paper's ridge normal equations with lambda = 0.1. |

## Evaluation

Held-out perturbations are scored as pseudobulk profiles. Phenocopy queries use each held-out profile as a target state and rank the whole library. The correct hit is the perturbation that produced the target. Its training cells were not used if it is in the eval split. Embeddings of eval perturbations are available; their outcomes are not used for fitting.

Local metrics, defined in `src/alz_perturb_gan/metrics.py`:

- **MAE** and **L2** between predicted and observed profiles
- **Pearson delta**: correlation of deviations from control
- **DES overlap**: Jaccard overlap of the top genes by absolute deviation from control. This is a stand-in for the Virtual Cell Challenge differential-expression score, not Arc's implementation.
- **PDS**: normalized rank of the matching true profile among all true profiles, by L2 of the delta. Ties use the average rank. This is a stand-in for perturbation discrimination, not Arc's implementation.
- **Phenocopy top-1 and MRR**. Top-1 requires the true perturbation to be the unique best score. An all-tied ranking is not a hit.

Arc's `cell-eval` package was not installed for the smoke run (`cell_eval` does not import). `cell_eval_adapter.py` calls `MetricsEvaluator` if a future environment exports it, and otherwise returns that reason. The official CLI, when you have both h5ad files and the package, is `cell-eval run -ap pred.h5ad -ar real.h5ad --profile full`.

## Setup

Python 3.11 or newer. The smoke run is CPU-only. ParticleGAN 0.8 requires PyTorch 2.6 or newer.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest
python -m alz_perturb_gan.smoke
```

`python -m alz_perturb_gan.smoke` trains the four baselines, the ParticleGAN decision model (40 updates), and the HyperGAN develop bridge (80 updates) on the built-in synthetic set. It writes `results/smoke_results.json`. No download.

## HyperGAN

**Released 1.0.0-rc2 does not run here.** The git tag `1.0.0-rc2` (9 August 2020) has `setup.py` version `1.0.6`, which is the PyPI release `hypergan==1.0.6` uploaded the same day. There is no PyPI version named `1.0.0-rc2`. On Python 3.12 the import fails in this order:

1. `ModuleNotFoundError: No module named 'hyperchamber'` from `hypergan/layers/operation.py`. The package declares no `install_requires`. Classifiers list Python 3.4–3.6 only. The 1.0 README still describes a TensorFlow framework.
2. After `hyperchamber==0.3.1`, `ModuleNotFoundError: No module named 'PIL'` from `hypergan/samplers/base_sampler.py`.
3. With Pillow present, `ModuleNotFoundError: No module named 'torchvision'` from `hypergan/configurable_component.py`. `StandardGAN.build` also exports ONNX onto `device='cuda'`.

That release is not on the training path. The failure is recorded in `src/alz_perturb_gan/hypergan_bridge.py` and in the smoke JSON.

The develop branch is a 2026 rebuild, package version `2.0.0a1`, and it is not a PyPI release. This experiment pins commit `01a3aaec45f49923922787b0412a30fd71a2c897`. Its training loop uses ParticleGAN 0.8.0. The bridge keeps that dependency isolated: `gan.py` does not import HyperGAN. The bridge writes a conditional recipe whose data factory reads the same training cells as the other models, then calls `hypergan.training.train`.

```bash
python -m pip install 'hndl>=0.8.0' \
  'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@01a3aaec45f49923922787b0412a30fd71a2c897'
```

`pip install -e '.[hypergan]'` is the same pin. Short smokes set `training.ema = 0` so sampling uses the live generator. The develop default decay of 0.995 does not move in a few dozen updates, and sampling reads the EMA copy.

## How to fetch the Virtual Cell data

Do this only when you want the real files. Tests and the default smoke do not touch them. The training h5ad is large.

The Arc layout is documented in the [Virtual Cell Challenge README](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md). The 2025 files are in `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`:

```text
2025/train/adata_Training.h5ad
2025/train/pert_counts_Training.csv
2025/validation/adata_Validation.h5ad
```

One way to copy the training matrix, if you have the Google Cloud CLI:

```bash
gcloud storage cp \
  gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad \
  data/adata_Training.h5ad
python -m pip install -e '.[vcc]'
python -m alz_perturb_gan.smoke --data data/adata_Training.h5ad
```

`load_virtual_cell_h5ad` expects AnnData with `obs['target_gene']`. Cells labeled `non-targeting` (also `control`) become the control mean and are not perturbations. Other labels are single knockdowns. Expression profiles are pseudobulk means. At most 32 training cells per perturbation are kept for the GAN, because these assays are unpaired: each perturbed cell is conditioned on the control mean, not on a matched pre-perturbation cell.

The loader's eval split is a lexical holdout of perturbation names. It is not the official challenge split. The official validation matrix is the separate file above. Default embeddings are one-hot over the names in that file, so the linear model cannot transfer to a perturbation identity that was absent from training. Pass `embeddings=` if you have gene or compound features.

Combinations are represented on the synthetic set (member names on a `Perturbation`). The 2025 Virtual Cell file is single knockdowns. The additive baseline still runs; with no combinations it predicts each training single by its own profile.

## Current results

Synthetic set, seed 0: 16 genes, 6 singles, 4 combinations, noise 0.02, 1 held-out single and 2 held-out combinations (3 eval profiles, 10 library candidates). ParticleGAN decision model: 40 updates. HyperGAN develop bridge: 80 updates. Command: `python -m alz_perturb_gan.smoke`. Full numbers are in `results/smoke_results.json`.

| model | MAE | Pearson delta | DES overlap | PDS | phenocopy top-1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_change | 0.399 | n/a | 0.000 | 0.500 | 0.000 | 0.182 |
| mean_effect | 0.413 | 0.437 | 0.374 | 0.500 | 0.000 | 0.182 |
| additive | 0.237 | 0.680 | 0.778 | 0.667 | 0.667 | 0.750 |
| linear | 0.174 | 0.911 | 0.793 | 1.000 | 1.000 | 1.000 |
| gan_generator | 0.385 | 0.491 | 0.463 | 0.500 | 0.333 | 0.667 |
| decision_noul | n/a | n/a | n/a | n/a | 0.000 | 0.145 |
| hypergan | 0.435 | 0.441 | 0.380 | 0.500 | n/a | n/a |

On this simulator the effects are linear in the embedding and combinations add, so the linear and additive baselines are the right answers. They win. The 40-step generator does not beat them. The decision noul does not retrieve the phenocopy (top-1 is 0). An in-sample check on reshuffled training targets is accuracy 0.714 and Brier 0.193; that is not a held-out calibration. The mean-effect baseline is slightly worse than predicting no change on this draw (MAE 0.413 versus 0.399). HyperGAN's 80-step conditional run is in the same band as the short ParticleGAN generator and behind both linear and additive. Pearson delta is undefined for `no_change` because the predicted delta is identically zero. PDS of 0.5 for a constant prediction is the tie, not a success.

These numbers are a toy CPU smoke. They are not evidence about Alzheimer's biology, the Virtual Cell Challenge leaderboard, or any therapeutic claim.

## Layout

| Path | Role |
| --- | --- |
| `src/alz_perturb_gan/data.py` | `PerturbationSet`, synthetic simulator, h5ad loader |
| `src/alz_perturb_gan/baselines.py` | no-change, mean effect, additive, linear |
| `src/alz_perturb_gan/gan.py` | ParticleGAN recipe loop and decision-head loss |
| `src/alz_perturb_gan/models.py` | generator, critic, decision head |
| `src/alz_perturb_gan/metrics.py` | MAE, Pearson delta, overlap, discrimination, phenocopy rank |
| `src/alz_perturb_gan/harness.py` | one comparison across models |
| `src/alz_perturb_gan/hypergan_bridge.py` | isolated develop-branch trainer, plus the 1.0.0-rc2 failure record |
| `src/alz_perturb_gan/cell_eval_adapter.py` | optional `cell-eval` hook |
| `src/alz_perturb_gan/smoke.py` | CPU end-to-end run |

## License

MIT. Copyright 2026 Differential AI Lab.
