# ParticleGAN 0.12.0 on NeuroKO at 1,200 updates

## Benchmark

Use the fixed neuronal benchmark from the v11 study. The conditional generator and critic each have two 96-unit hidden layers. Every fit uses 64 particles with 16 latent coordinates, a batch size of 64, and exactly 1,200 updates. The primary metric is the unweighted mean energy distance over GLCCI1, SLC2A4RG, PLEKHA1 and MAMSTR. Lower is better. Each score uses 256 draws per condition. Spread is the ratio of generated to observed root total variance, with a target of 1. [Previous formulation study](neuroko_formulation_1200.md)

In 0.12.0, the upstream default `unified_v12` inherits the same wheel’s `unified_v11` recipe and changes its optimizer family to `unified_v12_equil`. Both the effective critic formulation and the prior regularizer are named `unified_v11`. The benchmark pins the requested wheel, checks each instantiated optimizer and runs the released trainer. Its default v12 arm uses upstream optimizer and loss settings at the fixed NeuroKO capacity, including constant generator, critic and particle rates of 0.014, 0.021 and 0.0406. [Release](https://github.com/HyperGAN/ParticleGAN/releases/tag/v0.12.0) · [Upstream recipe](https://github.com/HyperGAN/ParticleGAN/blob/3256a05867688ab7ec0e5b4f50354c14d6972e6f/particlegan/recipes.py)

We specified 12 candidates before scoring the outer group. They include halving or doubling all rates, slowing the critic or particles independently, and applying the previous v11 choice of hollow self-coupling and relativistic weight 2. With those two settings and the slower critic and particles, we also halved or doubled the generator rate alone. At seed 0 we evaluated each candidate by leaving out each of `Screen1_1`, `Screen1_3` and `Screen1_4` in turn. The top three plus the default and the v11 transfer advanced to seed 1. The lowest average over the three groups and both seeds determined the v12 configuration. Each fold fits the expression projection using its training rows. [Candidate definitions and selection rule](../scripts/compare_neuroko_v12.py)

The outer evaluation uses neuronal `Screen1_0`, a group previously used for this project, with twelve new training seeds (72–83). Each seed’s arms use the same physical card, initialization, projected cells and minibatches. Controls include both v11 releases, EMPF at the selected v12 rates, and EMPF at its independently selected rates of 0.020, 0.030 and 0.058. The paired 95% Student intervals quantify seed variability conditional on this one screen group. Fit timings exclude preparation and evaluation, and include contention from other jobs on the two RTX A6000s.

## Reproduce on two GPUs

From the repository root, follow the Python, CUDA and H5AD setup in the [NeuroKO GPU README](../README.md#neuroko-gpu-comparison). With the environment active, run the following for the original public files. The driver reads `iNeurons.h5ad` and `iAstrocytes.h5ad` from the data directory. It pins each wheel in an isolated import directory and maps each paired holdout seed to GPU 0 or 1.

```bash
data=paper_watch/data/neuroko
python -m pip install --no-deps --target "$data/wheels/0.12.0" 'particlegan==0.12.0'
python paper_watch/scripts/compare_neuroko_v12.py \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" \
  --output "$data/v12-1200"
```

The v12 wheel SHA-256 is `8ae6ea792d61f7d3e0a6d19b90cd403b50b7eba54d6c5bac48e09324eeac418d`. For an already extracted pilot containing the first 10,000 cells of each group, pass its directory with `--data` and omit `--window-cells`. The same command resumes from completed fit receipts. It writes the search definition to `protocol.json`, the validation decision to `freeze.json`, raw run receipts to `validation/` and `heldout/`, and paired measurements to `report.json`.
