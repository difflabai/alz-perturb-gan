# ParticleGAN 0.12.0 on NeuroKO at 1,200 updates

Retuned v12 lowered mean held-out energy by **22.1%** against its release defaults at fixed capacity, winning on all 12 seeds. Mean generated spread increased from **0.648 to 1.031** relative to the observed neurons. Validation selected the default learning rates with hollow self-coupling and relativistic weight 2. Under hollow self-coupling, a generated draw cannot match itself when the within-batch transport weights are computed. Distinct draws from the same latent particle can still match. Both transport settings were available in 0.11.1.

The independently tuned EMPF control and selected v12 finished within **0.0013** mean energy of each other. At identical actual rates, EMPF scored **0.0265** lower on average and won 9 of 12 seeds. The same-wheel v11 optimizer scored **0.0282** lower on average and won 8 of 12. All three paired intervals include zero. V12 finished its 1,200 updates in a median **123 seconds**, about 18% less time than v11 on the same wheel. The evidence supports the gain over default v12 and leaves the quality ordering between the formulations unresolved on this holdout.

## Results on the held-out neurons

| Arm | Generator / critic / particles LR | Energy ↓ | Spread (target 1) | Median fit |
| --- | --- | ---: | ---: | ---: |
| v12, selected | 0.014 / 0.021 / 0.0406 | 1.3892 | 1.031 | 123 s |
| v12, release defaults at fixed capacity | 0.014 / 0.021 / 0.0406 | 1.7837 | 0.648 | 131 s |
| v12, previous v11 settings | 0.014 / 0.0075 / 0.0125 | 1.3866 | 1.073 | 124 s |
| v11, same wheel and selected settings | 0.014 / 0.021 / 0.0406 | 1.3610 | 1.040 | 150 s |
| v11, previously tuned on 0.11.1 | 0.014 / 0.0075 / 0.0125 | 1.3823 | 1.041 | 178 s |
| EMPF, matched rates on 0.12.0 | 0.014 / 0.021 / 0.0406 | 1.3627 | 1.047 | 149 s |
| EMPF, independently tuned rates on 0.12.0 | 0.02 / 0.03 / 0.058 | 1.3879 | 1.063 | 144 s |

The differences below subtract each control’s energy from selected v12 on the same seed. A negative difference favors v12.

| Control | Mean difference | Paired 95% interval | Seeds won by v12 |
| --- | ---: | ---: | ---: |
| v12, release defaults at fixed capacity | -0.3945 | [-0.4375, -0.3515] | 12/12 |
| v12, previous v11 settings | +0.0026 | [-0.0382, +0.0433] | 6/12 |
| v11, same wheel and selected settings | +0.0282 | [-0.0237, +0.0802] | 4/12 |
| v11, previously tuned on 0.11.1 | +0.0069 | [-0.0279, +0.0417] | 5/12 |
| EMPF, matched rates on 0.12.0 | +0.0265 | [-0.0061, +0.0592] | 3/12 |
| EMPF, independently tuned rates on 0.12.0 | +0.0013 | [-0.0424, +0.0450] | 7/12 |

### Held-out energy by perturbation

| Target | Selected v12 | Same-wheel v11 | Matched EMPF | Tuned EMPF |
| --- | ---: | ---: | ---: | ---: |
| GLCCI1 | 1.5458 | 1.4894 | 1.5165 | 1.5480 |
| SLC2A4RG | 1.4145 | 1.3816 | 1.3785 | 1.4041 |
| PLEKHA1 | 1.2169 | 1.1974 | 1.2170 | 1.2141 |
| MAMSTR | 1.3797 | 1.3756 | 1.3387 | 1.3853 |

### Training-group selection

| Finalist | Mean energy over two seeds and three groups |
| --- | ---: |
| `hollow-active2` | 0.9603 |
| `v11-transfer` | 0.9608 |
| `hollow` | 0.9924 |
| `default` | 1.2255 |

Selection used 36 first-seed fits and 12 second-seed fits across four finalists. Four additional second-seed validation fits ran on spare GPU capacity. The selection used only the prespecified finalists. All 84 held-out fits passed the pairing checks for projected data, starting weights and the full minibatch sequence. [Per-seed measurements, hashes and frozen selection](neuroko_v12_1200.json)


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
