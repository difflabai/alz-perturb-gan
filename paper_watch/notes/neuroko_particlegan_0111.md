# ParticleGAN 0.11.1 on NeuroKO

This is a matched replay of the [NeuroKO pilot](neuroko_revival.md) with the tested `0.10.0` wheel and the verified `0.11.1` wheel. Release `0.11.1` retains explicit EMPF and uses `unified_v11` as the public default. Its changelog includes a GPU memory fix and changes to the default's critic penalty and optimizer. The NeuroKO workload uses small CPU vector networks. [Release notes](https://github.com/HyperGAN/ParticleGAN/releases/tag/v0.11.1) · [PyPI wheel](https://pypi.org/project/particlegan/0.11.1/)

The [CUDA learning-rate sweep](neuroko_particlegan_0111_gpu_lr.md) follows this CPU comparison with separate training validation and GPU holdout fits.

## Results

The 0.10.0 replay reproduced the archived pilot exactly. With explicit EMPF, the 0.11.1 wheel reproduced every energy-distance, shift-cosine and spread-ratio score for both the model and its shuffled-label arm. The serialized EMPF recipes and non-model controls also matched exactly.

| Package and recipe | Neuron energy ↓ | Astrocyte energy ↓ | Neuron spread ratio | Astrocyte spread ratio | Median fit CPU ms/update, neurons / astrocytes |
|---|---:|---:|---:|---:|---:|
| 0.10.0 EMPF | 1.298 | 1.657 | 1.36 | 1.38 | 54 / 53 |
| 0.11.1 EMPF | 1.298 | 1.657 | 1.36 | 1.38 | 52 / 52 |
| 0.11.1 `unified_v11` | 1.522 | 2.002 | 0.61 | 0.62 | 102 / 104 |
| Empirical resampling | 1.049 | 1.468 | 1.21 | 1.21 | — |
| Diagonal Gaussian | 1.036 | 1.431 | 1.21 | 1.20 | — |

At the matched learning rate, `unified_v11` had 17% higher mean energy distance in neurons and 21% higher in astrocytes than EMPF. It beat EMPF on 9 of 72 paired target–holdout–seed rows, and empirical resampling on 0 of 72. Generated spread was 61–62% of observed spread. Mean shift cosine was 0.006 in neurons and 0.055 in astrocytes, compared with −0.041 and −0.020 for EMPF. The true-label `unified_v11` energy means were close to its shuffled-label means of 1.534 and 2.009. Its median CPU time per update was about twice that of 0.11.1 EMPF.

At seed 0, the published learning rate of 0.014 improved `unified_v11` on 23 of 24 paired rows. Mean energy distance fell from 1.518 to 1.458 in neurons and from 1.992 to 1.965 in astrocytes. Empirical resampling scored 1.050 and 1.467 on those same rows. The version upgrade reproduces the existing NeuroKO benchmark with explicit EMPF. The CPU pilot favors the simple baselines over either trainer.

## Matched design

The three primary arms use the same first 10,000 release rows per group, four targets, four neuronal replicate holdouts, two astrocyte sequencing-batch holdouts, and seeds 0, 1 and 2. The model and shuffled-label fit each receive 300 updates per fold. The harness uses the same architecture, batch size 64, 64 particles, 16 latent dimensions, prior sigma 0.025 and base learning rate 0.005. PCA and scaling are fitted on each training split. The public recipes supply their own critic, prior, optimizer modifiers and schedule.

The learning-rate sensitivity uses `unified_v11`'s published base rate of 0.014 with seed 0 on all six groups. Compare it only with the seed-0 results of the matched-rate run. The batch, particle and latent dimensions stay at the NeuroKO harness values.

CPU timings measure process CPU time inside model fitting, including the first fit update, on a Ryzen 9 5900X with two Torch threads per process. They exclude preprocessing, model construction, drawing samples and scoring. The 300-update comparisons and all score means are descriptive for these pilot windows.

## Reproduce

Use the two release wheels in separate environments with the same remaining packages. Wheel and H5AD digests are listed in the [comparison summary](neuroko_particlegan_0111.json). With the pilot H5AD slices under `paper_watch/data/neuroko/pilot`, run from the repository root.

```sh
# Run once in each ParticleGAN environment.
particle_version=$(python -c 'from importlib.metadata import version; print(version("particlegan"))')
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m paper_watch.alztest.neuroko evaluate --data paper_watch/data/neuroko/pilot \
  --recipe empf --out "paper_watch/data/neuroko/empf-${particle_version}.json"

# Run these two with the verified 0.11.1 wheel.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m paper_watch.alztest.neuroko evaluate --data paper_watch/data/neuroko/pilot \
  --recipe unified_v11 --out paper_watch/data/neuroko/unified-v11-comparison.json
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m paper_watch.alztest.neuroko evaluate --data paper_watch/data/neuroko/pilot \
  --recipe unified_v11 --learning-rate 0.014 --seeds 0 \
  --out paper_watch/data/neuroko/unified-v11-published-rate.json
```

With the original Zenodo files, use `--data paper_watch/data/neuroko --window-cells 10000` to select the pilot windows. `--seeds` also lets independent processes run individual seeds when reproducing the three-seed comparison.
