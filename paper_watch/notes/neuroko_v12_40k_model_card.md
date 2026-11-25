# NeuroKO neuronal ParticleGAN v12 at 40,000 updates

Two conditional generators model gene expression in human induced neurons after four gene knockouts. Both use ParticleGAN 0.12.0 and differ only in training seed. The original experiment saved scores and trajectory fingerprints. The downloadable weights come from deterministic GPU replays of both runs. [Public checkpoints and checksums](https://github.com/difflabai/alz-perturb-gan/releases/tag/neuroko-v12-40k) · [Study report](https://difflabai.github.io/alz-perturb-gan/neuroko-40k/)

| Mean energy distance on Screen1_1 | Seed 22 | Seed 251 | Mean |
| --- | ---: | ---: | ---: |
| v12 at 40,000 updates | 1.16185 | 1.14060 | 1.15122 |

Lower energy distance means the generated and observed distributions are closer in the fitted expression space. At 40,000 updates, v12 led on both seeds. Across seeds, matched v11 scored 1.20192, matched EMPF 1.20712, independently tuned v11 1.21361 and tuned EMPF 1.22364. V12’s mean effective covariance rank at the endpoint was 16.55, compared with 18.23 in the observed neurons. Matched EMPF was closer on that diversity measure for both seeds. [Five-way results, target metrics, timing and curves](https://github.com/difflabai/alz-perturb-gan/blob/neuroko-v12-40k/paper_watch/notes/neuroko_v12_shared_smoothing_40000.json)

In the original runs, v12 took 87.8 and 57.6 minutes for seeds 22 and 251. Matched EMPF took 127.2 and 86.2 minutes. These are fit times with intermediate scoring excluded, measured on shared A6000 GPUs.

Both long runs used one development group. Across six fresh seeds and three groups at 3,600 updates, tuned v11 led with 0.93298, followed by v12 at 0.93671. Matched EMPF scored 0.96530. The prespecified comparison on the reserved biological holdout required v12 to beat all four controls. That comparison did not qualify. These checkpoints support research on the measured knockout distributions. Generalization to a new screen, donor, or perturbation remains untested here. [Six-seed replication and holdout decision](https://github.com/difflabai/alz-perturb-gan/blob/neuroko-v12-40k/paper_watch/notes/neuroko_v12_shared_smoothing_3600_fresh.json)

## Download and sample on CPU

Run from Bash with Git, curl and Python 3.11. Sampling needs only the repository, one checkpoint and the Python packages below. This example downloads seed 22 and checks it against the release manifest.

```bash
git clone --branch neuroko-v12-40k --depth 1 https://github.com/difflabai/alz-perturb-gan.git
cd alz-perturb-gan
python3.11 -m venv .venv-neuroko
source .venv-neuroko/bin/activate
python -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'numpy==1.26.4' 'particlegan==0.12.0'

release=https://github.com/difflabai/alz-perturb-gan/releases/download/neuroko-v12-40k
model=neuroko-v12-40k-seed22.pt
curl -fLsS -O "$release/$model"
curl -fLsS -O "$release/SHA256SUMS"
awk -v file="$model" '$2 == file' SHA256SUMS | sha256sum -c -
python -m paper_watch.alztest.neuroko_checkpoint sample "$model" \
  --condition MAMSTR --draws 256 --seed 123 --out mamstr.npy
python -c "import numpy as np; x = np.load('mamstr.npy'); print(x.shape, np.isfinite(x).all())"
```

The output is a `(256, 30)` array of standardized principal component scores. Select `INT` for the intergenic control, or `GLCCI1`, `SLC2A4RG`, `PLEKHA1` or `MAMSTR` for a knockout. Use `neuroko-v12-40k-seed251.pt` to sample the other training seed. The CLI’s sample seed controls fresh draws from the trained model. CPU and CUDA use device-specific random streams.

## Check the published scores on CUDA

Use a CUDA-enabled environment for this section. The reference machine used Linux, Python 3.11, PyTorch 2.14.1+cu130, an NVIDIA RTX A6000 and the package versions below. Exact comparisons include the PCA fit, seeded samples and endpoint metrics. Numerical libraries, hardware and thread counts can affect bit-for-bit comparisons. The public fetcher downloads ~13.78 GB and verifies both original H5AD files against Zenodo’s checksums.

```bash
python -m pip install --force-reinstall 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'numpy==1.26.4' 'particlegan==0.12.0' 'scipy==1.17.1' \
  'pandas==2.3.3' 'h5py==3.16.0' 'anndata==0.12.7'
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
data=paper_watch/data/neuroko
python paper_watch/scripts/fetch_neuroko.py --out "$data"
python -m paper_watch.alztest.neuroko_checkpoint verify "$model" \
  --device cuda --data "$data" --window-cells 10000
```

The verifier reloads the served model weights, reconstructs the panel from the input files and draws 256 profiles per condition. It recomputes energy distance, perturbation-shift cosine, spread ratio and effective covariance rank for each knockout. It also compares the embedded replay trace and its minibatch fingerprints with the 34 published measurements from 1,200 through 40,000 updates. Only the endpoint is regenerated from the endpoint weights.

## Repeat training

From the same CUDA environment, these commands replay the two seeds on separate GPUs. On a single GPU, run the seeds in sequence with `CUDA_VISIBLE_DEVICES=0`. The output directories hold the training receipts and checkpoints. The JSON trial name is the identifier used in the original study. The recipe argument selects v12.

```bash
trial='{"name":"original-v11-critic-49","lr":0.014,"rhbi_self_coupling":"hollow","rhbi_relativistic_weight":2.0,"prior_sigma":0.15,"d_lr_mult":0.35,"prior_lr_mult":2.9,"prior_reg":0.1}'
train_seed() {
  CUDA_VISIBLE_DEVICES="$1" python -m paper_watch.alztest.neuroko_tune validation \
    --data "$data" --window-cells 10000 --output "runs/neuroko-40k/seed$2" \
    --seed "$2" --recipe unified_v12 --release 0.12.0 --trial "$trial" \
    --validation-group Screen1_1 --updates 40000 --trace-every 1200 \
    --deterministic --require-constant-lr --save-checkpoint
}
train_seed 0 22 & first=$!
train_seed 1 251 & second=$!
wait "$first" && wait "$second"
```

The benchmark fits 1,320 neuronal cells from `Screen1_3` and `Screen1_4` and evaluates 660 from `Screen1_1`, across four knockout conditions and their control. It uses the first 10,000 cells per group from each original file to choose the shared target panel. The model has 64 learned particles in 16 latent dimensions, batch size 64 and two 96-unit tanh hidden layers in both networks. The training recipe uses hollow self-coupling, relativistic weight 2, Gaussian prior noise 0.15, prior regularization 0.1 and native optimizer smoothing 0.0061. Both optimizers keep their learning rates constant for all 40,000 updates. The generator rate is 0.014, critic rate 0.0049 and particle rate 0.0406. V12 retains its native adaptive optimizer updates. [Commands for all controls](https://github.com/difflabai/alz-perturb-gan/blob/neuroko-v12-40k/paper_watch/README.md#follow-v12-and-the-four-original-controls-to-40000-updates)

## File contents and expression scale

Load either checkpoint with `torch.load(path, map_location="cpu", weights_only=True)`. Format version 1 contains the ordered condition labels, the scored generator, critic and prior under `served`, the native ParticleGAN state under `trainer`, the replay receipt, and the fitted PCA under `projection`. The native trainer state includes the training iterate, optimizers, EMA and random streams. Use the served parameters to reproduce the published samples.

The data pipeline normalizes counts by cell library size to 10,000, applies `log1p`, selects the 512 genes with highest training variance and fits 30 principal components on the training cells. To reconstruct the rank-30 approximation in the 512 selected genes, multiply each sampled component by its saved scale, rotate back and add the training mean. Columns follow the saved gene order. The values are approximate log-normalized expression and can be negative.

```python
import numpy as np
import torch

checkpoint = torch.load("neuroko-v12-40k-seed22.pt", map_location="cpu", weights_only=True)
p = checkpoint["projection"]
scores = np.load("mamstr.npy")
approximate_expression = (scores * p["scale"].numpy()) @ p["axes"].numpy() + p["center"].numpy()
gene_names = p["feature_names"]
```

## Attribution and reuse

Differential AI Lab releases these checkpoints under Creative Commons Attribution 4.0. Credit the lab, cite this release and identify modifications when redistributing the weights. Repository code uses the MIT license. [Checkpoint license](https://creativecommons.org/licenses/by/4.0/) · [Code license](https://github.com/difflabai/alz-perturb-gan/blob/neuroko-v12-40k/LICENSE)

Training data are from Sam Ogden and Geraldine Rodschinka, *Supplementary data – Transcriptomic and phenotypic characterization of Alzheimer's disease risk genes in neurons and astrocytes*, Zenodo record 23260389, CC BY 4.0. The benchmark uses their processed neuronal and astrocyte files to define the panel and trains on the neuronal subset. [Dataset and attribution](https://zenodo.org/records/23260389)
