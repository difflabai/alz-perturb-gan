# Real-data run 1

**The GAN loses to the expression baselines in this run. This was a CPU run because the requested GPUs were unavailable to this session.** CUDA PyTorch was installed, but `nvidia-smi` returned exit 9 (could not communicate with the NVIDIA driver), `torch.cuda.is_available()` was false, and CUDA allocation raised `No CUDA GPUs are available`. The GPU attempt stopped before loading/training. No GPU benchmark or GPU-memory measurement was completed.

Completed 2026-10-09 UTC (2026-10-08 in America/Denver). This is the repository's learning experiment on H1 embryonic stem-cell knockdowns, with no Alzheimer's or therapeutic inference. The README learning-experiment banner is unchanged.

All six model rows below use the same seeded split of **150 perturbation identities: 120 training, 30 unseen evaluation perturbations**. No evaluation perturbation's cells or pseudobulk enter training. These are local held-out identities within the public training release, rather than Arc's official validation/test split.

| Model | MAE ↓ | L2 ↓ | Pearson delta ↑ | Top-10 DE Jaccard ↑ | Local PDS ↑ | Top-1 ↑ | MRR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_change | 0.014079 | 3.948069 | n/a | 0.000000 | 0.500000 | 0.000000 | 0.013245 |
| mean_effect | 0.014333 | 3.976597 | 0.161309 | 0.028884 | 0.500000 | 0.000000 | 0.013245 |
| additive | 0.014333 | 3.976597 | 0.161309 | 0.028884 | 0.500000 | 0.000000 | 0.045935 |
| linear | 0.014333 | 3.976597 | 0.161309 | 0.028884 | 0.500000 | 0.000000 | 0.023881 |
| gan_generator | 0.017204 | 4.205802 | 0.098997 | 0.027325 | 0.500000 | 0.000000 | 0.027107 |
| decision_noul | n/a | n/a | n/a | n/a | n/a | 0.000000 | 0.013245 |

These local metrics use all 18,080 genes and ALL cells belonging to each held-out perturbation. Scores are macro-averaged over 30 held-out pseudobulks. The GAN profile is a 64-draw particle average. Phenocopy queries rank the full 150-perturbation library; top-1 requires a unique best score, and MRR averages tied ranks. Local DE overlap is top-10 absolute-delta Jaccard; local PDS uses L2 and average ties. They are separate implementations from Arc's metrics. Pearson delta is undefined for `no_change`'s zero predicted delta. The decision head produces probabilities/rankings, rather than expression, hence its n/a cells.

`additive` is reported as its implemented fallback: this dataset has single knockdowns and no combinatorial cases. On an unseen single it predicts mean effect; there is no real-data additive-combination test here. With one-hot identity embeddings, the ridge linear model predicts its training mean for an unseen identity. Different MRRs for additive/linear reflect their predictions for the known training candidates in the full library, not different held-out expression predictions.

**Arc cell-eval 0.8.2, VCC profile.** `MetricsEvaluator.compute(profile="vcc")` succeeded for all five expression models, using explicit log1p DE input and four threads. Below are its aggregate `mean` rows, without leaderboard normalization.

| Model | Arc MAE ↓ | Arc PDS (L1) ↑ | Arc DE overlap_at_N ↑ |
| --- | ---: | ---: | ---: |
| no_change | 0.017857 | 0.518889 | 0.002651 |
| mean_effect | 0.017790 | 0.524444 | 0.002255 |
| additive | 0.017790 | 0.524444 | 0.002255 |
| linear | 0.017790 | 0.524444 | 0.002255 |
| gan_generator | 0.019502 | 0.523333 | 0.006854 |
| decision_noul | n/a | n/a | n/a |

Arc evaluation uses up to 128 sampled real cells per held-out identity: **3,712 perturbed cells plus 128 non-targeting cells**. It uses 128 predicted cells per identity. Baselines repeat their deterministic profile; the generator supplies individual particle draws. The same observed non-targeting sample is available to every prediction as its control distribution. Generator predictions are clipped to zero for Arc's nonnegative input requirement; local metrics use the raw generator output. Thus Arc MAE is against sampled-cell pseudobulks, while local MAE is against all-cell pseudobulks. Arc PDS uses L1, target-gene exclusion, and its own rank normalization; its near-0.52 constant-baseline scores are not evidence of successful discrimination.

The generator's local MAE is **22.2% worse than no_change** and **20.0% worse than mean_effect**. Its Arc MAE is **9.6% worse than mean_effect**. The GAN's Arc DE overlap is slightly higher (0.006854 versus 0.002255), but remains extremely small. This small DE advantage does not rescue its expression error or zero retrieval hits. All models have zero top-1 retrieval on this split; this comparison provides no successful phenocopy result.

**The decision labels are degenerate on these unpaired data.** Every sampled real training label across 2,000 minibatches was negative. In the fresh held-out check (seed 45), all 3,712 real-cell labels were also negative. The head predicts mean probability 1.87749e-8, accuracy 1.0, and Brier 3.56308e-16. Always predicting zero achieves perfect accuracy and Brier zero on those labels. Its full-library MRR is 0.013245, the all-tied value 1/75.5, and top-1 is zero. The excellent binary accuracy is therefore an all-negative-label failure, not evidence of calibration or biological utility. Comparing a noisy individual cell to a clean pseudobulk target makes the control mean geometrically closer.

**Data and training configuration.** The [Arc dataset README](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md) identifies the public training file. This run used:

- GCS object: `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad`.
- Generation `1765904883947296`, updated `2025-12-16T17:08:04.075Z`, 15,482,497,461 bytes; published CRC32C `/Z0row==` matched the full local download.
- SHA-256: `a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b`.
- 221,273 cells, 18,080 genes, 150 knockdown identities, 38,176 non-targeting cells. The 30 evaluation identities contain 36,350 cells in total. The matrix contains raw integer counts, with no layers; normalize each cell to 10,000 total counts, then apply log1p. No gene selection or filtering was applied.
- Split and model initialization seed 42; single-cell sampling seed 43; local generator inference seed 44; held-out decision-query seed 45. The explicit split and counts are in [split.json](results/real_data_run_1/split.json).
- 15,123 sampled training cells, capped at 128 per training perturbation. All training cells condition on the common mean of all non-targeting cells; the assay is unpaired. All cells contribute to their identity's pseudobulk. Controls are shared available background data.
- Fixed one-hot perturbation embeddings, with no external gene features or outcomes for unseen identities. This is a test of the current implementation under that restriction; it does not compare models with informative transferable gene embeddings.
- ParticleGAN 0.8.0 recipe `gan`; 2,000 updates, batch 64, hidden width 128, latent width 16, 256 learnable particles, learning rate 0.0002. Reconstruction weight 1.0, generator adversarial weight 0.1, decision adversarial weight 1.0. The smoke run had 40 updates, batch 16, hidden width 64, latent width 4, and 64 particles. This run presents 128,000 training examples, approximately 8.46 passes over the capped cell pool.
- Linear baseline: 32 gene components and ridge 0.1. No tuning or selection used the held-out scores. This is one training seed and one split, without uncertainty intervals.
- Python 3.11.15; PyTorch 2.6.0+cu124 (CUDA 12.4 build); NumPy 2.4.6; AnnData 0.12.19; cell-eval 0.8.2. CPU: AMD Ryzen 9 5900X, with four PyTorch/OpenMP/BLAS threads. See the installed [requirements snapshot](results/real_data_run_1/requirements.txt).

| Operation | Seconds |
| --- | ---: |
| Stream and aggregate real data | 140.319 |
| no_change fit + local scoring | 0.111 |
| mean_effect fit + local scoring | 0.095 |
| additive fit + local scoring | 0.094 |
| linear fit + local scoring | 0.404 |
| Generator + critic + decision training (2,000 updates) | 420.098 |
| Generator + decision local scoring | 1.985 |
| Arc evaluation + export: no_change | 16.276 |
| Arc evaluation + export: mean_effect | 7.699 |
| Arc evaluation + export: additive | 7.900 |
| Arc evaluation + export: linear | 7.531 |
| Arc evaluation + export: gan_generator | 14.205 |
| SHA-256 of source h5ad | 42.080 |
| Total benchmark wall time, including exports and hash | 661.895 |

Total benchmark runtime was **11 minutes 1.9 seconds**, excluding environment installation, download, and tests. Generator/critic/decision training took **7 minutes 0.1 seconds**. The remaining total includes dataset aggregation, h5ad/checkpoint export, scoring, and source hashing. The benchmark used no GPU; GPU memory was not measurable in this session. The runner's CUDA path selects the UUID with the most free memory via `CUDA_VISIBLE_DEVICES` and caps its PyTorch allocator at 19 GiB, but that path was not validated on a GPU here.

**Exact commands used.** The original checkout's `.git` was read-only, so branch creation and the commit use an isolated writable clone inside the authorized root. Starting code commit: `e34230da52e2f04e0998b8f19aa98843ddb30b81`. All data, environment, caches, working copy, and run artifacts are under `/ml2/altgan`.

```bash
git clone --no-hardlinks /ml2/altgan/alz-perturb-gan /ml2/altgan/real-data-run-1
cd /ml2/altgan/real-data-run-1
git remote set-url origin https://github.com/difflabai/alz-perturb-gan.git
git switch -c real-data-run-1

mkdir -p /ml2/altgan/data /ml2/altgan/runs/real-data-run-1 /ml2/altgan/.cache/uv
UV_CACHE_DIR=/ml2/altgan/.cache/uv UV_PYTHON_INSTALL_DIR=/ml2/altgan/.python \
  uv venv /ml2/altgan/.venv --python 3.11 --managed-python
UV_CACHE_DIR=/ml2/altgan/.cache/uv TMPDIR=/ml2/altgan/.cache \
  uv pip install --python /ml2/altgan/.venv/bin/python 'torch==2.6.0' \
  --index-url https://download.pytorch.org/whl/cu124 \
  > /ml2/altgan/runs/real-data-run-1/install-torch.log 2>&1
UV_CACHE_DIR=/ml2/altgan/.cache/uv TMPDIR=/ml2/altgan/.cache \
  uv pip install --python /ml2/altgan/.venv/bin/python -e '.[dev,vcc]' \
  'torch==2.6.0+cu124' cell-eval google-crc32c fsspec requests \
  > /ml2/altgan/runs/real-data-run-1/install-package.log 2>&1
# Refresh the editable entry point after adding the benchmark CLI.
UV_CACHE_DIR=/ml2/altgan/.cache/uv TMPDIR=/ml2/altgan/.cache \
  uv pip install --python /ml2/altgan/.venv/bin/python --no-deps -e . \
  > /ml2/altgan/runs/real-data-run-1/install-entrypoint.log 2>&1

curl -sS -L --fail --retry 5 -C - \
  -D /ml2/altgan/data/adata_Training.headers.txt \
  https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad \
  -o /ml2/altgan/data/adata_Training.h5ad \
  > /ml2/altgan/runs/real-data-run-1/download.log 2>&1
curl -sS -L --fail --retry 3 \
  https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/pert_counts_Training.csv \
  -o /ml2/altgan/data/pert_counts_Training.csv
curl -sS --fail \
  'https://storage.googleapis.com/storage/v1/b/arc-institute-virtual-cell-atlas/o/virtual-cell-challenge%2F2025%2Ftrain%2Fadata_Training.h5ad' \
  -o /ml2/altgan/data/adata_Training.gcs.json

TMPDIR=/ml2/altgan/.cache \
  /ml2/altgan/.venv/bin/python -m alz_perturb_gan.real_data \
  --data /ml2/altgan/data/adata_Training.h5ad --input-scale counts \
  --device cuda --run-dir /ml2/altgan/runs/real-data-run-1/cuda-attempt \
  > /ml2/altgan/runs/real-data-run-1/cuda-attempt.log 2>&1
# Above attempt failed at GPU selection; no training occurred.

TMPDIR=/ml2/altgan/.cache XDG_CACHE_HOME=/ml2/altgan/.cache \
NUMBA_CACHE_DIR=/ml2/altgan/.cache/numba MPLCONFIGDIR=/ml2/altgan/.cache/matplotlib \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 \
  /ml2/altgan/.venv/bin/python -u -m alz_perturb_gan.real_data \
  --data /ml2/altgan/data/adata_Training.h5ad --input-scale counts --device cpu \
  --seed 42 --steps 2000 --batch-size 64 --hidden 128 --lr 0.0002 \
  --train-cells 128 --eval-cells 128 --eval-fraction 0.2 --draws 64 --cell-eval \
  --run-dir /ml2/altgan/runs/real-data-run-1/main \
  > /ml2/altgan/runs/real-data-run-1/main.log 2>&1

TMPDIR=/ml2/altgan/.cache XDG_CACHE_HOME=/ml2/altgan/.cache \
NUMBA_CACHE_DIR=/ml2/altgan/.cache/numba MPLCONFIGDIR=/ml2/altgan/.cache/matplotlib \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /ml2/altgan/.venv/bin/python -m pytest \
  --basetemp=/ml2/altgan/runs/real-data-run-1/pytest-tmp \
  > /ml2/altgan/runs/real-data-run-1/tests.log 2>&1
```

For a future exact-version download, append `?generation=1765904883947296` to the h5ad URL and verify the recorded checksum. For a GPU rerun, change `--device cpu` to `--device cuda` and use a fresh run directory. The directory should be new: training logs append and other outputs overwrite.

The committed [full result JSON](results/real_data_run_1/results.json), per-model Arc CSVs, source hashes, installed versions, split, and training log support the tables. Local heavy artifacts are in `/ml2/altgan/runs/real-data-run-1/main/`: checkpoint, real/predicted h5ad files, and diagnostic `predictions.pt`. The generator in `predictions.pt` is a second seeded 64-draw sample; local metrics use the first 64-draw sample, while Arc uses individual draws. Data and large binary artifacts are not in Git.

Validation: **21 tests passed**, including sparse streaming pseudobulk agreement across chunk sizes, seeded split reproducibility, exclusion of held-out identities from training tensors, explicit expression-scale checks, baseline evaluation, and ParticleGAN smoke training. The actual real run completed all 2,000 steps with finite logged losses. GPU execution remains unverified.

**Concrete next experiments.** Choose a separate identity-held-out validation set before changing hyperparameters; retain this split as a recorded test result.

1. Replace identity-only one-hot features with transferable gene features: begin with a control-only gene coexpression embedding or a documented pretrained gene embedding. Compare linear and GAN models on exactly the same features and splits. Otherwise unseen identities have no learned relationship to training identities.
2. Repair the decision-label design before interpreting probabilities. Train/query on comparable pseudobulk or bootstrap-distribution states, measure positive/negative label counts, and introduce balanced positive and negative targets. Evaluate Brier/accuracy against an always-zero classifier on an independent identity holdout.
3. Run reconstruction-only and adversarial ablations with identical architecture, embeddings, cell batches, seed, and update count: generator adversarial weight 0 versus 0.01 versus 0.1, and decision critic-gating weight 0 versus 1. This measures whether the GAN contributes beyond ordinary regression.
4. Use batch-specific available controls from `obs['batch']`; compare to the shared-control mean. For generative modeling, use nonnegative outputs and a distributional objective suited to unpaired cells, then check single-cell variance as well as pseudobulk error and DE recovery. Avoid treating randomly paired cells as reconstruction targets for a matched response.
5. Repeat with at least five fixed perturbation splits/seeds and bootstrap confidence intervals over perturbations. Select configurations only on the separate validation identities, then compare paired errors against no_change and mean_effect. Include combinatorial data in a separate benchmark before drawing conclusions about additive prediction.
