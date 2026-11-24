# Paper watch — test conversions

This directory holds the ALZ Perturb GAN **paper watch** test
conversions. The watch scans new Alzheimer's research (bioRxiv/medRxiv,
arXiv, PubMed, dataset hubs) on a Mon/Wed/Fri cadence and judges every
find by one question: does it give us a distribution to model, tell us
which distribution matters, or help us read what the model learned?

Top-ranked finds are converted here into small CPU tests — one PR per
find, research-note style, **nothing merged by the watch**. Each test
uses the released ParticleGAN EMPF API with small CPU-specific network
and optimizer settings. Refinements inside a 10-iteration loop may touch data handling,
conditioning, splits, and readouts — never the GAN. Minimum control
arms: a mean/linear baseline and a shuffled-label/condition arm.
"Proven" means the effect holds across ≥3 seeds with a clear margin
over every control on held-out data; after 10 unproven iterations the
PR ships as an honest negative/partial with the full log.

## Layout

- `alztest/` — shared harness (conditional ParticleGAN wrapper,
  distribution metrics, baselines) and dataset loaders. Loaders raise
  `DataUnavailable` with a precise reason, so tests **skip cleanly**
  when the data is not present.
- `tests/` — one test module per find.
- `notes/` — one iteration log per find (every round, with numbers).
- `scripts/run_once.py` — run one seed of one test module and print
  JSON margins: `python3 scripts/run_once.py <test_module> <seed> <steps>`

## Running locally

Use an isolated Python 3.11 environment from the repository root. The
paper-watch harness uses the same ParticleGAN release source as the
experimental microglia CLI.

```sh
python -m pip install -r paper_watch/requirements.txt
python -m pytest paper_watch/tests/test_harness.py
```

The regular repository suite discovers `tests/`. The data-dependent
experiments under `paper_watch/tests/` are opt-in. Their scientific
thresholds can fail when the model falls short of the research target.
For a single-seed diagnostic, run `scripts/run_once.py` from
`paper_watch/`, using the module name without the `.py` suffix.

## Data

Datasets are NOT in git. Tests look in `paper_watch/data/` (or env
overrides such as `PSYCHAD_H5AD`); see `alztest/data.py` for the exact
expected files per loader. SEA-AD multiregion files come from the
public `allen-brain-cell-atlas` bucket (Allen Brain Cell Atlas).

## Integrated experiments

| Experiment | Recorded finding |
|---|---|
| [SEA-AD cross-region ordering](notes/seaad_vulnerability_readout.md) | The Gaussian baseline leads at the tested CPU budget |
| [Lipid–lysosome microglia readout](notes/seaad_ll_microglia_readout.md) | The centroid leads. The default gene panel lacks both complete modules |
| [Cognition at matched pathology](notes/seaad_resilience_contrast.md) | Single-fit AUROC 0.580. A larger independent cohort is needed |
| [PsychAD cross-disorder specificity](notes/psychad_cross_disorder.md) | Mean shift and empirical donor sampling beat the 300-step EMPF fit on held-out disease specificity and distribution fit |
| [PsychAD lipid and lysosome programs](notes/psychad_program_readout.md) | With the configuration chosen on training donors, mean shift and empirical sampling matched the held-out program correlation better than EMPF in an exploratory three-split readout. Energy distance differences were within donor uncertainty |

## NeuroKO

The [versioned NeuroKO release](https://zenodo.org/records/23260389) includes
processed neuronal and astrocyte count matrices. The [NeuroKO evaluation](notes/neuroko_revival.md)
documents the release audit and the grouped perturbation tests that follow
closed PR #10.

### NeuroKO GPU comparison

Run from the repository root in Bash on Linux with Python 3.11, Git, curl and an NVIDIA GPU. The reference runs used an RTX A6000, a CUDA 13.0-compatible driver and PyTorch 2.14.1+cu130. Allow space for both H5ADs (13.78 GB total), the CUDA environment and the checkout. The two release wheels below use a shared Python environment and separate import directories. The top-level project's ParticleGAN pin is for a different experiment.

```bash
python3.11 -m venv .venv-neuroko
source .venv-neuroko/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'numpy==1.26.4' 'scipy==1.17.1' 'pandas==2.3.3' \
  'h5py==3.16.0' 'anndata==0.12.7' 'pytest>=8,<10'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'

data=paper_watch/data/neuroko
for version in 0.10.0 0.11.1; do
  python -m pip install --no-deps --target "$data/wheels/$version" "particlegan==$version"
done
python paper_watch/scripts/fetch_neuroko.py --out "$data"
```

The fetcher resumes interrupted transfers and verifies each original H5AD against its Zenodo byte count and MD5. Keep the filenames `iNeurons.h5ad` and `iAstrocytes.h5ad`. The benchmark reads the first 10,000 cells per group for a 60,000-cell pilot. Both files determine the shared target panel. [Versioned NeuroKO data](https://zenodo.org/records/23260389)

The following function uses one GPU and runs the two releases sequentially. Keep these commands in the same Bash session. On a machine with multiple GPUs, set `CUDA_VISIBLE_DEVICES` to the desired device before starting.

```bash
run_neuroko_pair() {
  local version recipe rate
  for version in 0.10.0 0.11.1; do
    case "$version" in
      0.10.0) recipe=empf; rate=0.005 ;;
      0.11.1) recipe=unified_v11; rate=0.028 ;;
    esac
    PYTHONPATH="$data/wheels/$version" python -c \
      'import sys; from importlib.metadata import version; assert version("particlegan") == sys.argv[1]' "$version" || return 1
    PYTHONPATH="$data/wheels/$version" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python -m paper_watch.alztest.neuroko_duration \
        --data "$data" --window-cells 10000 --device cuda:0 --group Screen1_0 --seed 1 \
        --recipe "$recipe" --rate "$rate" --checkpoints "$@" \
        --out "$data/results/$recipe.json" --state-file "$data/results/$recipe.pt" || return 1
  done
}
run_neuroko_pair 300
```

A successful smoke run writes one JSON and one trainer checkpoint per recipe in `paper_watch/data/neuroko/results/`. Check that both receipts report CUDA, the same fit and held-out cell counts, batch size 64, 64 particles, latent dimension 16 and a completed 300-update score. Our reference scores were about **1.470 for EMPF** and **1.742 for tuned Unified v11**. Hardware and numerical differences can affect later scores.

```bash
run_neuroko_pair 300 1200 2400 3600 {4000..40000..1000}
```

Use the long command again after an interruption. It resumes each arm from its saved trainer state. Compare `mean_energy_distance` in the final checkpoint (lower is better), each target's `spread_ratio` (1 matches the observed spread), and `effective_rank`. Report the seed and neuronal holdout with the results. [Reference runs and comparison](notes/neuroko_particlegan_0111_gpu_lr.md)

The 1,200-update hyperparameter study includes matched-capacity EMPF and a three-seed held-out comparison. [Results and reproduction commands](notes/neuroko_cuda_tuning_1200.md)

The fixed-network formulation comparison also pins the particle table, batch size and evaluation draws. Its two-GPU reproduction command covers the released Unified v11 recipe with tuned transport settings, same-wheel EMPF and historical EMPF. [1,200-update formulation comparison](notes/neuroko_formulation_1200.md)

The ParticleGAN 0.12.0 comparison tunes the generator, critic and particle rates and the transport settings on three training groups. It compares the frozen choice with same-wheel v11, v11 from 0.11.1, and EMPF on twelve paired seeds. [v12 benchmark and two-GPU reproduction](notes/neuroko_v12_1200.md)

A fixed-learning-rate follow-up compared two v12 finalists and matched controls at 1,200 updates. On 16 paired seeds the training-selected v12 scored 1.3665, matched v11 scored 1.3584 and matched EMPF scored 1.3531 (lower is better). [Full comparison and reproduction commands](notes/neuroko_v12_refinement.md)

#### Copy-paste prompt for the stronger-prior v12 comparison

> Clone `https://github.com/difflabai/alz-perturb-gan` on branch `paper-watch/neuroko-revival` and record the commit. Follow the NeuroKO GPU setup in `paper_watch/README.md`, including the data checksums, and install the pinned 0.12.0 wheel as shown in `paper_watch/notes/neuroko_v12_1200.md`. Use both GPUs. Run the “Prior penalty at latent noise 0.15” command in `paper_watch/notes/neuroko_v12_refinement.md`. Compare v12, v11 and EMPF at 1,200 updates with 64 particles, 16 latent dimensions, batch size 64, noise 0.15 and native prior penalty weight 0.1 for all three. Keep the generator LR at 0.014, critic LR at 0.021 and particle LR at 0.0406. Verify the flat learning rates and paired input hashes. Report the four-target mean energy, paired wins, spread ratio and fit time against the matched and tuned controls. Report the screening and independent training results separately. Run the biological holdout with `--stage confirm` if the selection qualifies.

The stronger-prior v12 led the screen, then scored 0.9481 on fresh training seeds against 0.9393 for matched EMPF. All matched runs used constant learning rates. Append `--stage compare-releases` to the prior-weight command to repeat the fresh-seed comparison with the original 0.10.0 EMPF and 0.11.1 v11 wheels. [Fresh-seed results and release comparison](notes/neuroko_v12_refinement.md#prior-penalty-at-latent-noise-015)

To continue the search from the same checkout, run the `--study prior-interaction` command. It compares both latent noise levels and prior weights under each of the two fixed learning-rate settings on the same training folds. [Eight configurations and the qualification rule](notes/neuroko_v12_refinement.md#separating-the-effects-of-latent-noise-and-role-rates)

In that comparison the stronger prior helped v12 at noise 0.15 under both rate settings. The next training command, `--study strong-prior-noise`, tests noise 0.25 against matched controls. A separate replay measures how much of the earlier improvement came from training with more noise and how much came from sampling the trained generator with more noise. [Higher-noise comparison](notes/neuroko_v12_refinement.md#increasing-noise-with-the-stronger-prior) · [Training and sampling replay](notes/neuroko_v12_refinement.md#separating-training-noise-from-sampling-noise)

The `--study critic-motion` command tests a fixed critic LR of 0.0035 while keeping generator and particle rates at 0.014 and 0.0406. This rate comes from measuring the change in each critic's scores at the same starting rate. All three formulations receive the reduced critic rate in the matched comparison. [Critic measurements and two-GPU reproduction commands](notes/neuroko_v12_refinement.md#calibrating-the-critic-learning-rate-from-its-measured-updates)

Two earlier replication commands compare the formulations at fixed rates of 0.020, 0.042 and 0.0203, then at EMPF’s tuned rates of 0.020, 0.030 and 0.058. [Paired results and commands](notes/neuroko_v12_refinement.md#replicated-comparison-of-learning-rates-and-latent-noise)

#### Copy-paste prompt for the 1,200-update formulation comparison

> Clone `https://github.com/difflabai/alz-perturb-gan` on branch `paper-watch/neuroko-revival` and record the commit. Follow the NeuroKO GPU setup in `paper_watch/README.md`, then the two-GPU commands in `paper_watch/notes/neuroko_formulation_1200.md`. Compare Unified v11 and EMPF after exactly 1,200 updates on the fixed network, 64 particles, 16 latent dimensions and batch size 64. Verify each pair’s initialization, data and minibatch hashes. Report lower-is-better held-out energy per seed, the paired mean difference, the spread ratio and timing. Identify the release and the actual generator, critic and particle learning rates for every arm.

#### Copy-paste prompt for the 40,000-update comparison

> Reproduce the NeuroKO GPU comparison. Clone https://github.com/difflabai/alz-perturb-gan on branch `paper-watch/neuroko-revival`, record the commit, and follow “NeuroKO GPU comparison” in `paper_watch/README.md`. Set up Python 3.11 and CUDA PyTorch, then download and checksum both NeuroKO H5ADs from Zenodo record 23260389. Run the matched seed-1 neuronal holdout (`Screen1_0`) with 0.10.0 EMPF at 0.005 and 0.11.1 Unified v11 at 0.028. Verify both 300-update smoke runs use CUDA and the same cells. Resume both to 40,000 updates. Save the receipts and checkpoints, then report which model has lower held-out energy distance, how many targets it wins, and how its spread compares with the observed cells.

## Blocked finds

The panAD and GNPC conversions are still waiting for data access. [Blocked finds](notes/blocked_finds.md)
records each access blocker, the defects in the closed designs, and what
a revival would need. HypoAD is tracked in an
[open pull request waiting for data access](https://github.com/difflabai/alz-perturb-gan/pulls?q=is%3Apr+is%3Aopen+label%3A%22waiting+for+data+access%22).
