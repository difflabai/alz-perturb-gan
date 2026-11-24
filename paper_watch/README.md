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

#### Copy-paste prompt for v12 versus the original v11 and EMPF releases

> Clone `https://github.com/difflabai/alz-perturb-gan` on branch `paper-watch/neuroko-revival` and record the commit. Follow the NeuroKO GPU setup in `paper_watch/README.md` to fetch and verify the public data and install the 0.10.0, 0.11.1 and 0.12.0 wheels in separate directories. Run the command below on two CUDA GPUs. Compare all five configurations at 1,200 updates with fixed learning rates. Report the four-target mean energy for each seed cohort, and how often v12 beat each original release at shared settings and at its independently tuned settings.

```bash
data=paper_watch/data/neuroko
python -m pip install --no-deps --target "$data/wheels/0.12.0" 'particlegan==0.12.0'
python paper_watch/scripts/neuroko_rate_noise.py --study original-v11-critic --stage compare-published-rates \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --output "$data/calibrated-original-releases"
```

This command reads the published calibration and computes the fits directly. The tuned controls had the lowest first-cohort mean among seven original v11 configurations and eight original EMPF configurations. In the output directory, `comparison_protocol.json` records the settings and `first_comparison.json` and `second_comparison.json` contain the paired scores for seeds 22–24 and 251–253. All six seeds use the same three development groups. [Control selection and scores](notes/neuroko_original_control_selection.json) · [Original-release results](notes/neuroko_v12_refinement.md#comparisons-with-the-original-releases)

#### Check whether v12 closes the gap by 3,600 updates

> Use the same checkout, data and three wheels as the original-release comparison above. Run all five configurations on both GPUs using the command below. Compare mean energy distance, generated-to-observed spread and effective covariance rank at 1,200, 2,400 and 3,600 updates, separately for training seeds 22 and 251. Lower energy is better. Each curve must reproduce its published 1,200-update energy. Report v12 against the rate-matched original controls and the independently tuned original controls.

```bash
python paper_watch/scripts/neuroko_convergence.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-convergence-3600"
```

The script checks the learning rates at every optimizer step, runs all three development groups, and writes the full comparison to `comparison.json` in the output directory. The 3,600-update endpoint is fixed before training. To run the same five configurations on six fresh seeds, add `--fresh` and use a separate output directory such as `"$data/v12-convergence-3600/fresh-838-843"`. This produces 18 paired group–seed comparisons and reports the mean difference against each original control at 3,600 updates.

To isolate the optimizer, compare v12 and v11 inside release 0.12.0. The following command reuses the pilot v12 fits and trains six matched v11 fits. For the six-seed replication, set the source to `"$data/v12-convergence-3600/fresh-838-843"` and choose a separate output directory. Run the replication whatever the pilot ranking. Both comparisons check the resolved wheel settings and pair the cells, initialization and minibatches at all three checkpoints.

```bash
python paper_watch/scripts/neuroko_convergence.py --same-wheel-from "$data/v12-convergence-3600" \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --output "$data/v12-optimizer-3600/pilot"
```

To give matched original EMPF the same optimizer smoothing (0.0061) as v12 and original v11, run the command below. It trains six EMPF fits, reuses the other four configurations, and verifies the pilot’s published 1,200-update energies and input fingerprints. The primary endpoint is the four-target mean energy distance at 3,600 updates. To repeat the comparison on the six fresh seeds, use the completed `fresh-838-843` source directory and a separate output.

```bash
python paper_watch/scripts/neuroko_convergence.py --shared-smoothing-from "$data/v12-convergence-3600" \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --output "$data/v12-smoothing-3600/pilot"
```

Once the six-seed original-release comparison finishes, use its results to decide whether to evaluate Screen1_0. The command verifies the 90 training records and checks that v12 beat every control on mean energy distance at 3,600 updates. If it did, all five configurations train on the three development groups and score Screen1_0 using seeds 844–859. The report includes each target and a paired 95% interval across training seeds for each comparison.

```bash
python paper_watch/scripts/neuroko_convergence.py --confirm-from "$data/v12-convergence-3600/fresh-838-843" \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --output "$data/v12-convergence-3600/heldout-844-859"
```

To test the same holdout at shared smoothing, pass the completed matched-smoothing replication as well. V12 must beat all four controls in that replication to qualify. Use the same output directory so the two comparisons can reuse the 64 identical fits across the 16 holdout seeds. The matched-smoothing decision and results are in its `smoothing-0.0061/` subdirectory.

```bash
python paper_watch/scripts/neuroko_convergence.py --confirm-from "$data/v12-convergence-3600/fresh-838-843" \
  --smoothing-replication "$data/v12-smoothing-3600/fresh-838-843" --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-convergence-3600/heldout-844-859"
```

#### Follow v12 and the four original controls to 40,000 updates

> Use the same checkout, NeuroKO data and pinned wheels. Run the five configurations on the first development group, Screen1_1, using pilot seeds 22 and 251 on separate GPUs. Compare energy distance at 40,000 updates and the average over the final 10,000 updates. Check whether spread and effective covariance rank move closer to the observed cells. Keep the learning rates constant and report each seed as well as the two-seed mean.

```bash
python paper_watch/scripts/neuroko_convergence.py --forty-thousand --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-convergence-40000"
```

The v12 generator, critic and particles use fixed learning rates of 0.014, 0.0049 and 0.0406. The released recipe sets both floors of its `cosine` schedule to 1.0, so the rates stay flat. The runner stops if a parameter group's learning rate changes at any optimizer step. [Per-step checks on both GPUs](notes/neuroko_v12_convergence_40000.json)

The run scores every 1,200 updates and again at 40,000. It checks each arm against its published 1,200-, 2,400- and 3,600-update scores and minibatch fingerprints. The last interval is 400 updates. The late-training mean uses linear interpolation between checkpoints to account for the shorter interval. [Observed cell ranks by group and target](notes/neuroko_v12_tuned_empf_geometry.json)

Once the two 40,000-update v12 fits have finished, use them to compare the v12 and v11 optimizers in release 0.12.0 on the same hardware, initialization and batches. This command trains two additional v11 fits and checks both optimizers against their published 3,600-update trajectories. The report includes the 40,000-update endpoint for each seed and the average energy over the final 10,000 updates.

```bash
python paper_watch/scripts/neuroko_convergence.py --same-wheel-from "$data/v12-convergence-40000" \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --output "$data/v12-optimizer-40000"
```

To compare the original releases through 40,000 at shared smoothing of 0.0061, use the original long run and the 3,600-update shared-smoothing pilot above. The command trains two additional original EMPF fits, reuses the other four configurations, and checks all five trajectories against their shorter runs. It reports both seeds at the endpoint and their time-weighted mean over the final 10,000 updates.

```bash
python paper_watch/scripts/neuroko_convergence.py --shared-smoothing-from "$data/v12-convergence-40000" \
  --smoothing-pilot "$data/v12-smoothing-3600/pilot" --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-smoothing-40000"
```

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
