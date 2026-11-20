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

The harness tests also check the statistics helpers: the unbiased energy
distance, train-only PCA, donor-grouped permutation p-values and donor
bootstrap intervals. The SEA-AD readouts score depth-matched donors and
report these p-values and intervals. Their recorded notes predate that
change.

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
| [Cognition at matched pathology](notes/seaad_resilience_contrast.md) | Exploratory. The committed 3-fit ensemble recorded mean AUROC 0.557 (the iteration 1 single fit recorded 0.580), within about one null SD of 0.5. Not rerun with donor-grouped inference. A larger independent cohort is needed |
| [PsychAD cross-disorder specificity](notes/psychad_cross_disorder.md) | Mean shift and empirical donor sampling beat the 300-step EMPF fit on held-out disease specificity and distribution fit |
| [PsychAD lipid and lysosome programs](notes/psychad_program_readout.md) | Longer EMPF fits reduced spurious program correlation. The two diagnostics favored different checkpoints |

## Blocked finds

The NeuroKO, panAD, and GNPC conversions were closed without a fit
because their data is not accessible. [Blocked finds](notes/blocked_finds.md)
records each access blocker, the defects in the closed designs, and what
a revival would need. HypoAD is tracked in an
[open pull request waiting for data access](https://github.com/difflabai/alz-perturb-gan/pulls?q=is%3Apr+is%3Aopen+label%3A%22waiting+for+data+access%22).
