# Paper watch — test conversions

This directory holds the ALZ Perturb GAN **paper watch** test
conversions. The watch scans new Alzheimer's research (bioRxiv/medRxiv,
arXiv, PubMed, dataset hubs) on a Mon/Wed/Fri cadence and judges every
find by one question: does it give us a distribution to model, tell us
which distribution matters, or help us read what the model learned?

Top-ranked finds are converted here into small CPU tests — one PR per
find, research-note style, **nothing merged by the watch**. Each test
uses the current company-standard ParticleGAN configuration
**unmodified** (latest particlegan / EMPF public API); refinements
inside a test's 10-iteration loop may touch data handling,
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

## Data

Datasets are NOT in git. Tests look in `paper_watch/data/` (or env
overrides such as `NEUROKO_H5AD`); see `alztest/data.py` for the exact
expected files per loader. SEA-AD multiregion files come from the
public `allen-brain-cell-atlas` bucket (Allen Brain Cell Atlas).

## Integrated experiments

| Experiment | Recorded finding |
|---|---|
| [SEA-AD cross-region ordering](notes/seaad_vulnerability_readout.md) | The Gaussian baseline leads at the tested CPU budget |
| [Lipid–lysosome microglia readout](notes/seaad_ll_microglia_readout.md) | The centroid leads. The default gene panel lacks both complete modules |
| [Cognition at matched pathology](notes/seaad_resilience_contrast.md) | Single-fit AUROC 0.580. A larger independent cohort is needed |

The NeuroKO, panAD, PsychAD, and GNPC designs remain in their
[closed pull requests](https://github.com/difflabai/alz-perturb-gan/pulls?q=is%3Apr+is%3Aclosed+%22Paper+watch%22).
