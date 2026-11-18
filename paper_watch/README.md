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
overrides such as `NEUROKO_H5AD`); see `alztest/data.py` for the exact
expected files per loader. SEA-AD multiregion files come from the
public `allen-brain-cell-atlas` bucket (Allen Brain Cell Atlas).

## Finds converted so far

| Find | Route | Status |
|---|---|---|
| NeuroKO CRISPR screen (Dominicus et al. 2026) | new distribution + target | blocked: printed Zenodo DOI does not resolve |
| SEA-AD multiregional vulnerability ordering (Travaglini et al. 2026) | new distribution + readout | loop running (iteration 8 of 10) |
| panAD atlas + MONET subtypes (Rahimzadeh et al. 2026) | new target | blocked: no open download of integrated object |
| PsychAD cross-disorder atlas (Lee et al. 2026) | new target | blocked: Synapse DUA required |
| GNPC longitudinal plasma proteomics (2026) | new distribution + target | blocked: consortium access |
| LL microglial remodeling axis (Arif et al. 2026) | new target + readout | loop running (iteration 2 of 10) |

Full briefs live in the watch ledger (kept outside this repo).
