# Project status

Updated 2026-11-20. This is the single current index of where the project
stands. Older handoff files (`CONTINUATION.txt`, `analysis/*CONTINUATION*.txt`,
`analysis/empf_comparison_20261117/TAKEOVER.txt`,
`analysis/particlegan_upgrade_20261117/README.txt` and `TAKEOVER.txt`) are
historical. Each now starts with a SUPERSEDED line pointing here; their content
is unchanged. If this page disagrees with a committed result record (a JSON
receipt or completion file), the record is right and this page needs fixing.

## Current default

The CLI default ([config](configs/microglia_default.json),
[CLI](scripts/microglia.py)) is `microglia-empf-countshape-v1`: ParticleGAN
0.10.0, native EMPF preset, `pbulk`, `hvg2000`, 20,000 updates, `lazy_k=1`,
generator/critic/prior learning rates 0.012/0.018/0.03. The readout is
`cellshape@0.5:biascal:stagehalf`.

- Status `owner-selected-experimental`. The default did not pass the
  prospective promotion rule and is not population-qualified
  ([selection record](analysis/default_empf_20261117/selection.json)).
- `generate` loads **one** retained fit: fold 0, seed 0 (25 training donors, 12
  held-out development donors). It is not a full-cohort refit.
- The published report and its biological findings (CLEC5A and others) still
  describe the frozen ParticleGAN 0.9.7 models. They were not recomputed with
  EMPF.

## Latest results

All of these come from the same single fold and seed (fold 0, seed 0) and the
same 12 held-out development donors.

- **EMPF against the previous 0.9.7 default**
  ([comparison](analysis/empf_comparison_20261117/completion/comparison.json)):
  profile RMSE, SD interval score and combined interval score improve by 1.57%,
  2.36% and 0.67%. Gene mean, detection and log-variance errors and coverage
  get worse.
- **Empirical resampling still wins** both interval scores (SD .6186 against
  .8063 for EMPF; combined .8448 against .9452). The broad goal of beating
  empirical resampling is not met.
- **Coverage pilot** ([README](analysis/coverage_empf_20261117/README.md)): a
  quarter-contraction readout raises SD coverage 58.3% to 62.2% and combined
  coverage 69.2% to 71.5% on the same weights. SD-only calibration on disjoint
  six-donor halves widens the SD bounds: SD and combined coverage reach 86.7%
  and 83.7% on top of the quarter readout, and 85.6% and 82.8% on the default
  readout. With six calibration donors and a nominal 80% level, the conformal
  rank is 6 (the largest score), so the nominal split-conformal level is 6/7
  (85.7%), not 80%. That level is marginal and per coordinate, and assumes
  exchangeable donors; these donors were also reused during development. The
  default readout is unchanged.
- **Conditional-residual probe**
  ([README](analysis/particlegan_review_20261117/README.md)): in a controlled
  two-group CPU example, EMPF's transport residual reports a perfect match after
  two conditions swap their spreads when inputs are wider than 64 features
  (tested at 65 and 2,150). The default trains with 2,150. Its
  effect on trained samples, including on the undercoverage above, has not
  been tested. `tests/test_empf_conditional_probe.py` pins the probe values
  against the frozen 0.10.0 release, with two strict expected failures for the
  wide widths.

## In progress: fold/seed confirmation

[analysis/coverage_empf_confirm_20261120](analysis/coverage_empf_confirm_20261120/PROTOCOL.md)
was declared in commit `d578e83` before any of its fits existed and started on
2026-11-20. It trains 17 new native EMPF fits (folds 0-8, seeds 0-1, without the
fold 0/seed 0 pilot) through the committed CLI, then scores the default readout,
the quarter-contraction readout and SD calibration for each fit against gates
fixed in the protocol. It runs on the shared GPUs with outputs under the
git-ignored `runs/coverage_empf_confirm_20261120/` and the CLI fit root.

As of this update, no confirmation result exists. Until it finishes, every
statement above rests on one fold and one seed. While it runs, do not edit
`configs/microglia_default.json`, `scripts/microglia*.py`, or any file listed
in [analysis/default_empf_20261117/runtime_source_manifest.json](analysis/default_empf_20261117/runtime_source_manifest.json)
or the confirmation's own source manifest: the CLI and the scorer check them by
hash. That manifest is not yet committed; it will land at
`analysis/coverage_empf_confirm_20261120/source_manifest.json`, and this list
should be rechecked against it then. The protocol also records a start-up race in the EMPF host, handled by
staggering process starts and rerunning any step that stops on it.

## Open experiments (not started)

- A training comparison for the wide-width conditional residual: native EMPF
  against one change to how it measures conditional discrepancy, first on a
  synthetic grouped distribution, then on ALZ fold 0/seed 0.
- The PPARG test: matched fresh training that changes what the critic observes
  while keeping the ParticleGAN update rule, with the count-gradient ablation
  kept separate (see the PPARG section of the [README](README.md)).
- Recomputing the published report with EMPF.

## Pull requests and paper watch

As of 2026-11-20:

- #18, the coverage pilot, wide-width probe and confirmation protocol, is open
  as a draft against `main`.
- Paper-watch conversions merged: #9 (scaffolding), #11 (SEA-AD vulnerability
  ordering), #13 (PsychAD specificity), #15 (lipid-lysosome microglia), #16
  (SEA-AD resilience). Their findings are listed in
  [paper_watch/README.md](paper_watch/README.md).
- Paper-watch conversions closed unmerged on 2026-11-20 as data- or
  access-blocked, following owner comments from 2026-11-18: #10 NeuroKO, #12
  panAD and #14 GNPC.
- Still open and blocked: #17 HypoAD (access/format), against `main`.

## Run the checks

From the repository root, with the research interpreter:

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
# Default data-free selection: tests/ and paper_watch/tests, legacy tests deselected.
# Missing optional packages, data and workspace paths are skips with reasons
# (tests/README.md). The isolated ParticleGAN 0.10.0 release is found automatically.
$PY -B -m pytest -p no:cacheprovider
# EMPF wide-width probe alone (frozen 0.10.0 release; 2 expected failures are by design):
$PY -B -m pytest -p no:cacheprovider tests/test_empf_conditional_probe.py
# Read-only GPU0 calibration binding tests (this workspace only):
$PY -B -m pytest -p no:cacheprovider runtime
# Published report consistency:
$PY -B scripts/report/build_report.py --check
# Analysis index freshness:
$PY -B scripts/hygiene/index_analysis.py --check
# Interpreter against the pinned research recipe:
$PY -B scripts/check_env.py envs/microglia-research-py311.txt
```

The legacy smoke tests run with `pytest -m legacy_pg080` and need ParticleGAN
0.8.x ([envs/legacy-pg080.txt](envs/legacy-pg080.txt)). CPU-only GitHub
Actions checks are defined in
[.github/workflows/checks.yml](.github/workflows/checks.yml); they use no data,
checkpoints or GPU.

## Reading order

1. This page.
2. [README.md](README.md): the project, the default and the frozen report findings.
3. [analysis/default_empf_20261117/README.md](analysis/default_empf_20261117/README.md): runtime setup and checkpoint paths.
4. [analysis/coverage_empf_20261117/README.md](analysis/coverage_empf_20261117/README.md) and [analysis/coverage_empf_confirm_20261120/PROTOCOL.md](analysis/coverage_empf_confirm_20261120/PROTOCOL.md).
5. [analysis/particlegan_review_20261117/README.md](analysis/particlegan_review_20261117/README.md).
6. [analysis/INDEX.md](analysis/INDEX.md) and [docs/LEGACY_LAYOUT.md](docs/LEGACY_LAYOUT.md) to find older records.
