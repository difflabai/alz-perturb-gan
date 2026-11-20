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

The items below come from the single pilot fold and seed (fold 0, seed 0) and
its 12 held-out development donors; see the fold/seed confirmation section for
the 17-fit results.

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

## Fold/seed confirmation (completed 2026-11-20)

[analysis/coverage_empf_confirm_20261120](analysis/coverage_empf_confirm_20261120/README.md),
declared in commit `d578e83` before any of its fits existed: 17 new native EMPF
fits (folds 0-8, seeds 0-1, without the pilot), all trained and scored, no
failures. The scorer reproduces the pilot exactly.

- **Quarter-contraction readout: PASS.** All ten pre-declared gates pass on the
  mean and both interval scores improve in every repeat. SD interval score
  .720 to .640 (15/17 fits), combined .937 to .898 (16/17), SD coverage 60.2%
  to 63.5%, combined 70.5% to 72.5%, profile RMSE unchanged. Costs inside the
  limits: slightly more PC-space over-dispersion and cell energy +.009.
- **SD calibration: CONFIRMED** on both readouts. On the default readout SD
  score .720 to .605 and combined .937 to .879 (17/17 each), coverage 87.5% SD
  and 84.1% combined (nominal level about 86-88% with six or seven
  calibration donors).
- **EMPF against the previous 0.9.7 default does not hold up across folds**
  (descriptive): profile RMSE .4895 against .4853 (EMPF better in 6/17), SD
  score .720 against .710 (6/17), combined .937 against .923 (7/17), all within
  the previous default's own stream-to-stream spread. The pilot's advantage
  was specific to fold 0/seed 0.
- Against stage-blind empirical resampling: raw readouts still lose the
  combined score (.898 for the quarter readout against .876); the calibrated
  default readout beats it on SD (.605 against .665) and ties on combined
  (.879 against .876).

No default was changed. Whether to adopt the quarter readout and/or the SD
calibration, and whether to keep EMPF as the default, are owner decisions.

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
# Current application and repository documentation (CPU, no data):
$PY -B -m pytest -p no:cacheprovider -q tests/test_microglia_*.py tests/test_repo_docs.py
# EMPF wide-width probe (frozen 0.10.0 release; 2 expected failures are by design):
$PY -B -m pytest -p no:cacheprovider -q tests/test_empf_conditional_probe.py
# Paper-watch harness (needs the isolated ParticleGAN 0.10.0 release; data tests skip):
PYTHONPATH=artifacts/native-releases/particlegan-0.10.0/site-packages \
  $PY -B -m pytest -p no:cacheprovider -q paper_watch/tests
# Published report consistency:
$PY -B scripts/report/build_report.py --check
# Analysis index freshness:
$PY -B scripts/hygiene/index_analysis.py --check
```

Plain `pytest tests` does not work in this environment: the legacy smoke tests
in `tests/` need the root `src/` package and ParticleGAN 0.8.0. The repository
has no CI.

## Reading order

1. This page.
2. [README.md](README.md): the project, the default and the frozen report findings.
3. [analysis/default_empf_20261117/README.md](analysis/default_empf_20261117/README.md): runtime setup and checkpoint paths.
4. [analysis/coverage_empf_20261117/README.md](analysis/coverage_empf_20261117/README.md) and [analysis/coverage_empf_confirm_20261120/README.md](analysis/coverage_empf_confirm_20261120/README.md).
5. [analysis/particlegan_review_20261117/README.md](analysis/particlegan_review_20261117/README.md).
6. [analysis/INDEX.md](analysis/INDEX.md) and [docs/LEGACY_LAYOUT.md](docs/LEGACY_LAYOUT.md) to find older records.
