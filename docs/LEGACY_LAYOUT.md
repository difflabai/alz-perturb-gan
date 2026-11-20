# Legacy repository layout

The repository root holds many files and directories from the 0.9.x research
lineage (2026-11-03 to 2026-11-08) next to the current code. This page explains
what they are. **Nothing has been moved, renamed or deleted.** Hundreds of
committed records pin these files by absolute path and SHA256, so moving them
would break the checks that make the history verifiable. For the current state
of the project, read [STATUS.md](../STATUS.md).

Counts below were taken at commit `a662ad3` (2026-11-20).

## Current code, for comparison

| Path | Role |
| --- | --- |
| `configs/`, `scripts/microglia.py`, `scripts/microglia_runtime.py` | Current EMPF application: config and CLI |
| `tests/test_microglia_*.py` | Tests for the current application |
| `paper_watch/` | Literature-conversion CPU harness, tests and notes |
| `scripts/report/`, `docs/` | Report builder and the published static report |
| `analysis/<study>_<date>/` | Dated studies; see [analysis/INDEX.md](../analysis/INDEX.md) |

Everything else at the root is described below.

## Root legacy families

The root has 169 tracked entries: 71 files and 98 directories.

| Family | Count | What it is |
| --- | ---: | --- |
| `*_runtime/` | 82 directories, 1,249 files | Per-experiment runtime snapshots (launchers, runners, fingerprints, plans) copied before each GPU job. Last changed 2026-11-03 to 2026-11-08. |
| `model/`, `frozen_model/`, `halloween_model/`, `routed_model/`, `stage_model/`, `ema_readout_model/` | 6 directories | Model code snapshots for the corresponding Nov 3-4 experiments |
| `runtime/` | 9 files | The shared runtime helpers those snapshots were copied from. It has since drifted from some of the copies, so a copy, not `runtime/`, is the record of what a given job ran. |
| `gpu1_science_snapshot/` | 18 files | Nov 3 snapshot of a GPU1 experiment's sources |
| `prepare_*.py` | 12 files | Job preparation scripts for the Nov 3-4 calibration-optimization experiments |
| `test_*.py` at the root | 10 files | Tests for those scripts. In a clean checkout, 8 of the 10 fail at collection: they import modules that are not on the root path (for example `pg_adapter`, `bcap_adapter`, `bench_core`; some exist only as copies inside snapshot directories, others not at all) or ParticleGAN itself. |
| Other root `*.py` | 9 files | `calibration.py`, `continuation.py`, `ensemble_execution.py`, `ensemble_scope.py`, `hybrid.py`, `monitor_first_fold.py`, `pure_suite.py`, `selection.py` and `verify_first_fold.py`: Nov 3-4 experiment code |
| Root `*.json` | 33 files | Plans, protocols, incumbents and results of those experiments (`*_plan.json`, `*_protocol.json`, `incumbent.json`, ...) |
| `CONTINUATION.txt`, `PURE_PARTICLEGAN_RESULTS.txt` | 2 files | Nov 4 handoff and results notes. `CONTINUATION.txt` is superseded by `STATUS.md`. |
| `src/alz_perturb_gan/`, `tests/` (non-microglia files), `results/smoke_results.json`, `pyproject.toml` | | The earlier synthetic perturbation smoke project (ParticleGAN 0.8.0); see [its archived README](archive/early-perturbation-smoke.md) |

The root `pyproject.toml` installs only the smoke project and pins
ParticleGAN 0.8.0. It is not the environment for the current model; see
[analysis/default_empf_20261117/README.md](../analysis/default_empf_20261117/README.md)
for the runtime setup.

## Why the files stay where they are

- More than 100 tracked JSON records name root-level scripts by path, and more
  than 450 name files inside `*_runtime/` directories.
- About 790 tracked JSON records name top-level files in `analysis/`, and
  `scripts/report/build_report.py` reads several large JSON inputs from there.
- Many records also store the SHA256 of the script that produced them and
  refuse to verify if it changes.

A move would need a path map (old path, new path, hash, last commit) and a
pinned commit for the old layout, so that every record stays checkable. That
is a possible later step, not part of this page.

## Intentional copies

The 82 `*_runtime/` directories hold 1,249 tracked files but only 784 distinct
contents. For example, one version of `project_runner_gpu0.py` appears in 74
directories. Under `analysis/`, the `model/` folders of the 0.9.7 studies and
the `improve_20261109/r2`-`r5` rounds repeat the same files; the
[analysis index](../analysis/INDEX.md#byte-identical-copies) lists the largest
groups.

These copies are deliberate: each experiment snapshotted the code it executed,
and its manifest pins that snapshot by hash. Do not deduplicate them. New
experiments should import shared code from a single versioned location, or
record the source by commit and hash, rather than add another copy.

## Local-only directories

`runs/`, `artifacts/` and `data/` are git-ignored. `runs/` holds training and
generation outputs (including the retained EMPF fits), `artifacts/` holds the
isolated ParticleGAN native releases and calibration inputs, and `data/` holds
raw cells. Some tests and runners need them and pass only on the machine that
has them.
