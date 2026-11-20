# Running the tests

From the repository root:

```bash
python -m pytest            # or plain `pytest`
```

pytest 8.4 or newer is required (`minversion` in `pyproject.toml`). Older
versions load the `-p alz_pytest_plugin` plugin before applying `pythonpath`
and stop at startup with an ImportError.

The default selection needs only `numpy` and `pytest` (CI installs `numpy==1.26.4 pytest==9.1.1`; the report hash check, `scripts/report/build_report.py --check`, additionally needs `scripts/report/requirements.txt`). It collects `tests/` and `paper_watch/tests/`, as configured in `pyproject.toml`. What it does:

- runs the microglia default/runtime/interval tests in `tests/test_microglia_*.py`;
- deselects the legacy synthetic smoke tests (`-m "not legacy_pg080"`). Their modules are still imported during collection, so when torch or ParticleGAN 0.8.x is missing they also appear in the `-rs` summary as skipped with that reason;
- in the files that are known to need optional packages, reports a test module, or test body, that needs a missing optional package as **skipped, with the package named**. Those files are `paper_watch/tests/`, the seven `legacy_pg080` modules and `test_empf_conditional_probe.py` (`OPTIONAL_DEPENDENCY_SCOPE` in the plugin). The packages are torch, ParticleGAN, scipy, pandas, anndata, h5py and scikit-learn. Every other test file, including the core `test_microglia_*.py` and `test_repo_*.py` tests and any new file, stays strict: a missing package there is an error, not a skip. Any other import failure is always an error;
- skips paper-watch data tests with their own `DataUnavailable` reason when the public file is not downloaded.

When ParticleGAN is not installed, the repository plugin ([`support/alz_pytest_plugin.py`](support/alz_pytest_plugin.py)) appends the native release target `artifacts/native-releases/particlegan-0.10.0/site-packages` to `sys.path` if it exists. Set `ALZ_PARTICLEGAN_0_10_SITE_PACKAGES` to use another target. The session header prints which ParticleGAN was used.

## Other selections

| Command | What it runs |
| --- | --- |
| `pytest -m legacy_pg080` | Legacy `src/alz_perturb_gan` tests. They need torch and ParticleGAN 0.8.x ([`envs/legacy-pg080.txt`](../envs/legacy-pg080.txt)). Under the native 0.10.0 release, `test_gan.py` and `test_smoke.py` are skipped with the version found. |
| `pytest paper_watch/tests` | Paper-watch tests. They need [`envs/paper-watch.txt`](../envs/paper-watch.txt) or the native ParticleGAN target, plus their public data files. |
| `pytest runtime` | Read-only GPU0 calibration binding tests. They need the local workspace. |

The `runtime/` tests are not in the default selection. Outside the original workspace they are skipped with the missing path named:

- All of them read absolute `/ml2/codex-storage/...` paths and the deployed calibration authorization.
- Seven of them build a full calibration source closure. That closure reads the ignored `data/calibration_fold0` inventory at the absolute path the deployed authorization pins. These seven pass only in the checkout that authorization names.

The plugin lives in `tests/support/` and is loaded with `-p`, so no new `conftest.py` is added to the repository root or to `runtime/`. `runtime/binding.py` requires every `*.py` file in those two directories to be pinned in a calibration authorization.

## Markers

| Marker | Meaning |
| --- | --- |
| `legacy_pg080` | Older synthetic smoke study. Deselected by default. |
| `needs_local_workspace` | Needs ignored local inventories or absolute workspace paths. Applied to `runtime/` tests. Skipped with a reason when the workspace is absent, never silently deselected. |
| `needs_particlegan` | Needs the frozen ParticleGAN 0.10.0 site-packages (`tests/test_empf_conditional_probe.py`). Skipped when absent. |
| `needs_data` | Registered for tests that need a local or downloaded dataset. The current paper-watch tests skip through their own data checks. |
| `gpu` | Needs CUDA. Never run in CI. |

## Not collected

The `test_*.py` files in the repository root are protocol scripts for earlier GPU0 calibration studies; `runtime/binding.py` pins them by hash. Eight of the ten import adapters or ParticleGAN builds that are not on any path in a clean checkout. They are not part of any selection, and this setup leaves them unchanged.
