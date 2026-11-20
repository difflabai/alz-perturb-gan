# Environment recipes

Three separate environments are used in this repository. They are not interchangeable.

| Recipe | Used for | ParticleGAN |
| --- | --- | --- |
| [`microglia-research-py311.txt`](microglia-research-py311.txt) | Human-microglia native EMPF default: `scripts/microglia.py`, `analysis/*_2026111*` | 0.10.0 at commit `952c3b6`, installed into `artifacts/native-releases/particlegan-0.10.0/site-packages`, not into the environment |
| [`legacy-pg080.txt`](legacy-pg080.txt) | Older synthetic smoke study in `src/alz_perturb_gan` (tests marked `legacy_pg080`) | 0.8.0 from PyPI, as pinned in `pyproject.toml` |
| [`paper-watch.txt`](paper-watch.txt) | `paper_watch/` tests and readouts | 0.10.0 at commit `952c3b6`, installed into the environment from git |

The microglia recipe pins the exact versions observed in the research interpreter on 2026-11-20 ([record](../analysis/default_empf_20261117/runtime_environment_0.10.0.json)). The other two give ranges, because exact versions were not recorded for them. Paper-watch needs pandas >= 2.2 and anndata, and the research interpreter has pandas 1.5.3 and no anndata. Keep it as a separate environment.

## Microglia research environment

```bash
python3.11 -m venv .venv-microglia
.venv-microglia/bin/pip install -r envs/microglia-research-py311.txt
.venv-microglia/bin/python -m pip install --no-deps --no-compile \
  --target artifacts/native-releases/particlegan-0.10.0/site-packages \
  'particlegan @ git+https://github.com/HyperGAN/ParticleGAN.git@952c3b62ba434f7da194067923351ff5748ef8dd'
.venv-microglia/bin/python -B scripts/check_env.py envs/microglia-research-py311.txt
```

The torch pin is the CUDA 13.0 build. Use `--allow-local-variant` with `check_env.py` when a CPU build of the same version was installed on purpose. `scripts/microglia.py config` and `--dry-run` need none of these packages. Training and generation also need the machine-local caches, checkpoints and the external model helpers named in [`runtime_source_manifest.json`](../analysis/default_empf_20261117/runtime_source_manifest.json). A recipe cannot supply those.

## check_env.py

`scripts/check_env.py RECIPE` reads a recipe as a pip requirements file:

- `name==version` lines must match exactly.
- Range lines are checked when `packaging` is importable.
- `name @ git+...@sha` lines are compared with the recorded install commit.
- `-r` includes are followed.

`# check_env:` comment lines add a Python version check and the isolated native-target check. The script exits 1 on any mismatch. It is a readable early check. The model loaders still verify the native version, origin and pinned source hashes themselves.
