"""Checkout-relative EMPF host for the application runtime.

analysis/empf_comparison_20261117/host.py derives the native EMPF trainer from
the round-5 trainer. The round-5 trainer puts two absolute /ml2/codex-storage
directories on sys.path, from which the runtime imports pg_model, critic, usage,
contracts and stage_law (SCIENCE) and bench_core (BENCH; via contracts.B, which
analysis/faithful097/model/released_*.py put first on sys.path). The 0.9.7
release_env pins SITE to /ml2/alz. That host belongs to a concluded study whose
source manifest pins every .py file in its directory, so it stays unchanged.

vendored/ holds byte-identical copies of those six modules. This module applies
the same derivation as host.activate (it reuses host.replace_once and
adapter.derive) plus these replace_once rewrites, so every path that the
derived files name is inside the current checkout:

  trainer.py      MODEL, SCIENCE, BENCH, RUNS -> ROOT-based paths; SCIENCE and
                  BENCH both name vendored/
  release_env.py  SITE -> ROOT/artifacts/native-releases/particlegan-0.10.0/site-packages
  contracts.py    B -> vendored/ (derived from vendored/contracts.py and
                  registered as the `contracts` module before the trainer loads)

The derived files go to a new directory (alz-host-empf-<lane>-vendored), so
the existing alz-host-empf-<lane> files and their "existing derived host
differs" check are untouched. Model code is unchanged: only import locations
differ, and every vendored file keeps the sha256 recorded for its original.
"""
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT / 'analysis/empf_comparison_20261117'
VENDORED = HERE / 'vendored'
VENDORED_FILES = ('pg_model.py', 'critic.py', 'usage.py', 'contracts.py', 'stage_law.py', 'bench_core.py')
HEADER = '# Derived by analysis/default_empf_20261117/vendored_host.py; see derivation.json.\n'

# Exact lines in the unchanged originals that name a location outside the checkout.
TRAINER_MODEL = "MODEL = Path('/ml2/alz/analysis/faithful097/model')"
TRAINER_SCIENCE = "SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')"
TRAINER_BENCH = "BENCH = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')"
TRAINER_RUNS = "RUNS = Path('/ml2/alz/runs/improve_20261109')"
RELEASE_SITE = "SITE=Path('/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages')"
CONTRACTS_B = "B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')"


def _host():
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    import host
    return host


def runtime_dir(lane):
    return _host().RELEASE / ('alz-host-empf-' + lane + '-vendored')


def derive_bodies(lane='native'):
    """Return ({name: derived body}, {name: original path}) without touching the filesystem."""
    if lane not in ('native', 'matched'):
        raise ValueError('unknown EMPF lane')
    H = _host()
    originals = {'release_env.py': H.ROOT/'analysis/faithful097/model/release_env.py',
                 'contracts.py': VENDORED/'contracts.py',
                 'trainer.py': H.OLD/'trainer.py', 'generate.py': H.OLD/'generate.py'}
    bodies = {n: p.read_text() for n, p in originals.items()}
    # The same derivation as host.activate ...
    bodies['release_env.py'] = bodies['release_env.py'].replace('particlegan-0.9.7', 'particlegan-0.10.0').replace("VERSION='0.9.7'", "VERSION='0.10.0'")
    bodies['trainer.py'] = H.replace_once(bodies['trainer.py'],
        'HERE = Path(__file__).resolve().parent', f'HERE = Path({str(H.OLD)!r})')
    bodies['trainer.py'] = H.replace_once(bodies['trainer.py'],
        "DEFAULT_OUT_ROOT = RUNS / 'fits_r5'", f'DEFAULT_OUT_ROOT = Path({str(H.RUNS/"fits"/lane)!r})')
    from adapter import derive
    bodies['trainer.py'] = derive(bodies['trainer.py'], lane)
    # ... plus the checkout-relative locations.
    for before, after in ((TRAINER_MODEL, f"MODEL = Path({str(H.ROOT/'analysis/faithful097/model')!r})"),
                          (TRAINER_SCIENCE, f'SCIENCE = Path({str(VENDORED)!r})'),
                          (TRAINER_BENCH, f'BENCH = Path({str(VENDORED)!r})'),
                          (TRAINER_RUNS, f"RUNS = Path({str(H.ROOT/'runs/improve_20261109')!r})")):
        bodies['trainer.py'] = H.replace_once(bodies['trainer.py'], before, after)
    bodies['release_env.py'] = H.replace_once(bodies['release_env.py'], RELEASE_SITE,
        f"SITE=Path({str(H.RELEASE/'site-packages')!r})")
    bodies['contracts.py'] = H.replace_once(bodies['contracts.py'], CONTRACTS_B, f'B=Path({str(VENDORED)!r})')
    return {n: HEADER + b for n, b in bodies.items()}, originals


def _write_atomic(path, text):
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def write_host(lane='native'):
    """Write (or confirm) the derived host files and derivation.json; return (directory, provenance).

    Missing files are written atomically; an existing file that differs is an error, never overwritten.
    """
    H = _host()
    bodies, originals = derive_bodies(lane)
    runtime = runtime_dir(lane); runtime.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in bodies.items():
        path = runtime/name
        if path.exists():
            if path.read_text() != expected:
                raise RuntimeError('existing derived EMPF host differs: ' + str(path))
        else:
            _write_atomic(path, expected)
        records[name] = dict(original=str(originals[name]), original_sha256=H.sha(originals[name]),
                             derived=str(path), derived_sha256=H.sha(path))
    vendored = {name: dict(path=str(VENDORED/name), sha256=H.sha(VENDORED/name)) for name in VENDORED_FILES}
    provenance = dict(release='0.10.0', commit=H.COMMIT, lane=lane,
                      host_derivation_sha256=H.sha(H.__file__), vendored_host_sha256=H.sha(__file__),
                      vendored=vendored, files=records)
    _write_atomic(runtime/'derivation.json', json.dumps(provenance, indent=2)+'\n')
    return runtime, provenance


def activate(lane='native'):
    """Derive (or confirm) the checkout-relative host and import its trainer and generator.

    Callers that may start together must serialize this (microglia_runtime holds a file lock).
    """
    H = _host()
    if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env', *(n[:-3] for n in VENDORED_FILES))):
        raise RuntimeError('EMPF lane requires a fresh subprocess')
    runtime, provenance = write_host(lane)
    H.load('release_env', runtime/'release_env.py')
    H.load('contracts', runtime/'contracts.py')
    return H.load('trainer', runtime/'trainer.py'), H.load('generate', runtime/'generate.py'), provenance
