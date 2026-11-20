"""Checkout-relative, versioned EMPF host for the application runtime.

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

Fit-lifecycle patches (also replace_once, so the pinned round-5 trainer.py and
generate.py stay byte-identical; see FIT_LIFECYCLE_PATCHES):

  trainer.py      FIT_EVENTS hook (set by scripts/microglia_runtime.py) called at
                  configure, begin, checkpoint, final_written, error and
                  receipt_written; checkpoint_receipt.json is written before the
                  checkpoint is renamed into place, so a crash right after the
                  rename leaves a receipt that matches it; the generation-only
                  branch (final.pt present, no fit_receipt.json) rebuilds the
                  receipt instead of returning silently (reconstructed: true,
                  failure.json -> failure.resolved.json); temp names carry the pid.
  generate.py     stops rewriting fit_receipt.status to 'completed' (adds
                  generated: true instead), so 'training_completed' holds on
                  every path.

Host activation is a small explicit state machine (HOST_MACHINE). The derived
files go to a versioned directory alz-host-empf-<lane>-vendored-<sha12>, where
sha12 starts the sha256 of every derivation input (derived bodies, vendored
digests, the deriving sources). derivation.json is written last and is the
commit marker:

  ABSENT   -derive->   DERIVING  (exclusive FileLock on the directory)
  DERIVING -commit->   DERIVED   (derivation.json written, all digests match)
  DERIVING -crash->    CORRUPT   (files without a marker)
  DERIVED  -damaged->  CORRUPT   (a file no longer matches its digest)
  CORRUPT  -rederive-> DERIVING  (only under the lock, and only when the
                                  directory's derivation sha is the requested one)

Loading holds a shared lock, deriving an exclusive one, so concurrent processes
load intact files and exactly one derives. A refreeze that changes any input
yields a new directory; the old one stays, so in-flight processes and earlier
epochs keep loading their own host. The unversioned legacy directory
alz-host-empf-<lane>-vendored/ is never written; it is used read-only, as
DERIVED, only when its files are byte-identical to the requested derivation.
"""
from pathlib import Path
import enum
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT / 'analysis/empf_comparison_20261117'
SRC = ROOT / 'src'
VENDORED = HERE / 'vendored'
VENDORED_FILES = ('pg_model.py', 'critic.py', 'usage.py', 'contracts.py', 'stage_law.py', 'bench_core.py')
DERIVED_FILES = ('release_env.py', 'contracts.py', 'trainer.py', 'generate.py')
MARKER = 'derivation.json'
HEADER = '# Derived by analysis/default_empf_20261117/vendored_host.py; see derivation.json.\n'
LOCKS_ITSELF = True  # scripts/microglia_runtime.activate_host need not serialize activate()

# Exact lines in the unchanged originals that name a location outside the checkout.
TRAINER_MODEL = "MODEL = Path('/ml2/alz/analysis/faithful097/model')"
TRAINER_SCIENCE = "SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')"
TRAINER_BENCH = "BENCH = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')"
TRAINER_RUNS = "RUNS = Path('/ml2/alz/runs/improve_20261109')"
RELEASE_SITE = "SITE=Path('/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages')"
CONTRACTS_B = "B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')"

# --- fit-lifecycle derivation patches -------------------------------------------------------------

TRAINER_HOOKS = '''FIT_EVENTS = None  # set by scripts/microglia_runtime.py; called as FIT_EVENTS(event, fit_dir, info)


def _fit_event(event, out, quiet=False, **info):
    """Report a fit-lifecycle event to the runtime (derived patch; a no-op when no runtime listens)."""
    if FIT_EVENTS is None:
        return
    try:
        FIT_EVENTS(event, Path(out), info)
    except Exception as exc:
        if not quiet:
            raise
        print(f'[trainer] fit-state event {event} was not recorded: {exc!r}', flush=True)


def _reconstruct_receipt(out, args, ctx):
    """Rebuild a missing fit_receipt.json from config.json, final.pt and checkpoint_receipt.json (derived patch).

    Guard: checkpoint_receipt.json names final.pt with final.pt's sha256 and status 'completed', and its
    step equals the requested number of updates (run() has already compared config.json). The receipt is
    stamped reconstructed: true, and failure.json is renamed to failure.resolved.json.
    """
    out = Path(out)
    record = out / 'checkpoint_receipt.json'
    if not record.exists():
        raise ValueError(f'cannot reconstruct fit_receipt.json in {out}: checkpoint_receipt.json is missing')
    cr = json.loads(record.read_text())
    final_sha = sha(out / 'final.pt')
    if cr.get('path') != 'final.pt' or cr.get('SHA256') != final_sha or cr.get('status') != 'completed':
        raise ValueError(f'cannot reconstruct fit_receipt.json in {out}: checkpoint_receipt.json does not describe final.pt')
    if cr.get('completed_updates') != args.updates:
        raise ValueError(f'cannot reconstruct fit_receipt.json in {out}: checkpoint step '
                         f'{cr.get("completed_updates")} is not the requested {args.updates} updates')
    q = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
    wall_seconds = q.get('wall_seconds')
    del q
    snaps = sorted(set(snapshot_updates(args.updates)))
    snap_files = {u_: (sha(out / f'snap_{u_}.pt') if (out / f'snap_{u_}.pt').exists() else None) for u_ in snaps}
    receipt = {'status': 'training_completed', 'completed_updates': cr['completed_updates'], 'endpoint': args.updates,
               'full_20k_endpoint': args.updates == RECIPE_STEPS, 'full_horizon_endpoint': args.updates == ctx.topt.steps,
               'wall_seconds': wall_seconds, 'seconds_per_update_this_session': None, 'updates_this_session': 0,
               'resumed_from': None, 'device': args.device, 'threads': args.threads, 'peak_memory': None,
               'final_SHA256': final_sha, 'snapshots_SHA256': {str(k): v for k, v in snap_files.items()},
               'snapshots_missing': [k for k, v in snap_files.items() if v is None],
               'reconstructed': True,
               'reconstructed_from': {'config_SHA256': sha(out / 'config.json'), 'final_SHA256': final_sha,
                                      'checkpoint_receipt_SHA256': sha(record)}}
    dump(out / 'fit_receipt.json', receipt)
    if (out / 'failure.json').exists():
        (out / 'failure.json').replace(out / 'failure.resolved.json')
    print(f'[trainer] reconstructed fit_receipt.json from final.pt and checkpoint_receipt.json: {out}', flush=True)
    _fit_event('reconstruct_receipt', out, step=cr['completed_updates'])
    return receipt


def run(args):
    check_out_root(args.out_root)'''

SAVE_BEFORE = """        tmp = p.with_suffix('.partial.pt')
        torch.save(snapshot(ctx, b, g, r, u, args.seed, streams, status, endpoint=args.updates, stream_root=stream_root, wall_seconds=wall()), tmp)
        tmp.replace(p)
        dump(out / 'checkpoint_receipt.json', {'path': p.name, 'SHA256': sha(p), 'status': status, 'completed_updates': policy.completed_steps})
        return p"""
SAVE_AFTER = """        tmp = p.with_name(f'.{p.stem}.partial.{os.getpid()}.pt')
        torch.save(snapshot(ctx, b, g, r, u, args.seed, streams, status, endpoint=args.updates, stream_root=stream_root, wall_seconds=wall()), tmp)
        # Receipt first: after the rename below, checkpoint_receipt.json already matches the checkpoint.
        dump(out / 'checkpoint_receipt.json', {'path': p.name, 'SHA256': sha(tmp), 'status': status, 'completed_updates': policy.completed_steps})
        tmp.replace(p)
        _fit_event('final_written' if status == 'completed' else 'checkpoint', out, step=policy.completed_steps)
        return p"""

GENERATION_ONLY_BEFORE = """        print(f'[trainer] final.pt present without generated.npz; generation only: {out}', flush=True)"""
GENERATION_ONLY_AFTER = """        if not (out / 'fit_receipt.json').exists():
            _reconstruct_receipt(out, args, ctx)
        print(f'[trainer] final.pt present without generated.npz; generation only: {out}', flush=True)"""

FIT_LIFECYCLE_PATCHES = {
    'trainer.py': (
        ("    tmp = p.with_name('.' + p.name + '.tmp')", "    tmp = p.with_name(f'.{p.name}.{os.getpid()}.tmp')"),
        (SAVE_BEFORE, SAVE_AFTER),
        ("        tmp = out / f'.snap_{step}.partial.pt'", "        tmp = out / f'.snap_{step}.partial.{os.getpid()}.pt'"),
        ("        dump(out / 'config.json', config)\n",
         "        dump(out / 'config.json', config)\n        _fit_event('configure', out)\n"),
        ("    curve = out / 'learning_curve.jsonl'\n",
         "    _fit_event('begin', out, recovery=(out / 'recovery.pt').exists())\n    curve = out / 'learning_curve.jsonl'\n"),
        ("'phase': policy._phase})\n        raise",
         "'phase': policy._phase})\n        _fit_event('error', out, quiet=True, error=repr(exc), step=policy.completed_steps)\n        raise"),
        ("    dump(out / 'fit_receipt.json', receipt)\n",
         "    dump(out / 'fit_receipt.json', receipt)\n    _fit_event('receipt_written', out, step=policy.completed_steps)\n"),
        (GENERATION_ONLY_BEFORE, GENERATION_ONLY_AFTER),
        # Last, so the helpers' own text never matches an earlier patch.
        ('def run(args):\n    check_out_root(args.out_root)', TRAINER_HOOKS),
    ),
    'generate.py': (
        ("        receipt['status'] = 'completed'\n", "        receipt['generated'] = True\n"),
    ),
}


def _host():
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    import host
    return host


def _ds():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from alz_perturb_gan import durable_state
    return durable_state


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


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
    # ... plus the checkout-relative locations ...
    for before, after in ((TRAINER_MODEL, f"MODEL = Path({str(H.ROOT/'analysis/faithful097/model')!r})"),
                          (TRAINER_SCIENCE, f'SCIENCE = Path({str(VENDORED)!r})'),
                          (TRAINER_BENCH, f'BENCH = Path({str(VENDORED)!r})'),
                          (TRAINER_RUNS, f"RUNS = Path({str(H.ROOT/'runs/improve_20261109')!r})")):
        bodies['trainer.py'] = H.replace_once(bodies['trainer.py'], before, after)
    bodies['release_env.py'] = H.replace_once(bodies['release_env.py'], RELEASE_SITE,
        f"SITE=Path({str(H.RELEASE/'site-packages')!r})")
    bodies['contracts.py'] = H.replace_once(bodies['contracts.py'], CONTRACTS_B, f'B=Path({str(VENDORED)!r})')
    # ... plus the fit-lifecycle patches.
    for name, patches in FIT_LIFECYCLE_PATCHES.items():
        for before, after in patches:
            bodies[name] = H.replace_once(bodies[name], before, after)
    return {n: HEADER + b for n, b in bodies.items()}, originals


def derivation_record(lane='native'):
    """Everything the derived host depends on, and its sha256 (the version of the derived directory)."""
    H = _host()
    bodies, originals = derive_bodies(lane)
    data = {n: bodies[n].encode() for n in DERIVED_FILES}
    inputs = dict(release='0.10.0', commit=H.COMMIT, lane=lane,
                  host_derivation_sha256=H.sha(H.__file__), adapter_sha256=H.sha(STUDY/'adapter.py'),
                  vendored_host_sha256=H.sha(__file__),
                  vendored={n: H.sha(VENDORED/n) for n in VENDORED_FILES},
                  originals={n: dict(path=str(originals[n]), sha256=H.sha(originals[n])) for n in DERIVED_FILES},
                  derived_sha256={n: _sha_bytes(data[n]) for n in DERIVED_FILES})
    digest = _sha_bytes(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode())
    return dict(derivation_sha256=digest, inputs=inputs), data


def runtime_dir(lane, derivation_sha256=None):
    """The versioned derived directory for a derivation (the current one when no sha is given)."""
    if derivation_sha256 is None:
        derivation_sha256 = derivation_record(lane)[0]['derivation_sha256']
    return _host().RELEASE / f'alz-host-empf-{lane}-vendored-{derivation_sha256[:12]}'


def legacy_dir(lane):
    """The unversioned directory written by the previous vendored host; read-only here."""
    return _host().RELEASE / ('alz-host-empf-' + lane + '-vendored')


class HostState(enum.Enum):
    ABSENT = 'absent'
    DERIVING = 'deriving'
    DERIVED = 'derived'
    CORRUPT = 'corrupt'


def _machine():
    ds = _ds()
    S = HostState
    return ds.Machine('derived_host', S, S.ABSENT, (), {
        (S.ABSENT, 'derive'): S.DERIVING,
        (S.DERIVING, 'commit'): S.DERIVED,
        (S.DERIVING, 'crash'): S.CORRUPT,
        (S.DERIVED, 'damaged'): S.CORRUPT,
        (S.CORRUPT, 'rederive'): S.DERIVING,
    })


HOST_MACHINE = _machine()


def provenance(record, directory):
    """The commit-marker document; derived_sha256 values are the sha256 of the bytes written."""
    inputs = record['inputs']
    files = {n: dict(original=inputs['originals'][n]['path'], original_sha256=inputs['originals'][n]['sha256'],
                     derived=str(Path(directory)/n), derived_sha256=inputs['derived_sha256'][n])
             for n in DERIVED_FILES}
    return dict(schema='vendored_host/2', derivation_sha256=record['derivation_sha256'],
                release=inputs['release'], commit=inputs['commit'], lane=inputs['lane'],
                host_derivation_sha256=inputs['host_derivation_sha256'], adapter_sha256=inputs['adapter_sha256'],
                vendored_host_sha256=inputs['vendored_host_sha256'],
                vendored={n: dict(path=str(VENDORED/n), sha256=d) for n, d in inputs['vendored'].items()},
                files=files)


def inspect(directory, record):
    """(HostState, reason) of a derived directory, read without a lock.

    DERIVING is not visible from the files: a directory that is being derived reads as ABSENT or CORRUPT
    until its marker is committed. Callers hold the directory lock when the answer matters.
    """
    directory = Path(directory)
    marker = directory/MARKER
    present = [n for n in DERIVED_FILES if (directory/n).exists()]
    if not marker.exists():
        if not present:
            return HostState.ABSENT, 'no derived files'
        return HostState.CORRUPT, f'commit marker {MARKER} missing beside {present}'
    try:
        recorded = json.loads(marker.read_text())
    except ValueError as exc:
        return HostState.CORRUPT, f'unreadable commit marker: {exc}'
    if recorded.get('derivation_sha256') != record['derivation_sha256']:
        return HostState.CORRUPT, f'marker records derivation {recorded.get("derivation_sha256")}'
    for name in DERIVED_FILES:
        path = directory/name
        if not path.exists():
            return HostState.CORRUPT, f'{name} missing'
        if _sha_bytes(path.read_bytes()) != record['inputs']['derived_sha256'][name]:
            return HostState.CORRUPT, f'{name} does not match its derived digest'
    return HostState.DERIVED, 'all derived digests match'


def _legacy_matches(lane, record, data):
    """The legacy unversioned directory, when its files are exactly this derivation's bytes; else None."""
    directory = legacy_dir(lane)
    if not (directory/MARKER).exists():
        return None
    for name in DERIVED_FILES:
        path = directory/name
        if not path.exists() or path.read_bytes() != data[name]:
            return None
    return directory


def _step(state, event):
    target = HOST_MACHINE.next_state(state, event)
    if target is None:
        raise RuntimeError(f'derived host: no edge {state.name} -{event}->')
    return target


def ensure_host(lane='native'):
    """Bring the versioned directory to DERIVED; return (directory, provenance, derived_here).

    Takes the directory's exclusive lock itself, so direct callers are serialized too.
    """
    ds = _ds()
    record, data = derivation_record(lane)
    directory = runtime_dir(lane, record['derivation_sha256'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    with ds.FileLock(directory, purpose='derive EMPF host'):
        state, reason = inspect(directory, record)
        if state is HostState.DERIVED:
            return directory, provenance(record, directory), False
        if state is HostState.CORRUPT:
            marker = directory/MARKER
            if marker.exists():
                try:
                    recorded = json.loads(marker.read_text()).get('derivation_sha256')
                except ValueError:
                    recorded = None
                if recorded not in (None, record['derivation_sha256']):
                    raise RuntimeError(f'refusing to rederive {directory}: it holds derivation {recorded}, '
                                       f'not {record["derivation_sha256"]}')
            if not directory.name.endswith(record['derivation_sha256'][:12]):
                raise RuntimeError(f'refusing to rederive {directory}: not the directory of this derivation')
            print(f'derived EMPF host {directory.name} is CORRUPT ({reason}); rederiving under the lock', flush=True)
            state = _step(state, 'rederive')
            marker.unlink(missing_ok=True)  # uncommit before touching any file
        else:
            state = _step(state, 'derive')
        directory.mkdir(exist_ok=True)
        ds.sweep_temps(directory, min_age_s=0)  # leftovers of dead derivers; no live writer without the lock
        for name in DERIVED_FILES:
            path = directory/name
            if not (path.exists() and path.read_bytes() == data[name]):
                ds.atomic_write_bytes(path, data[name])
        prov = provenance(record, directory)
        ds.atomic_write_bytes(directory/MARKER, (json.dumps(prov, indent=2) + '\n').encode())  # commit
        state = _step(state, 'commit')
        check, why = inspect(directory, record)
        if check is not state:
            raise RuntimeError(f'derived EMPF host did not verify after commit: {why}')
        return directory, prov, True


def write_host(lane='native'):
    """Write (or confirm) the derived host files and derivation.json; return (directory, provenance)."""
    directory, prov, _ = ensure_host(lane)
    return directory, prov


def _default_loader(directory):
    H = _host()
    H.load('release_env', directory/'release_env.py')
    H.load('contracts', directory/'contracts.py')
    return H.load('trainer', directory/'trainer.py'), H.load('generate', directory/'generate.py')


def activate(lane='native', loader=None, attempts=3):
    """Derive (or confirm) the versioned host and import it under a shared lock.

    Returns (trainer, generate, provenance) with the default loader, else (*loader(directory), provenance).
    """
    if loader is None:
        if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env', *(n[:-3] for n in VENDORED_FILES))):
            raise RuntimeError('EMPF lane requires a fresh subprocess')
        loader = _default_loader
    ds = _ds()
    record, data = derivation_record(lane)
    legacy = _legacy_matches(lane, record, data)
    if legacy is not None:
        with ds.FileLock(legacy, exclusive=False):  # the previous runtime's activation lock
            if _legacy_matches(lane, record, data) is not None:
                prov = provenance(record, legacy)
                prov['legacy_directory'] = True
                return (*loader(legacy), prov)
    directory = runtime_dir(lane, record['derivation_sha256'])
    for _ in range(attempts):
        if (directory/MARKER).exists():
            with ds.FileLock(directory, exclusive=False):
                state, reason = inspect(directory, record)
                if state is HostState.DERIVED:
                    return (*loader(directory), provenance(record, directory))
        ensure_host(lane)
    raise RuntimeError(f'derived EMPF host {directory} did not reach DERIVED after {attempts} attempts')
