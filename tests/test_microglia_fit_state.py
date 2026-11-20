"""Explicit fit lifecycle: fit_state.json, the fit lock and the derived trainer's events (CPU only, no data).

The trainer here is the REAL derived trainer.run() (from vendored_host.derive_bodies, so it carries the
fit-lifecycle derivation patches), extracted with ast and executed with stub model collaborators. Its
checkpoints are pickled dicts shaped so that the runtime's real check_checkpoint, check_weights and
microglia.check_fit accept them. scripts/microglia_runtime.main() drives it as in production, with the
source manifest, cache anchors and host activation stubbed.

(a) the transition table, through the durable_state property shim (random event sequences vs a model)
(b) crash injection (os._exit(137)) at every write of a fit's lifecycle; each rerun converges
(c) two processes training one fit: exactly one runs, the other exits 75
(e) legacy fit directories are adopted read-only
(f) a refrozen manifest in a tmp checkout passes check_unchanged_sources
"""
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load('microglia_runtime_fit_state', ROOT/'scripts/microglia_runtime.py')
M = R.application()
ds = R.ds
Fit = R.Fit
CFG = M.load_config()
SPEC = json.loads(M.RECIPE.read_text())['native']
UPDATES = 20000
SNAPS = [5000, 10000, 15000, 16000, 17000, 18000, 19000]
CHECKPOINTS = list(range(2000, UPDATES, 2000))  # recovery.pt writes at checkpoint_every=2000


# --------------------------------------------------------------------------------------------------
# (a) the transition table

# The table as specified; FIT_MACHINE must equal it exactly.
EXPECTED_TABLE = {
    ('ABSENT', 'configure'): 'CONFIGURED',
    ('CONFIGURED', 'start'): 'TRAINING',
    ('TRAINING', 'checkpoint'): 'TRAINING',
    ('TRAINING', 'error'): 'FAILED_RESUMABLE',
    ('FAILED_RESUMABLE', 'resume'): 'TRAINING',
    ('TRAINING', 'final_written'): 'TRAINED',
    ('FAILED_RESUMABLE', 'final_written'): 'TRAINED',  # reconcile: error recorded after final.pt landed
    ('TRAINED', 'receipt_written'): 'RECEIPTED',
    ('TRAINED', 'reconstruct_receipt'): 'RECEIPTED',
    ('RECEIPTED', 'audit_passed'): 'AUDITED',
    ('RECEIPTED', 'audit_failed'): 'RECEIPTED',
    **{(s, 'epoch_mismatch'): 'STRANDED'
       for s in ('ABSENT', 'CONFIGURED', 'TRAINING', 'FAILED_RESUMABLE', 'TRAINED', 'RECEIPTED')},
    ('FAILED_RESUMABLE', 'adopt_epoch'): 'FAILED_RESUMABLE',
    ('TRAINED', 'adopt_epoch'): 'TRAINED',
    ('RECEIPTED', 'adopt_epoch'): 'RECEIPTED',
    ('STRANDED', 'adopt_epoch'): 'FAILED_RESUMABLE',
    ('STRANDED', 'adopt_epoch_trained'): 'TRAINED',
    ('STRANDED', 'audit_under_epoch'): 'AUDITED',  # guarded: the live digest is the fit's own epoch
    **{(s, 'damaged'): 'DAMAGED' for s in ('ABSENT', 'CONFIGURED', 'TRAINED', 'RECEIPTED', 'AUDITED')},
    **{('DAMAGED', f'repair_to_{s.lower()}'): s
       for s in ('ABSENT', 'CONFIGURED', 'FAILED_RESUMABLE', 'TRAINED', 'RECEIPTED')},
}


def test_fit_machine_is_the_specified_table():
    table = {(s.name, e): t.name for (s, e), t in R.FIT_MACHINE.transitions.items()}
    assert table == EXPECTED_TABLE
    # No terminal state: AUDITED leaves only through 'damaged' (its files no longer verify), never relabelled.
    assert R.FIT_MACHINE.terminal == frozenset()
    assert R.FIT_MACHINE.initial is Fit.ABSENT
    assert R.FIT_MACHINE.allowed_events(Fit.AUDITED) == ['damaged']
    assert ('AUDITED', 'epoch_mismatch') not in table  # an audited fit is never stranded


def test_fit_transition_table_property(tmp_path):
    """Random event sequences through durable_state's property shim (tests/test_durable_state.py)."""
    import test_durable_state as shim
    machine = R.FIT_MACHINE
    rng = random.Random(20261120)
    events = sorted(machine.events) + ['undeclared']
    for i in range(400):
        sf = ds.StateFile(tmp_path/f'fit-{i}.json', machine)
        sf.create()
        model = shim.Model(machine)
        for _ in range(rng.randint(1, 14)):
            legal = machine.allowed_events(model.state)
            event = rng.choice(legal) if legal and rng.random() < 0.7 else rng.choice(events)
            shim._step(sf, model, event, rng.random() >= 0.1)
            shim._check(sf, model)


def test_start_and_resume_guards(tmp_path):
    digest, other = 'd'*64, 'e'*64
    fit = tmp_path/'fit'
    fit.mkdir()
    sf = R.fit_state_file(fit)
    R.adopt_fit(sf, fit)
    (fit/'config.json').write_text('{}')
    sf.transition('configure')
    (fit/'train_start.json').write_text(json.dumps({'application_source_manifest_sha256': other}))
    hook = R.fit_events(sf, fit, digest, lambda: {'application_source_manifest_sha256': digest})
    with pytest.raises(ValueError, match='refusing to start'):
        hook('begin', fit, {})
    assert sf.read().state is Fit.CONFIGURED
    (fit/'train_start.json').unlink()
    hook('begin', fit, {})  # absent start record: written, then TRAINING
    assert sf.read().state is Fit.TRAINING and R.start_epoch_of(fit) == digest
    hook('error', fit, {'error': 'boom', 'step': 4})
    assert sf.read().state is Fit.FAILED_RESUMABLE
    wrong = R.fit_events(sf, fit, other, lambda: {})
    with pytest.raises(ds.GuardFailed):
        wrong('begin', fit, {})
    hook('begin', fit, {})
    assert sf.read().state is Fit.TRAINING
    with pytest.raises(ValueError, match='not the locked fit'):
        hook('checkpoint', tmp_path/'elsewhere', {})


# --------------------------------------------------------------------------------------------------
# Harness: the derived trainer.run() with stub model collaborators

WANT = ('run', 'dump', 'sha', 'fit_dir', '_fit_event', '_reconstruct_receipt')


def derived_trainer_source(root=ROOT):
    host_dir = str(root/'analysis/default_empf_20261117')
    if host_dir not in sys.path:
        sys.path.insert(0, host_dir)
    import vendored_host
    return vendored_host.derive_bodies('native')[0]['trainer.py']


def fit_config(seed, cache):
    """A config.json that the real microglia.check_fit accepts for fold 0."""
    t = CFG['training']
    return {'study': 'improve_20261109', 'round': 3, 'arm': t['arm'], 'flags': ['pbulk'], 'panel': t['panel'],
            'fold': 0, 'seed': seed, 'cache': str(cache), 'cache_manifest_SHA256': R.sha(Path(cache)/'manifest.json'),
            'protocol_split': True, 'hyperparameters': SPEC['hyperparameters'], 'recipe': SPEC['recipe'],
            'native_recipe': 'empf', 'particlegan_version': '0.10.0', 'updates': t['updates'], 'batch': [32, 128],
            'critic_features': 2150, 'checkpoint_every': 2000, 'snapshot_updates': SNAPS}


def checkpoint(cfg, status, steps, pid=None):
    """A pickled checkpoint dict the runtime's real check_checkpoint accepts when completed."""
    record = {'calls': steps, 'observed_steps': steps,
              'empf': {'schema': 2, 'initialized': True, 'proto_initialized': False, 'proto_sum': None,
                       'proto_count': None}}
    q = {k: cfg[k] for k in ('arm', 'flags', 'panel', 'fold', 'seed', 'cache_manifest_SHA256', 'recipe',
                             'particlegan_version', 'native_recipe')}
    q.update(status=status, completed_updates=steps, fixed_endpoint=UPDATES,
             released_hyperparameters=cfg['hyperparameters'], penalty_options={'input_domain': 'continuous'},
             round=3, wall_seconds=float(steps), pid=os.getpid() if pid is None else pid,
             policy={'recipe': cfg['recipe'], 'completed_steps': steps,
                     'optimizers': [{}, {'regularizer': {'record': record}}]})
    return q


class FakeTorch:
    @staticmethod
    def save(obj, path):
        Path(path).write_bytes(pickle.dumps(obj))

    @staticmethod
    def load(path, **kw):
        return pickle.loads(Path(path).read_bytes())

    @staticmethod
    def set_num_threads(n):
        pass


class Policy:
    def __init__(self, steps=0):
        self.completed_steps = steps
        self._phase = 'ready'


def make_trainer(root=ROOT, sleep_at_first_step=0.0):
    """(T, G): the derived trainer's run() and helpers in a module with stub collaborators."""
    source = derived_trainer_source(root)
    tree = ast.parse(source)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANT)
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'FIT_EVENTS' for t in n.targets))]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == set(WANT)
    T = types.ModuleType('trainer')
    G = SimpleNamespace(generate_fit=lambda *a, **kw: pytest.fail('generation was not deferred'),
                        GEN_STREAMS=('a', 'b', 'c', 'd', 'e'))
    sys.modules['generate'] = G
    state = {'config': None}

    def make_config(ctx, args, stream_root):
        state['config'] = fit_config(args.seed, ctx.cache)
        return state['config']

    def build(ctx, seed, **kw):
        return [Policy()], None, None, None, {'streams': seed}

    def restore(ctx, q, seed=None):
        return [Policy(q['completed_updates'])], None, None, None, {'streams': seed}

    def advance(ctx, b, g, r, u, **kw):
        b[-1].completed_steps += 1
        if sleep_at_first_step and b[-1].completed_steps == 1:
            time.sleep(sleep_at_first_step)
        return {'step': b[-1].completed_steps}

    def snapshot(ctx, b, g, r, u, seed, streams, status, *, endpoint, stream_root, wall_seconds):
        return checkpoint(state['config'], status, b[-1].completed_steps)

    def parser():
        def parse_args(argv):
            kv = {argv[i].lstrip('-').replace('-', '_'): argv[i+1] for i in range(0, len(argv), 2)}
            return SimpleNamespace(cache=kv['cache'], arm=kv['arm'], seed=int(kv['seed']), updates=int(kv['updates']),
                                   out_root=Path(kv['out_root']), device=kv['device'], threads=int(kv['threads']),
                                   gpu_mem_fraction=float(kv['gpu_mem_fraction']), checkpoint_every=2000,
                                   stop_after=None, stream_root=1, allow_nonprotocol_split=False)
        return SimpleNamespace(parse_args=parse_args)

    def context(cache, device, flags, **kw):
        return SimpleNamespace(panel=CFG['training']['panel'], fold=0, cache=Path(cache), device='cpu', flags=flags,
                               topt=SimpleNamespace(steps=UPDATES))

    T.__dict__.update(
        Path=Path, json=json, hashlib=hashlib, signal=__import__('signal'), time=time, os=os, torch=FakeTorch,
        ROUND=3, _CONFIG_MUTABLE={'cache', 'checkpoint_every', 'folds_file'}, RECIPE_STEPS=UPDATES,
        check_out_root=lambda o: o, parse_arm=lambda a: ('pbulk',), arm_name=lambda f: 'pbulk',
        arms=SimpleNamespace(train_options=lambda f: SimpleNamespace(steps=UPDATES)),
        passive_particle_diagnostics=lambda b, u: {}, peak_memory=lambda: {}, snapshot_updates=lambda u: list(SNAPS),
        Context=context, make_config=make_config, build=build, restore=restore, advance=advance, snapshot=snapshot,
        model_snapshot=lambda ctx, b, seed, **kw: {'pid': os.getpid(), 'step': b[-1].completed_steps},
        parser=parser, recipe=lambda: SimpleNamespace(to_dict=lambda: SPEC['recipe']))
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'derived trainer.py', 'exec'), T.__dict__)
    return T, G


def install(setattr_, work, T=None, G=None, sleep_at_first_step=0.0):
    """Stub the runtime's environment: manifest, sources, cache anchors and host activation."""
    if T is None:
        T, G = make_trainer(sleep_at_first_step=sleep_at_first_step)
    setattr_(R, 'SOURCE_MANIFEST', work/'manifest.json')
    setattr_(R, 'check_sources', lambda: {'sources': {}})
    setattr_(R, 'check_cache', lambda cache, fold, held=False: {'anchor': fold})
    setattr_(R, 'application', lambda: M)
    setattr_(R, 'activate', lambda device, memory: (T, G, {'host': 'harness'}, {'cpus': [0, 1, 2], 'source': 'test'}))
    return T, G


def prepare_work(work, manifest='{"sources": {"epoch": 1}}'):
    work.mkdir(parents=True, exist_ok=True)
    (work/'manifest.json').write_text(manifest)
    (work/'cache').mkdir(exist_ok=True)
    (work/'cache/manifest.json').write_text('{"inner_fold": 0}')
    return work


def fit_path(work, seed=0):
    return work/'fits'/CFG['training']['arm']/CFG['training']['panel']/'fold0'/f'seed{seed}'


def _install_crash(spec, scope):
    """os._exit(137) at a named os.replace inside `scope`.

    replace:<name>[:n]  after the n-th replace onto <name>
    before:<name>[:n]   instead of the n-th replace onto <name> (its temp file is left behind)
    state:<STATE>       after fit_state.json is replaced with <STATE>
    """
    kind, _, rest = spec.partition(':')
    real = os.replace
    counts = {}

    def replace(src, dst, *a, **kw):
        path = Path(os.fspath(dst))
        inside = str(path).startswith(str(scope))
        target, _, nth = rest.partition(':')
        if inside and kind == 'before' and path.name == target:
            counts[target] = counts.get(target, 0) + 1
            if counts[target] == int(nth or 1):
                os._exit(137)
        real(src, dst, *a, **kw)
        if not inside:
            return
        if kind == 'replace' and path.name == target:
            counts[target] = counts.get(target, 0) + 1
            if counts[target] == int(nth or 1):
                os._exit(137)
        if kind == 'state' and path.name == R.FIT_STATE and json.loads(path.read_text())['state'] == rest:
            os._exit(137)
    os.replace = replace


def _install_event_failure(spec):
    """Fail the first FIT_EVENTS call for <event>: OSError(ENOSPC) by default, KeyboardInterrupt with ':interrupt'."""
    event, _, kind = spec.partition(':')
    real = R.fit_events

    def fit_events(*a, **kw):
        on = real(*a, **kw)
        fired = []

        def hook(name, out, info):
            if name == event and not fired:
                fired.append(1)
                if kind == 'interrupt':
                    raise KeyboardInterrupt('injected')
                raise OSError(28, 'No space left on device (injected)')
            return on(name, out, info)
        return hook
    R.fit_events = fit_events


def child_main():
    """Entry point of the subprocess that runs microglia_runtime.main(['train', ...])."""
    work = Path(os.environ['FIT_WORK'])
    if os.environ.get('FIT_CRASH'):
        _install_crash(os.environ['FIT_CRASH'], work/'fits')
    if os.environ.get('FIT_FAIL_EVENT'):
        _install_event_failure(os.environ['FIT_FAIL_EVENT'])
    install(lambda obj, name, value: setattr(obj, name, value), work,
            sleep_at_first_step=float(os.environ.get('FIT_SLEEP', '0')))
    extra = json.loads(os.environ.get('FIT_ARGS', '[]'))
    try:
        rc = R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits'), *extra])
    except R.Stranded as exc:  # as in the CLI's __main__
        print(f'microglia runtime: Stranded: {exc}', file=sys.stderr, flush=True)
        rc = R.EX_STRANDED
    except (ValueError, OSError, RuntimeError, KeyError, ds.DurableStateError) as exc:
        print(f'microglia runtime: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def run_child(work, crash=None, args=(), sleep=0.0, timeout=120, start=False, fail_event=None):
    env = dict(os.environ, FIT_WORK=str(work), FIT_ARGS=json.dumps(list(args)), FIT_SLEEP=str(sleep),
               PYTHONDONTWRITEBYTECODE='1')
    for key in ('FIT_CRASH', 'FIT_FAIL_EVENT', ds.FAULT_ENV):
        env.pop(key, None)
    if crash:
        env['FIT_CRASH'] = crash
    if fail_event:
        env['FIT_FAIL_EVENT'] = fail_event
    code = f'import sys; sys.path.insert(0, {str(TESTS)!r}); import test_microglia_fit_state as H; H.child_main()'
    proc = subprocess.Popen([sys.executable, '-B', '-c', code], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if start:
        return proc
    out, err = proc.communicate(timeout=timeout)
    return proc.returncode, out, err


def state_of(work, seed=0):
    return R.fit_state_file(fit_path(work, seed)).read()


def events_of(work, seed=0):
    return [e['event'] for e in R.fit_state_file(fit_path(work, seed)).journal()]


# --------------------------------------------------------------------------------------------------
# A clean run, and the in-process derivation patches

def test_clean_run_reaches_audited_then_is_already_done(tmp_path):
    work = prepare_work(tmp_path/'w')
    rc, out, err = run_child(work)
    assert rc == 0, err
    fit = fit_path(work)
    snap = state_of(work)
    assert snap.state is Fit.AUDITED and snap.data['audited_epoch'] == R.sha(work/'manifest.json')
    assert events_of(work) == ['__adopt__', 'configure', 'start', *['checkpoint']*len(CHECKPOINTS),
                               'final_written', 'receipt_written', 'audit_passed']
    receipt = json.loads((fit/'fit_receipt.json').read_text())
    assert receipt['status'] == 'training_completed' and 'reconstructed' not in receipt
    M.check_fit(fit, CFG, 0, 0)
    assert not [p.name for p in fit.iterdir() if '.tmp' in p.name or '.partial' in p.name]
    before = snap.version
    rc, out, err = run_child(work)
    assert rc == R.ALREADY_DONE, err
    assert state_of(work).version == before  # the AUDITED -train-> no-op writes nothing


def test_derivation_patches_reach_the_derived_sources():
    sys.path.insert(0, str(ROOT/'analysis/default_empf_20261117'))
    import vendored_host as V
    bodies = V.derive_bodies('native')[0]
    trainer, generate = bodies['trainer.py'], bodies['generate.py']
    assert "receipt['status'] = 'completed'" not in generate and "receipt['generated'] = True" in generate
    assert "'.partial.pt'" not in trainer and "'.' + p.name + '.tmp'" not in trainer
    assert 'FIT_EVENTS = None' in trainer
    for event in ("'configure'", "'begin'", "'final_written' if status == 'completed' else 'checkpoint'", "'error'",
                  "'receipt_written'", "'reconstruct_receipt'"):
        assert trainer.count(f'_fit_event({event}') == 1, event
    # The pinned originals are untouched.
    assert "receipt['status'] = 'completed'" in (ROOT/'analysis/improve_20261109/r5/generate.py').read_text()


# --------------------------------------------------------------------------------------------------
# (b) crash injection

CRASHES = [
    'replace:config.json',                                        # after config, before CONFIGURED
    'state:TRAINING',                                             # after start
    *[f'replace:recovery.pt:{n}' for n in range(1, len(CHECKPOINTS)+1)],  # after each recovery.pt
    'before:final.pt',                                            # final.pt staged, not yet renamed
    'replace:final.pt',                                           # final.pt renamed, state still TRAINING
    'state:TRAINED',                                              # inside save('completed'): recorded, recovery.pt not yet removed
    'before:fit_receipt.json',                                    # crash_before_receipt
    'replace:fit_receipt.json',                                   # receipt written, state still TRAINED
    'state:RECEIPTED',                                            # after receipt, before audit
    'replace:no_skip_audit.json',                                 # after audit, before state
]


@pytest.mark.parametrize('crash', CRASHES)
def test_crash_then_rerun_converges(crash, tmp_path):
    work = prepare_work(tmp_path/'w')
    rc, out, err = run_child(work, crash=crash)
    assert rc == 137, (rc, out, err)
    for attempt in range(3):  # bounded: a rerun must converge, never loop
        rc, out, err = run_child(work)
        if rc != 1 or 'OSError' not in err:
            break
    # After the audit landed, the rerun only reconciles (RECEIPTED -audit_passed-> AUDITED) and exits 3.
    assert rc == (R.ALREADY_DONE if crash == 'replace:no_skip_audit.json' else 0), (crash, out, err)
    assert attempt == 0, f'needed {attempt+1} reruns'
    fit = fit_path(work)
    assert state_of(work).state is Fit.AUDITED
    M.check_fit(fit, CFG, 0, 0)
    # No crash debris survives the rerun: recovery.pt beside final.pt, staged checkpoints, state temps.
    assert not (fit/'recovery.pt').exists(), crash
    assert not [p.name for p in fit.iterdir() if '.tmp' in p.name or '.partial' in p.name], crash
    receipt = json.loads((fit/'fit_receipt.json').read_text())
    assert receipt['snapshots_missing'] == [] and receipt['final_SHA256'] == R.sha(fit/'final.pt')
    if crash == 'before:fit_receipt.json':
        assert receipt['reconstructed'] is True and receipt['status'] == 'training_completed'
        assert 'reconstruct_receipt' in events_of(work)
    assert run_child(work)[0] == R.ALREADY_DONE


def test_crash_before_receipt_was_a_retry_loop_without_reconstruction(tmp_path):
    """Regression: the unpatched generation-only branch returned silently, so every rerun failed on the
    missing fit_receipt.json (an OSError the CLI reports and a queue retries forever)."""
    work = prepare_work(tmp_path/'w')
    assert run_child(work, crash='before:fit_receipt.json')[0] == 137
    fit = fit_path(work)
    assert (fit/'final.pt').exists() and not (fit/'fit_receipt.json').exists()
    assert state_of(work).state is Fit.TRAINED
    rc, out, err = run_child(work)
    assert rc == 0, err
    assert 'reconstructed fit_receipt.json' in out
    assert state_of(work).state is Fit.AUDITED


def test_unreconstructable_receipt_fails_with_a_reason_and_does_not_advance(tmp_path):
    work = prepare_work(tmp_path/'w')
    assert run_child(work, crash='before:fit_receipt.json')[0] == 137
    fit = fit_path(work)
    record = json.loads((fit/'checkpoint_receipt.json').read_text())
    (fit/'checkpoint_receipt.json').write_text(json.dumps(dict(record, SHA256='0'*64)))
    for _ in range(2):
        rc, out, err = run_child(work)
        assert rc == 1 and 'does not describe final.pt' in err
        assert state_of(work).state is Fit.TRAINED and not (fit/'fit_receipt.json').exists()


def test_failure_json_is_resolved_by_reconstruction(tmp_path):
    work = prepare_work(tmp_path/'w')
    assert run_child(work, crash='before:fit_receipt.json')[0] == 137
    fit = fit_path(work)
    (fit/'failure.json').write_text('{"error": "earlier interruption"}')
    assert run_child(work)[0] == 0
    assert not (fit/'failure.json').exists()
    assert json.loads((fit/'failure.resolved.json').read_text()) == {'error': 'earlier interruption'}


def test_epoch_change_mid_fit_strands_until_explicit_adoption(tmp_path):
    work = prepare_work(tmp_path/'w')
    assert run_child(work, crash='replace:recovery.pt:3')[0] == 137
    (work/'manifest.json').write_text('{"sources": {"epoch": 2}}')  # a refreeze between runs
    for _ in range(2):
        rc, out, err = run_child(work)
        assert rc == R.EX_STRANDED and ('refusing to resume' in err or 'STRANDED' in err), err
        assert state_of(work).state is Fit.STRANDED
    rc, out, err = run_child(work, args=['--adopt-epoch'])
    assert rc == 0, err
    snap = state_of(work)
    assert snap.state is Fit.AUDITED and snap.data['epoch_adoptions'][0]['digest'] == R.sha(work/'manifest.json')
    audit = json.loads((fit_path(work)/'no_skip_audit.json').read_text())
    assert audit['epoch_adoptions'] and audit['training_start_source_manifest_sha256'] != audit['application_source_manifest_sha256']


# --------------------------------------------------------------------------------------------------
# (c) concurrency

def _race(index, base):
    work = prepare_work(base/f'r{index}')
    rng = random.Random(index)
    procs = []
    for k in range(2):
        time.sleep(rng.uniform(0, .15))
        procs.append(run_child(work, sleep=2.0, start=True))
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=180)
        results.append((proc.returncode, out, err, proc.pid))
    return work, results


def test_two_trainers_on_one_fit_one_runs_one_exits_75(tmp_path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        races = list(pool.map(lambda i: _race(i, tmp_path), range(20)))
    for work, results in races:
        codes = sorted(r[0] for r in results)
        assert codes == [0, R.EX_TEMPFAIL], results
        winner = next(r for r in results if r[0] == 0)
        loser = next(r for r in results if r[0] == R.EX_TEMPFAIL)
        assert f'owner pid {winner[3]}' in loser[2], loser[2]
        fit = fit_path(work)
        pids = {pickle.loads((fit/'final.pt').read_bytes())['pid']}
        pids |= {pickle.loads((fit/f'snap_{s}.pt').read_bytes())['pid'] for s in SNAPS}
        journal = R.fit_state_file(fit).journal()
        pids |= {e['pid'] for e in journal if e['event'] in ('start', 'final_written', 'receipt_written', 'audit_passed')}
        assert pids == {winner[3]}
        assert json.loads((fit/'fit_receipt.json').read_text())['final_SHA256'] == R.sha(fit/'final.pt')
        assert state_of(work).state is Fit.AUDITED


# --------------------------------------------------------------------------------------------------
# (e) legacy adoption is read-only

IGNORED = (R.FIT_STATE, R.FIT_STATE + '.journal.jsonl', R.FIT_STATE + '.lock', 'fit.lock')


def tree(fit):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(fit).iterdir())
            if p.is_file() and p.name not in IGNORED}


def legacy_receipt(fit, status='training_completed', **over):
    receipt = {'status': status, 'completed_updates': UPDATES, 'endpoint': UPDATES,
               'final_SHA256': R.sha(fit/'final.pt'),
               'snapshots_SHA256': {str(s): R.sha(fit/f'snap_{s}.pt') for s in SNAPS}, 'snapshots_missing': []}
    receipt.update(over)
    (fit/'fit_receipt.json').write_text(json.dumps(receipt))


def legacy_audit(fit, digest='a'*64):
    (fit/'no_skip_audit.json').write_text(json.dumps({
        'status': R.AUDIT_PASSED, 'penalty_calls': UPDATES, 'observed_steps': UPDATES, 'completed_updates': UPDATES,
        'lazy_k': 1, 'final_checkpoint_sha256': R.sha(fit/'final.pt'), 'application_source_manifest_sha256': digest}))


def build_legacy(fit, cache, layout):
    fit.mkdir(parents=True)
    if layout == 'empty':
        return
    cfg = fit_config(0, cache)
    (fit/'config.json').write_text(json.dumps(cfg))
    if layout == 'config_only':
        return
    if layout.startswith('recovery'):
        FakeTorch.save(checkpoint(cfg, 'incomplete', 8000), fit/'recovery.pt')
        FakeTorch.save({'step': 5000}, fit/'snap_5000.pt')
        (fit/'checkpoint_receipt.json').write_text(json.dumps({'path': 'recovery.pt', 'SHA256': R.sha(fit/'recovery.pt'),
                                                               'status': 'incomplete', 'completed_updates': 8000}))
        (fit/'failure.json').write_text('{"error": "KeyboardInterrupt()"}')
        if layout == 'recovery_pinned':
            (fit/'train_start.json').write_text(json.dumps({'application_source_manifest_sha256': 'b'*64}))
        return
    FakeTorch.save(checkpoint(cfg, 'completed', UPDATES), fit/'final.pt')
    for s in SNAPS:
        FakeTorch.save({'step': s}, fit/f'snap_{s}.pt')
    if layout == 'final_only':
        return
    legacy_receipt(fit, status='completed' if layout == 'native_generated' else 'training_completed',
                   **({'snapshots_missing': [19000]} if layout == 'audited_but_check_fails' else {}))
    if layout in ('audited', 'audited_but_check_fails'):
        legacy_audit(fit)


LEGACY = {
    'empty': (Fit.ABSENT, {}),
    'config_only': (Fit.CONFIGURED, {}),
    'recovery_pinned': (Fit.FAILED_RESUMABLE, {'step': 8000, 'start_epoch': 'b'*64}),
    'recovery_unpinned': (Fit.FAILED_RESUMABLE, {'step': 8000, 'start_epoch': None}),  # pre-#19
    'final_only': (Fit.TRAINED, {}),
    'receipted': (Fit.RECEIPTED, {'receipt_status': 'training_completed'}),
    'native_generated': (Fit.RECEIPTED, {'receipt_status': 'completed'}),
    'audited': (Fit.AUDITED, {'audit_source_manifest_sha256': 'a'*64}),
    'audited_but_check_fails': (Fit.RECEIPTED, {}),
}


@pytest.fixture
def in_process(tmp_path, monkeypatch):
    work = prepare_work(tmp_path/'w')
    install(lambda obj, name, value: monkeypatch.setattr(obj, name, value), work)
    return work


@pytest.mark.parametrize('layout', sorted(LEGACY))
def test_legacy_layout_is_adopted_read_only(layout, in_process, capsys):
    work = in_process
    fit = fit_path(work)
    build_legacy(fit, work/'cache', layout)
    before = tree(fit)
    listing = {p.name for p in fit.iterdir()}
    assert R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits'), '--dry-run']) == 0
    expected, evidence = LEGACY[layout]
    snap = R.fit_state_file(fit).read()
    assert snap.state is expected
    assert snap.version == 1 and [e['event'] for e in R.fit_state_file(fit).journal()] == ['__adopt__']
    for key, value in evidence.items():
        assert snap.data['adopted']['evidence'][key] == value, key
    if layout == 'audited_but_check_fails':
        assert 'check_fit failed' in snap.data['adopted']['evidence']['reason']
    assert tree(fit) == before  # no legacy byte changed
    assert {p.name for p in fit.iterdir()} - listing <= set(IGNORED)
    # Adoption happens once; a second touch reads the state file and infers nothing again.
    assert R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits'), '--dry-run']) == 0
    assert R.fit_state_file(fit).read().version == 1 and tree(fit) == before


def test_audited_legacy_fit_is_already_done_and_never_relabelled(in_process):
    work = in_process
    fit = fit_path(work)
    build_legacy(fit, work/'cache', 'audited')
    before = tree(fit)
    assert R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits')]) == R.ALREADY_DONE
    assert tree(fit) == before


def test_native_generated_receipt_is_audited_without_rewriting_it(in_process):
    work = in_process
    fit = fit_path(work)
    build_legacy(fit, work/'cache', 'native_generated')
    receipt = (fit/'fit_receipt.json').read_bytes()
    (fit/'train_start.json').write_text(json.dumps({'application_source_manifest_sha256': R.sha(work/'manifest.json')}))
    assert R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits')]) == 0
    assert R.fit_state_file(fit).read().state is Fit.AUDITED
    assert (fit/'fit_receipt.json').read_bytes() == receipt


def test_pre_pinning_recovery_needs_adopt_epoch(in_process):
    work = in_process
    fit = fit_path(work)
    build_legacy(fit, work/'cache', 'recovery_unpinned')
    with pytest.raises(ValueError, match='--adopt-epoch'):
        R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits')])
    assert R.fit_state_file(fit).read().state is Fit.FAILED_RESUMABLE
    assert R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits'), '--adopt-epoch']) == 0
    events = [e['event'] for e in R.fit_state_file(fit).journal()]
    assert events[:3] == ['__adopt__', 'adopt_epoch', 'resume'] and events[-1] == 'audit_passed'
    assert json.loads((fit/'fit_receipt.json').read_text())['resumed_from'] == 8000


def test_lock_held_elsewhere_exits_75(in_process, capsys):
    work = in_process
    fit = fit_path(work)
    fit.mkdir(parents=True)
    script = (f'import sys, time; sys.path.insert(0, {str(ROOT/"src")!r})\n'
              'from alz_perturb_gan.durable_state import FileLock\n'
              f'FileLock({str(fit/"fit")!r}).acquire(); print("held", flush=True); time.sleep(30)\n')
    holder = subprocess.Popen([sys.executable, '-B', '-c', script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'held'
        assert R.main(['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits')]) == R.EX_TEMPFAIL
        assert f'owner pid {holder.pid}' in capsys.readouterr().err
        assert not (fit/R.FIT_STATE).exists()  # nothing happens without the lock
    finally:
        holder.kill()
        holder.wait()


def test_generate_refuses_a_fit_that_is_not_audited(in_process):
    work = in_process
    fit = fit_path(work)
    build_legacy(fit, work/'cache', 'receipted')
    R.adopt_fit(R.fit_state_file(fit), fit)
    with pytest.raises(ValueError, match='only an AUDITED fit generates'):
        R.main(['generate', '--cache', str(work/'cache'), '--fit', str(fit), '--out', str(work/'g.npz')])


# --------------------------------------------------------------------------------------------------
# (f) a refrozen manifest passes check_unchanged_sources

def test_refrozen_manifest_in_a_tmp_checkout_passes_check_unchanged_sources(tmp_path):
    F = _load('refreeze_runtime_manifest_fit_state', ROOT/'scripts/refreeze_runtime_manifest.py')
    manifest = json.loads(R.SOURCE_MANIFEST.read_text())
    for name in ('src/alz_perturb_gan/durable_state.py', 'src/alz_perturb_gan/__init__.py',
                 'scripts/microglia_runtime.py', 'analysis/default_empf_20261117/vendored_host.py'):
        assert name in manifest['sources'], f'{name} is imported by the runtime and must be pinned'
    root = tmp_path/'alz'
    artifacts = [k for k in manifest['sources'] if k.startswith('artifacts/')]
    artifact_root = next((Path(p) for p in (os.environ.get('ALZ_ARTIFACT_ROOT'), ROOT)
                          if p and all((Path(p)/k).exists() for k in artifacts)), None)
    for rel in [k for k in manifest['sources'] if k not in artifacts] + [
            str(R.SOURCE_MANIFEST.relative_to(ROOT)), manifest['archived_previous_entrypoint']['archive']]:
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, root/rel)
    pinned = root/R.SOURCE_MANIFEST.relative_to(ROOT)
    if artifact_root is None:  # no local artifacts here: drop their keys from the tmp manifest only
        tmp_manifest = json.loads(pinned.read_text())
        tmp_manifest['sources'] = {k: v for k, v in tmp_manifest['sources'].items() if k not in artifacts}
        pinned.write_text(json.dumps(tmp_manifest, indent=2))
    else:
        (root/'artifacts').symlink_to(artifact_root/'artifacts')
    copy = _load('microglia_runtime_tmp_checkout', root/'scripts/microglia_runtime.py')
    started, digest = copy.check_sources(), copy.sha(copy.SOURCE_MANIFEST)
    copy.check_unchanged_sources(started, digest)
    with open(root/'scripts/microglia_runtime.py', 'a') as handle:  # an edit to a pinned source
        handle.write('# edited\n')
    with pytest.raises(ValueError):
        copy.check_sources()
    assert F.refreeze(pinned, root, 'test refreeze')['changed'] == ['scripts/microglia_runtime.py']
    started, digest = copy.check_sources(), copy.sha(copy.SOURCE_MANIFEST)
    copy.check_unchanged_sources(started, digest)
