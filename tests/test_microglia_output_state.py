"""Generation output promotion as an explicit state machine (<stem>.state.json; CPU only, no data).

scripts/microglia_runtime.main('generate') drives a stub host whose generator mirrors
analysis/improve_20261109/r5/generate.py (np.savez staged through a hidden .partial.npz, and its own
receipt dumped beside `out`). Source manifest, cache anchors, weights and check_fit are stubbed.

(a) the output transition table, through the durable_state property shim
(b) crash injection (os._exit(137)) at each promotion step: every crash rolls forward to the new
    COMMITTED pair or back to the previous one, and never leaves a new receipt beside an old NPZ;
    the G1 regression (interrupted between the receipt and the NPZ under --force)
(c) 2 to 6 concurrent --force runs onto one --out: one promotes per lock hold, the others exit 75 or
    serialize, the final pair always matches; a rerun after commit exits 3; the G2 regression
(g) legacy outputs are adopted read-only (bytes unchanged), #19 pending sidecars are recognized
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
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


R = _load('microglia_runtime_output_state', ROOT/'scripts/microglia_runtime.py')
M = R.application()
ds = R.ds
Out = R.Out
DEFAULT_MODE = M.load_config()['generation']['mode']
NATIVE_RECIPE = json.loads(M.RECIPE.read_text())['native']['recipe']


# --------------------------------------------------------------------------------------------------
# Stub host (shared with test_microglia_epoch_lease.py and test_microglia_cli_exitcodes.py)

class Stub:
    """Stub host: generate_fit writes `payload` values; train run() emits the derived trainer's events."""

    def __init__(self, payload=3, delay=0.0, during_training=None):
        self.payload = payload
        self.delay = delay
        self.during_training = during_training
        self.activated = 0
        self.generated = []
        self.observed_holders = []

    def activate(self, device, memory):
        self.activated += 1
        stub = self

        def generate_fit(fit, cache, *, mode=None, out=None, device='cpu', **kw):
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            stub.observed_holders.append(R.live_holders())
            if stub.delay:
                time.sleep(stub.delay)
            tmp = out.with_name('.' + out.stem + '.partial.npz')
            with open(tmp, 'wb') as fh:
                np.savez_compressed(fh, x=np.arange(stub.payload))
            tmp.replace(out)
            stub.generated.append(out)
            stats = {'generated_SHA256': R.sha(out), 'fit': str(fit), 'mode': mode, 'payload': stub.payload}
            if mode is not None:
                out.with_name(out.stem + '.receipt.json').write_text(json.dumps(stats))
            return None, stats

        def run(args):
            fit = Path(args.out_root)/'pbulk/hvg2000/fold0'/f'seed{args.seed}'
            fit.mkdir(parents=True, exist_ok=True)
            hook = getattr(T, 'FIT_EVENTS', None) or (lambda *a: None)
            if not (fit/'config.json').exists():
                (fit/'config.json').write_text('{}')
                hook('configure', fit, {})
            if not (fit/'final.pt').exists():
                hook('begin', fit, {'recovery': (fit/'recovery.pt').exists()})
                (fit/'final.pt').write_text('{}')
                hook('final_written', fit, {'step': 20000})
            if not (fit/'fit_receipt.json').exists():
                (fit/'fit_receipt.json').write_text(json.dumps({'status': 'training_completed'}))
                hook('receipt_written', fit, {'step': 20000})
            if stub.during_training is not None:
                stub.during_training()
            return fit

        record = {'calls': 20000, 'observed_steps': 20000}
        q = {'completed_updates': 20000, 'policy': {'optimizers': [{}, {'regularizer': {'record': record}}]}}
        T = SimpleNamespace(torch=SimpleNamespace(load=lambda *a, **kw: q), np=np,
                            recipe=lambda: SimpleNamespace(to_dict=lambda: NATIVE_RECIPE),
                            parser=lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(
                                out_root=argv[argv.index('--out-root')+1], seed=int(argv[argv.index('--seed')+1]))),
                            run=run)
        G = SimpleNamespace(generate_fit=generate_fit, GEN_STREAMS=('a', 'b', 'c', 'd', 'e'))
        return T, G, {'stub': True}, {'cpus': [3, 4, 5], 'source': 'slot_lock'}


def prepare(work, manifest_text='{"sources": {}}'):
    """A work tree: manifest.json, cache/, an audited-looking legacy fit/ (no fit_state.json)."""
    work = Path(work)
    (work/'cache').mkdir(parents=True, exist_ok=True)
    manifest = work/'manifest.json'
    if not manifest.exists():
        manifest.write_text(manifest_text)
    (work/'cache/manifest.json').write_text('{"inner_fold": 0}')
    (work/'fit').mkdir(exist_ok=True)
    (work/'fit/config.json').write_text(json.dumps({'cache_manifest_SHA256': R.sha(work/'cache/manifest.json')}))
    (work/'fit/fit_receipt.json').write_text('{}')
    return work


def install(setattr_, work, stub, module=None, manifest=None):
    """Point the runtime at `work` and the stub host (setattr_ is monkeypatch.setattr or plain setattr)."""
    module = R if module is None else module
    setattr_(module, 'SOURCE_MANIFEST', Path(manifest) if manifest is not None else Path(work)/'manifest.json')
    setattr_(module, 'check_sources', lambda: {'sources': {}})
    setattr_(module, 'check_cache', lambda cache, fold, held=False: {'anchor': fold})
    setattr_(module, 'check_weights', lambda fit, receipt: {})
    setattr_(module, 'check_checkpoint', lambda q, saved: None)
    setattr_(module, 'activate', stub.activate)
    setattr_(M, 'check_fit', lambda *a, **kw: None)
    setattr_(module, 'application', lambda: M)
    return stub


def gen_argv(work, *extra, out=None):
    out = Path(out) if out is not None else Path(work)/'generated/out.npz'
    return ['generate', '--cache', str(Path(work)/'cache'), '--fit', str(Path(work)/'fit'), '--out', str(out),
            '--mode', DEFAULT_MODE, *extra]


def out_of(work):
    return Path(work)/'generated/out.npz'


def payload_of(path):
    return len(np.load(path)['x'])


def pair(work):
    """(payload of out.npz, payload the receipt describes, receipt matches the NPZ)."""
    out = out_of(work)
    receipt = json.loads(R.receipt_path(out).read_text())
    return payload_of(out), receipt['payload'], receipt['generated_SHA256'] == R.sha(out)


def staging_files(work):
    names = R._pending_names(out_of(work))
    return [p.name for p in names['run'] + names['legacy']]


def journal_events(work):
    return [e['event'] for e in R.output_state_file(out_of(work)).journal()]


def child_main():
    """Subprocess entry: microglia_runtime.cli(generate ...) with the stub host; exits with its code."""
    work = Path(os.environ['OUT_WORK'])
    stub = Stub(payload=int(os.environ.get('OUT_PAYLOAD', '3')), delay=float(os.environ.get('OUT_DELAY', '0')))
    install(setattr, work, stub)
    go = os.environ.get('OUT_GO')
    if go:
        while not Path(go).exists():
            time.sleep(0.002)
    rc = R.cli(gen_argv(work, *json.loads(os.environ.get('OUT_ARGS', '[]'))))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def run_child(work, *args, payload=3, fault=None, delay=0.0, go=None, start=False, timeout=120):
    env = dict(os.environ, OUT_WORK=str(work), OUT_ARGS=json.dumps(list(args)), OUT_PAYLOAD=str(payload),
               OUT_DELAY=str(delay), PYTHONDONTWRITEBYTECODE='1')
    env.pop(ds.FAULT_ENV, None)
    env.pop('OUT_GO', None)
    if fault:
        env[ds.FAULT_ENV] = fault
    if go:
        env['OUT_GO'] = str(go)
    code = f'import sys; sys.path.insert(0, {str(TESTS)!r}); import test_microglia_output_state as H; H.child_main()'
    proc = subprocess.Popen([sys.executable, '-B', '-c', code], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if start:
        return proc
    out, err = proc.communicate(timeout=timeout)
    return proc.returncode, out, err


@pytest.fixture
def work(tmp_path, monkeypatch):
    work = prepare(tmp_path/'w')
    stub = install(monkeypatch.setattr, work, Stub())
    return SimpleNamespace(path=work, stub=stub, out=out_of(work))


# --------------------------------------------------------------------------------------------------
# (a) the transition table

STABLE = ('NONE', 'COMMITTED', 'INCONSISTENT', 'ORPHAN_RECEIPT', 'UNRECEIPTED_NPZ')
EXPECTED_TABLE = {
    **{(s, 'stage'): 'STAGING' for s in STABLE},
    ('STAGING', 'post_checks_passed'): 'STAGED',
    **{(src, f'abort_to_{s.lower()}'): s for src in ('STAGING', 'STAGED') for s in STABLE},
    ('STAGED', 'begin_promote'): 'PROMOTING',
    ('PROMOTING', 'commit'): 'COMMITTED',
    ('PROMOTING', 'lost'): 'INCONSISTENT',
    **{(a, f'observed_{b.lower()}'): b for a in STABLE for b in STABLE if a != b or a in ('COMMITTED', 'INCONSISTENT')},
}


def test_output_machine_is_the_specified_table():
    table = {(s.name, e): t.name for (s, e), t in R.OUT_MACHINE.transitions.items()}
    assert table == EXPECTED_TABLE
    assert R.OUT_MACHINE.initial is Out.NONE and R.OUT_MACHINE.terminal == frozenset()
    # PROMOTING only rolls forward (or is journaled lost); it never returns to a staging state.
    assert R.OUT_MACHINE.allowed_events(Out.PROMOTING) == ['commit', 'lost']
    assert set(R.OUT_MACHINE.adopt_entries) == {Out.COMMITTED, Out.INCONSISTENT, Out.ORPHAN_RECEIPT,
                                                Out.UNRECEIPTED_NPZ}


def test_output_transition_table_property(tmp_path):
    """Random event sequences through durable_state's property shim (tests/test_durable_state.py)."""
    import test_durable_state as shim
    machine = R.OUT_MACHINE
    rng = random.Random(20261120)
    events = sorted(machine.events) + ['undeclared']
    for i in range(300):
        sf = ds.StateFile(tmp_path/f'out-{i}.state.json', machine)
        sf.create()
        model = shim.Model(machine)
        for _ in range(rng.randint(1, 12)):
            legal = machine.allowed_events(model.state)
            event = rng.choice(legal) if legal and rng.random() < 0.7 else rng.choice(events)
            shim._step(sf, model, event, rng.random() >= 0.1)
            shim._check(sf, model)


def test_stage_guard_requires_force_outside_none(tmp_path):
    out = tmp_path/'o.npz'
    sf = R.output_state_file(out)
    sf.create()
    snap = sf.transition('observed_committed')
    with pytest.raises(ds.GuardFailed):
        R.stage_output(sf, out, snap, False, R.new_run_id())
    assert sf.read().state is Out.COMMITTED
    snap = R.stage_output(sf, out, snap, True, R.new_run_id())
    assert snap.state is Out.STAGING and snap.data['previous']['state'] == 'COMMITTED'


def test_pending_names_are_unique_per_run(tmp_path):
    out = tmp_path/f'generated_{DEFAULT_MODE}.npz'
    a, b = R.new_run_id(), R.new_run_id()
    assert a != b
    pa, pb = R.run_pending(out, a), R.run_pending(out, b)
    assert pa[0] != pb[0] and pa[1] != pb[1]
    assert pa[0].name == f'.{out.stem}.{a}.pending.npz' and pa[1].name == f'.{out.stem}.{a}.pending.receipt.json'
    for path in (*pa, R.pending_path(out), out.with_name('.' + R.pending_path(out).stem + '.partial.npz')):
        path.write_text('x')
    names = R._pending_names(out)
    assert sorted(p.name for p in names['run']) == sorted(p.name for p in pa)
    assert sorted(p.name for p in names['legacy']) == sorted([R.pending_path(out).name,
                                                               '.' + R.pending_path(out).stem + '.partial.npz'])
    # Another output's staging files in the same directory are never claimed.
    other = tmp_path/'generated_other.npz'
    R.run_pending(other, a)[0].write_text('x')
    assert all('other' not in p.name for p in R._pending_names(out)['run'])


# --------------------------------------------------------------------------------------------------
# A clean run, and the readers

def test_clean_generation_commits_and_rerun_is_already_done(work):
    assert R.main(gen_argv(work.path)) == 0
    snap = R.output_state_file(work.out).read()
    assert snap.state is Out.COMMITTED and snap.data['npz_sha'] == R.sha(work.out)
    assert snap.data['receipt_sha'] == R.sha(R.receipt_path(work.out))
    assert journal_events(work.path) == ['__adopt__', 'stage', 'post_checks_passed', 'begin_promote', 'commit']
    assert R.is_committed(work.out) and M.output_state(work.out) == 'COMMITTED'
    assert staging_files(work.path) == []
    activated = work.stub.activated
    assert R.main(gen_argv(work.path)) == R.ALREADY_DONE  # exit 3, distinguishable from a usage error
    assert work.stub.activated == activated
    assert R.main(gen_argv(work.path, '--force')) == 0
    assert journal_events(work.path)[-4:] == ['stage', 'post_checks_passed', 'begin_promote', 'commit']


def test_failed_post_check_rolls_back_to_the_previous_pair(work, monkeypatch):
    assert R.main(gen_argv(work.path)) == 0
    before = work.out.read_bytes(), R.receipt_path(work.out).read_bytes()
    work.stub.payload = 9

    def failing(cache, fold, held=False):
        if held and len(work.stub.generated) > 1:  # only as this run's post-check
            raise ValueError('development cache payload changed: heldout/counts.npz')
        return {}
    monkeypatch.setattr(R, 'check_cache', failing)
    with pytest.raises(ValueError, match='payload changed'):
        R.main(gen_argv(work.path, '--force'))
    assert (work.out.read_bytes(), R.receipt_path(work.out).read_bytes()) == before
    snap = R.output_state_file(work.out).read()
    assert snap.state is Out.COMMITTED and journal_events(work.path)[-1] == 'abort_to_committed'
    assert staging_files(work.path) == [] and R.is_committed(work.out)


def test_is_committed_requires_the_recorded_pair(work):
    assert R.main(gen_argv(work.path)) == 0
    receipt = R.receipt_path(work.out)
    original = receipt.read_bytes()
    receipt.write_bytes(original.replace(b'}', b', "edited": true}', 1))  # still names the NPZ's sha
    assert R.infer_output(work.out)[0] is Out.COMMITTED  # the legacy rule alone would accept it
    assert not R.is_committed(work.out) and M.output_state(work.out) == 'INCONSISTENT'
    receipt.write_bytes(original)
    assert R.is_committed(work.out)


# --------------------------------------------------------------------------------------------------
# (b) crash injection at each promotion step

ROLL_BACK = ('out_after_pending_npz', 'out_after_pending_receipt')
ROLL_FORWARD = ('out_after_begin_promote', 'out_after_npz_replace', 'out_after_receipt_replace', 'out_before_commit')
assert set(ROLL_BACK + ROLL_FORWARD) == set(R.OUTPUT_FAULTS)


@pytest.mark.parametrize('fault', R.OUTPUT_FAULTS)
def test_crash_at_each_promotion_step_converges(tmp_path, fault):
    work = prepare(tmp_path/'w')
    assert run_child(work, payload=3)[0] == 0
    old = pair(work)
    assert old == (3, 3, True)
    rc, out, err = run_child(work, '--force', payload=7, fault=fault)
    assert rc == ds.FAULT_EXIT_CODE, err
    # Whatever was on disk at the crash: never a new receipt beside an old NPZ.
    npz, described, _ = pair(work)
    assert not (described == 7 and npz == 3)
    recorded = M.output_state(out_of(work))
    assert recorded in ('STAGING', 'STAGED', 'PROMOTING')
    assert not R.is_committed(out_of(work))  # an interrupted promotion is never reported done
    rc, out, err = run_child(work, payload=11)  # the next open recovers, then finds a COMMITTED output
    assert rc == R.ALREADY_DONE, err
    expected = 3 if fault in ROLL_BACK else 7
    assert pair(work) == (expected, expected, True)
    assert R.output_state_file(out_of(work)).read().state is Out.COMMITTED and R.is_committed(out_of(work))
    events = journal_events(work)
    assert events[-1] == ('abort_to_committed' if fault in ROLL_BACK else 'commit')
    assert staging_files(work) == []


@pytest.mark.parametrize('fault', ['out_after_pending_npz', 'out_after_begin_promote', 'out_after_npz_replace'])
def test_crash_during_a_first_generation(tmp_path, fault):
    work = prepare(tmp_path/'w')
    assert run_child(work, payload=5, fault=fault)[0] == ds.FAULT_EXIT_CODE
    assert not R.is_committed(out_of(work))
    rc, out, err = run_child(work, payload=5)
    if fault == 'out_after_pending_npz':  # rolled back to NONE, then generated
        assert rc == 0, err
        assert journal_events(work)[-5:] == ['abort_to_none', 'stage', 'post_checks_passed', 'begin_promote', 'commit']
    else:  # rolled forward, then already done
        assert rc == R.ALREADY_DONE, err
    assert pair(work) == (5, 5, True) and staging_files(work) == []


def test_g1_interrupt_between_receipt_and_npz_under_force(work, monkeypatch):
    """G1: a --force run stopped (SIGINT) between the receipt write and the NPZ replace."""
    assert R.main(gen_argv(work.path)) == 0
    work.stub.payload = 8
    real_fault = ds._fault

    def interrupt(name):
        if name == 'out_after_begin_promote':
            raise KeyboardInterrupt('simulated SIGINT between receipt write and NPZ replace')
        real_fault(name)
    monkeypatch.setattr(R.ds, '_fault', interrupt)
    with pytest.raises(KeyboardInterrupt):
        R.main(gen_argv(work.path, '--force'))
    monkeypatch.setattr(R.ds, '_fault', real_fault)
    # The old pair is still in place; the state says PROMOTING, so it is not reported as done.
    assert pair(work.path) == (3, 3, True)
    assert M.output_state(work.out) == 'PROMOTING' and not R.is_committed(work.out)
    assert R.main(gen_argv(work.path)) == R.ALREADY_DONE  # rolled forward from the retained pending files
    assert pair(work.path) == (8, 8, True) and staging_files(work.path) == []


def test_g1_legacy_mismatched_pair_is_inconsistent_never_done(work):
    """A pre-state G1 leftover (new receipt beside the old NPZ) is INCONSISTENT; only --force replaces it."""
    assert R.main(gen_argv(work.path)) == 0
    for name in ('out.state.json', 'out.state.json.journal.jsonl'):
        (work.out.parent/name).unlink()
    receipt = json.loads(R.receipt_path(work.out).read_text())
    receipt.update(generated_SHA256='f'*64, payload=8)
    R.receipt_path(work.out).write_text(json.dumps(receipt))
    before = work.out.read_bytes(), R.receipt_path(work.out).read_bytes()
    assert not R.is_committed(work.out) and M.output_state(work.out) == 'INCONSISTENT'
    with pytest.raises(FileExistsError, match='INCONSISTENT'):
        R.main(gen_argv(work.path))
    assert (work.out.read_bytes(), R.receipt_path(work.out).read_bytes()) == before  # adoption wrote neither
    assert R.output_state_file(work.out).read().state is Out.INCONSISTENT
    assert R.main(gen_argv(work.path, '--force')) == 0
    assert pair(work.path) == (3, 3, True) and R.is_committed(work.out)


# --------------------------------------------------------------------------------------------------
# (c) concurrency

@pytest.mark.parametrize('n', [2, 3, 6])
def test_concurrent_force_runs_onto_one_out(tmp_path, n):
    work = prepare(tmp_path/'w')
    assert run_child(work, payload=3)[0] == 0
    go = tmp_path/'go'
    procs = [run_child(work, '--force', payload=10 + i, delay=0.4, go=go, start=True) for i in range(n)]
    time.sleep(0.5)
    go.write_text('go')
    results = [(p.wait(timeout=120), *p.communicate()) for p in procs]
    codes = sorted(r[0] for r in results)
    assert set(codes) <= {0, R.EX_TEMPFAIL}, results
    assert codes.count(0) >= 1, results
    for rc, out, err in results:
        if rc == R.EX_TEMPFAIL:
            assert 'is busy' in err
    npz, described, matches = pair(work)
    assert matches and npz == described and npz in range(10, 10 + n)
    # One promotion per lock hold: the journal never interleaves two runs' promotion steps.
    events = journal_events(work)[5:]  # after the first, uncontended generation
    assert events == ['stage', 'post_checks_passed', 'begin_promote', 'commit'] * codes.count(0)
    assert staging_files(work) == []
    assert run_child(work)[0] == R.ALREADY_DONE  # a rerun after commit


def test_g2_second_force_run_during_the_first_is_busy(work, monkeypatch):
    """G2: two --force runs onto one --out used to share pending names; now the second exits 75."""
    assert R.main(gen_argv(work.path)) == 0
    entered, release = threading.Event(), threading.Event()
    real_activate = work.stub.activate

    def slow_activate(device, memory):
        T, G, prov, cpus = real_activate(device, memory)
        inner = G.generate_fit

        def gen(*a, **kw):
            entered.set()
            release.wait(10)
            return inner(*a, **kw)
        G.generate_fit = gen
        return T, G, prov, cpus
    monkeypatch.setattr(R, 'activate', slow_activate)
    work.stub.payload = 7
    results = {}
    a = threading.Thread(target=lambda: results.setdefault('A', R.main(gen_argv(work.path, '--force'))))
    a.start()
    assert entered.wait(10)
    results['B'] = R.main(gen_argv(work.path, '--force'))
    release.set()
    a.join(20)
    assert results == {'A': 0, 'B': R.EX_TEMPFAIL}
    assert pair(work.path) == (7, 7, True) and staging_files(work.path) == []


# --------------------------------------------------------------------------------------------------
# (g) legacy outputs

def _legacy(out, kind):
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind in ('pair', 'mismatch', 'npz'):
        with open(out, 'wb') as fh:
            np.savez_compressed(fh, x=np.arange(4))
    if kind in ('pair', 'mismatch', 'receipt'):
        digest = R.sha(out) if kind == 'pair' else 'a'*64
        R.receipt_path(out).write_text(json.dumps({'generated_SHA256': digest, 'payload': 4}))


@pytest.mark.parametrize('kind,state', [('pair', Out.COMMITTED), ('mismatch', Out.INCONSISTENT),
                                        ('receipt', Out.ORPHAN_RECEIPT), ('npz', Out.UNRECEIPTED_NPZ),
                                        ('none', Out.NONE)])
def test_legacy_output_is_adopted_read_only(work, kind, state):
    _legacy(work.out, kind)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in work.out.parent.glob('out*')} \
        if work.out.parent.exists() else {}
    assert R.infer_output(work.out)[0] is state
    assert M.output_state(work.out) == state.name  # the CLI's read-only reader agrees, writing nothing
    assert not R.output_state_path(work.out).exists()
    rc = None
    try:
        rc = R.main(gen_argv(work.path))
    except FileExistsError:
        pass
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in work.out.parent.glob('out*')
             if p.name in before}
    snap = R.output_state_file(work.out).read()
    if state is Out.NONE:
        assert rc == 0 and snap.state is Out.COMMITTED
        return
    assert after == before  # no legacy NPZ or receipt byte (or mtime) changed
    assert snap.state is state and R.output_state_file(work.out).journal()[0]['event'] == '__adopt__'
    assert rc == (R.ALREADY_DONE if state is Out.COMMITTED else None)
    assert R.is_committed(work.out) is (state is Out.COMMITTED)


def test_legacy_pending_sidecars_are_swept_only_when_old(work):
    work.out.parent.mkdir(parents=True)
    old = R.pending_path(work.out)
    young = R.receipt_path(old)
    old.write_bytes(b'legacy pending npz')
    young.write_bytes(b'{}')
    stale = time.time() - R.LEGACY_SIDECAR_MIN_AGE_S - 60
    os.utime(old, (stale, stale))
    dead_run = R.run_pending(work.out, R.new_run_id())[0]
    dead_run.write_bytes(b'a dead run')
    assert R.main(gen_argv(work.path)) == 0
    assert not old.exists() and not dead_run.exists()
    assert young.exists()  # a pre-lock (#19) runtime may still own it


def test_derived_generator_refuses_to_overwrite_without_force():
    VH = _load('vendored_host_output_state', ROOT/'analysis/default_empf_20261117/vendored_host.py')
    bodies, _ = VH.derive_bodies('native')
    assert '    if out.exists() and not force:\n' in bodies['generate.py']
    assert 'and mode is not None:\n' not in bodies['generate.py'].split('def generate_fit', 1)[1].split('T = targets')[0]
    original = (ROOT/'analysis/improve_20261109/r5/generate.py').read_text()
    assert '    if out.exists() and not force and mode is not None:\n' in original  # the pinned original is unchanged
