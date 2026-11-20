"""Regressions from the adversarial review of the output-promotion and source-epoch state machines.

Each test is a repro from the review (break-sm-runtime-output-epoch/b1..b7), now fixed:
(1) scripts/microglia.py hands an output an interrupted run left STAGING/STAGED/PROMOTING to the runtime
    without --force (roll back and generate, or roll forward and exit 3) instead of a usage error
(2) a receipt that is not a JSON object (and a state file that is a JSON list) never raises from
    is_committed or generate; it reads as INCONSISTENT (or UNREADABLE) and --force replaces it
(3) a consistent pair the machine did not commit (receipt edited in place, pair copied in) is observed as
    INCONSISTENT, never laundered into COMMITTED by a routine run
(4) refreeze and runtime resolve the same lease directory ($MICROGLIA_EPOCH_DIR honoured by both)
(5) in-process consumers (check_sources/activate, or `with source_epoch_lease()`) hold the lease
(6) a waiting refreeze is not starved by overlapping runtimes (intent gate)
(7) an unwritable epoch directory: --dry-run proceeds, real runs fail with a clear message, an existing
    lease file is locked read-only
(8) damaged output state files exit 4 and `generate --readopt` recovers them; an unknown previous state or
    a PROMOTING record without its pending entry no longer crashes; is_committed validates the state file
(9) PROMOTING never moves a file named by an absolute or foreign pending path
(10) temp files of dead writers beside an output, and holder records of dead runtimes, are swept
(11) a short random-SIGKILL fuzzer (the b1 fuzzer, bounded) converges to a matching COMMITTED pair
CPU only; no data, GPU or torch.
"""
import errno
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_microglia_output_state as H  # noqa: E402  (shared stub host)

R, M, ds, Out = H.R, H.M, H.ds, H.Out


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _load('refreeze_runtime_manifest_review', ROOT/'scripts/refreeze_runtime_manifest.py')


@pytest.fixture
def work(tmp_path, monkeypatch):
    work = H.prepare(tmp_path/'w')
    stub = H.install(monkeypatch.setattr, work, H.Stub())
    return type('W', (), {'path': work, 'stub': stub, 'out': H.out_of(work)})


def cli(argv):
    """R.cli; any exception escaping it is a failure (the review found AttributeError tracebacks)."""
    return R.cli(argv)


def state_doc(out):
    return json.loads(R.output_state_path(out).read_text())


def put_state(out, doc):
    R.output_state_path(out).write_text(json.dumps(doc))


def dead_pid():
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


# --------------------------------------------------------------------------------------------------
# (1) the CLI lets the runtime recover an interrupted output

@pytest.fixture
def cli_root(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'ROOT', tmp_path/'root')
    cache = tmp_path/'root/runs/improve_20261109/caches/hvg2000/fold0'
    cache.mkdir(parents=True)
    (cache/'manifest.json').write_text('{"inner_fold": 0}')
    monkeypatch.setattr(M, 'check_fit', lambda *a, **kw: None)
    return tmp_path


def route_to_runtime(monkeypatch, work):
    """M.subprocess.run -> the runtime (stub host) in a child process; records each command."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        rc, out, err = H.run_child(work, *(['--force'] if '--force' in cmd else []), payload=5)
        return subprocess.CompletedProcess(cmd, rc, out, err)
    monkeypatch.setattr(M.subprocess, 'run', run)
    return calls


@pytest.mark.parametrize('fault', R.OUTPUT_FAULTS)
def test_cli_recovers_an_interrupted_first_generation(tmp_path, cli_root, monkeypatch, fault):
    work = H.prepare(tmp_path/'w')
    assert H.run_child(work, payload=5, fault=fault)[0] == ds.FAULT_EXIT_CODE
    out = H.out_of(work)
    state = M.output_state(out)
    assert state in M.RECOVERABLE_OUTPUT
    calls = route_to_runtime(monkeypatch, work)
    rc = M.main(['generate', '--fit', str(work/'fit'), '--out', str(out)])
    assert len(calls) == 1 and '--force' not in calls[0]  # recovered without discarding a roll-forward
    assert rc == (0 if state in ('STAGING', 'STAGED') else M.ALREADY_DONE)
    assert H.pair(work) == (5, 5, True) and R.is_committed(out) and M.output_state(out) == 'COMMITTED'
    assert H.staging_files(work) == []


def test_cli_recovers_an_interrupted_force_run_back_to_the_committed_pair(tmp_path, cli_root, monkeypatch):
    work = H.prepare(tmp_path/'w')
    assert H.run_child(work, payload=3)[0] == 0
    assert H.run_child(work, '--force', payload=7, fault='out_after_pending_receipt')[0] == ds.FAULT_EXIT_CODE
    out = H.out_of(work)
    assert M.output_state(out) == 'STAGING'
    calls = route_to_runtime(monkeypatch, work)
    assert M.main(['generate', '--fit', str(work/'fit'), '--out', str(out)]) == M.ALREADY_DONE
    assert '--force' not in calls[0] and H.pair(work) == (3, 3, True) and R.is_committed(out)


def test_cli_refusal_names_only_existing_files(tmp_path, cli_root, monkeypatch, capsys):
    monkeypatch.setattr(M.subprocess, 'run', lambda *a, **kw: pytest.fail('subprocess started'))
    out = tmp_path/'o/out.npz'
    out.parent.mkdir(parents=True)
    M.receipt_path(out).write_text('{}')  # ORPHAN_RECEIPT
    with pytest.raises(SystemExit) as usage:
        M.main(['generate', '--fit', str(tmp_path/'fit'), '--out', str(out)])
    err = capsys.readouterr().err
    assert usage.value.code == M.EX_USAGE and 'ORPHAN_RECEIPT' in err
    assert str(M.receipt_path(out)) in err and f'{out},' not in err and not err.rstrip().endswith(str(out))
    M.receipt_path(out).unlink()
    Path(str(R.output_state_path(out)) + ds.JOURNAL_SUFFIX).write_text('')  # journal without state: UNREADABLE
    with pytest.raises(SystemExit) as usage:
        M.main(['generate', '--fit', str(tmp_path/'fit'), '--out', str(out)])
    err = capsys.readouterr().err
    assert usage.value.code == M.EX_USAGE and 'no output files' in err and '--readopt' in err


# --------------------------------------------------------------------------------------------------
# (2) receipts and state files that are not JSON objects

@pytest.mark.parametrize('body', ['[1, 2]', '"x"', '5', 'null', '{"generated_SHA'])
def test_non_object_receipt_is_inconsistent_and_force_replaces_it(work, body):
    work.out.parent.mkdir(parents=True)
    with open(work.out, 'wb') as fh:
        np.savez_compressed(fh, x=np.arange(4))
    R.receipt_path(work.out).write_text(body)
    assert R.is_committed(work.out) is False
    assert R.infer_output(work.out)[0] is Out.INCONSISTENT and M.output_state(work.out) == 'INCONSISTENT'
    assert cli(H.gen_argv(work.path)) == 1  # FileExistsError, not an AttributeError traceback
    assert R.output_state_file(work.out).read().state is Out.INCONSISTENT
    assert cli(H.gen_argv(work.path, '--force')) == 0
    assert H.pair(work.path) == (3, 3, True) and R.is_committed(work.out)


@pytest.mark.parametrize('mutate', ['list', 'data_list', 'bogus_state', 'wrong_machine', 'bad_schema', 'bad_version',
                                    'truncated'])
def test_invalid_state_file_is_never_committed(work, mutate):
    assert R.main(H.gen_argv(work.path)) == 0
    doc = state_doc(work.out)
    changed = {'list': [doc], 'data_list': {**doc, 'data': []}, 'bogus_state': {**doc, 'state': 'BOGUS'},
               'wrong_machine': {**doc, 'machine': 'microglia_fit'}, 'bad_schema': {**doc, 'schema': 'x/0'},
               'bad_version': {**doc, 'version': 0}, 'truncated': None}[mutate]
    if changed is None:
        R.output_state_path(work.out).write_text(R.output_state_path(work.out).read_text()[:40])
    else:
        put_state(work.out, changed)
    assert R.is_committed(work.out) is False and M.output_state(work.out) == 'UNREADABLE'
    assert cli(H.gen_argv(work.path)) == R.EX_STRANDED  # damaged: exit 4 with the --readopt hint


# --------------------------------------------------------------------------------------------------
# (3) foreign consistent pairs are INCONSISTENT, never laundered

def test_edited_receipt_is_not_laundered_by_a_routine_run(work):
    """b7: forge fields of a COMMITTED receipt but keep its generated_SHA256."""
    assert R.main(H.gen_argv(work.path)) == 0
    receipt = json.loads(R.receipt_path(work.out).read_text())
    receipt['application_default']['source_manifest_sha256'] = '0'*64
    receipt['cache_anchor'] = {'forged': True}
    R.receipt_path(work.out).write_text(json.dumps(receipt))
    assert not R.is_committed(work.out) and M.output_state(work.out) == 'INCONSISTENT'
    for _ in range(2):  # sticky: a second routine run does not flip it back either
        assert cli(H.gen_argv(work.path)) == 1
        assert not R.is_committed(work.out) and M.output_state(work.out) == 'INCONSISTENT'
        assert R.output_state_file(work.out).read().state is Out.INCONSISTENT
    assert H.journal_events(work.path)[-1] == 'observed_inconsistent'
    assert cli(H.gen_argv(work.path, '--force')) == 0 and R.is_committed(work.out)


@pytest.mark.parametrize('recorded', ['COMMITTED', 'NONE'])
def test_pair_copied_in_is_inconsistent(work, recorded):
    assert R.main(H.gen_argv(work.path)) == 0
    if recorded == 'NONE':  # outputs deleted; a routine run records NONE before the copy arrives
        work.out.unlink()
        R.receipt_path(work.out).unlink()
        R.open_output(work.out)
        assert R.output_state_file(work.out).read().state is Out.NONE
    with open(work.out, 'wb') as fh:
        np.savez_compressed(fh, x=np.arange(6))
    R.receipt_path(work.out).write_text(json.dumps({'generated_SHA256': R.sha(work.out), 'payload': 6}))
    assert not R.is_committed(work.out) and M.output_state(work.out) == 'INCONSISTENT'
    assert cli(H.gen_argv(work.path)) == 1
    assert not R.is_committed(work.out) and R.output_state_file(work.out).read().state is Out.INCONSISTENT
    assert R.output_state_file(work.out).read().data['observed']['foreign_pair'] is True


def test_legacy_consistent_pair_is_still_adopted_committed(work):
    work.out.parent.mkdir(parents=True)
    with open(work.out, 'wb') as fh:
        np.savez_compressed(fh, x=np.arange(3))
    R.receipt_path(work.out).write_text(json.dumps({'generated_SHA256': R.sha(work.out), 'payload': 3}))
    assert R.is_committed(work.out) and M.output_state(work.out) == 'COMMITTED'
    assert cli(H.gen_argv(work.path)) == R.ALREADY_DONE and R.is_committed(work.out)


# --------------------------------------------------------------------------------------------------
# (4) refreeze and runtime use one lease directory

def test_refreeze_and_runtime_resolve_the_same_epoch_dir(monkeypatch, tmp_path):
    assert F.epoch_dir() == R.epoch_dir() == Path(os.environ[R.EPOCH_DIR_ENV])
    assert F.EPOCH_DIR_ENV == R.EPOCH_DIR_ENV
    monkeypatch.delenv(R.EPOCH_DIR_ENV)
    assert F.epoch_dir() == R.epoch_dir() == ROOT/'runs/.source-epoch'
    assert F.epoch_dir(tmp_path) == R.epoch_dir(tmp_path) == tmp_path/'runs/.source-epoch'


def test_refreeze_main_honours_the_env_lease_dir(capsys):
    """b5 part 1: a runtime lease under $MICROGLIA_EPOCH_DIR blocks refreeze main() without --epoch-dir."""
    lease = R.EpochLease(['probe']).acquire()
    try:
        assert F.main(['--reason', 'probe']) == F.EX_BUSY
        assert f'pid {os.getpid()}' in capsys.readouterr().err
    finally:
        lease.release()


def test_refreeze_of_a_manifest_outside_the_checkout_needs_epoch_dir(tmp_path, capsys):
    with pytest.raises(SystemExit) as usage:
        F.main(['--reason', 'x', '--manifest', str(tmp_path/'other/m.json')])
    assert usage.value.code == 2 and '--epoch-dir' in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------
# (5) in-process consumers hold the lease

LIBRARY = textwrap.dedent('''
    import importlib.util, json, sys, time
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('rt', sys.argv[1])
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    R.SOURCE_MANIFEST = Path(sys.argv[2])
    R.check_sources()   # a scorer that imports the runtime: no main(), no explicit lease
    print('checked', flush=True)
    time.sleep(600)
''')


def test_library_consumer_holds_the_lease_from_check_sources(tmp_path):
    entry = tmp_path/'old.py'
    entry.write_text('old\n')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'sources': {}, 'archived_previous_entrypoint':
                                    {'archive': str(entry), 'sha256': R.sha(entry)}}))
    epoch = Path(os.environ[R.EPOCH_DIR_ENV])
    proc = subprocess.Popen([sys.executable, '-B', '-c', LIBRARY, str(ROOT/'scripts/microglia_runtime.py'),
                             str(manifest)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == 'checked', proc.stderr.read()
        assert [h['pid'] for h in R.live_holders(epoch)] == [proc.pid]
        with pytest.raises(F.EpochBusy):
            F.exclusive_epoch(epoch)
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    F.exclusive_epoch(epoch).release()


def test_source_epoch_lease_context_manager(tmp_path):
    epoch = Path(os.environ[R.EPOCH_DIR_ENV])
    with R.source_epoch_lease(['score.py']) as lease:
        assert [h['cmd'] for h in R.live_holders(epoch)] == [['score.py']]
        with R.source_epoch_lease(['nested']) as inner:  # reentrant: the held lease is reused
            assert inner is lease
        assert R.hold_source_epoch() is lease
        with pytest.raises(F.EpochBusy):
            F.exclusive_epoch(epoch)
    assert R.live_holders(epoch) == []
    F.exclusive_epoch(epoch).release()


# --------------------------------------------------------------------------------------------------
# (6) a waiting refreeze is not starved

def test_waiting_refreeze_is_not_starved_by_overlapping_runtimes(tmp_path):
    """b5 part 2: a new runtime starts every 0.1 s and lives 0.25 s; refreeze --wait still gets in."""
    epoch = tmp_path/'epoch'
    got = {}
    stop = threading.Event()
    errors = []

    def runtime():
        try:
            lease = R.EpochLease(['churn'], directory=epoch, timeout=30).acquire()
            time.sleep(0.25)
            lease.release()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def churn():
        threads = []
        while not stop.is_set():
            t = threading.Thread(target=runtime)
            t.start()
            threads.append(t)
            time.sleep(0.1)
        for t in threads:
            t.join()

    first = R.EpochLease(['first'], directory=epoch).acquire()
    churner = threading.Thread(target=churn)
    churner.start()
    time.sleep(0.3)

    def waiter():
        lock = F.exclusive_epoch(epoch, wait=True)
        got['at'] = time.monotonic()
        time.sleep(0.05)
        lock.release()
    start = time.monotonic()
    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.3)
    first.release()
    t.join(5)
    stop.set()
    churner.join(60)
    assert 'at' in got and got['at'] - start < 3, 'refreeze --wait starved by overlapping runtimes'
    assert errors == []  # runtimes queued behind the refreeze rather than failing


# --------------------------------------------------------------------------------------------------
# (7) an unwritable epoch directory

@pytest.fixture
def read_only_dir(tmp_path):
    if os.geteuid() == 0:
        pytest.skip('root ignores directory permissions')
    ro = tmp_path/'ro'
    ro.mkdir()
    yield ro
    os.chmod(ro, 0o755)
    for path in ro.rglob('*'):
        if path.is_dir():
            os.chmod(path, 0o755)


def test_unwritable_epoch_dir(read_only_dir, monkeypatch, capsys):
    os.chmod(read_only_dir, 0o555)
    missing = read_only_dir/'epoch'
    lease = R.EpochLease(['dry'], directory=missing, optional=True).acquire()
    assert lease.degraded == 'no lease' and lease.held
    lease.release()
    with pytest.raises(PermissionError, match='--dry-run works without it'):
        R.EpochLease(['real'], directory=missing).acquire()
    # train --dry-run proceeds (exit of _main), generate fails with exit 1 and the message.
    monkeypatch.setenv(R.EPOCH_DIR_ENV, str(missing))
    monkeypatch.setattr(R, '_main', lambda p, args, lease: 0)
    assert R.cli(['train', '--cache', 'c', '--out-root', 'o', '--dry-run']) == 0
    assert R.cli(['generate', '--cache', 'c', '--fit', 'f', '--out', 'o.npz']) == 1
    assert 'train and generate need write access' in capsys.readouterr().err


def test_existing_lease_in_an_unwritable_dir_is_held_read_only(read_only_dir):
    epoch = read_only_dir/'epoch'
    F.exclusive_epoch(epoch).release()  # a refreeze once created the lease and gate files
    os.chmod(epoch/'holders', 0o555)
    os.chmod(epoch, 0o555)
    lease = R.EpochLease(['ro'], directory=epoch, timeout=0.2).acquire()
    try:
        assert lease.degraded == 'read-only' and lease.lock is not None and lease.record is None
        probe = ds.FileLock(epoch/'lease', exclusive=True, blocking=False)
        with pytest.raises(ds.Busy):
            probe.acquire()
    finally:
        lease.release()


# --------------------------------------------------------------------------------------------------
# (8) damaged output state files and --readopt

def damage(out, how):
    state = R.output_state_path(out)
    journal = Path(str(state) + ds.JOURNAL_SUFFIX)
    if how == 'state_deleted':
        state.unlink()
    elif how == 'journal_ahead':
        lines = journal.read_text().splitlines()
        last = json.loads(lines[-1])
        last['seq'] += 1
        journal.write_text('\n'.join(lines + [json.dumps(last)]) + '\n')
    elif how == 'wrong_machine':
        put_state(out, {**state_doc(out), 'machine': 'microglia_fit'})
    elif how == 'bogus_state':
        put_state(out, {**state_doc(out), 'state': 'BOGUS'})


@pytest.mark.parametrize('how', ['state_deleted', 'journal_ahead', 'wrong_machine', 'bogus_state'])
def test_damaged_state_exits_4_and_readopt_recovers(work, how, capsys):
    assert R.main(H.gen_argv(work.path)) == 0
    damage(work.out, how)
    assert not R.is_committed(work.out) and M.output_state(work.out) == 'UNREADABLE'
    capsys.readouterr()
    assert cli(H.gen_argv(work.path)) == R.EX_STRANDED
    assert cli(H.gen_argv(work.path, '--force')) == R.EX_STRANDED
    assert '--readopt' in capsys.readouterr().err
    assert cli(H.gen_argv(work.path, '--readopt')) == R.ALREADY_DONE  # adopted again: receipt names the NPZ
    assert R.is_committed(work.out) and M.output_state(work.out) == 'COMMITTED'
    archived = sorted(p.name for p in work.out.parent.glob('out.state.json*.superseded.*'))
    assert archived and all('.superseded.' in name for name in archived)
    assert H.journal_events(work.path) == ['__adopt__']
    assert cli(H.gen_argv(work.path, '--force')) == 0


@pytest.mark.parametrize('previous', [{'state': 'BOGUS'}, None, {'state': 'STAGING'}, ['not', 'a', 'dict']])
def test_staging_with_an_unusable_previous_state_rolls_back_to_the_files(work, previous):
    assert R.main(H.gen_argv(work.path)) == 0
    doc = state_doc(work.out)
    doc['state'] = 'STAGING'
    doc['data'].update(previous=previous, run_id='r20260101T000000-1-deadbeef')
    put_state(work.out, doc)
    assert cli(H.gen_argv(work.path)) == 1  # rolled back; the pair's origin is unknown: INCONSISTENT
    assert R.output_state_file(work.out).read().state is Out.INCONSISTENT and not R.is_committed(work.out)
    assert cli(H.gen_argv(work.path, '--force')) == 0 and R.is_committed(work.out)


@pytest.mark.parametrize('pending', [None, {'npz': '.out.rX.pending.npz', 'receipt': '.out.rX.pending.receipt.json'},
                                     'a string'])
def test_promoting_without_a_valid_pending_entry_is_lost(work, pending):
    assert R.main(H.gen_argv(work.path)) == 0
    before = work.out.read_bytes()
    doc = state_doc(work.out)
    doc['state'] = 'PROMOTING'
    doc['data'].update(pending=pending, run_id='rX')
    put_state(work.out, doc)
    assert cli(H.gen_argv(work.path)) == 1
    assert 'lost' in H.journal_events(work.path)
    assert R.output_state_file(work.out).read().state is Out.INCONSISTENT and work.out.read_bytes() == before
    assert cli(H.gen_argv(work.path, '--force')) == 0 and R.is_committed(work.out)


# --------------------------------------------------------------------------------------------------
# (9) pending paths are confined to this run's own staging names

@pytest.mark.parametrize('kind', ['absolute', 'dotdot'])
def test_promoting_never_moves_a_file_from_outside(work, tmp_path, kind):
    assert R.main(H.gen_argv(work.path)) == 0
    victim = tmp_path/'victim.npz'
    victim.write_bytes(work.out.read_bytes())  # same sha as the recorded npz
    doc = state_doc(work.out)
    run_id = 'r20260101T000000-1-deadbeef'
    name = str(victim) if kind == 'absolute' else os.path.relpath(victim, work.out.parent)
    doc['state'] = 'PROMOTING'
    doc['data'].update(run_id=run_id, pending={'npz': name, 'receipt': R.run_pending(work.out, run_id)[1].name})
    put_state(work.out, doc)
    cli(H.gen_argv(work.path))
    assert victim.exists()
    assert 'lost' in H.journal_events(work.path)


# --------------------------------------------------------------------------------------------------
# (10) temp files of dead writers and holder records of dead runtimes

def test_dead_writer_temps_beside_the_output_are_swept(work):
    assert R.main(H.gen_argv(work.path)) == 0
    pid = dead_pid()
    names = [f'.out.state.json.abcdefgh.{pid}.tmp', f'.out.npz.lock.owner.abcdefgh.{pid}.tmp',
             f'.out.state.json.lock.owner.abcdefgh.{pid}.tmp',
             f'..out.r20260101T000000-1-deadbeef.pending.receipt.json.abcdefgh.{pid}.tmp']
    keep = [f'.out.state.json.abcdefgh.{os.getppid()}.tmp',  # a live writer
            f'.other.state.json.abcdefgh.{pid}.tmp']        # another output's temp
    for name in names + keep:
        (work.out.parent/name).write_text('x')
    assert cli(H.gen_argv(work.path)) == R.ALREADY_DONE
    left = {p.name for p in work.out.parent.iterdir()}
    assert not set(names) & left and set(keep) <= left


def test_dead_holder_records_are_pruned_on_acquire(tmp_path):
    epoch = tmp_path/'epoch'
    (epoch/'holders').mkdir(parents=True)
    pid = dead_pid()
    (epoch/'holders/dead.json').write_text(json.dumps({'pid': pid, 'pid_starttime': 1}))
    (epoch/'holders/garbled.json').write_text('{')  # unparseable: left for a refreeze
    lease = R.EpochLease(['x'], directory=epoch).acquire()
    try:
        names = {p.name for p in (epoch/'holders').iterdir()}
        assert 'dead.json' not in names and 'garbled.json' in names and lease.record.name in names
    finally:
        lease.release()


# --------------------------------------------------------------------------------------------------
# (11) bounded random-SIGKILL fuzzer

def test_random_sigkill_converges(tmp_path):
    rng = random.Random(20261120)
    for trial in range(6):
        work = H.prepare(tmp_path/f't{trial}')
        if trial % 2:
            assert H.run_child(work, payload=3)[0] == 0
        proc = H.run_child(work, '--force', payload=7, delay=rng.uniform(0, 0.05), start=True)
        time.sleep(rng.uniform(0.3, 1.2))
        proc.send_signal(signal.SIGKILL)
        proc.communicate()
        out = H.out_of(work)
        assert not (R.is_committed(out) and not H.pair(work)[2])
        rc, _, err = H.run_child(work, payload=7)
        assert rc in (0, R.ALREADY_DONE), err
        assert H.pair(work)[2] and R.is_committed(out) and M.output_state(out) == 'COMMITTED'
        assert H.staging_files(work) == []
