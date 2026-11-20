"""Tests for analysis/coverage_empf_confirm_20261120/queue_state.py (CPU only, stdlib only).

Synthetic fixtures only: queue.log lines in both legacy formats are written by the
tests, fit/generation/score trees are fake, and the stub job is a tiny Python
script. Nothing here reads runs/coverage_empf_confirm_20261120.

(a) job transition table: seeded random claim/start/exit/owner_dead/retry walks
    (hypothesis is not required); invariants: never claimed twice concurrently,
    never lost, tries <= 3
(b) crash injection: the `run` driver is killed at each fault point and restarted
(c) concurrency: 8 driver processes, 50 stub jobs, each runs exactly once to completion
(d) PID reuse: a recorded pid that is alive with another starttime is dead
(e) classify() over sample log tails
(f) legacy adoption over a fixture mirroring the 05:37 situation, and the gen-2 formats
(r) adversarial-review regressions: a stale adopted ledger catching up (and `run` never
    retraining a legacy-finished job), gen-1 failures re-queued in jobs_queue2.txt, adopted
    jobs with no tries left, the requeue cap, a refused start, an abandon racing a retry, a
    failing worker, a lingering descendant holding the run lock, a child that never ran,
    ledger job lists that only grow, concurrent --write-ledger, corrupt/partial inputs
"""

from __future__ import annotations

import enum
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QS_PATH = ROOT/'analysis/coverage_empf_confirm_20261120/queue_state.py'
spec = importlib.util.spec_from_file_location('confirm_queue_state', QS_PATH)
QS = importlib.util.module_from_spec(spec)
sys.modules['confirm_queue_state'] = QS
spec.loader.exec_module(QS)
from alz_perturb_gan import durable_state as ds  # noqa: E402  (queue_state put src/ on sys.path)

J = QS.JobState
PY = sys.executable
MODE_A = QS.MODE_A


# --------------------------------------------------------------------------- fixtures
def make_fit(repo, k, s, audited=True):
    fit = repo/QS.FIT_SUBDIR/f'fold{k}/seed{s}'
    fit.mkdir(parents=True, exist_ok=True)
    (fit/'config.json').write_text('{}')
    (fit/'final.pt').write_bytes(b'x')
    (fit/'fit_receipt.json').write_text(json.dumps(dict(completed_updates=20000, endpoint=20000, final_SHA256='abc')))
    if audited:
        (fit/'no_skip_audit.json').write_text(json.dumps(dict(final_checkpoint_sha256='abc', completed_updates=20000)))
    return fit


def make_readout(repo, k, s):
    gen = repo/QS.GENERATED_SUBDIR/f'fold{k}/seed{s}'
    gen.mkdir(parents=True, exist_ok=True)
    (gen/f'generated_{MODE_A}.npz').write_bytes(b'npz')
    (gen/f'generated_{MODE_A}.receipt.json').write_text(json.dumps(dict(
        mode=MODE_A, fit=str(repo/QS.FIT_SUBDIR/f'fold{k}/seed{s}'), generated_SHA256='def', application_default={})))


def make_score(run_root, k, s):
    d = run_root/f'scores/fold{k}/seed{s}'
    d.mkdir(parents=True, exist_ok=True)
    (d/'score.json').write_text(json.dumps(dict(status='scored', fold=k, seed=s)))


STUB = r'''
import json, os, sys, time
from pathlib import Path
repo, k, s, counts, sleep, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
counts.mkdir(parents=True, exist_ok=True)
running = counts/f"fold{k}_seed{s}.running"
try:
    fd = os.open(running, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
except FileExistsError:
    (counts/f"fold{k}_seed{s}.OVERLAP").write_text("x")
with open(counts/f"fold{k}_seed{s}.count", "a") as fh:
    fh.write(f"{os.getpid()}\n")
n = len((counts/f"fold{k}_seed{s}.count").read_text().split())
time.sleep(sleep)
os.unlink(running)
if mode == "race-once" and n == 1:
    print("RuntimeError: existing derived EMPF host differs: /x/alz-host-empf-native/release_env.py", flush=True)
    sys.exit(1)
if mode == "oom":
    print("torch.OutOfMemoryError: CUDA out of memory.", flush=True)
    sys.exit(1)
fit = Path(repo)/"runs/microglia_empf/fits/pbulk/hvg2000"/f"fold{k}/seed{s}"
fit.mkdir(parents=True, exist_ok=True)
(fit/"config.json").write_text("{}")
(fit/"final.pt").write_bytes(b"x")
(fit/"fit_receipt.json").write_text(json.dumps(dict(completed_updates=20000, endpoint=20000, final_SHA256="abc")))
(fit/"no_skip_audit.json").write_text(json.dumps(dict(final_checkpoint_sha256="abc", completed_updates=20000)))
gen = Path(repo)/"runs/microglia_empf/generated"/f"fold{k}/seed{s}"
gen.mkdir(parents=True, exist_ok=True)
mode_a = "cellshape@0.5:biascal:stagehalf"
(gen/f"generated_{mode_a}.npz").write_bytes(b"npz")
(gen/f"generated_{mode_a}.receipt.json").write_text(json.dumps(dict(mode=mode_a, fit=str(fit), generated_SHA256="def", application_default={})))
'''


class Bench:
    """A fake repo + run root + ledger with a stub job command."""

    def __init__(self, tmp, jobs, sleep=0.05, mode='ok'):
        self.tmp = Path(tmp)
        self.repo = self.tmp/'repo'
        self.run_root = self.repo/'runs/coverage_empf_confirm_20261120'
        (self.run_root/'logs').mkdir(parents=True)
        self.ledger = self.tmp/'ledger'
        self.counts = self.tmp/'counts'
        self.stub = self.tmp/'stub.py'
        self.stub.write_text(STUB)
        self.jobfile = self.tmp/'jobs.txt'
        self.jobfile.write_text(''.join(f'{k} {s}\n' for k, s in jobs))
        self.jobs = list(jobs)
        self.sleep, self.mode = sleep, mode

    def argv(self, workers=1, offset=0, extra=()):
        cmd = [PY, '-B', str(self.stub), str(self.repo), '{fold}', '{seed}', str(self.counts), str(self.sleep),
               self.mode]
        return [PY, '-B', str(QS_PATH), 'run', '--jobfile', str(self.jobfile), '--ledger-dir', str(self.ledger),
                '--run-root', str(self.run_root), '--repo-root', str(self.repo), '--workers', str(workers),
                '--worker-offset', str(offset), '--start-spacing', '0', '--poll', '0.05',
                '--job-cmd', json.dumps(cmd), *extra]

    def run(self, fault=None, timeout=120, **kw):
        env = {k: v for k, v in os.environ.items() if k != QS.FAULT_ENV}
        if fault:
            env[QS.FAULT_ENV] = fault
        return subprocess.run(self.argv(**kw), env=env, capture_output=True, text=True, timeout=timeout)

    def count(self, job):
        f = self.counts/f'fold{job[0]}_seed{job[1]}.count'
        return len(f.read_text().split()) if f.exists() else 0

    def overlaps(self):
        return sorted(p.name for p in self.counts.glob('*.OVERLAP')) if self.counts.exists() else []

    def views(self):
        return QS.Ledger(self.ledger, QS.Layout(self.run_root, self.repo)).views()


def dead_identity():
    """pid + starttime of a process that has exited."""
    p = subprocess.Popen([PY, '-c', 'import time; time.sleep(5)'])
    try:
        for _ in range(100):
            st = ds.pid_starttime(p.pid)
            if st is not None:
                break
            time.sleep(0.01)
        ident = dict(pid=p.pid, pid_starttime=st, host=os.uname().nodename, worker=None)
    finally:
        p.kill(); p.wait()
    return ident


# --------------------------------------------------------------------------- the machine itself
def test_machine_is_valid_and_matches_the_design():
    m = QS.JOB_MACHINE
    assert m.terminal == {J.SCORED, J.ABANDONED}
    assert m.next_state(J.UNCLAIMED, 'claim') is J.CLAIMED
    assert m.next_state(J.CLAIMED, 'owner_dead') is J.UNCLAIMED
    assert m.next_state(J.RUNNING, 'owner_dead') is J.UNCLAIMED
    assert m.next_state(J.FAILED, 'retry') is J.RETRY_WAIT
    assert m.next_state(J.RETRY_WAIT, 'start') is J.RUNNING
    chain = [(J.EXITED_OK, 'verify_fit', J.TRAINED), (J.TRAINED, 'verify_cli_a', J.CLI_GENERATED),
             (J.CLI_GENERATED, 'scored', J.SCORED)]
    for a, e, b in chain:
        assert m.next_state(a, e) is b
    assert m.next_state(J.UNCLAIMED, 'start') is None  # must be claimed first
    assert m.next_state(J.CLAIMED, 'claim') is None    # never claimed twice
    assert set(QS.REASONS) == {'HOST_DIFFERS', 'HOST_PARTIAL_IMPORT', 'KILLED', 'OOM', 'OUTPUT_EXISTS',
                               'SOURCE_CHANGED', 'GENERATION_FAILED', 'TRAIN_ERROR'}
    assert {r for r, ok in QS.REASONS.items() if ok} == {'HOST_DIFFERS', 'HOST_PARTIAL_IMPORT', 'KILLED',
                                                         'GENERATION_FAILED'}


# --------------------------------------------------------------------------- (a) property walks
LOGS = {'HOST_DIFFERS': ('RuntimeError: existing derived EMPF host differs: /a/b', 1),
        'KILLED': ('', 'missing'), 'OOM': ('CUDA out of memory', 1), 'TRAIN_ERROR': ('ValueError: x', 1)}


def _walk(tmp, seed, n_jobs=3, n_owners=3, steps=60, monkeypatch=None, seen=None):
    seen = set() if seen is None else seen
    rng = random.Random(seed)
    ledger = QS.Ledger(tmp/f'walk{seed}')
    jobs = [(k, 0) for k in range(n_jobs)]
    ledger.init(jobs, f'digest{seed}')
    owners = [dict(pid=10_000 + i, pid_starttime=1, host='h', worker=i) for i in range(n_owners)]
    live = {o['pid'] for o in owners}
    monkeypatch.setattr(QS, 'alive', lambda info: bool(info) and info.get('pid') in live)
    nxt = 10_000 + n_owners

    def owned(o, states):
        out = []
        for job, v in ledger.views().items():
            if v['state'] in states and (v['owner'] or {}).get('pid') == o['pid']:
                out.append(job)
        return out

    for _ in range(steps):
        live_owners = [o for o in owners if o['pid'] in live]
        if not live_owners:
            o = dict(pid=nxt, pid_starttime=1, host='h', worker=0); nxt += 1
            owners.append(o); live.add(o['pid']); continue
        o = rng.choice(live_owners)
        op = rng.choice(['claim', 'claim', 'start', 'exit', 'exit', 'owner_dead', 'retry', 'recover'])
        before = ledger.views()
        if op == 'claim':
            job = ledger.claim(o)
            if job is not None:
                was = before[job]
                assert was['state'] == 'UNCLAIMED' or (was['state'] == 'FAILED' and was['retryable']
                                                       and was['tries'] < QS.MAX_TRIES)
                # a second claimer is refused by the table
                with pytest.raises(ds.IllegalTransition):
                    ledger.sf(job).transition('claim')
        elif op == 'start':
            mine = owned(o, ('CLAIMED', 'RETRY_WAIT'))
            if mine:
                ledger.start(rng.choice(mine), o)
        elif op == 'exit':
            mine = owned(o, ('RUNNING',))
            if mine:
                job = rng.choice(mine)
                if rng.random() < 0.3:
                    ledger.exit(job, 0)
                else:
                    text, rc = LOGS[rng.choice(['HOST_DIFFERS', 'KILLED', 'HOST_DIFFERS', 'OOM', 'TRAIN_ERROR'])]
                    ledger.exit(job, rc, text)
        elif op == 'owner_dead':
            live.discard(o['pid'])
        elif op == 'retry':
            failed = [j for j, v in before.items() if v['state'] == 'FAILED']
            if failed:
                job = rng.choice(failed)
                v = before[job]
                if v['retryable'] and v['tries'] < QS.MAX_TRIES:
                    ledger.retry(job, o)
                else:
                    with pytest.raises(ds.GuardFailed):
                        ledger.retry(job, o)
        else:
            ledger.recover(o)
            for job, v in ledger.views().items():
                if v['state'] in ('CLAIMED', 'RUNNING', 'RETRY_WAIT'):
                    assert (v['owner'] or {}).get('pid') in live, 'a dead owner kept a job after recovery'
        after = ledger.views()
        assert set(after) == set(jobs), 'a job was lost'
        for job, v in after.items():
            seen.add(v['state'])
            if v['tries'] == QS.MAX_TRIES:
                seen.add('tries=3')
            assert v['tries'] <= QS.MAX_TRIES
            if v['state'] in ('CLAIMED', 'RUNNING', 'RETRY_WAIT'):
                assert v['owner'] and v['owner'].get('pid'), 'a live-state job without an owner'
            journal = ledger.sf(job).journal()
            assert [e['seq'] for e in journal] == list(range(1, v['version'] + 1))
    # Everything still in flight is recoverable: kill every owner, recover, and no job is stuck.
    live.clear()
    final = dict(pid=99_999, pid_starttime=1, host='h', worker=0); live.add(final['pid'])
    ledger.recover(final)
    for job, v in ledger.views().items():
        assert v['state'] not in ('CLAIMED', 'RUNNING', 'RETRY_WAIT'), (job, v)


def test_transition_table_property(tmp_path, monkeypatch):
    seen = set()
    for seed in range(30):
        _walk(tmp_path, seed, monkeypatch=monkeypatch, seen=seen)
    # the walks reach every state the driver can, including exhausted retries
    assert {'UNCLAIMED', 'CLAIMED', 'RUNNING', 'EXITED_OK', 'FAILED', 'RETRY_WAIT', 'tries=3'} <= seen, seen


# --------------------------------------------------------------------------- (b) crash injection
@pytest.mark.parametrize('point', QS.FAULT_POINTS)
def test_crash_at_each_point_then_restart_runs_each_job_once(tmp_path, point):
    jobs = [(8, 0), (1, 1)]
    b = Bench(tmp_path, jobs, sleep=1.0 if point == 'while_running' else 0.05)
    crashed = b.run(fault=point)
    assert crashed.returncode == 137, crashed.stderr
    # no waiting: for while_running the child is still alive and the restart must reown it
    resumed = b.run()
    assert resumed.returncode == 0, resumed.stderr
    views = b.views()
    for job in jobs:
        assert b.count(job) == 1, (point, job, b.count(job))
        assert views[job]['state'] == 'CLI_GENERATED', (point, job, views[job])
        assert views[job]['tries'] == 1
    assert not b.overlaps()
    assert json.loads(resumed.stdout.strip().splitlines()[-1])['training_drained'] is True


def test_restart_requeues_claimed_jobs_regression_8_0_and_1_1(tmp_path):
    """The legacy cursor dropped '8 0' and '1 1' when the queue died between claim and start.
    In the ledger, a claim belongs to pid + starttime; a dead owner's claims are re-queued."""
    jobs = [(8, 0), (1, 1), (2, 1)]
    b = Bench(tmp_path, jobs)
    ledger = QS.Ledger(b.ledger)
    ledger.init(jobs, QS.read_jobfile(b.jobfile)[1])
    dead = dead_identity()
    assert ledger.claim(dead) == (8, 0)
    assert ledger.claim(dead) == (1, 1)
    assert {j: v['state'] for j, v in ledger.views().items()} == {(8, 0): 'CLAIMED', (1, 1): 'CLAIMED',
                                                                (2, 1): 'UNCLAIMED'}
    r = b.run(workers=2)
    assert r.returncode == 0, r.stderr
    views = b.views()
    assert all(b.count(j) == 1 for j in jobs)
    assert all(views[j]['state'] == 'CLI_GENERATED' for j in jobs)
    assert views[(8, 0)]['requeues'] == 1 and views[(2, 1)]['requeues'] == 0


def test_retryable_failure_is_retried_and_terminal_is_not(tmp_path):
    b = Bench(tmp_path, [(3, 1)], mode='race-once')
    assert b.run().returncode == 0
    v = b.views()[(3, 1)]
    assert b.count((3, 1)) == 2 and v['state'] == 'CLI_GENERATED' and v['tries'] == 2
    hist = QS.Ledger(b.ledger).sf((3, 1)).read().data['history']
    assert [h['failure'] and h['failure']['reason'] for h in hist] == ['HOST_DIFFERS', None]

    b2 = Bench(tmp_path/'oom', [(4, 1)], mode='oom')
    r = b2.run()
    assert r.returncode == 0
    v = b2.views()[(4, 1)]
    assert b2.count((4, 1)) == 1 and v['state'] == 'FAILED' and v['reason'] == 'OOM' and v['retryable'] is False
    assert json.loads(r.stdout.strip().splitlines()[-1])['queue'] == 'DRAINED'


def test_jobfile_digest_guards_a_stale_ledger(tmp_path):
    b = Bench(tmp_path, [(1, 1)])
    assert b.run().returncode == 0
    b.jobfile.write_text('1 1\n2 1\n')
    refused = b.run()
    assert refused.returncode != 0 and 'replace-jobfile' in refused.stderr
    ok = b.run(extra=('--replace-jobfile',))
    assert ok.returncode == 0, ok.stderr
    assert b.count((1, 1)) == 1 and b.count((2, 1)) == 1  # 1 1 is not rerun: state is per job, not positional


# --------------------------------------------------------------------------- (c) concurrency
def test_eight_driver_processes_fifty_jobs_each_exactly_once(tmp_path):
    jobs = [(k, s) for s in range(5) for k in range(10)]
    b = Bench(tmp_path, jobs, sleep=0.05, mode='ok')
    env = {k: v for k, v in os.environ.items() if k != QS.FAULT_ENV}
    # every fifth job hits the activation race once and must be retried exactly once
    race = {(k, s) for k, s in jobs if k % 5 == 0}
    procs = [b.argv(workers=1, offset=w) for w in range(8)]
    # per-job mode: the stub reads it from a file
    modes = tmp_path/'modes.json'
    modes.write_text(json.dumps({f'{k}_{s}': 'race-once' for k, s in race}))
    b.stub.write_text(STUB.replace(
        'repo, k, s, counts, sleep, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), '
        'float(sys.argv[5]), sys.argv[6]',
        'repo, k, s, counts, sleep, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), '
        f'float(sys.argv[5]), json.load(open({str(modes)!r})).get(f"{{sys.argv[2]}}_{{sys.argv[3]}}", "ok")'))
    running = [subprocess.Popen(a, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for a in procs]
    for p in running:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err
    views = b.views()
    assert set(views) == set(jobs)
    for job in jobs:
        expected = 2 if job in race else 1
        assert b.count(job) == expected, (job, b.count(job))
        assert views[job]['state'] == 'CLI_GENERATED', (job, views[job])
        assert views[job]['tries'] == expected
    assert not b.overlaps()


# --------------------------------------------------------------------------- (d) PID reuse
def test_pid_reuse_owner_is_dead(tmp_path):
    ledger = QS.Ledger(tmp_path/'l')
    ledger.init([(5, 1)], 'd')
    me = QS.identity()
    impostor = dict(pid=os.getppid(), pid_starttime=(ds.pid_starttime(os.getppid()) or 0) + 12345,
                    host=me['host'], worker=0)
    assert ds._pid_alive(impostor['pid'])
    assert not QS.alive(impostor)
    assert QS.alive(me)
    ledger.claim(impostor)
    ledger.recover(me)
    v = ledger.views()[(5, 1)]
    assert v['state'] == 'UNCLAIMED' and v['requeues'] == 1


def test_legacy_pid_file_pointing_at_an_unrelated_or_older_process(tmp_path):
    sleeper = subprocess.Popen([PY, '-c', 'import time; time.sleep(30)'])
    named = subprocess.Popen([PY, '-c', 'import time; time.sleep(30)', 'microglia.py', 'train', '--fold', '3',
                              '--seed', '1', '--device', 'cuda'])
    try:
        time.sleep(0.2)
        assert not QS.legacy_pid_alive(sleeper.pid, 3, 1)               # cmdline names no microglia job
        assert QS.legacy_pid_alive(named.pid, 3, 1)                     # cmdline matches
        assert not QS.legacy_pid_alive(named.pid, 3, 0)                 # other seed
        assert not QS.legacy_pid_alive(named.pid, 3, 1, not_before=time.time() + 60)  # started before the log line
        assert QS.legacy_pid_alive(named.pid, 3, 1, not_before=time.time() - 60)
        # through the reader: pid file for try1 points at the unrelated sleeper
        run_root = tmp_path/'runs/coverage_empf_confirm_20261120'
        logs = run_root/'logs'; logs.mkdir(parents=True)
        (logs/'queue.log').write_text(f'{now_iso(-10)} start fold3 seed1 cores6-8 try1\n')
        (logs/'pid_fold3_seed1.try1').write_text(f'{sleeper.pid}\n')
        r = QS.adopt_legacy(run_root, tmp_path, jobs=[(3, 1)], live_scan={}, queue_alive=False)
        assert r['jobs'][(3, 1)]['state'] == 'UNCLAIMED'
        (logs/'pid_fold3_seed1.try1').write_text(f'{named.pid}\n')
        r = QS.adopt_legacy(run_root, tmp_path, jobs=[(3, 1)], live_scan={}, queue_alive=True)
        assert r['jobs'][(3, 1)]['state'] == 'RUNNING' and r['jobs'][(3, 1)]['pid'] == named.pid
    finally:
        for p in (sleeper, named):
            p.kill(); p.wait()


def now_iso(offset=0.0):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(time.time() + offset, timezone.utc).isoformat(timespec='seconds')


# --------------------------------------------------------------------------- (e) classify
HOST_FILE = '/ml2/alz/artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native/trainer.py'


@pytest.mark.parametrize('text,rc,fit,expected', [
    (f'RuntimeError: existing derived EMPF host differs: {HOST_FILE}', 1, False, ('HOST_DIFFERS', True)),
    (f'  File "{HOST_FILE}", line 812\n    def run(ar\n           ^\nSyntaxError: \'(\' was never closed', 1, False,
     ('HOST_PARTIAL_IMPORT', True)),
    (f"ImportError: cannot import name 'recipe' from 'trainer' ({HOST_FILE})", 1, False,
     ('HOST_PARTIAL_IMPORT', True)),
    (f"  File \"{HOST_FILE}\", line 3\nModuleNotFoundError: No module named 'release_en'", 1, False,
     ('HOST_PARTIAL_IMPORT', True)),
    ("ModuleNotFoundError: No module named 'torch'", 1, False, ('TRAIN_ERROR', False)),  # not the host
    ('', 'missing', False, ('KILLED', True)),
    ('', None, False, ('KILLED', True)),
    ('[trainer] step 9000', 137, False, ('KILLED', True)),
    ('[trainer] step 9000', 9, False, ('KILLED', True)),
    ('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB', 1, False, ('OOM', False)),
    ('usage: microglia.py\nmicroglia.py: error: --out must be outside the fit directory', 2, False,
     ('OUTPUT_EXISTS', False)),
    ('FileExistsError: completed fit exists, refusing to overwrite: /x', 1, False, ('OUTPUT_EXISTS', False)),
    ('microglia runtime: pinned EMPF runtime source missing or changed: scripts/x.py', 1, False,
     ('SOURCE_CHANGED', False)),
    ('microglia runtime: runtime source manifest changed during execution', 1, False, ('SOURCE_CHANGED', False)),
    ("ValueError: readout is outside the declared application profiles", 1, True, ('GENERATION_FAILED', True)),
    ("ValueError: recovery checkpoint endpoint differs from --updates", 1, False, ('TRAIN_ERROR', False)),
])
def test_classify_table(tmp_path, text, rc, fit, expected):
    fit_dir = make_fit(tmp_path, 1, 1) if fit else tmp_path/'nofit'
    assert QS.classify(text, rc, fit_dir) == expected


# --------------------------------------------------------------------------- (f) legacy adoption
GEN1_0537 = '''2026-11-20T03:31:39+00:00 start fold3 seed0 gpu0
2026-11-20T03:31:39+00:00 start fold2 seed0 gpu1
2026-11-20T03:31:39+00:00 start fold4 seed0 gpu1
2026-11-20T03:31:39+00:00 start fold1 seed0 gpu0
2026-11-20T03:31:39+00:00 start fold5 seed0 gpu0
2026-11-20T03:31:39+00:00 start fold6 seed0 gpu1
2026-11-20T03:31:39+00:00 start fold8 seed0 gpu1
2026-11-20T03:31:39+00:00 start fold7 seed0 gpu0
2026-11-20T03:31:39+00:00 end fold8 seed0 rc=1
2026-11-20T03:31:40+00:00 start fold0 seed1 gpu0
2026-11-20T03:48:07+00:00 queue finished
'''
JOBS_QUEUE2 = '8 0\n1 1\n2 1\n3 1\n4 1\n5 1\n6 1\n7 1\n8 1\n'
GEN1_LIVE = [(k, 0) for k in range(1, 8)] + [(0, 1)]
START_0331 = QS._epoch('2026-11-20T03:31:39+00:00')


def legacy_fixture(tmp_path):
    repo = tmp_path/'repo'
    run_root = repo/'runs/coverage_empf_confirm_20261120'
    logs = run_root/'logs'
    logs.mkdir(parents=True)
    (logs/'queue.log').write_text(GEN1_0537)
    (logs/'train_fold8_seed0.log').write_text(
        'Traceback (most recent call last):\n  File "/ml2/alz/analysis/empf_comparison_20261117/host.py", line 55\n'
        f'RuntimeError: existing derived EMPF host differs: {HOST_FILE}\n')
    (logs/'jobs_queue2.txt').write_text(JOBS_QUEUE2)
    (logs/'queue.next').write_text('1\n')
    pids = {job: 2_751_700 + i for i, job in enumerate(GEN1_LIVE)}
    cmds = {str(p): f'/py -B scripts/microglia.py train --fold {k} --seed {s} --device cuda'
            for (k, s), p in pids.items()}
    kw = dict(read_cmdline=lambda pid: cmds.get(str(pid)),
              live_scan={job: [p] for job, p in pids.items()},
              start_epoch=lambda pid: START_0331 + 0.5, queue_alive=True)
    return repo, run_root, logs, pids, kw


def test_legacy_adoption_at_0537_is_not_drained(tmp_path):
    repo, run_root, logs, pids, kw = legacy_fixture(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    report = QS.status(run_root, repo, **kw)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before, 'status wrote files'
    jobs = report['jobs']
    assert report['derived']['queue'] == 'NOT_DRAINED'
    assert report['derived']['training_drained'] is False
    assert [m['kind'] for m in report['markers']] == ['finished1']   # recorded, never terminal
    for job in GEN1_LIVE:
        v = jobs[job]
        assert v['state'] == 'RUNNING' and v['generation'] == 1 and v['pid'] == pids[job], (job, v)
        assert 'no retry' in v['blocked']
    # 8 0's gen-1 attempt failed, but jobs_queue2.txt re-queued it: waiting, not FAILED
    f8 = jobs[(8, 0)]
    assert (f8['state'], f8['tries']) == ('UNCLAIMED', 1)
    assert f8['earlier'] == dict(gen=1, rc=1, reason='HOST_DIFFERS', retryable=True)
    assert 're-queued in jobs_queue2.txt after gen-1 rc=1 (HOST_DIFFERS)' in f8['blocked']
    for k in range(1, 9):
        assert jobs[(k, 1)]['state'] == 'UNCLAIMED', (k, jobs[(k, 1)])
    assert report['cursor']['next'] == 1 and len(report['cursor']['sha256']) == 64
    assert report['derived']['counts'] == {'RUNNING': 8, 'UNCLAIMED': 9}


def test_legacy_adoption_gen2_formats(tmp_path):
    repo, run_root, logs, pids, kw = legacy_fixture(tmp_path)
    # gen-1 orphans have finished meanwhile: end lines arrive after 'queue finished'
    gen2 = '\n'.join([
        '2026-11-20T05:43:29+00:00 end fold2 seed0 rc=0',
        '2026-11-20T05:43:55+00:00 start fold8 seed0 cores18-20 try1',
        '2026-11-20T05:44:30+00:00 retry fold8 seed0 rc=1 (activation race)',
        '2026-11-20T05:44:31+00:00 start fold8 seed0 cores18-20 try2',
        '2026-11-20T06:44:31+00:00 end fold8 seed0 rc=0',
        '2026-11-20T05:49:55+00:00 start fold1 seed1 cores9-11 try1',
        '2026-11-20T06:00:00+00:00 end fold1 seed1 rc=missing',
        '2026-11-20T06:06:56+00:00 start fold2 seed1 cores3-5 try1',
        '2026-11-20T06:07:30+00:00 retry fold2 seed1 rc=1 (activation race)',
        '2026-11-20T06:26:56+00:00 start fold4 seed1 cores6-8 try1',
        '2026-11-20T06:30:00+00:00 end fold1 seed0 rc=1',
        '2026-11-20T07:00:00+00:00 queue2 finished', ''])
    (logs/'queue.log').write_text(GEN1_0537 + gen2)
    (logs/'queue.next').write_text('6\n')                          # 8 0, 1 1, 2 1, 3 1, 4 1 claimed
    (logs/'rc_fold4_seed1.try1').write_text('1\n')                 # exited; end line not yet written
    (logs/'train_fold4_seed1.try1.log').write_text('torch.OutOfMemoryError: CUDA out of memory.\n')
    (logs/'train_fold1_seed0.log').write_text('ValueError: something else\n')
    for job in [(2, 0), (8, 0)]:
        make_fit(repo, *job)
        make_readout(repo, *job)
    make_score(run_root, 2, 0)
    make_fit(repo, 3, 0)                                            # trained, readout A missing
    kw = dict(kw, live_scan={job: p for job, p in kw['live_scan'].items() if job not in [(2, 0), (1, 0)]},
              queue_alive=False)
    jobs = QS.status(run_root, repo, **kw)['jobs']
    assert jobs[(2, 0)]['state'] == 'SCORED'
    assert jobs[(8, 0)]['state'] == 'CLI_GENERATED' and jobs[(8, 0)]['blocked'] == 'awaiting score.py'
    # score_all.sh gave up on it: the reader says so, and the queue cannot drain on its own
    (logs/'score_queue.log').write_text(
        '2026-11-20T07:00:00+00:00 start scorer fold8 seed0 pid 999999\n'
        '2026-11-20T07:02:00+00:00 scorer fold8 seed0 exited without score.json, attempt 1 of 2 (see x)\n'
        '2026-11-20T07:04:00+00:00 start scorer fold8 seed0 pid 999998\n'
        '2026-11-20T07:06:00+00:00 scorer fold8 seed0 exited without score.json, attempt 2 of 2 (see x)\n')
    jobs = QS.status(run_root, repo, **kw)['jobs']
    assert jobs[(8, 0)]['state'] == 'CLI_GENERATED' and 'gave up' in jobs[(8, 0)]['blocked']
    assert (jobs[(1, 1)]['state'], jobs[(1, 1)]['reason'], jobs[(1, 1)]['retryable']) == ('FAILED', 'KILLED', True)
    assert jobs[(2, 1)]['state'] == 'RETRY_WAIT'
    assert jobs[(3, 1)]['state'] == 'UNCLAIMED' and 'dropped' in jobs[(3, 1)]['blocked']
    assert (jobs[(4, 1)]['state'], jobs[(4, 1)]['reason']) == ('FAILED', 'OOM')
    assert jobs[(4, 1)]['terminal'] is True
    assert (jobs[(1, 0)]['state'], jobs[(1, 0)]['reason']) == ('FAILED', 'TRAIN_ERROR')
    assert jobs[(3, 0)]['state'] == 'RUNNING'                       # still alive (gen-1 orphan)
    assert jobs[(5, 1)]['state'] == 'UNCLAIMED'


def test_write_ledger_adopts_into_the_chosen_directory_only(tmp_path):
    repo, run_root, logs, pids, kw = legacy_fixture(tmp_path)
    ledger_dir = tmp_path/'elsewhere'
    before = sorted(p.relative_to(tmp_path) for p in run_root.rglob('*'))
    report = QS.status(run_root, repo, ledger_dir=ledger_dir, write_ledger=True, **kw)
    assert sorted(p.relative_to(tmp_path) for p in run_root.rglob('*')) == before
    assert report['derived']['queue'] == 'NOT_DRAINED'
    ledger = QS.Ledger(ledger_dir)
    views = ledger.views()
    assert views[(8, 0)]['state'] == 'UNCLAIMED' and views[(8, 0)]['legacy_tracked'] is True
    assert ledger.sf((8, 0)).read().data['legacy_earlier']['reason'] == 'HOST_DIFFERS'
    assert views[(3, 0)]['state'] == 'RUNNING' and views[(3, 0)]['owner']['legacy'] is True
    assert views[(4, 1)]['state'] == 'UNCLAIMED'
    snap = ledger.sf((3, 0)).read()
    assert snap.data['adopted']['evidence']['legacy_state'] == 'RUNNING'
    # a second status run reads the ledger and reports no disagreement
    again = QS.status(run_root, repo, ledger_dir=ledger_dir, **kw)
    assert again['jobs'][(3, 0)]['source'] == 'ledger' and not again['jobs'][(3, 0)]['blocked']


def test_drained_is_derived_from_job_states_only():
    v = lambda st, **kw: dict(state=st, **kw)  # noqa: E731
    assert QS.derive_queue({(1, 1): v('SCORED'), (2, 1): v('ABANDONED'),
                            (3, 1): v('FAILED', retryable=False, tries=1)})['queue'] == 'DRAINED'
    assert QS.derive_queue({(1, 1): v('SCORED'), (3, 1): v('FAILED', retryable=True, tries=3)})['queue'] == 'DRAINED'
    for st in ('UNCLAIMED', 'CLAIMED', 'RUNNING', 'RETRY_WAIT', 'EXITED_OK', 'TRAINED', 'CLI_GENERATED'):
        assert QS.derive_queue({(1, 1): v('SCORED'), (2, 1): v(st)})['queue'] == 'NOT_DRAINED'
    assert QS.derive_queue({(1, 1): v('FAILED', retryable=True, tries=1)})['queue'] == 'NOT_DRAINED'
    d = QS.derive_queue({(1, 1): v('CLI_GENERATED'), (2, 1): v('TRAINED')})
    assert d['training_drained'] is True and d['queue'] == 'NOT_DRAINED'


def test_status_cli_is_read_only_and_prints_json(tmp_path):
    repo, run_root, logs, pids, kw = legacy_fixture(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    r = subprocess.run([PY, '-B', str(QS_PATH), 'status', '--run-root', str(run_root), '--repo-root', str(repo),
                        '--json'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report['derived']['queue'] == 'NOT_DRAINED' and report['wrote_ledger'] is False
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before


def test_abandon_is_manual_with_a_reason(tmp_path):
    ledger = QS.Ledger(tmp_path/'l')
    ledger.init([(6, 1)], 'd')
    r = subprocess.run([PY, '-B', str(QS_PATH), 'abandon', '--ledger-dir', str(tmp_path/'l'), '--fold', '6',
                        '--seed', '1', '--reason', 'owner dropped fold 6'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    snap = ledger.sf((6, 1)).read()
    assert snap.state is J.ABANDONED and snap.data['abandoned']['reason'] == 'owner dropped fold 6'
    assert QS.derive_queue(ledger.views())['queue'] == 'DRAINED'


def test_advance_verifies_files_in_order(tmp_path):
    repo = tmp_path/'repo'; run_root = repo/'runs/x'
    layout = QS.Layout(run_root, repo)
    ledger = QS.Ledger(tmp_path/'l', layout)
    ledger.init([(7, 1)], 'd')
    me = QS.identity()
    ledger.claim(me); ledger.start((7, 1), me); ledger.exit((7, 1), 0)
    assert ledger.advance((7, 1)).state is J.EXITED_OK            # no fit yet
    make_fit(repo, 7, 1, audited=False)
    assert ledger.advance((7, 1)).state is J.TRAINED              # RECEIPTED counts
    make_readout(repo, 7, 1)
    assert ledger.advance((7, 1)).state is J.CLI_GENERATED
    make_score(run_root, 7, 1)
    assert ledger.advance((7, 1)).state is J.SCORED


def test_scorer_blocked_reasons():
    row = dict(pid=4242, failures=0, of=2)
    scoring = lambda pid: 'python3 -B analysis/coverage_empf_confirm_20261120/score.py --fold 3 --seed 1'  # noqa: E731
    assert QS.scorer_blocked(row, 3, 1, scoring).startswith('scoring (score.py pid 4242')
    assert QS.scorer_blocked(row, 3, 0, scoring) == 'awaiting score.py'          # pid runs another fit
    assert QS.scorer_blocked(dict(row, failures=1), 3, 1, lambda pid: None).startswith('scorer failed 1 of 2')
    assert QS.scorer_blocked(None, 3, 1) == 'awaiting score.py'


# --------------------------------------------------------------------------- adversarial-review regressions
def gen2_fixture(tmp_path, rows, nxt):
    """A run root with jobs_queue2.txt + queue.next and an empty queue.log."""
    repo = tmp_path/'repo'
    run_root = repo/'runs/coverage_empf_confirm_20261120'
    logs = run_root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    (logs/'jobs_queue2.txt').write_text(''.join(f'{k} {s}\n' for k, s in rows))
    (logs/'queue.next').write_text(f'{nxt}\n')
    (logs/'queue.log').touch()
    return repo, run_root, logs


def fake_proc(*args, cwd=None):
    """A sleeping process whose cmdline carries ``args`` (for cmdline matching)."""
    p = subprocess.Popen([PY, '-c', 'import time; time.sleep(60)', *map(str, args)], cwd=cwd)
    for _ in range(100):
        if QS.cmdline(p.pid) and str(args[-1]) in QS.cmdline(p.pid):
            break
        time.sleep(0.01)
    return p


def test_adopted_ledger_follows_legacy_to_scored_and_run_never_retrains(tmp_path):
    """t3: status --write-ledger while the bash queue runs a job; the job then ends, is trained,
    generated and scored by the legacy pipeline. The ledger must catch up (read-only status shows
    it at once, --write-ledger records it) and a post-drain `run` must not run the job again.
    The same holds for a job that was still waiting in jobs_queue2.txt at adoption."""
    b = Bench(tmp_path, [(3, 1), (4, 1)], sleep=0.01)
    logs = b.run_root/'logs'
    (logs/'jobs_queue2.txt').write_text('3 1\n4 1\n')
    (logs/'queue.next').write_text('2\n')
    job_proc = fake_proc('scripts/microglia.py', 'train', '--fold', '3', '--seed', '1', '--device', 'cuda')
    try:
        (logs/'queue.log').write_text(f'{now_iso(-5)} start fold3 seed1 cores6-8 try1\n')
        (logs/'pid_fold3_seed1.try1').write_text(f'{job_proc.pid}\n')
        kw = dict(jobs=[(3, 1), (4, 1)], live_scan={})
        r = QS.status(b.run_root, b.repo, ledger_dir=b.ledger, write_ledger=True, queue_alive=True, **kw)
        assert r['jobs'][(3, 1)]['state'] == 'RUNNING' and r['jobs'][(4, 1)]['state'] == 'UNCLAIMED'
        assert all(v['legacy_tracked'] for v in b.views().values())
    finally:
        job_proc.kill(); job_proc.wait()
    # the bash queue finishes both jobs; the legacy pipeline trains, generates and scores them
    (logs/'rc_fold3_seed1.try1').write_text('0\n')
    with (logs/'queue.log').open('a') as fh:
        fh.write(f'{now_iso(-4)} end fold3 seed1 rc=0\n{now_iso(-3)} start fold4 seed1 cores9-11 try1\n'
                 f'{now_iso(-2)} end fold4 seed1 rc=0\n{now_iso(-1)} queue2 finished\n')
    (logs/'queue.next').write_text('3\n')
    for job in [(3, 1), (4, 1)]:
        make_fit(b.repo, *job); make_readout(b.repo, *job)
    make_score(b.run_root, 3, 1)
    kw['queue_alive'] = False
    # read-only status: shows the legacy truth, writes nothing
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    r = QS.status(b.run_root, b.repo, ledger_dir=b.ledger, **kw)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before
    assert r['jobs'][(3, 1)]['state'] == 'SCORED' and 'behind the legacy files' in r['jobs'][(3, 1)]['blocked']
    assert r['jobs'][(4, 1)]['state'] == 'CLI_GENERATED'
    assert r['derived']['training_drained'] is True
    # --write-ledger records it, and re-running it is stable
    for _ in range(2):
        QS.status(b.run_root, b.repo, ledger_dir=b.ledger, write_ledger=True, **kw)
        views = b.views()
        assert views[(3, 1)]['state'] == 'SCORED' and views[(4, 1)]['state'] == 'CLI_GENERATED'
    versions = {j: v['version'] for j, v in views.items()}
    QS.status(b.run_root, b.repo, ledger_dir=b.ledger, write_ledger=True, **kw)
    assert {j: v['version'] for j, v in b.views().items()} == versions, 'an unchanged legacy view wrote again'
    r = b.run()
    assert r.returncode == 0, r.stderr
    assert b.count((3, 1)) == 0 and b.count((4, 1)) == 0, 'run retrained a job the legacy queue finished'


def test_run_syncs_legacy_itself_and_leaves_jobs_to_a_live_bash_queue(tmp_path):
    """`run` alone (no status --write-ledger) mirrors the legacy files first; while a bash
    queue script inside this repo runs, its tracked jobs are not claimed."""
    b = Bench(tmp_path, [(5, 1), (6, 1)], sleep=0.01)
    logs = b.run_root/'logs'
    (logs/'jobs_queue2.txt').write_text('5 1\n6 1\n')
    (logs/'queue.next').write_text('1\n')
    QS.status(b.run_root, b.repo, ledger_dir=b.ledger, write_ledger=True, jobs=[(5, 1), (6, 1)], live_scan={},
              queue_alive=True)
    queue = fake_proc(b.repo/'analysis/coverage_empf_confirm_20261120/train_queue.sh')
    try:
        assert QS.queue_process_alive(repo_root=b.repo)
        assert not QS.queue_process_alive(repo_root=tmp_path/'another-checkout')
        r = b.run()
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout.strip().splitlines()[-1])
        assert out['legacy_busy'] is True and out['queue'] == 'NOT_DRAINED'
        assert b.count((5, 1)) == 0 and b.count((6, 1)) == 0
    finally:
        queue.kill(); queue.wait()
    # the bash queue ran 5 1 to completion, then died before 6 1
    (logs/'queue.log').write_text(f'{now_iso(-3)} start fold5 seed1 cores0-2 try1\n{now_iso(-2)} end fold5 seed1 rc=0\n')
    (logs/'queue.next').write_text('2\n')
    make_fit(b.repo, 5, 1); make_readout(b.repo, 5, 1); make_score(b.run_root, 5, 1)
    r = b.run()
    assert r.returncode == 0, r.stderr
    views = b.views()
    assert views[(5, 1)]['state'] == 'SCORED' and b.count((5, 1)) == 0
    assert views[(6, 1)]['state'] == 'CLI_GENERATED' and b.count((6, 1)) == 1
    assert views[(6, 1)]['legacy_tracked'] is False


def test_queue_process_scoping():
    cmds = {'1': 'bash /repo/analysis/x/train_queue.sh /repo/logs/jobs', '2': 'bash train_all.sh',
            '3': 'zsh -c eval \'pgrep -f "train_queue.sh|x"\''}
    cwds = {'2': '/repo'}
    read = lambda pid: cmds.get(str(pid))  # noqa: E731
    cwd = lambda pid: cwds.get(str(pid))  # noqa: E731
    orig = Path.iterdir
    try:
        Path.iterdir = lambda self: iter([Path('/proc')/p for p in cmds]) if str(self) == '/proc' else orig(self)
        assert QS.queue_process_alive(read, '/repo', cwd)
        assert not QS.queue_process_alive(read, '/elsewhere', cwd)
        cmds.pop('1')
        assert QS.queue_process_alive(read, '/repo', cwd)          # relative path resolved against cwd
        cwds['2'] = '/other'
        assert not QS.queue_process_alive(read, '/repo', cwd)
    finally:
        Path.iterdir = orig


@pytest.mark.parametrize('log,reason', [
    ('torch.OutOfMemoryError: CUDA out of memory.\n', 'OOM'),
    ('ValueError: x\n', 'TRAIN_ERROR'),
    ('FileExistsError: completed fit exists, refusing to overwrite\n', 'OUTPUT_EXISTS'),
    ('microglia runtime: runtime source manifest changed during execution\n', 'SOURCE_CHANGED'),
])
def test_gen1_failure_requeued_in_jobs_queue2_is_never_terminal(tmp_path, log, reason):
    """t5 G: a non-retryable gen-1 failure for a job jobs_queue2.txt re-queued is waiting
    (or claimed by the cursor), never FAILED/terminal and never DRAINED."""
    repo, run_root, logs = gen2_fixture(tmp_path, [(8, 0)], 1)
    ts = '2026-11-20T03:31:39+00:00'
    (logs/'queue.log').write_text(f'{ts} start fold8 seed0 gpu1\n{ts} end fold8 seed0 rc=1\n')
    (logs/'train_fold8_seed0.log').write_text(log)
    kw = dict(jobs=[(8, 0)], live_scan={})
    for nxt, alive_, state in [(1, True, 'UNCLAIMED'), (2, True, 'CLAIMED'), (2, False, 'UNCLAIMED')]:
        (logs/'queue.next').write_text(f'{nxt}\n')
        r = QS.status(run_root, repo, queue_alive=alive_, **kw)
        v = r['jobs'][(8, 0)]
        assert v['state'] == state and not QS.is_terminal_view(v), (nxt, alive_, v)
        assert r['derived']['queue'] == 'NOT_DRAINED'
        assert v['earlier']['reason'] == reason and f'({reason})' in v['blocked']
    # once the gen-2 attempt exists, it decides
    (logs/'queue.log').write_text(f'{ts} start fold8 seed0 gpu1\n{ts} end fold8 seed0 rc=1\n'
                                  f'{ts} start fold8 seed0 cores0-2 try1\n{ts} end fold8 seed0 rc=1\n')
    (logs/'train_fold8_seed0.try1.log').write_text(log)
    v = QS.status(run_root, repo, queue_alive=False, **kw)['jobs'][(8, 0)]
    assert (v['state'], v['reason'], v['terminal']) == ('FAILED', reason, True)


def test_adopted_job_with_no_tries_left_is_failed_not_stuck(tmp_path):
    """t10: gen-1 failure + two activation-race retries + an open, dead try3 adopts as
    UNCLAIMED with tries=3. claim() used to take it and start() refuse it, killing the worker
    and spending a requeue per run forever. Now the first claim pass marks it FAILED."""
    b = Bench(tmp_path, [(8, 0)], sleep=0.01)
    logs = b.run_root/'logs'
    ts = '2026-11-20T05:00:00+00:00'
    (logs/'queue.log').write_text('\n'.join([
        f'{ts} start fold8 seed0 gpu1', f'{ts} end fold8 seed0 rc=1',
        f'{ts} start fold8 seed0 cores18-20 try1', f'{ts} retry fold8 seed0 rc=1 (activation race)',
        f'{ts} start fold8 seed0 cores18-20 try2', f'{ts} retry fold8 seed0 rc=1 (activation race)',
        f'{ts} start fold8 seed0 cores18-20 try3']) + '\n')
    (logs/'pid_fold8_seed0.try3').write_text('999999\n')
    (logs/'jobs_queue2.txt').write_text('8 0\n'); (logs/'queue.next').write_text('2\n')
    r = QS.status(b.run_root, b.repo, ledger_dir=b.ledger, write_ledger=True, jobs=[(8, 0)], queue_alive=False,
                  live_scan={})
    v = r['jobs'][(8, 0)]
    assert (v['state'], v['tries']) == ('UNCLAIMED', 3) and 'TRIES_EXHAUSTED' in v['blocked']
    b.jobfile.write_text('8 0\n')
    for _ in range(2):
        res = b.run()
        assert res.returncode == 0 and 'Traceback' not in res.stderr, res.stderr
        assert json.loads(res.stdout.strip().splitlines()[-1])['queue'] == 'DRAINED'
    v = b.views()[(8, 0)]
    assert (v['state'], v['reason'], v['retryable'], v['requeues']) == ('FAILED', 'TRIES_EXHAUSTED', False, 0)
    assert b.count((8, 0)) == 0


def test_requeue_limit_is_terminal_with_a_reason(tmp_path):
    """t5 D: after MAX_REQUEUES dead owners the job used to stay UNCLAIMED forever, unclaimable
    and never terminal, with no reason shown."""
    b = Bench(tmp_path, [(1, 0)])
    ledger = QS.Ledger(b.ledger, QS.Layout(b.run_root, b.repo))
    ledger.init([(1, 0)], QS.read_jobfile(b.jobfile)[1])
    for _ in range(QS.MAX_REQUEUES):
        job = ledger.claim(dead_identity())
        ledger.owner_dead(job, ledger.sf(job).read())
    v = ledger.views()[(1, 0)]
    assert v['state'] == 'UNCLAIMED' and 'REQUEUE_LIMIT' in v['blocked']
    res = b.run()
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1])['queue'] == 'DRAINED'
    v = b.views()[(1, 0)]
    assert (v['state'], v['reason'], v['retryable']) == ('FAILED', 'REQUEUE_LIMIT', False)
    assert 'operator must fix or abandon' in v['blocked'] and b.count((1, 0)) == 0


def test_start_refused_releases_the_claim(tmp_path, monkeypatch):
    b = Bench(tmp_path, [(2, 1)])
    ledger = QS.Ledger(b.ledger, QS.Layout(b.run_root, b.repo))
    ledger.init([(2, 1)], 'd')
    driver = QS.Driver(ledger, workers=1, spacing=0, poll=0.01)
    assert ledger.claim(driver.me) == (2, 1)

    def refuse(job, owner):
        raise ds.GuardFailed('guard rejected CLAIMED -start->')
    monkeypatch.setattr(ledger, 'start', refuse)
    driver.run_job((2, 1), '0-2')
    v = ledger.views()[(2, 1)]
    assert v['state'] == 'UNCLAIMED' and v['requeues'] == 1 and driver.notes and not driver.errors


def test_operator_abandon_during_retry_does_not_kill_the_worker(tmp_path, monkeypatch):
    """t5 E: abandon lands between the driver reading FAILED and its retry. The worker used to
    die with VersionConflict and the driver still exited 0."""
    b = Bench(tmp_path, [(2, 0)])
    ledger = QS.Ledger(b.ledger, QS.Layout(b.run_root, b.repo))
    ledger.init([(2, 0)], 'd')
    driver = QS.Driver(ledger, workers=1, spacing=0, poll=0.01)
    ledger.claim(driver.me); ledger.start((2, 0), driver.me)
    ledger.exit((2, 0), 1, 'RuntimeError: existing derived EMPF host differs: /x')
    real_retry = ledger.retry

    def retry_after_abandon(job, owner, expect_version=None):
        ledger.abandon(job, 'operator: dropped fold 2')
        return real_retry(job, owner, expect_version=expect_version)
    monkeypatch.setattr(ledger, 'retry', retry_after_abandon)
    driver.run_job((2, 0), '0-2')
    assert ledger.views()[(2, 0)]['state'] == 'ABANDONED'
    assert len(driver.notes) == 1 and 'VersionConflict' in driver.notes[0] and not driver.errors


def test_a_failing_worker_makes_the_run_exit_nonzero(tmp_path, monkeypatch):
    b = Bench(tmp_path, [(3, 0)])
    ledger = QS.Ledger(b.ledger, QS.Layout(b.run_root, b.repo))
    ledger.init([(3, 0)], 'd')
    driver = QS.Driver(ledger, workers=1, spacing=0, poll=0.01)
    monkeypatch.setattr(driver, 'run_job', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    result = driver.run()
    assert result['worker_errors'] == 1 and 'boom' in driver.errors[0]


def test_lingering_descendant_does_not_hold_the_attempt_open(tmp_path):
    """t9: the job exits 0 at once but leaves a background descendant holding the inherited
    run-lock fd. The attempt is settled from <base>.rc instead of waiting for the lock."""
    marker = f'qs_lingering_{os.getpid()}_{int(time.time() * 1e6)}'
    b = Bench(tmp_path, [(6, 1)])
    argv = b.argv()
    argv[argv.index('--job-cmd') + 1] = json.dumps(['bash', '-c', f'(exec -a {marker} sleep 8) & exit 0'])
    t0 = time.time()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        took = time.time() - t0
    finally:
        subprocess.run(['pkill', '-f', marker])
    assert r.returncode == 0, r.stderr
    assert took < 5, f'driver waited {took:.1f}s for a lingering descendant'
    v = b.views()[(6, 1)]
    assert v['state'] == 'EXITED_OK' and v['tries'] == 1


def test_recovery_settles_from_rc_while_a_descendant_holds_the_lock(tmp_path):
    ledger = QS.Ledger(tmp_path/'l')
    ledger.init([(7, 0)], 'd')
    dead = dead_identity()
    ledger.claim(dead)
    snap = ledger.start((7, 0), dead)
    base = ledger.child_base((7, 0), snap.data['try'])
    holder = subprocess.Popen([PY, '-c', 'import fcntl, sys, time\nfh = open(sys.argv[1], "a")\n'
                               'fcntl.flock(fh, fcntl.LOCK_EX)\nprint("locked", flush=True)\ntime.sleep(60)',
                               f'{base}.run'], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'locked'
        ds.atomic_write_json(f'{base}.rc', dict(rc=0, launch_id=snap.data['launch_id'], at=time.time()))
        assert ledger.recover(QS.identity()) == []                   # not reowned: the attempt is over
        assert ledger.views()[(7, 0)]['state'] == 'EXITED_OK'
    finally:
        holder.kill(); holder.wait()


def test_child_that_never_ran_the_command_is_requeued_without_a_try(tmp_path):
    """A wrapper that exits 75 (lock busy or superseded launch) never ran the job command:
    the attempt is released, not counted as KILLED."""
    ledger = QS.Ledger(tmp_path/'l')
    ledger.init([(4, 0)], 'd')
    driver = QS.Driver(ledger, workers=1, spacing=0, poll=0.01)
    ledger.claim(driver.me)
    snap = ledger.start((4, 0), driver.me)
    proc = subprocess.Popen([PY, '-c', 'raise SystemExit(75)'])
    snap = driver.wait_child((4, 0), snap, proc)
    assert snap.state is J.UNCLAIMED and snap.data['tries'] == 0 and snap.data['requeues'] == 1
    assert 'never started' in snap.data['released']['why']


def test_init_never_drops_jobs_from_the_ledger(tmp_path):
    """t5 C: `run --jobfile` with the adoption digest rewrote queue.json with only the
    jobfile's rows, orphaning the other state files."""
    ledger = QS.Ledger(tmp_path/'l')
    ledger.init([(1, 1), (2, 1), (3, 1)], 'same')
    ledger.init([(2, 1)], 'same')
    assert ledger.jobs() == [(1, 1), (2, 1), (3, 1)]
    # a state file queue.json does not list still counts
    QS.Ledger(tmp_path/'l').sf((9, 1)).create({'fold': 9, 'seed': 1, 'tries': 0, 'requeues': 0})
    assert (9, 1) in ledger.jobs() and (9, 1) in ledger.views()
    ledger.init([(2, 1)], 'same')
    assert QS.Ledger(tmp_path/'l').header()['jobs'] == [[1, 1], [2, 1], [3, 1], [9, 1]]


def test_concurrent_write_ledger_processes_do_not_crash(tmp_path):
    """t4: concurrent status --write-ledger used to race sf.exists() / sf.create()."""
    rows = list(QS.CONFIRMATION)
    for rnd in range(2):
        repo, run_root, logs = gen2_fixture(tmp_path/f'r{rnd}', rows, 1)
        argv = [PY, '-B', str(QS_PATH), 'status', '--run-root', str(run_root), '--repo-root', str(repo),
                '--ledger-dir', str(tmp_path/f'r{rnd}/ledger'), '--write-ledger', '--json']
        procs = [subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                 for _ in range(8)]
        for p in procs:
            _, err = p.communicate(timeout=120)
            assert p.returncode == 0, err
        views = QS.Ledger(tmp_path/f'r{rnd}/ledger').views()
        assert set(views) == set(rows) and all(v['state'] == 'UNCLAIMED' for v in views.values())


def test_status_reports_corrupt_or_partial_inputs_instead_of_crashing(tmp_path):
    """t5 F and L."""
    repo, run_root, logs = gen2_fixture(tmp_path, [(4, 0), (5, 0)], 1)
    ledger_dir = tmp_path/'ledger'
    ledger = QS.Ledger(ledger_dir)
    ledger.init([(4, 0), (5, 0)], 'y' * 64)
    ledger.abandon((5, 0), 'x')
    kw = dict(ledger_dir=ledger_dir, jobs=[(4, 0), (5, 0)], queue_alive=False, live_scan={})
    # a truncated job file
    good = (ledger_dir/'fold4_seed0.json').read_text()
    (ledger_dir/'fold4_seed0.json').write_text('{"schema": ')
    r = QS.status(run_root, repo, **kw)
    assert r['jobs'][(4, 0)]['state'] == 'CORRUPT' and r['derived']['queue'] == 'NOT_DRAINED'
    assert r['jobs'][(5, 0)]['state'] == 'ABANDONED'
    # an unknown state name
    (ledger_dir/'fold4_seed0.json').write_text(good.replace('"UNCLAIMED"', '"DONE"'))
    r = QS.status(run_root, repo, **kw)
    assert r['jobs'][(4, 0)]['state'] == 'CORRUPT' and 'DONE' in r['jobs'][(4, 0)]['blocked']
    (ledger_dir/'fold4_seed0.json').write_text(good)
    # a truncated queue.json, then one without jobs: the job files are still read
    for broken in ('{"jobs": [[4, 0]', '{"schema": "confirm_queue/1"}'):
        (ledger_dir/'queue.json').write_text(broken)
        r = QS.status(run_root, repo, **kw)
        assert r['jobs'][(5, 0)]['state'] == 'ABANDONED' and r['jobs'][(5, 0)]['source'] == 'ledger'
        assert any('queue.json' in a for a in r['anomalies']), r['anomalies']
    # partial / malformed jobs_queue2.txt lines
    for content, bad in [('8 0\n1 ', 'line 2'), ('8 0\n1', 'line 2'), ('8 0\nx y\n', 'line 2'),
                         ('8 0 extra\n', 'extra columns')]:
        (logs/'jobs_queue2.txt').write_text(content)
        r = QS.status(run_root, repo, jobs=[(8, 0)], queue_alive=True, live_scan={})
        assert r['jobs'][(8, 0)]['state'] == 'UNCLAIMED'
        assert any(bad in a for a in r['anomalies']), (content, r['anomalies'])
    # an empty queue.next is unknown, not 1
    (logs/'jobs_queue2.txt').write_text('8 0\n')
    (logs/'queue.next').write_text('')
    r = QS.status(run_root, repo, jobs=[(8, 0)], queue_alive=True, live_scan={})
    assert r['cursor']['next'] is None and 'unknown' in r['jobs'][(8, 0)]['blocked']
    assert any('queue.next' in a for a in r['anomalies'])


def test_run_skips_a_corrupt_job_file(tmp_path):
    b = Bench(tmp_path, [(4, 0), (5, 0)], sleep=0.01)
    ledger = QS.Ledger(b.ledger)
    ledger.init(b.jobs, QS.read_jobfile(b.jobfile)[1])
    (b.ledger/'fold4_seed0.json').write_text('{"schema": ')
    r = b.run()
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out['counts'] == {'CLI_GENERATED': 1, 'CORRUPT': 1} and out['queue'] == 'NOT_DRAINED'
    assert b.count((5, 0)) == 1 and b.count((4, 0)) == 0
