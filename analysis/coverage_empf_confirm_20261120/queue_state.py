"""Explicit job ledger for the EMPF confirmation training queue (see QUEUE_STATE.md).

The live queue (train_all.sh, then train_queue.sh) keeps its job state implicitly:
start/end lines in queue.log, a positional cursor in queue.next, pid_/rc_ files
and whatever the CLI left in the fit and generation directories. This module
makes that state an explicit per-job machine on top of
``alz_perturb_gan.durable_state``:

    UNCLAIMED -claim-> CLAIMED -start-> RUNNING -exit_ok-> EXITED_OK
        -verify_fit-> TRAINED -verify_cli_a-> CLI_GENERATED -scored-> SCORED
    RUNNING -exit_failed-> FAILED(reason, retryable)
    FAILED -retry-> RETRY_WAIT -start-> RUNNING     (retryable and tries < 3)
    CLAIMED | RUNNING | RETRY_WAIT -owner_dead-> UNCLAIMED
    RUNNING -reown-> RUNNING                         (driver died, job alive)
    UNCLAIMED | FAILED | RETRY_WAIT | EXITED_OK | TRAINED | CLI_GENERATED
        -abandon-> ABANDONED                         (manual, with a reason)

Commands:

    status --run-root R [--json] [--write-ledger [--ledger-dir D]]
        Read-only by default. Adopts the legacy layout (queue.log in both
        formats, queue.next + jobs_queue2.txt, pid_/rc_ files, fit dirs,
        generated readout A, scores/) and prints per-job states, blocked
        reasons and the derived queue state. Writes only with --write-ledger,
        and only into the ledger directory (default <R>/logs/queue_state/).

    run --jobfile J --ledger-dir D --log-dir L [...]
        The future driver: a Python replacement for train_queue.sh, to be used
        only after the live queue has drained. Never run it against the live
        run directory while train_queue.sh is running.

    abandon --ledger-dir D --fold K --seed S --reason TEXT

The queue-level state is derived, never stored as free text: DRAINED iff every
job is SCORED, ABANDONED or FAILED with no retry left.

Stdlib only; CPU only. Python 3.11+, Linux (/proc).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import enum
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))

from alz_perturb_gan import durable_state as ds  # noqa: E402
from alz_perturb_gan.durable_state import (  # noqa: E402
    FileLock, IllegalTransition, GuardFailed, Machine, StateFile, VersionConflict,
)

DEFAULT_RUN_ROOT = REPO / 'runs/coverage_empf_confirm_20261120'
PYTHON = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
MODE_A = 'cellshape@0.5:biascal:stagehalf'
FIT_SUBDIR = 'runs/microglia_empf/fits/pbulk/hvg2000'
GENERATED_SUBDIR = 'runs/microglia_empf/generated'
PILOT = (0, 0)
CONFIRMATION = tuple((k, s) for s in (0, 1) for k in range(9) if (k, s) != PILOT)
MAX_TRIES = 3
MAX_REQUEUES = 5
HOST_RACE = 'existing derived EMPF host differs'
HOST_PATH = re.compile(r'alz-host-empf-native|native-releases/particlegan')
FAULT_ENV = 'QUEUE_STATE_FAULT'
FAULT_POINTS = ('after_claim', 'after_start', 'while_running', 'after_exit')
SCHEMA = 'confirm_queue/1'


# --------------------------------------------------------------------------- the job machine
class JobState(enum.Enum):
    UNCLAIMED = 'UNCLAIMED'
    CLAIMED = 'CLAIMED'
    RUNNING = 'RUNNING'
    EXITED_OK = 'EXITED_OK'
    FAILED = 'FAILED'
    RETRY_WAIT = 'RETRY_WAIT'
    TRAINED = 'TRAINED'
    CLI_GENERATED = 'CLI_GENERATED'
    SCORED = 'SCORED'
    ABANDONED = 'ABANDONED'


J = JobState
TRANSITIONS = {
    (J.UNCLAIMED, 'claim'): J.CLAIMED,
    (J.CLAIMED, 'start'): J.RUNNING,
    (J.RETRY_WAIT, 'start'): J.RUNNING,
    (J.CLAIMED, 'owner_dead'): J.UNCLAIMED,
    (J.RUNNING, 'owner_dead'): J.UNCLAIMED,
    (J.RETRY_WAIT, 'owner_dead'): J.UNCLAIMED,
    (J.RUNNING, 'reown'): J.RUNNING,
    (J.RUNNING, 'exit_ok'): J.EXITED_OK,
    (J.RUNNING, 'exit_failed'): J.FAILED,
    (J.FAILED, 'retry'): J.RETRY_WAIT,
    (J.EXITED_OK, 'verify_fit'): J.TRAINED,
    (J.TRAINED, 'verify_cli_a'): J.CLI_GENERATED,
    (J.CLI_GENERATED, 'scored'): J.SCORED,
    **{(s, 'abandon'): J.ABANDONED for s in (J.UNCLAIMED, J.FAILED, J.RETRY_WAIT, J.EXITED_OK,
                                             J.TRAINED, J.CLI_GENERATED)},
}
JOB_MACHINE = Machine('confirm_queue_job', JobState, J.UNCLAIMED, {J.SCORED, J.ABANDONED}, TRANSITIONS,
                      adopt_entries=set(JobState) - {J.UNCLAIMED})
LIVE_STATES = (J.CLAIMED, J.RUNNING, J.RETRY_WAIT)

# Failure reasons -> retryable. HOST_* and KILLED are transient; GENERATION_FAILED
# reruns as a generation-only step (the CLI resumes a complete fit to generation).
REASONS = {'HOST_DIFFERS': True, 'HOST_PARTIAL_IMPORT': True, 'KILLED': True, 'OOM': False,
           'OUTPUT_EXISTS': False, 'SOURCE_CHANGED': False, 'GENERATION_FAILED': True,
           'TRAIN_ERROR': False}


def fault(point):
    """Crash-injection hook: QUEUE_STATE_FAULT=<point> kills this process at <point>."""
    if os.environ.get(FAULT_ENV) == point:
        os._exit(137)


# --------------------------------------------------------------------------- process identity
def identity(worker=None):
    pid = os.getpid()
    return dict(pid=pid, pid_starttime=ds.pid_starttime(pid), host=socket.gethostname(), worker=worker)


def alive(info):
    """pid + /proc starttime liveness; a reused pid (other starttime) is dead."""
    if not info or info.get('pid') is None:
        return False
    if info.get('pid_starttime') is None:
        return False  # never trust a bare pid in the ledger
    return ds.owner_status(info) == 'alive'


def cmdline(pid):
    try:
        raw = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except (OSError, ValueError):
        return None
    return raw.replace(b'\0', b' ').decode(errors='replace')


def cmd_matches_job(cmd, fold, seed):
    if not cmd or 'microglia' not in cmd:
        return False
    return bool(re.search(rf'--fold\s+{fold}\s(?:.*\s)?--seed\s+{seed}(\s|$)', cmd))


def legacy_pid_alive(pid, fold, seed, not_before=None, read_cmdline=cmdline, start_epoch=None):
    """A legacy pid (no starttime recorded) counts as alive only when its cmdline
    names microglia and this fold/seed, and it started no earlier than not_before."""
    cmd = read_cmdline(pid)
    if not cmd_matches_job(cmd, fold, seed):
        return False
    if not_before is not None:
        started = (start_epoch or ds._process_start_epoch)(int(pid))
        if started is None or started < not_before - 2.0:
            return False
    return True


def scan_live_jobs(read_cmdline=cmdline):
    """{(fold, seed): [pid, ...]} for live processes whose cmdline names a microglia job."""
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        cmd = read_cmdline(proc.name)
        if not cmd or 'microglia' not in cmd:
            continue
        m = re.search(r'--fold\s+(\d+)\s(?:.*\s)?--seed\s+(\d+)(\s|$)', cmd)
        if m:
            found.setdefault((int(m.group(1)), int(m.group(2))), []).append(int(proc.name))
    return found


# --------------------------------------------------------------------------- classify
def classify(log_tail, rc, fit_dir=None):
    """(reason, retryable) for a finished, unsuccessful attempt.

    rc is an int, the string 'missing' or None. fit_dir, when given, is checked
    for final.pt + fit_receipt.json (a complete fit whose CLI generation failed).
    """
    text = log_tail or ''
    if HOST_RACE in text:
        return 'HOST_DIFFERS', True
    if re.search(r'^\s*(SyntaxError|IndentationError|ImportError|ModuleNotFoundError)\b', text, re.M) \
            and HOST_PATH.search(text):
        return 'HOST_PARTIAL_IMPORT', True
    if rc in (None, 'missing', '', 137, 9, -9, '137', '9', '-9'):
        return 'KILLED', True
    if 'CUDA out of memory' in text or 'OutOfMemoryError' in text:
        return 'OOM', False
    rc_i = int(rc) if str(rc).lstrip('-').isdigit() else None
    if 'FileExistsError' in text or (rc_i == 2 and re.search(
            r'exists|refusing to overwrite|must be outside', text)):
        return 'OUTPUT_EXISTS', False
    if re.search(r'source manifest changed|source missing or changed|source changed|'
                 r'confirmation input changed|entrypoint archive changed', text):
        return 'SOURCE_CHANGED', False
    if fit_dir is not None and (Path(fit_dir)/'final.pt').exists() and (Path(fit_dir)/'fit_receipt.json').exists():
        return 'GENERATION_FAILED', True
    return 'TRAIN_ERROR', False


def tail(path, limit=20000):
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - limit))
            return fh.read().decode(errors='replace')
    except OSError:
        return ''


# --------------------------------------------------------------------------- fit / readout / score evidence
def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def fit_state(fit_dir):
    """Legacy fit-dir rules (duplicated from the sm-runtime-fit-lifecycle design as a pure,
    read-only function): FRESH, TRAINING, INVALID, FINAL_UNRECEIPTED, RECEIPTED or AUDITED.
    An explicit fit_state.json (durable_state document) wins when present."""
    fit = Path(fit_dir)
    explicit = _json(fit/'fit_state.json')
    if isinstance(explicit, dict) and isinstance(explicit.get('state'), str):
        return explicit['state'], {'source': 'fit_state.json'}
    if not fit.is_dir():
        return 'FRESH', {}
    has = {n: (fit/n).exists() for n in ('config.json', 'final.pt', 'recovery.pt', 'fit_receipt.json',
                                         'no_skip_audit.json')}
    if not has['final.pt']:
        return ('TRAINING' if has['recovery.pt'] or has['config.json'] else 'FRESH'), has
    if not has['config.json']:
        return 'INVALID', has
    if not has['fit_receipt.json']:
        return 'FINAL_UNRECEIPTED', has
    receipt = _json(fit/'fit_receipt.json') or {}
    if receipt.get('completed_updates') != 20000 or receipt.get('snapshots_missing'):
        return 'INVALID', dict(has, reason='receipt is not a complete 20000-update fit')
    audit = _json(fit/'no_skip_audit.json')
    if isinstance(audit, dict) and audit.get('final_checkpoint_sha256') == receipt.get('final_SHA256') \
            and audit.get('completed_updates') == 20000:
        return 'AUDITED', has
    return 'RECEIPTED', has


def cli_a_state(gen_dir, fold, seed, deep=False):
    """(ok, detail) for the CLI's readout A: NPZ + receipt for this fit and mode."""
    npz = Path(gen_dir)/f'generated_{MODE_A}.npz'
    receipt_path = npz.with_name(npz.stem + '.receipt.json')
    if not npz.exists():
        return False, 'readout A NPZ missing'
    receipt = _json(receipt_path)
    if not isinstance(receipt, dict):
        return False, 'readout A receipt missing or unreadable'
    if receipt.get('mode') != MODE_A or not str(receipt.get('fit', '')).rstrip('/').endswith(f'fold{fold}/seed{seed}'):
        return False, 'readout A receipt names another fit or mode'
    if 'application_default' not in receipt or not receipt.get('generated_SHA256'):
        return False, 'readout A receipt is not a CLI receipt'
    if deep and hashlib.sha256(npz.read_bytes()).hexdigest() != receipt['generated_SHA256']:
        return False, 'readout A NPZ hash differs from its receipt'
    return True, 'readout A NPZ + receipt'


def score_state(score_dir, fold, seed):
    value = _json(Path(score_dir)/'score.json')
    if not isinstance(value, dict):
        return False, 'score.json missing'
    if value.get('status') != 'scored' or value.get('fold') != fold or value.get('seed') != seed:
        return False, 'score.json is not a scored record for this fit'
    return True, 'score.json'


class Layout:
    """Where one job's evidence lives. repo_root holds the CLI's fit and generation roots."""

    def __init__(self, run_root, repo_root=None):
        self.run_root = Path(run_root)
        self.repo_root = Path(repo_root) if repo_root else self.run_root.parents[1]
        self.logs = self.run_root/'logs'

    def fit(self, k, s):
        return self.repo_root/FIT_SUBDIR/f'fold{k}/seed{s}'

    def generated(self, k, s):
        return self.repo_root/GENERATED_SUBDIR/f'fold{k}/seed{s}'

    def score(self, k, s):
        return self.run_root/f'scores/fold{k}/seed{s}'


def evidence_state(layout, k, s, deep=False):
    """The furthest verified stage from files alone: SCORED, CLI_GENERATED, TRAINED or None."""
    fstate, _ = fit_state(layout.fit(k, s))
    trained = fstate in ('RECEIPTED', 'AUDITED')
    cli_ok, cli_detail = cli_a_state(layout.generated(k, s), k, s, deep=deep)
    scored, _ = score_state(layout.score(k, s), k, s)
    if scored:
        return 'SCORED', dict(fit=fstate, readout_a=cli_detail)
    if trained and cli_ok:
        return 'CLI_GENERATED', dict(fit=fstate, readout_a=cli_detail)
    if trained:
        return 'TRAINED', dict(fit=fstate, readout_a=cli_detail)
    return None, dict(fit=fstate, readout_a=cli_detail)


# --------------------------------------------------------------------------- legacy reader (read-only)
TS = r'(?P<ts>\S+)'
LINE_PATTERNS = [
    ('start1', re.compile(TS + r' start fold(?P<k>\d+) seed(?P<s>\d+) gpu(?P<gpu>\d+)$')),
    ('start2', re.compile(TS + r' start fold(?P<k>\d+) seed(?P<s>\d+) cores(?P<c>\d+-\d+) try(?P<n>\d+)$')),
    ('retry', re.compile(TS + r' retry fold(?P<k>\d+) seed(?P<s>\d+) rc=(?P<rc>\S+) \(activation race\)$')),
    ('end', re.compile(TS + r' end fold(?P<k>\d+) seed(?P<s>\d+) rc=(?P<rc>\d+|missing)$')),
    ('finished1', re.compile(TS + r' queue finished$')),
    ('finished2', re.compile(TS + r' queue2 finished$')),
]


def _epoch(ts):
    from datetime import datetime
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def _rc(value):
    return 'missing' if value in (None, 'missing') else int(value)


def parse_queue_log(text):
    """Attempts per job from queue.log, both generations. Returns (attempts, markers, anomalies).

    attempts[(k, s)] is a list of dicts {gen, try, start_ts, cores|gpu, rc, outcome}
    where outcome is None (open), 'end' or 'retry'. 'queue finished' lines are
    recorded as markers only; they never close a job."""
    attempts, markers, anomalies = {}, [], []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        for kind, pat in LINE_PATTERNS:
            m = pat.match(line)
            if m:
                break
        else:
            anomalies.append(f'line {lineno}: unrecognised: {line}')
            continue
        if kind.startswith('finished'):
            markers.append(dict(kind=kind, ts=m['ts'], line=lineno))
            continue
        job = (int(m['k']), int(m['s']))
        rows = attempts.setdefault(job, [])
        if kind in ('start1', 'start2'):
            if any(r['outcome'] is None for r in rows):
                anomalies.append(f'line {lineno}: fold{job[0]} seed{job[1]} started while an attempt is open')
            gen = 1 if kind == 'start1' else 2
            rows.append(dict(gen=gen, try_=int(m['n']) if gen == 2 else 1, start_ts=m['ts'],
                             start_epoch=_epoch(m['ts']), slot=m['gpu'] if gen == 1 else m['c'],
                             rc=None, outcome=None, end_ts=None, line=lineno))
            continue
        open_rows = [r for r in rows if r['outcome'] is None]
        if not open_rows:
            anomalies.append(f'line {lineno}: {kind} for fold{job[0]} seed{job[1]} without an open attempt')
            continue
        if len(open_rows) > 1:
            anomalies.append(f'line {lineno}: several open attempts for fold{job[0]} seed{job[1]}; closing the latest')
        row = open_rows[-1]
        row.update(rc=_rc(m['rc']), outcome=kind, end_ts=m['ts'])
    return attempts, markers, anomalies


SCORER_START = re.compile(TS + r' start scorer fold(?P<k>\d+) seed(?P<s>\d+) pid (?P<pid>\d+)$')
SCORER_FAIL = re.compile(TS + r' scorer fold(?P<k>\d+) seed(?P<s>\d+) exited without score.json, '
                         r'attempt (?P<n>\d+) of (?P<of>\d+)')


def parse_score_queue_log(text):
    """score_all.sh's log: {(k, s): {'pid': last scorer pid, 'failures': n, 'of': limit}}."""
    out = {}
    for line in text.splitlines():
        m = SCORER_START.match(line.strip())
        if m:
            out.setdefault((int(m['k']), int(m['s'])), dict(pid=None, failures=0, of=2))['pid'] = int(m['pid'])
            continue
        m = SCORER_FAIL.match(line.strip())
        if m:
            row = out.setdefault((int(m['k']), int(m['s'])), dict(pid=None, failures=0, of=2))
            row.update(failures=int(m['n']), of=int(m['of']))
    return out


def scorer_blocked(row, k, s, read_cmdline=cmdline):
    """Why a trained, generated fit has no score.json yet, from score_all.sh's log."""
    if not row:
        return 'awaiting score.py'
    cmd = read_cmdline(row['pid']) if row.get('pid') else None
    if cmd and 'score.py' in cmd and re.search(rf'--fold\s+{k}\s+--seed\s+{s}(\s|$)', cmd):
        return f"scoring (score.py pid {row['pid']})"
    if row['failures'] >= row['of']:
        return f"scorer gave up after {row['failures']} attempts: operator must fix or abandon"
    if row['failures']:
        return f"scorer failed {row['failures']} of {row['of']}; score_all.sh retries"
    return 'awaiting score.py'


def read_cursor(logs):
    """The gen-2 positional cursor: (jobfile rows, next index, jobfile sha256) or None."""
    jobfile, idx = Path(logs)/'jobs_queue2.txt', Path(logs)/'queue.next'
    if not jobfile.exists():
        return None
    raw = jobfile.read_bytes()
    rows = [tuple(int(x) for x in line.split()) for line in raw.decode().splitlines() if line.strip()]
    try:
        nxt = int(idx.read_text().strip())
    except (OSError, ValueError):
        nxt = 1
    return dict(jobfile=str(jobfile), sha256=hashlib.sha256(raw).hexdigest(), rows=rows, next=nxt,
                claimed=rows[:max(0, nxt - 1)], unclaimed=rows[max(0, nxt - 1):])


def queue_process_alive(read_cmdline=cmdline):
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            cmd = read_cmdline(proc.name)
            if cmd and ('train_queue.sh' in cmd or 'train_all.sh' in cmd):
                return True
    return False


def adopt_legacy(run_root, repo_root=None, jobs=None, *, read_cmdline=cmdline, live_scan=None,
                 queue_alive=None, start_epoch=None, deep=False):
    """Per-job states inferred from the legacy layout. Pure reader: never writes.

    Returns {'jobs': {(k, s): view}, 'markers', 'anomalies', 'cursor'}; a view is
    {state, try, generation, pid, reason, retryable, evidence, blocked}.
    """
    layout = Layout(run_root, repo_root)
    logs = layout.logs
    attempts, markers, anomalies = parse_queue_log(tail(logs/'queue.log', limit=10**7))
    cursor = read_cursor(logs)
    scorers = parse_score_queue_log(tail(logs/'score_queue.log', limit=10**7))
    live = live_scan if live_scan is not None else scan_live_jobs(read_cmdline)
    q_alive = queue_alive if queue_alive is not None else queue_process_alive(read_cmdline)
    universe = list(jobs) if jobs is not None else list(CONFIRMATION)
    for job in list(attempts) + (cursor['rows'] if cursor else []):
        if job not in universe:
            universe.append(job)
    out = {}
    for k, s in universe:
        rows = attempts.get((k, s), [])
        view = dict(state=None, try_=len(rows) and rows[-1]['try_'], generation=rows[-1]['gen'] if rows else None,
                    pid=None, reason=None, retryable=None, tries=sum(1 for r in rows if r['outcome']),
                    attempts=len(rows), evidence={}, blocked=None)
        stage, evidence = evidence_state(layout, k, s, deep=deep)
        view['evidence'] = evidence
        last = rows[-1] if rows else None
        # liveness of the last open attempt
        running_pid = None
        if last is not None and last['outcome'] is None:
            if last['gen'] == 2:
                pid_file = logs/f"pid_fold{k}_seed{s}.try{last['try_']}"
                try:
                    pid = int(pid_file.read_text().strip())
                except (OSError, ValueError):
                    pid = None
                if pid and legacy_pid_alive(pid, k, s, last['start_epoch'], read_cmdline, start_epoch):
                    running_pid = pid
            else:
                for pid in live.get((k, s), []):
                    if legacy_pid_alive(pid, k, s, last['start_epoch'], read_cmdline, start_epoch):
                        running_pid = pid
                        break
        if stage == 'SCORED':
            view['state'] = 'SCORED'
        elif running_pid:
            view.update(state='RUNNING', pid=running_pid,
                        retry=last['gen'] == 2)
            if last['gen'] == 1:
                view['blocked'] = 'gen-1 orphan: runs outside any live queue; no retry if it fails'
        elif last is not None and last['outcome'] is None:
            rc_file = logs/f"rc_fold{k}_seed{s}.try{last['try_']}"
            rc = None
            if last['gen'] == 2 and rc_file.exists():
                try:
                    rc = _rc(rc_file.read_text().strip() or 'missing')
                except ValueError:
                    rc = 'missing'
            if rc is not None:
                last = dict(last, rc=rc, outcome='end')
                view['tries'] += 1
            elif stage in ('TRAINED', 'CLI_GENERATED'):
                view['state'] = stage
            else:
                view.update(state='UNCLAIMED', blocked='attempt open but no live process and no rc: '
                            'owner dead, re-queue (owner_dead)')
        if view['state'] is None and stage == 'CLI_GENERATED':
            view['state'] = 'CLI_GENERATED'
            view['blocked'] = scorer_blocked(scorers.get((k, s)), k, s, read_cmdline)
        elif view['state'] is None and last is not None and last['outcome'] == 'retry':
            view.update(state='RETRY_WAIT', blocked=None if q_alive else 'retry logged but queue process is dead')
        elif view['state'] is None and last is not None and last['outcome'] == 'end':
            if last['rc'] == 0:
                view['state'] = 'TRAINED' if stage == 'TRAINED' else 'EXITED_OK'
                view['blocked'] = ('CLI readout A missing after rc=0' if stage == 'TRAINED'
                                   else f"rc=0 but fit not verifiable ({evidence['fit']})")
            else:
                log = logs/(f"train_fold{k}_seed{s}.try{last['try_']}.log" if last['gen'] == 2
                            else f'train_fold{k}_seed{s}.log')
                reason, retryable = classify(tail(log), last['rc'], layout.fit(k, s))
                view.update(state='FAILED', reason=reason, retryable=retryable, rc=last['rc'])
        if view['state'] is None and stage == 'TRAINED':
            view.update(state='TRAINED', blocked='CLI readout A missing')
        if view['state'] is None:
            claimed = cursor is not None and (k, s) in cursor['claimed']
            if claimed and not rows_since_cursor(rows):
                if q_alive:
                    view.update(state='CLAIMED', blocked='claimed by queue.next, waiting for the start lock')
                else:
                    view.update(state='UNCLAIMED', blocked='claimed by queue.next but never started; queue dead '
                                '(dropped job, re-queue)')
            else:
                view['state'] = 'UNCLAIMED'
                if cursor is not None and (k, s) in cursor['unclaimed']:
                    view['blocked'] = None if q_alive else 'waiting in jobs_queue2.txt but the queue process is dead'
                elif cursor is None or (k, s) not in cursor['rows']:
                    view['blocked'] = 'not in any live job file'
        if view['state'] == 'FAILED':
            terminal = not view['retryable'] or view['tries'] >= MAX_TRIES
            view['terminal'] = terminal
            view['blocked'] = (f"terminal failure {view['reason']}: operator must fix or abandon" if terminal
                               else f"retryable failure {view['reason']}: the legacy queue does not retry this; "
                                    'rerun the job (generation-only if the fit is complete)')
        out[(k, s)] = view
    return dict(jobs=out, markers=markers, anomalies=anomalies, cursor=cursor, queue_process_alive=q_alive)


def rows_since_cursor(rows):
    """Whether a gen-2 start exists for the job (gen-1 attempts do not count for the gen-2 cursor)."""
    return any(r['gen'] == 2 for r in rows)


# --------------------------------------------------------------------------- derived queue state
def is_terminal_view(view):
    if view['state'] in ('SCORED', 'ABANDONED'):
        return True
    return view['state'] == 'FAILED' and (not view.get('retryable') or view.get('tries', 0) >= MAX_TRIES)


def derive_queue(views):
    states = {job: v['state'] for job, v in views.items()}
    drained = bool(views) and all(is_terminal_view(v) for v in views.values())
    training_open = [job for job, v in views.items()
                     if v['state'] in ('UNCLAIMED', 'CLAIMED', 'RUNNING', 'RETRY_WAIT')
                     or (v['state'] == 'FAILED' and not is_terminal_view(v))]
    counts = {}
    for st in states.values():
        counts[st] = counts.get(st, 0) + 1
    return dict(queue='DRAINED' if drained else 'NOT_DRAINED', training_drained=not training_open,
                counts=dict(sorted(counts.items())), training_open=sorted(training_open))


# --------------------------------------------------------------------------- the ledger
def retry_left(data):
    return bool((data.get('failure') or {}).get('retryable')) and data.get('tries', 0) < MAX_TRIES


def job_name(job):
    return f'fold{job[0]}_seed{job[1]}'


def _view_from_snapshot(snap):
    d = snap.data
    failure = d.get('failure') or {}
    return dict(state=snap.state.name, try_=d.get('try'), tries=d.get('tries', 0), reason=failure.get('reason'),
                retryable=failure.get('retryable'), owner=d.get('owner'), child=d.get('child'),
                requeues=d.get('requeues', 0), version=snap.version, blocked=None)


class Ledger:
    """Per-job StateFiles in one directory, plus queue.json (jobfile digest and job order)."""

    def __init__(self, directory, layout=None, deep=False, read_only=False):
        self.dir = Path(directory)
        self.layout = layout
        self.deep = deep
        self.read_only = read_only

    # -- files
    def sf(self, job):
        # read_only: never reconcile a journal on open (status must not write).
        return StateFile(self.dir/f'{job_name(job)}.json', JOB_MACHINE, recover=not self.read_only)

    def child_base(self, job, n):
        return self.dir/'children'/f'{job_name(job)}.try{n}'

    def header(self):
        return _json(self.dir/'queue.json')

    def jobs(self):
        h = self.header()
        return [tuple(j) for j in h['jobs']] if h else []

    def init(self, jobs, jobfile_sha256, jobfile=None, replace=False):
        """Record the job list and the jobfile digest. A different jobfile is refused unless
        replace=True: per-job states are keyed by (fold, seed), never by position."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir/'children').mkdir(exist_ok=True)
        with FileLock(self.dir/'queue.json', purpose='init'):
            h = self.header()
            if h and h['jobfile_sha256'] != jobfile_sha256 and not replace:
                raise SystemExit(f'ledger {self.dir} belongs to jobfile {h["jobfile_sha256"][:12]} '
                                 f'({h.get("jobfile")}); refusing {jobfile_sha256[:12]} without --replace-jobfile')
            ordered = list(dict.fromkeys([*(tuple(j) for j in (h['jobs'] if h and replace else [])), *jobs]))
            if not h or h['jobfile_sha256'] != jobfile_sha256 or [tuple(j) for j in h['jobs']] != ordered:
                ds.atomic_write_json(self.dir/'queue.json', dict(
                    schema=SCHEMA, jobfile=str(jobfile) if jobfile else None, jobfile_sha256=jobfile_sha256,
                    jobs=[list(j) for j in ordered], history=[*((h or {}).get('history', [])),
                                                            dict(jobfile_sha256=jobfile_sha256, at=time.time())]))
        for job in jobs:
            self.sf(job).create({'fold': job[0], 'seed': job[1], 'tries': 0, 'requeues': 0}, exist_ok=True)

    # -- reading
    def views(self):
        out = {}
        for job in self.jobs():
            snap = self.sf(job).read()
            if snap is not None:
                out[job] = _view_from_snapshot(snap)
        return out

    # -- transitions
    def claim(self, owner):
        """Claim the first UNCLAIMED job in queue order (or take over a retryable FAILED job
        through FAILED -retry-> RETRY_WAIT); None when nothing is claimable."""
        for job in self.jobs():
            sf = self.sf(job)
            snap = sf.read()
            if snap is None:
                continue
            try:
                if snap.state is J.UNCLAIMED and snap.data.get('requeues', 0) < MAX_REQUEUES:
                    sf.transition('claim', expect_version=snap.version,
                                  update=lambda d: {**d, 'owner': owner, 'claimed_at': time.time()})
                elif snap.state is J.FAILED and retry_left(snap.data):
                    self.retry(job, owner, expect_version=snap.version)
                else:
                    continue
            except (IllegalTransition, VersionConflict, GuardFailed):
                continue
            return job
        return None

    def start(self, job, owner):
        """CLAIMED|RETRY_WAIT -start-> RUNNING; records try n and a launch id. Returns the snapshot."""
        def guard(snap):
            return (snap.data.get('owner') or {}).get('pid') == owner['pid'] and \
                snap.data.get('tries', 0) < MAX_TRIES

        def update(d):
            d['try'] = d.get('tries', 0) + 1
            d['launch_id'] = uuid.uuid4().hex
            d['child'] = None
            d['started_at'] = time.time()
            return d
        return self.sf(job).transition('start', guard=guard, update=update)

    def exit(self, job, rc, log_tail='', launch_id=None):
        snap = self.sf(job).read()

        def guard(s):
            return launch_id is None or s.data.get('launch_id') == launch_id

        def update(d):
            d['tries'] = d.get('tries', 0) + 1
            d['rc'] = rc
            d['ended_at'] = time.time()
            d['history'] = [*d.get('history', []), dict(try_=d.get('try'), rc=rc,
                                                        failure=None if rc == 0 else d.get('failure'))]
            return d
        if rc == 0:
            return self.sf(job).transition('exit_ok', guard=guard, update=update, expect_version=snap.version)
        fit = self.layout.fit(*job) if self.layout else None
        reason, retryable = classify(log_tail, rc, fit)

        def update_failed(d):
            d = update(d)
            d['failure'] = dict(reason=reason, retryable=retryable)
            d['history'][-1]['failure'] = d['failure']
            return d
        return self.sf(job).transition('exit_failed', guard=guard, update=update_failed,
                                       expect_version=snap.version)

    def retry(self, job, owner, expect_version=None):
        return self.sf(job).transition('retry', guard=lambda s: retry_left(s.data), expect_version=expect_version,
                                       update=lambda d: {**d, 'owner': owner})

    def owner_dead(self, job, snap):
        def update(d):
            d['requeues'] = d.get('requeues', 0) + 1
            d['owner'] = None
            d['child'] = None
            return d
        return self.sf(job).transition('owner_dead', update=update, expect_version=snap.version)

    def reown(self, job, snap, owner):
        return self.sf(job).transition('reown', expect_version=snap.version,
                                       update=lambda d: {**d, 'owner': owner})

    def abandon(self, job, reason):
        return self.sf(job).transition('abandon', update=lambda d: {**d, 'abandoned': dict(
            reason=reason, at=time.time(), by=identity())})

    def advance(self, job):
        """EXITED_OK -> TRAINED -> CLI_GENERATED -> SCORED as far as the files verify."""
        if self.layout is None:
            return self.sf(job).read()
        k, s = job
        sf = self.sf(job)
        while True:
            snap = sf.read()
            if snap.state is J.EXITED_OK:
                st, _ = fit_state(self.layout.fit(k, s))
                if st not in ('RECEIPTED', 'AUDITED'):
                    return snap
                event, ev = 'verify_fit', dict(fit_state=st)
            elif snap.state is J.TRAINED:
                ok, detail = cli_a_state(self.layout.generated(k, s), k, s, deep=self.deep)
                if not ok:
                    return snap
                event, ev = 'verify_cli_a', dict(readout_a=detail)
            elif snap.state is J.CLI_GENERATED:
                ok, detail = score_state(self.layout.score(k, s), k, s)
                if not ok:
                    return snap
                event, ev = 'scored', dict(score=detail)
            else:
                return snap
            try:
                sf.transition(event, expect_version=snap.version,
                              update=lambda d, ev=ev, event=event: {**d, event: ev})
            except VersionConflict:
                continue

    # -- recovery
    @contextlib.contextmanager
    def child_lock(self, job, snap):
        """Yields None while the attempt's child is alive, else ('exited', rc) | ('vanished', None) |
        ('never', None) with the child's run lock held. The child holds <base>.run's flock for its
        whole life (the fd is passed on to the job command), so a free lock means no live child, and
        a late-starting child blocks on the lock and then sees the ledger has moved on."""
        base = self.child_base(job, snap.data.get('try'))
        lock = FileLock(f'{base}.run', blocking=False, record_owner=False)
        try:
            lock.acquire()
        except ds.Busy:
            yield None
            return
        try:
            launch = snap.data.get('launch_id')
            rc = _json(f'{base}.rc')
            started = _json(f'{base}.child')
            if isinstance(rc, dict) and rc.get('launch_id') == launch:
                yield 'exited', rc['rc']
            elif isinstance(started, dict) and started.get('launch_id') == launch:
                yield 'vanished', 'missing'
            else:
                yield 'never', None
        finally:
            lock.release()

    def settle(self, job, snap, outcome):
        """Apply a dead child's outcome (lock held by the caller)."""
        kind, rc = outcome
        if kind == 'never':
            return self.owner_dead(job, snap)
        base = self.child_base(job, snap.data.get('try'))
        return self.exit(job, rc, tail(f'{base}.log'), launch_id=snap.data.get('launch_id'))

    def owner_alive(self, job, owner):
        if owner.get('legacy'):
            return bool(owner.get('pid')) and legacy_pid_alive(owner['pid'], *job)
        return alive(owner)

    def recover(self, me, adopt_running=True):
        """Re-queue claimed-but-unfinished jobs whose owner is dead (pid + starttime), apply exits a
        dead driver never recorded, and return RUNNING jobs whose live child this driver now owns."""
        adopted = []
        for job in self.jobs():
            snap = self.sf(job).read()
            if snap is None or snap.state not in LIVE_STATES:
                continue
            owner = snap.data.get('owner') or {}
            if owner.get('pid') == me['pid'] and owner.get('pid_starttime') == me['pid_starttime']:
                continue
            if self.owner_alive(job, owner):
                continue
            try:
                if snap.state is not J.RUNNING:
                    self.owner_dead(job, snap)
                    continue
                with self.child_lock(job, snap) as outcome:
                    if outcome is not None:
                        self.settle(job, snap, outcome)
                    elif adopt_running:
                        self.reown(job, snap, me)
                        adopted.append(job)
            except (VersionConflict, IllegalTransition, GuardFailed):
                continue
        return adopted


# --------------------------------------------------------------------------- the driver
def _pdeathsig():
    try:
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def child_main(argv):
    """Internal: run one job attempt. Holds <base>.run's flock (passed on to the job command),
    records itself in <base>.child, runs the command with its output in <base>.log and writes
    <base>.rc atomically. Refuses to run unless the ledger still says RUNNING with this launch id."""
    p = argparse.ArgumentParser(prog='queue_state.py _child')
    p.add_argument('--ledger', required=True); p.add_argument('--fold', type=int, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--try', dest='n', type=int, required=True)
    p.add_argument('--launch-id', required=True); p.add_argument('cmd', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    ledger = Ledger(a.ledger)
    job = (a.fold, a.seed)
    base = ledger.child_base(job, a.n)
    lock = FileLock(f'{base}.run', blocking=False, record_owner=False)
    try:
        lock.acquire()
    except ds.Busy:
        return 75
    snap = ledger.sf(job).read()
    if snap is None or snap.state is not J.RUNNING or snap.data.get('launch_id') != a.launch_id:
        return 75  # superseded launch: recovery re-queued the job before this child started
    ds.atomic_write_json(f'{base}.child', dict(identity(), launch_id=a.launch_id, cmd=cmd))
    with open(f'{base}.log', 'ab') as log:
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock._fd,),
                                    preexec_fn=_pdeathsig)
            rc = proc.wait()
        except OSError as exc:
            log.write(f'queue_state: cannot start job: {exc}\n'.encode())
            rc = 127
    if rc < 0:
        rc = 128 - rc
    ds.atomic_write_json(f'{base}.rc', dict(rc=rc, launch_id=a.launch_id, at=time.time()))
    return 0


def default_command(job, cores, python=PYTHON):
    k, s = job
    gpu = (int(cores.split('-')[0]) // 3) % 2
    return ['env', f'CUDA_VISIBLE_DEVICES={gpu}', 'taskset', '-c', cores, python, '-B',
            'scripts/microglia.py', 'train', '--fold', str(k), '--seed', str(s), '--device', 'cuda']


class Driver:
    def __init__(self, ledger, workers=8, worker_offset=0, spacing=25.0, poll=30.0, command=None,
                 cwd=None):
        self.ledger = ledger
        self.workers = workers
        self.offset = worker_offset
        self.spacing = spacing
        self.poll = poll
        self.command = command or default_command
        self.cwd = cwd
        self.me = identity()
        self.stop = threading.Event()

    def launch(self, job, snap, cores):
        n, launch_id = snap.data['try'], snap.data['launch_id']
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), '_child', '--ledger', str(self.ledger.dir),
                '--fold', str(job[0]), '--seed', str(job[1]), '--try', str(n), '--launch-id', launch_id, '--',
                *self.command(job, cores)]
        env = {k: v for k, v in os.environ.items() if k != FAULT_ENV}
        # start_new_session: the job survives a crash of this driver; a restart reowns it.
        with FileLock(self.ledger.dir/'start', purpose='spaced start'):
            proc = subprocess.Popen(argv, cwd=self.cwd, env=env, start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if self.spacing:
                time.sleep(self.spacing)
        return proc

    def wait_child(self, job, snap, proc=None):
        """Wait until the attempt's child is gone, then apply its exit. Returns the new snapshot."""
        first = True
        while True:
            if proc is not None and proc.poll() is None:
                outcome = None
            else:
                with self.ledger.child_lock(job, snap) as outcome:
                    if outcome is not None:
                        fault('after_exit')
                        return self.ledger.settle(job, snap, outcome if outcome[0] != 'never'
                                                  else ('vanished', 'missing'))
            if first:
                fault('while_running')
            first = False
            time.sleep(self.poll)

    def run_job(self, job, cores, snap=None):
        """Drive one claimed (or reowned RUNNING) job until it leaves the retry loop."""
        while True:
            proc = None
            if snap is None or snap.state is not J.RUNNING:
                snap = self.ledger.start(job, self.me)
                fault('after_start')
                proc = self.launch(job, snap, cores)
            snap = self.wait_child(job, snap, proc)
            if snap.state is J.EXITED_OK:
                self.ledger.advance(job)
                return
            if retry_left(snap.data):
                self.ledger.retry(job, self.me, expect_version=snap.version)
                snap = None
                continue
            return

    def worker(self, w, adopted):
        cores = f'{3 * (w + self.offset)}-{3 * (w + self.offset) + 2}'
        while not self.stop.is_set():
            with adopted['lock']:
                job = adopted['jobs'].pop(0) if adopted['jobs'] else None
            if job is not None:
                self.run_job(job, cores, snap=self.ledger.sf(job).read())
                continue
            with FileLock(self.ledger.dir/'claim', purpose='claim'):
                self.ledger.recover(self.me, adopt_running=False)
                job = self.ledger.claim(self.me)
            if job is None:
                return
            fault('after_claim')
            self.run_job(job, cores)

    def run(self):
        with FileLock(self.ledger.dir/'claim', purpose='recover'):
            adopted = {'jobs': self.ledger.recover(self.me), 'lock': threading.Lock()}
        for job in self.ledger.jobs():
            self.ledger.advance(job)
        threads = [threading.Thread(target=self.worker, args=(w, adopted), daemon=True)
                   for w in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for job in self.ledger.jobs():
            self.ledger.advance(job)
        return derive_queue(self.ledger.views())


def read_jobfile(path):
    raw = Path(path).read_bytes()
    jobs = [tuple(int(x) for x in line.split()[:2]) for line in raw.decode().splitlines() if line.strip()]
    return jobs, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- status
def status(run_root, repo_root=None, ledger_dir=None, write_ledger=False, deep=False, **legacy_kw):
    layout = Layout(run_root, repo_root)
    legacy = adopt_legacy(run_root, repo_root, deep=deep, **legacy_kw)
    ledger_dir = Path(ledger_dir) if ledger_dir else layout.logs/'queue_state'
    ledger = Ledger(ledger_dir, layout, deep=deep, read_only=not write_ledger)
    views = {job: dict(v, source='legacy') for job, v in legacy['jobs'].items()}
    if write_ledger:
        jobs = list(legacy['jobs'])
        digest = legacy['cursor']['sha256'] if legacy['cursor'] else hashlib.sha256(
            json.dumps(sorted(jobs)).encode()).hexdigest()
        ledger.dir.mkdir(parents=True, exist_ok=True)
        with FileLock(ledger.dir/'queue.json', purpose='adopt'):
            h = ledger.header()
            if not h:
                ds.atomic_write_json(ledger.dir/'queue.json', dict(
                    schema=SCHEMA, jobfile=legacy['cursor']['jobfile'] if legacy['cursor'] else None,
                    jobfile_sha256=digest, jobs=[list(j) for j in jobs], adopted_from=str(run_root),
                    history=[dict(jobfile_sha256=digest, at=time.time())]))
        for job, v in legacy['jobs'].items():
            sf = ledger.sf(job)
            if sf.exists():
                continue
            data = {'fold': job[0], 'seed': job[1], 'tries': v.get('tries', 0), 'requeues': 0,
                    'legacy': {k: v[k] for k in ('generation', 'pid', 'attempts') if v.get(k) is not None}}
            if v['state'] == 'FAILED':
                data['failure'] = dict(reason=v['reason'], retryable=v['retryable'])
            if v['state'] in ('RUNNING', 'CLAIMED'):
                data['owner'] = dict(legacy=True, pid=v.get('pid'), pid_starttime=None)
            if v['state'] == 'RUNNING':
                data['try'] = v.get('try_')
            evidence = dict(v['evidence'], legacy_state=v['state'], blocked=v['blocked'])
            if v['state'] == 'UNCLAIMED':
                sf.create(dict(data, adopted={'evidence': evidence}))
            else:
                sf.adopt(lambda _d, st=v['state'], ev=evidence: (st, ev), legacy_dir=str(layout.logs), data=data)
        for job in jobs:
            ledger.advance(job)
    if ledger.header():
        for job, v in ledger.views().items():
            leg = views.get(job, {})
            v['source'] = 'ledger'
            v['legacy_state'] = leg.get('state')
            if leg.get('state') and leg['state'] != v['state']:
                v['blocked'] = f"ledger says {v['state']}, legacy files say {leg['state']}"
            v['evidence'] = leg.get('evidence', {})
            views[job] = v
    return dict(run_root=str(run_root), ledger_dir=str(ledger_dir), wrote_ledger=bool(write_ledger),
                derived=derive_queue(views), jobs=views, markers=legacy['markers'],
                anomalies=legacy['anomalies'], cursor=legacy['cursor'] and {
                    k: legacy['cursor'][k] for k in ('jobfile', 'sha256', 'next')},
                queue_process_alive=legacy['queue_process_alive'])


def print_status(report, out=sys.stdout):
    d = report['derived']
    print(f"queue: {d['queue']}  training_drained={d['training_drained']}  counts={d['counts']}", file=out)
    if report['cursor']:
        c = report['cursor']
        print(f"legacy cursor: {c['jobfile']} next={c['next']} sha256={c['sha256'][:12]}", file=out)
    for m in report['markers']:
        print(f"marker (not terminal): {m['kind']} at {m['ts']}", file=out)
    print(f"{'job':<14}{'state':<15}{'try':<5}{'detail':<28}blocked", file=out)
    for (k, s), v in sorted(report['jobs'].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        detail = ''
        if v['state'] == 'FAILED':
            detail = f"{v['reason']} {'retryable' if v['retryable'] else 'terminal'} tries={v.get('tries')}"
        elif v['state'] == 'RUNNING':
            detail = f"pid={v.get('pid') or (v.get('child') or {}).get('pid')} gen={v.get('generation')}"
        print(f"fold{k} seed{s:<6}{v['state']:<15}{str(v.get('try_') or ''):<5}{detail:<28}{v.get('blocked') or ''}",
              file=out)
    for a in report['anomalies']:
        print(f'anomaly: {a}', file=out)


def _jsonable(report):
    out = dict(report)
    out['jobs'] = {job_name(j): v for j, v in report['jobs'].items()}
    out['derived'] = dict(report['derived'], training_open=[job_name(j) for j in report['derived']['training_open']])
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_child':
        return child_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    st = sub.add_parser('status', help='read-only status (adopts the legacy layout)')
    st.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    st.add_argument('--repo-root', type=Path)
    st.add_argument('--ledger-dir', type=Path)
    st.add_argument('--json', action='store_true')
    st.add_argument('--write-ledger', action='store_true', help='write adopted states into the ledger dir')
    st.add_argument('--deep', action='store_true', help='hash readout-A NPZs against their receipts')
    rn = sub.add_parser('run', help='the post-drain driver (Python replacement for train_queue.sh)')
    rn.add_argument('--jobfile', type=Path, required=True)
    rn.add_argument('--ledger-dir', type=Path, required=True)
    rn.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    rn.add_argument('--repo-root', type=Path)
    rn.add_argument('--workers', type=int, default=8)
    rn.add_argument('--worker-offset', type=int, default=0)
    rn.add_argument('--start-spacing', type=float, default=25.0)
    rn.add_argument('--poll', type=float, default=30.0)
    rn.add_argument('--replace-jobfile', action='store_true')
    rn.add_argument('--job-cmd', help='testing: JSON argv template with {fold} {seed} {cores}')
    ab = sub.add_parser('abandon', help='manually abandon one job, with a reason')
    ab.add_argument('--ledger-dir', type=Path, required=True)
    ab.add_argument('--fold', type=int, required=True); ab.add_argument('--seed', type=int, required=True)
    ab.add_argument('--reason', required=True)
    a = p.parse_args(argv)
    if a.command == 'status':
        report = status(a.run_root, a.repo_root, a.ledger_dir, a.write_ledger, a.deep)
        if a.json:
            json.dump(_jsonable(report), sys.stdout, indent=2, default=str)
            print()
        else:
            print_status(report)
        return 0
    if a.command == 'abandon':
        Ledger(a.ledger_dir).abandon((a.fold, a.seed), a.reason)
        return 0
    jobs, digest = read_jobfile(a.jobfile)
    layout = Layout(a.run_root, a.repo_root)
    ledger = Ledger(a.ledger_dir, layout)
    ledger.init(jobs, digest, a.jobfile, replace=a.replace_jobfile)
    command = None
    if a.job_cmd:
        template = json.loads(a.job_cmd)
        command = lambda job, cores: [x.format(fold=job[0], seed=job[1], cores=cores) for x in template]  # noqa: E731
    driver = Driver(ledger, a.workers, a.worker_offset, a.start_spacing, a.poll, command, cwd=layout.repo_root)
    result = driver.run()
    print(json.dumps(dict(result, training_open=[job_name(j) for j in result['training_open']])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
