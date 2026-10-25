"""v4.5: the arm watcher is separate from the group it signals (TERM, wait, KILL, verify empty), and killed/stopped
runs are never reported as clean. CPU only; every process here is started by the test; test locks under tmp."""
import fcntl, hashlib, json, os, re, select, signal, subprocess, sys, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / 'study' / 'supervise_v4.py'
CHAIN = REPO / 'study' / 'cal_chain_v4.sh'
PY = sys.executable


def _env(tmp_path, **kw):
    return dict(os.environ, ALZ_SCOPE_ROOT=str(tmp_path), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(REPO / 'src'), **kw)


def _stat(pid):
    try: f = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
    except OSError: return None
    return {'state': f[0], 'ppid': int(f[1]), 'pgid': int(f[2]), 'sid': int(f[3]), 'starttime': int(f[19])}


def _alive(pid):
    s = _stat(pid); return s is not None and s['state'] not in ('Z', 'X')


def _members(pgid):
    return [int(d) for d in os.listdir('/proc') if d.isdigit() and (_stat(int(d)) or {}).get('pgid') == pgid and _alive(int(d))]


def _children(pid):
    return sorted(int(d) for d in os.listdir('/proc') if d.isdigit() and (_stat(int(d)) or {}).get('ppid') == pid)


def _lock_free(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); return True
        except BlockingIOError: return False
    finally: os.close(fd)


def _wait(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond(): return True
        time.sleep(0.1)
    return cond()


def _reap_group(pgid, leader_start):
    """Cleanup for pre-fix code only: SIGKILL the group THIS test created, verified by its recorded leader."""
    if not pgid or pgid in (os.getpgrp(), os.getsid(0)): return
    s = _stat(pgid)
    if _members(pgid) and (s is None or s['starttime'] == leader_start):
        try: os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError: pass


# ---- 1. separate watcher: a child that ignores SIGTERM is killed, the lock is released --------------
ARM = r'''
import fcntl, os, signal, subprocess, sys, time
lock = sys.argv[1]
fd = os.open(lock, os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
c = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                      'print("ready", flush=True); time.sleep(600)'], pass_fds=(fd,), stdout=subprocess.PIPE, text=True)
assert c.stdout.readline().strip() == 'ready'
from alz_perturb_gan import parent_watch
w = parent_watch.start(interval=0.1, grace=0.5)
print('child', c.pid, flush=True)
print('watcher', getattr(w, 'pid', 'thread'), flush=True)
time.sleep(600)
'''


def _fake_arm(tmp_path):
    lock = tmp_path / 'slot.test.lock'; lock.touch()
    sup = subprocess.Popen(['sleep', '120'])
    ident = f"{sup.pid}:{_stat(sup.pid)['starttime']}:{os.getpgid(sup.pid)}:{os.getsid(sup.pid)}"
    log = open(tmp_path / 'arm.log', 'w')
    arm = subprocess.Popen([PY, '-c', ARM, str(lock)], stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True,
                           env=_env(tmp_path, ALZ_SUPERVISOR_IDENT=ident))
    info = {'pgid': arm.pid, 'leader_start': _stat(arm.pid)['starttime']}
    for _ in range(2):
        k, v = arm.stdout.readline().split()
        info[k] = v
    return sup, arm, lock, info


def test_v45_term_ignoring_child_is_killed_and_lock_released(tmp_path):
    sup, arm, lock, info = _fake_arm(tmp_path)
    try:
        child = int(info['child'])
        assert _stat(child)['pgid'] == info['pgid'] and not _lock_free(lock)
        sup.kill(); sup.wait()                         # the supervisor dies without any teardown
        assert _wait(lambda: not _members(info['pgid']), 15), \
            f'arm group survived: {_members(info["pgid"])}; log:\n{(tmp_path / "arm.log").read_text()}'
        assert not _alive(child) and _lock_free(lock)
    finally:
        if sup.poll() is None: sup.kill(); sup.wait()
        _reap_group(info['pgid'], info['leader_start'])
        arm.wait(timeout=10)


def test_v45_watcher_is_outside_the_group_and_survives_its_own_term(tmp_path):
    sup, arm, lock, info = _fake_arm(tmp_path)
    try:
        assert info['watcher'] != 'thread', 'watcher must be a separate process'
        w = int(info['watcher']); ws = _stat(w)
        assert ws['pgid'] != info['pgid'] and ws['sid'] == w and ws['pgid'] == w, 'watcher must lead its own session'
        held = [l for l in os.listdir(f'/proc/{w}/fd') if os.path.realpath(f'/proc/{w}/fd/{l}') == str(lock)]
        assert not held, 'watcher must not hold the lock FD'
        sup.kill(); sup.wait()
        assert _wait(lambda: not _members(info['pgid']), 15)
        log = (tmp_path / 'arm.log').read_text()
        # it sent SIGTERM to the group, was still alive to send SIGKILL, then verified the group empty
        i_t, i_k, i_e = (log.find(s) for s in ('sent SIGTERM to arm group', 'sent SIGKILL to arm group', 'verified empty'))
        assert -1 < i_t < i_k < i_e, log
        assert _wait(lambda: not _alive(w), 10), 'watcher should exit by itself once the group is empty'
    finally:
        if sup.poll() is None: sup.kill(); sup.wait()
        _reap_group(info['pgid'], info['leader_start'])
        arm.wait(timeout=10)


def test_v45_watcher_refuses_group_whose_leader_identity_changed(tmp_path):
    from alz_perturb_gan import parent_watch
    victim = subprocess.Popen(['sleep', '30'], start_new_session=True)
    try:
        st = _stat(victim.pid)['starttime']
        sup = (999999, 1, 999999, 999999)
        rc = parent_watch.cleanup_group(victim.pid, (victim.pid, st - 1), (victim.pid, st - 1), sup, grace=0.5)
        assert rc == 98 and victim.poll() is None, 'an unverified group was signalled'
        rc = parent_watch.cleanup_group(victim.pid, (victim.pid, st), (victim.pid, st), sup, grace=0.5)
        assert rc == 0 and _wait(lambda: victim.poll() is not None, 5)
    finally:
        if victim.poll() is None: victim.kill()
        victim.wait()


# ---- 2. dry supervisor end to end ---------------------------------------------------------------------
def _read_json_line(p, want, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r, _, _ = select.select([p.stdout], [], [], 0.2)
        if r:
            line = p.stdout.readline()
            if not line: break
            if line.startswith('{'):
                d = json.loads(line)
                if want(d): return d
    return None


def _start_dry(tmp_path, *extra, arm_seconds=120):
    cmd = [PY, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'), '--seeds', '42',
           '--time-scale', '1', '--dry-arm-seconds', str(arm_seconds), '--arm-timeout', '300', *extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(tmp_path), start_new_session=True, bufsize=1)
    arm = {}
    def started():
        kids = _children(p.pid)
        if kids and _children(kids[0]):
            s = _stat(kids[0]); arm.update(pid=kids[0], pgid=s['pgid'], starttime=s['starttime'], inner=_children(kids[0])[0]); return True
        return False
    assert _wait(started, 30), 'arm did not start'
    time.sleep(1.0)
    return p, arm, [str(tmp_path / 'dry' / f'gpu1-slot{i}.test.lock') for i in (0, 1)]


def _probe(tmp_path, *args):
    r = subprocess.run([PY, str(SUP), '--post-run-probe', *args, '--probe-timeout', '20'], capture_output=True, text=True,
                       timeout=60, env=_env(tmp_path))
    lines = [l for l in r.stdout.splitlines() if l.startswith('{')]
    return r.returncode, (json.loads(Path(json.loads(lines[-1])['post_run_receipt']).read_text()) if lines else None), r


def _probe_receipt(tmp_path, rec):
    return _probe(tmp_path, '--supervisor-receipt', str(rec), '--sha256', hashlib.sha256(Path(rec).read_bytes()).hexdigest())


@pytest.mark.parametrize('layers', [[], ['--dry-no-pdeathsig']])
def test_v45_dry_parent_death_kills_term_ignoring_descendant(tmp_path, layers):
    p, arm, locks = _start_dry(tmp_path, '--dry-arm-term-ignoring-child', *layers)
    try:
        p.kill(); p.communicate(timeout=30)
        assert _wait(lambda: not _members(arm['pgid']), 20), f'arm group survived: {_members(arm["pgid"])}'
        assert all(_lock_free(l) for l in locks)
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_group(arm.get('pgid'), arm.get('starttime'))


def test_v45_supervisor_sigkill_no_receipt_probe_is_incomplete(tmp_path):
    p, arm, locks = _start_dry(tmp_path)
    try:
        start = _read_json_line(p, lambda d: d.get('event') == 'start', 10)
        assert start and Path(start['cohort']).is_dir(), 'supervisor must announce its cohort before running arms'
        p.kill(); p.communicate(timeout=30)            # SIGKILL (manual kill or kernel OOM kill): no receipt possible
        assert _wait(lambda: not _members(arm['pgid']), 20)
        cohort = Path(start['cohort'])
        assert not (cohort / 'supervisor_receipt.json').exists()
        prc, P, r = _probe(tmp_path, '--cohort', str(cohort))
        assert prc == 8 and P['verdict'] == 'incomplete' and P['run_status'] == 'incomplete', (prc, P, r.stdout, r.stderr)
        assert P['mode'] == 'no_supervisor_receipt' and P['residue'] == 'clean' and all(l['state'] == 'free' for l in P['locks'])
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_group(arm.get('pgid'), arm.get('starttime'))


def test_v45_no_receipt_probe_without_lease_is_unknown(tmp_path):
    c = tmp_path / 'dry' / 'dry-v4-x'; c.mkdir(parents=True)
    prc, P, r = _probe(tmp_path, '--cohort', str(c))
    assert prc == 8 and P['verdict'] == 'unknown' and P['run_status'] == 'unknown', (prc, P, r.stdout, r.stderr)


def test_v45_arm_oom_like_sigkill_is_labelled_and_incomplete(tmp_path):
    """The kernel OOM killer delivers SIGKILL; simulate it on CPU by SIGKILLing the arm's inner process."""
    p, arm, locks = _start_dry(tmp_path)
    try:
        os.kill(arm['inner'], signal.SIGKILL)
        out, err = p.communicate(timeout=60)
        line = json.loads([l for l in out.splitlines() if l.startswith('{') and '"receipt"' in l][-1])
        R = json.loads(Path(line['receipt']).read_text())
        a = R['arms'][0]
        assert a['status'] == 'killed_sigkill' and 'OOM' in a['kill_cause'] and 'oom_kill_counter' in a, a
        assert p.returncode == 3 and R['rc'] == 3 and R['completeness'] == 'incomplete' and R['stop_reason'] == 'killed_sigkill'
        assert line['completeness'] == 'incomplete'
        prc, P, _ = _probe_receipt(tmp_path, line['receipt'])
        assert prc == 8 and P['verdict'] == 'incomplete' and P['residue'] == 'clean' and P['run_status_reason'] == 'killed_sigkill'
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_group(arm.get('pgid'), arm.get('starttime'))


def test_v45_systemctl_stop_like_term_to_all_is_incomplete(tmp_path):
    """systemctl stop (KillMode=control-group) sends SIGTERM to every process of the unit at once."""
    p, arm, locks = _start_dry(tmp_path)
    try:
        for pg in (os.getpgid(p.pid), arm['pgid']): os.killpg(pg, signal.SIGTERM)
        out, err = p.communicate(timeout=60)
        line = json.loads([l for l in out.splitlines() if l.startswith('{') and '"receipt"' in l][-1])
        R = json.loads(Path(line['receipt']).read_text())
        assert p.returncode == 3 and R['completeness'] == 'incomplete' and R['stop_reason'] == 'sigterm'
        assert R['cleanup']['all_children_gone'] and all(_lock_free(l) for l in locks)
        prc, P, _ = _probe_receipt(tmp_path, line['receipt'])
        assert prc == 8 and P['verdict'] == 'incomplete' and P['residue'] == 'clean'
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_group(arm.get('pgid'), arm.get('starttime'))


def test_v45_in_process_supervisor_main_restores_signal_dispositions(tmp_path):
    """An in-process supervisor run must not leave SIGTERM ignored: bash cannot trap a signal ignored at entry, so a
    wrapper started afterwards would silently lose its stop label."""
    code = ('import importlib.util, signal, sys\n'
            'spec = importlib.util.spec_from_file_location("sv45", %r); m = importlib.util.module_from_spec(spec)\n'
            'spec.loader.exec_module(m)\n'
            'rc = m.main(sys.argv[1:])\n'
            'print("rc", rc, "TERM", signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, "HUP", signal.getsignal(signal.SIGHUP) is signal.SIG_DFL)\n') % str(SUP)
    r = subprocess.run([sys.executable, '-c', code, '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'),
                        '--seeds', '42', '--time-scale', '0.01', '--dry-arm-seconds', '0.3'],
                       capture_output=True, text=True, timeout=120, env=_env(tmp_path))
    assert 'TERM True HUP True' in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])


def test_v45_completed_run_is_still_clean(tmp_path):
    cmd = [PY, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'), '--seeds', '42',
           '--time-scale', '0.01', '--dry-arm-seconds', '0.3']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=_env(tmp_path))
    line = json.loads([l for l in r.stdout.splitlines() if l.startswith('{') and '"receipt"' in l][-1])
    assert r.returncode == 0 and line['completeness'] == 'complete'
    prc, P, _ = _probe_receipt(tmp_path, line['receipt'])
    assert prc == 0 and P['verdict'] == 'clean' and P['run_status'] == 'complete' and not P['watchers_alive']


# ---- 3. wrapper labels -------------------------------------------------------------------------------
def _chain(tmp_path, sup_body):
    root = tmp_path / 'scope'; study = root / 'study'; cache = root / 'cache'; cohort = root / 'cohort'
    for d in (study, cache, cohort): d.mkdir(parents=True)
    (study / 'study_bundle_v2.pt').write_bytes(b'x')
    text = CHAIN.read_text(); name = re.search(r'^SUPERVISOR_NAME=(\S+)$', text, re.M).group(1)
    marker = root / 'probe_argv.json'
    (study / name).write_text(
        'import json, os, signal, sys, time\n'
        f'COHORT, REC, READY = {str(cohort)!r}, {str(cohort / "supervisor_receipt.json")!r}, {str(root / "ready")!r}\n'
        'if "--post-run-probe" in sys.argv:\n'
        f'    open({str(marker)!r}, "w").write(json.dumps(sys.argv))\n'
        '    open(COHORT+"/post_run_probe_observation.json", "w").write("{}"); sys.exit(0)\n'
        'if "--finalize-probe" in sys.argv:\n'
        '    v = "clean" if os.path.exists(REC) and json.load(open(REC)).get("rc") == 0 and "--stop-requested" not in sys.argv else "incomplete"\n'
        '    print(json.dumps({"verdict": v})); sys.exit(0 if v == "clean" else 8)\n'
        'print(json.dumps({"event": "start", "cohort": COHORT}), flush=True)\n'
        'def finish(rc, stop):\n'
        '    open(REC, "w").write(json.dumps({"rc": rc, "stop_reason": stop}))\n'
        '    print(json.dumps({"receipt": REC, "rc": rc}), flush=True); sys.exit(rc)\n'
        + sup_body)
    for k, v in (('SCOPE', root), ('S', study), ('PY', PY), ('CACHE', cache)):
        text = re.sub(rf'^{k}=/ml2/altgan\S*$', f'{k}={v}', text, count=1, flags=re.M)
    sh = tmp_path / 'chain.sh'; sh.write_text(text.replace('@PINNED_COMMIT@', 'a' * 40))
    return sh, root, marker


def _run_chain(sh, root, on_ready=None):
    w = subprocess.Popen(['bash', str(sh), '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    if on_ready:
        assert _wait(lambda: (root / 'ready').exists(), 20); on_ready(w)
    out, err = w.communicate(timeout=60)
    return w.returncode, out, err


def test_v45_wrapper_labels_supervisor_sigkill_incomplete_and_probes_cohort(tmp_path):
    sh, root, marker = _chain(tmp_path, 'os.kill(os.getpid(), signal.SIGKILL)\n')
    rc, out, err = _run_chain(sh, root)
    assert rc != 0 and 'CHAIN_DONE' not in out and 'CHAIN_INCOMPLETE' in out, (rc, out, err)
    assert 'no_supervisor_receipt' in out and 'supervisor_sigkill' in out
    argv = json.loads(marker.read_text()); assert '--cohort' in argv and argv[argv.index('--cohort') + 1] == str(root / 'cohort')


@pytest.mark.parametrize('who', ['whole_unit', 'wrapper_only'])
def test_v45_wrapper_systemctl_stop_runs_probe_and_is_incomplete(tmp_path, who):
    body = ('signal.signal(signal.SIGTERM, lambda *a: finish(3, "sigterm"))\n'
            'open(READY, "w").write("1")\n'
            'time.sleep(3 if %r == "wrapper_only" else 30)\n'
            'finish(0, None)\n' % who)
    sh, root, marker = _chain(tmp_path, body)
    def stop(w):
        if who == 'whole_unit': os.killpg(w.pid, signal.SIGTERM)      # like systemctl stop: every process of the unit
        else: os.kill(w.pid, signal.SIGTERM)
    rc, out, err = _run_chain(sh, root, stop)
    assert rc != 0 and 'CHAIN_DONE' not in out and 'CHAIN_INCOMPLETE' in out, (rc, out, err)
    assert 'wrapper_sigterm' in out and marker.exists(), 'the probe must still run after SIGTERM'


def test_v45_supervisor_teardown_kills_term_ignoring_descendant(tmp_path):
    """Supervisor alive: its own identity-checked teardown must TERM, then KILL a descendant that ignores SIGTERM."""
    p, arm, locks = _start_dry(tmp_path, '--dry-arm-term-ignoring-child')
    try:
        p.send_signal(signal.SIGTERM)
        out, err = p.communicate(timeout=90)
        line = json.loads([l for l in out.splitlines() if l.startswith('{') and '"receipt"' in l][-1])
        R = json.loads(Path(line['receipt']).read_text())
        sent = [s['signal'] for s in R['signals_sent'] if s['pgid'] == arm['pgid']]
        assert sent == ['SIGTERM', 'SIGKILL'] and R['cleanup']['all_children_gone'] and not R['cleanup']['refused_signals'], R['cleanup']
        assert R['completeness'] == 'incomplete' and not _members(arm['pgid']) and all(_lock_free(l) for l in locks)
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_group(arm.get('pgid'), arm.get('starttime'))
