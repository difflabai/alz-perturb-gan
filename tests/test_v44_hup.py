"""v4.4: hangup and parent-death safety (CPU only; test locks under tmp; every process here is started by the test).

Each test fails on 2c46d160 and passes after the fix (see the before/after logs)."""
import fcntl, hashlib, importlib, json, os, re, signal, subprocess, sys, termios, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / 'study' / 'supervise_v4.py'
CHAIN = REPO / 'study' / 'cal_chain_v4.sh'


def _env(tmp_path):
    return dict(os.environ, ALZ_SCOPE_ROOT=str(tmp_path), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


def _stat(pid):
    try: f = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
    except OSError: return None
    return {'state': f[0], 'ppid': int(f[1]), 'pgid': int(f[2]), 'starttime': int(f[19])}


def _members(pgid):
    out = []
    for d in os.listdir('/proc'):
        if d.isdigit():
            s = _stat(int(d))
            if s and s['pgid'] == pgid and s['state'] not in ('Z', 'X'): out.append(int(d))
    return out


def _children(pid):
    return sorted(int(d) for d in os.listdir('/proc') if d.isdigit() and (_stat(int(d)) or {}).get('ppid') == pid)


def _locks_free(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return False
        finally: os.close(fd)     # released by close
    return True


def _wait(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond(): return True
        time.sleep(0.1)
    return cond()


def _start_dry(tmp_path, *extra):
    cmd = [sys.executable, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'), '--seeds', '42',
           '--time-scale', '1', '--dry-arm-seconds', '120', '--arm-timeout', '300', *extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(tmp_path), start_new_session=True)
    arm = {}
    def started():   # the arm leader (timeout) is a child of the supervisor and already has its inner child
        kids = _children(p.pid)
        if kids and _children(kids[0]):
            s = _stat(kids[0]); arm.update(pid=kids[0], pgid=s['pgid'], starttime=s['starttime'], inner=_children(kids[0])[0]); return True
        return False
    assert _wait(started, 30), 'arm did not start'
    time.sleep(0.5)
    locks = [str(tmp_path / 'dry' / f'gpu1-slot{i}.test.lock') for i in (0, 1)]
    return p, arm, locks


def _reap_test_arm(arm):
    """Cleanup for the pre-fix code: kill only the arm group this test started, verified by its recorded leader."""
    if not arm or arm['pgid'] == os.getpgrp(): return
    s = _stat(arm['pid'])
    if _members(arm['pgid']) and (s is None or s['starttime'] == arm['starttime']):
        try: os.killpg(arm['pgid'], signal.SIGKILL)
        except ProcessLookupError: pass


# ---- 1. SIGHUP mid-arm: teardown by identity, receipt written, locks released, no orphans ------
def test_v44_sighup_mid_arm_writes_receipt_releases_locks_no_orphans(tmp_path):
    p, arm, locks = _start_dry(tmp_path)
    try:
        assert not _locks_free(locks)
        p.send_signal(signal.SIGHUP)
        stdout, stderr = p.communicate(timeout=60)
        lines = [l for l in stdout.splitlines() if l.startswith('{')]
        assert lines, f'no receipt line; rc={p.returncode} stderr={stderr[-500:]}'
        out = json.loads(lines[-1]); rec = Path(out['receipt'])
        assert rec.is_file() and hashlib.sha256(rec.read_bytes()).hexdigest() == out['sha256']
        R = json.loads(rec.read_text())
        assert p.returncode == 3 and R['rc'] == 3 and R['stop_reason'] == 'sighup'
        a = R['arms'][0]
        assert a['status'] == 'interrupted_sighup' and a['pgid'] == arm['pgid'] and a['inner']['pid'] == arm['inner']
        sent = [s for s in R['signals_sent'] if s['pgid'] == arm['pgid']]
        assert sent and sent[0]['signal'] == 'SIGTERM' and sent[0]['leader'] == a['leader'] and not R['cleanup']['refused_signals']
        assert R['cleanup']['all_children_gone'] and R['final_wait']['survivors_after'] == {}
        assert _locks_free(locks), 'locks still held after the supervisor exited'
        assert not _members(arm['pgid']) and _stat(arm['inner']) is None, 'orphaned arm processes remain'
        h = hashlib.sha256(rec.read_bytes()).hexdigest()
        pr = subprocess.run([sys.executable, str(SUP), '--post-run-probe', '--supervisor-receipt', str(rec), '--sha256', h,
                             '--probe-timeout', '20'], capture_output=True, text=True, timeout=60, env=_env(tmp_path))
        P = json.loads((rec.parent / 'post_run_receipt.json').read_text())
        assert pr.returncode == 8 and P['verdict'] == 'incomplete' and P['residue'] == 'clean'   # v4.5: interrupted -> incomplete
        assert all(l['state'] == 'free' for l in P['locks'])
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_test_arm(arm)


# ---- 2. supervisor dies (SIGKILL, no teardown possible): the arm exits and releases its locks ----
@pytest.mark.parametrize('layers', ['pdeathsig_and_poll', 'pdeathsig_only', 'poll_only'])
def test_v44_parent_death_arm_exits_and_releases_inherited_locks(tmp_path, layers):
    extra = {'pdeathsig_and_poll': [], 'pdeathsig_only': ['--dry-no-parent-poll'], 'poll_only': ['--dry-no-pdeathsig']}[layers]
    p, arm, locks = _start_dry(tmp_path, *extra)
    try:
        assert not _locks_free(locks)
        p.kill(); p.communicate(timeout=30)            # no receipt is possible after SIGKILL; the arm must still go
        assert _wait(lambda: not _members(arm['pgid']), 20), f'arm group {arm["pgid"]} outlived its supervisor'
        assert _stat(arm['inner']) is None and _locks_free(locks), 'inherited lock FDs were not released'
        log = next((tmp_path / 'dry').glob('dry-v4-*/seed42/B0.log')).read_text()
        if layers == 'poll_only': assert 'gone; cleaning up arm group' in log and 'verified empty' in log, log   # v4.5 watcher process
    finally:
        if p.poll() is None: p.kill(); p.wait()
        _reap_test_arm(arm)


def test_v44_parent_watch_refuses_to_signal_a_group_it_does_not_lead(tmp_path):
    """Layer 2 never signals a group outside its own arm session: here the arm shares the test's group, so the (separate)
    watcher refuses the group and terminates only the recorded inner process (v4.5)."""
    sup = subprocess.Popen(['sleep', '30'])
    try:
        ident = f"{sup.pid}:{_stat(sup.pid)['starttime']}:{os.getpgid(sup.pid)}:{os.getsid(sup.pid)}"
        code = 'import time\nfrom alz_perturb_gan import parent_watch\nparent_watch.start(interval=0.1)\nprint("watching", flush=True)\ntime.sleep(30)\n'
        arm = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=dict(_env(tmp_path), ALZ_SUPERVISOR_IDENT=ident, PYTHONPATH=str(REPO / 'src')))
        assert arm.stdout.readline().strip() == 'watching'
        sup.kill(); sup.wait()
        _, err = arm.communicate(timeout=20)
        assert arm.returncode == -signal.SIGTERM and 'refusing to signal group' in err, (arm.returncode, err)
    finally:
        if sup.poll() is None: sup.kill(); sup.wait()


def test_v44_real_mode_refuses_parent_watch_test_flags(tmp_path):
    for flag in ('--dry-no-pdeathsig', '--dry-no-parent-poll'):
        p = subprocess.run([sys.executable, str(SUP), flag, '--authorization-reference', 'x', '--ownership-reference', 'y',
                            '--lease-valid-until', '2099-01-01T00:00:00+00:00', '--harness-commit', 'a', '--bundle', '/x',
                            '--bundle-sha256', 'b', '--protocol-sha256', 'c'], capture_output=True, text=True, timeout=30, env=_env(tmp_path))
        assert p.returncode == 2 and 'dry-only options' in p.stdout, (flag, p.stdout, p.stderr)


# ---- 3. wrapper: no controlling terminal unless under a systemd unit; ignores SIGHUP --------------
def _fake_chain(tmp_path, sup_sleep=0.0):
    root = tmp_path / 'scope'; study = root / 'study'; cache = root / 'cache'
    for d in (study, cache): d.mkdir(parents=True)
    (study / 'study_bundle_v2.pt').write_bytes(b'x')
    rec = root / 'rec'; rec.mkdir(); (rec / 'supervisor_receipt.json').write_text('{}')
    text = CHAIN.read_text(); name = re.search(r'^SUPERVISOR_NAME=(\S+)$', text, re.M).group(1)
    (study / name).write_text(
        'import json,sys,time\n'
        f'if "--post-run-probe" in sys.argv: open({str(rec / "post_run_probe_observation.json")!r}, "w").write("{{}}"); sys.exit(0)\n'
        'if "--finalize-probe" in sys.argv: print(json.dumps({"verdict": "clean"})); sys.exit(0)\n'
        f'time.sleep({sup_sleep})\n'
        f'print(json.dumps({{"receipt": "{rec / "supervisor_receipt.json"}"}})); sys.exit(0)\n')
    for k, v in (('SCOPE', root), ('S', study), ('PY', sys.executable), ('CACHE', cache)):
        text = re.sub(rf'^{k}=/ml2/altgan\S*$', f'{k}={v}', text, count=1, flags=re.M)
    sh = tmp_path / 'chain.sh'; sh.write_text(text.replace('@PINNED_COMMIT@', 'a' * 40))
    return sh


def _with_ctty(argv, env=None):
    """Run argv as a CHILD of a session leader that owns a pty as its controlling terminal."""
    master, slave = os.openpty()
    def pre():
        os.setsid(); fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    try:
        return subprocess.run(['bash', '-c', '"$@"; exit $?', '_', *argv], stdin=slave, capture_output=True, text=True,
                              timeout=60, preexec_fn=pre, env=env)
    finally:
        os.close(slave); os.close(master)


def test_v44_wrapper_refuses_controlling_terminal(tmp_path):
    p = _with_ctty(['bash', str(_fake_chain(tmp_path)), '--dry-run'])
    assert p.returncode == 2 and 'controlling terminal' in p.stderr and 'CHAIN_DONE' not in p.stdout, (p.returncode, p.stdout, p.stderr)


def test_v44_wrapper_refuses_ctty_with_spoofed_invocation_id(tmp_path):
    if open('/proc/self/cgroup').read().strip().endswith('.service'): pytest.skip('test process itself runs in a .service cgroup')
    p = _with_ctty(['bash', str(_fake_chain(tmp_path)), '--dry-run'], env=dict(os.environ, INVOCATION_ID='0' * 32))
    assert p.returncode == 2 and 'controlling terminal' in p.stderr, (p.returncode, p.stderr)


def test_v44_wrapper_runs_under_setsid_and_reports_launch_mode(tmp_path):
    p = subprocess.run(['bash', str(_fake_chain(tmp_path)), '--dry-run'], capture_output=True, text=True, timeout=60, start_new_session=True)
    assert p.returncode == 0 and 'CHAIN_DONE' in p.stdout and 'launch=setsid' in p.stdout, (p.stdout, p.stderr)


def test_v44_wrapper_survives_sighup_and_still_probes(tmp_path):
    w = subprocess.Popen(['bash', str(_fake_chain(tmp_path, sup_sleep=3)), '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    time.sleep(1.0); os.kill(w.pid, signal.SIGHUP)
    out, err = w.communicate(timeout=60)
    assert w.returncode == 0 and 'CHAIN_DONE' in out and 'probe rc=0' in out, (w.returncode, out, err)


# ---- 4. select creates its analysis directory itself: scoped, no-follow, exclusive --------------
def test_v44_select_creates_its_analysis_dir(tmp_path):
    from test_v4 import _synthetic_cohort, _sel
    runs, rec, proto = _synthetic_cohort(tmp_path / 'cohort'); out = tmp_path / 'analysis'
    assert not out.exists()
    path, digest, doc = _sel(runs, rec, out, proto)
    assert out.is_dir() and not out.is_symlink() and Path(path).parent == out and (out / 'ledger.jsonl').is_file()
    assert out.stat().st_uid == os.geteuid()


@pytest.mark.parametrize('case', ['existing', 'symlink_to_dir', 'dangling_symlink', 'outside_scope', 'missing_parent', 'relative', 'empty'])
def test_v44_select_refuses_unsafe_analysis_dir_before_binding_work(tmp_path, monkeypatch, case):
    from alz_perturb_gan import gene_analysis
    def no_bindings(*a, **k): raise AssertionError('binding work ran before the out-dir was validated')
    monkeypatch.setattr(gene_analysis, 'bindings', no_bindings)
    real = tmp_path / 'real'; real.mkdir()
    out = {'existing': real, 'symlink_to_dir': tmp_path / 'link', 'dangling_symlink': tmp_path / 'dangling',
           'outside_scope': Path(str(tmp_path) + '-outside') / 'analysis', 'missing_parent': tmp_path / 'a' / 'b',
           'relative': 'analysis-rel', 'empty': ''}[case]
    if case == 'symlink_to_dir': (tmp_path / 'link').symlink_to(real)
    if case == 'dangling_symlink': (tmp_path / 'dangling').symlink_to(tmp_path / 'nowhere')
    with pytest.raises((FileExistsError, PermissionError)):
        gene_analysis.select({42: str(tmp_path)}, str(tmp_path / 'r.json'), str(out), expected_seeds=[42], bundle_sha256='b', protocol_sha256='p')
    assert not list(real.iterdir()) and not (tmp_path / 'nowhere').exists() and not (tmp_path / 'a').exists()
    assert not Path(str(tmp_path) + '-outside').exists() and not Path('analysis-rel').exists()
