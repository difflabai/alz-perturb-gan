"""v4.6 (Ember's hold on 2e1bbf8a): (1) the watcher re-checks every recorded identity against its pre-SIGTERM
snapshot just before SIGKILL and withholds SIGKILL if anything changed; (2) a SIGTERM received by the chain wrapper
reaches the probe, whose OWN verdict and exit code are then incomplete. CPU only. The watcher tests use fake pids
above pid_max with /proc access and os.killpg replaced, so nothing real is ever signalled."""
import json, os, re, signal, subprocess, sys, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / 'study' / 'supervise_v4.py'
CHAIN = REPO / 'study' / 'cal_chain_v4.sh'
sys.path.insert(0, str(REPO / 'src'))
from alz_perturb_gan import parent_watch as pw  # noqa: E402

PG = 4400001                       # above the kernel's maximum pid_max (4194304): can never be a real process
SUP_ID = (4400900, 11, 4400900, 4400900)
LEADER, INNER, KID = (PG, 1000), (PG + 1, 1001), (PG + 2, 1002)


def _env(tmp_path, **kw):
    return dict(os.environ, ALZ_SCOPE_ROOT=str(tmp_path), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(REPO / 'src'), **kw)


def _wait(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond(): return True
        time.sleep(0.1)
    return cond()


def _fake_group(monkeypatch, before, after):
    """Fake /proc view of group PG: `before` until SIGTERM is sent, `after` from then on. Records signals."""
    state = {'termed': False, 'sent': []}
    def cur(): return dict(after if state['termed'] else before)
    def members(pgid):
        assert pgid == PG; return sorted(cur().items())
    def stat(pid):
        c = cur()
        if pid in c: return {'state': 'S', 'pgid': PG, 'sid': PG, 'starttime': c[pid]}
        if pid == os.getpid(): return {'state': 'R', 'pgid': 4400777, 'sid': 4400777, 'starttime': 5}
        return None
    def killpg(pgid, sig):
        assert pgid == PG, pgid                              # never a real group
        state['sent'].append(signal.Signals(sig).name)
        if sig == signal.SIGTERM: state['termed'] = True
    monkeypatch.setattr(pw, 'group_members', members)
    monkeypatch.setattr(pw, '_stat', stat)
    monkeypatch.setattr(pw.os, 'killpg', killpg)
    return state


@pytest.mark.parametrize('case', ['leader_replaced', 'member_replaced', 'group_emptied_and_reused'])
def test_v46_watcher_withholds_sigkill_when_identity_changed_between_term_and_kill(monkeypatch, capsys, case):
    before = [LEADER, INNER, KID]
    after = {'leader_replaced': [(PG, 9000), KID],                       # new process with the leader's pid
             'member_replaced': [LEADER, (KID[0], 9002)],                # new process with a member's pid
             'group_emptied_and_reused': [(PG, 9000), (PG + 7, 9007)]}[case]
    st = _fake_group(monkeypatch, before, after)
    rc = pw.cleanup_group(PG, LEADER, INNER, SUP_ID, grace=0.2, known=[KID])
    err = capsys.readouterr().err
    assert st['sent'] == ['SIGTERM'], (st['sent'], err)
    assert rc == 98 and 'refusing SIGKILL' in err and 'identity changed since SIGTERM' in err, (rc, err)


@pytest.mark.parametrize('case', ['survivor_unchanged', 'survivor_plus_new_descendant'])
def test_v46_watcher_still_kills_when_identities_unchanged(monkeypatch, capsys, case):
    """Control (passes before and after): an unchanged TERM-ignoring member, or one that started a new descendant
    after SIGTERM (same group by continuity), is still killed."""
    after = {'survivor_unchanged': [KID], 'survivor_plus_new_descendant': [KID, (PG + 3, 1003)]}[case]
    st = _fake_group(monkeypatch, [LEADER, INNER, KID], after)
    pw.cleanup_group(PG, LEADER, INNER, SUP_ID, grace=0.2, known=[KID])
    assert st['sent'] == ['SIGTERM', 'SIGKILL'], (st['sent'], capsys.readouterr().err)


def _dry_receipt(tmp_path):
    cmd = [sys.executable, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'), '--seeds', '42',
           '--time-scale', '0.01', '--dry-arm-seconds', '0.3']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=_env(tmp_path))
    line = json.loads([l for l in r.stdout.splitlines() if l.startswith('{') and '"receipt"' in l][-1])
    assert r.returncode == 0 and line['completeness'] == 'complete', r.stdout[-2000:]
    return Path(line['receipt'])


def test_v46_probe_with_stop_requested_is_incomplete_for_a_complete_receipt(tmp_path):
    rec = _dry_receipt(tmp_path)
    h = __import__('hashlib').sha256(rec.read_bytes()).hexdigest()
    pr = subprocess.run([sys.executable, str(SUP), '--post-run-probe', '--supervisor-receipt', str(rec), '--sha256', h,
                         '--probe-timeout', '20', '--stop-requested', 'wrapper_sigterm'],
                        capture_output=True, text=True, timeout=60, env=_env(tmp_path))
    assert pr.returncode == 8, (pr.returncode, pr.stdout, pr.stderr[-1500:])
    P = json.loads((rec.parent / 'post_run_receipt.json').read_text())
    assert P['verdict'] == 'incomplete' and P['run_status'] == 'incomplete' and P['stop_requested'] == 'wrapper_sigterm'
    assert P['residue'] == 'clean' and 'wrapper_sigterm' in P['run_status_reason']


def test_v46_wrapper_only_sigterm_probe_itself_reports_incomplete(tmp_path):
    """Only the wrapper gets SIGTERM; the (real, dry) supervisor finishes normally with rc 0 and a complete receipt.
    The (real) probe's own receipt, verdict and exit code must say incomplete, not clean."""
    root = tmp_path / 'scope'; study = root / 'study'; cache = root / 'cache'; dry = root / 'dry'; ready = root / 'ready'
    for d in (study, cache, dry): d.mkdir(parents=True)
    (study / 'study_bundle_v2.pt').write_bytes(b'x')
    text = CHAIN.read_text(); name = re.search(r'^SUPERVISOR_NAME=(\S+)$', text, re.M).group(1)
    (study / name).write_text(                       # shim: real supervisor in dry mode, real probe
        'import os, sys\n'
        f'REAL, READY, DRY = {str(SUP)!r}, {str(ready)!r}, {str(dry)!r}\n'
        'if "--post-run-probe" in sys.argv: os.execv(sys.executable, [sys.executable, REAL] + sys.argv[1:])\n'
        'open(READY, "w").write("1")\n'
        'os.execv(sys.executable, [sys.executable, REAL, "--dry-run", "--dry-skip-static-checks", "--dry-dir", DRY,\n'
        '                          "--seeds", "42", "--time-scale", "0.01", "--dry-arm-seconds", "1.0"])\n')
    for k, v in (('SCOPE', root), ('S', study), ('PY', sys.executable), ('CACHE', cache)):
        text = re.sub(rf'^{k}=/ml2/altgan\S*$', f'{k}={v}', text, count=1, flags=re.M)
    sh = tmp_path / 'chain.sh'; sh.write_text(text.replace('@PINNED_COMMIT@', 'a' * 40))
    w = subprocess.Popen(['bash', str(sh), '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True, env=_env(tmp_path))
    try:
        assert _wait(ready.exists, 30)
        os.kill(w.pid, signal.SIGTERM)                # the wrapper only; the supervisor is not signalled
        out, err = w.communicate(timeout=120)
    finally:
        if w.poll() is None: w.kill(); w.wait()
    assert 'supervisor rc=0' in out, (out, err[-2000:])
    R = json.loads(next(dry.glob('dry-v4-*/supervisor_receipt.json')).read_text())
    assert R['rc'] == 0 and R['completeness'] == 'complete'
    probe_line = json.loads([l for l in out.splitlines() if l.startswith('{') and '"post_run_receipt"' in l][-1])
    P = json.loads(Path(probe_line['post_run_receipt']).read_text())
    prc = int(re.search(r'^probe rc=(\d+)$', out, re.M).group(1))
    assert prc == 8 and probe_line['verdict'] == 'incomplete', (prc, probe_line, out)
    assert P['verdict'] == 'incomplete' and P['run_status'] == 'incomplete' and P['stop_requested'] == 'wrapper_sigterm', P
    assert 'CHAIN_INCOMPLETE' in out and 'CHAIN_DONE' not in out and w.returncode == 8
