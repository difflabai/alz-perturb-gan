"""Ember HOLD on af70a8dc: one regression test per item (each fails on af70a8dc, passes after the fix)."""
import hashlib, importlib.util, json, os, re, signal, subprocess, sys, time
from pathlib import Path
import pytest, torch

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / 'study' / 'supervise_v4.py'
CHAIN = REPO / 'study' / 'cal_chain_v4.sh'


def _env(tmp_path):
    return dict(os.environ, ALZ_SCOPE_ROOT=str(tmp_path), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


def _module():
    spec = importlib.util.spec_from_file_location(f'sup_{time.monotonic_ns()}', SUP)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _dry(tmp_path, *extra, timeout=120):
    cmd = [sys.executable, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'), '--seeds', '42', *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_env(tmp_path))
    out = json.loads([l for l in p.stdout.splitlines() if l.startswith('{')][-1])
    return p.returncode, out, json.loads(Path(out['receipt']).read_text())


# ---- 1. probe: any GPU query failure means NOT clean ---------------------------------------
@pytest.mark.parametrize('failure', ['TimeoutExpired', 'CalledProcessError(9)'])
def test_hold1_probe_gpu_query_failure_is_not_clean(tmp_path, failure):
    rc, out, R = _dry(tmp_path, '--time-scale', '0.001', '--dry-arm-seconds', '0.1')
    assert rc == 0
    m = _module(); m.nvidia_smi = lambda args, timeout: (None, failure)
    prc = m.post_run_probe(out['receipt'], hashlib.sha256(Path(out['receipt']).read_bytes()).hexdigest(), 10)
    P = json.loads((Path(out['receipt']).parent / 'post_run_receipt.json').read_text())
    assert prc != 0 and P['verdict'] != 'clean'


# ---- 2. select binds checkpoint <-> training.json and refuses dummy/untrained -----------------
@pytest.mark.parametrize('fault', ['dummy_bytes', 'untrained', 'history_mismatch', 'steps_mismatch', 'wrong_arm_in_training'])
def test_hold2_select_refuses_ineligible_checkpoint(tmp_path, fault):
    from test_v4 import _synthetic_cohort, _sel, cal_study
    from alz_perturb_gan.gene_models import ModelConfig, build_models
    from alz_perturb_gan.gene_checkpoint import save_checkpoint
    runs, rec, proto = _synthetic_cohort(tmp_path / 'cohort'); out = tmp_path / 'analysis'   # select creates it (v4.4)
    d = Path(runs[43]) / 'GH'; ck = d / 'checkpoint.pt'; tj = d / 'training.json'
    if fault == 'dummy_bytes':
        ck.chmod(0o644); ck.write_bytes(b'not a checkpoint')
    if fault == 'untrained':
        ck.chmod(0o644); ck.unlink()
        save_checkpoint(ck, build_models(ModelConfig(4, 12, hidden=8, z_dim=2, particles=8, seed=43, steps=2, batch_size=4, arm='B0')),
                        cal_study().protocol)
    t = json.loads(tj.read_text()); tj.chmod(0o644)
    if fault == 'history_mismatch': t['history'] = t['history'][:1]; tj.write_text(json.dumps(t))
    if fault == 'steps_mismatch': t['steps'] = 999; tj.write_text(json.dumps(t))
    if fault == 'wrong_arm_in_training': t['arm'] = 'B0'; tj.write_text(json.dumps(t))
    with pytest.raises(PermissionError):
        _sel(runs, rec, out, proto)
    assert not list(out.glob('selection_frozen_*'))


# ---- 3. cleanup never claims a group without a verified surviving recorded identity ----------
def test_hold3_cleanup_does_not_claim_unverified_group(tmp_path):
    m = _module()
    s = m.Supervisor(m.parse(['--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'd')]))
    # leader exits, a later process remains in the group: the recorded identities (leader, inner) are not it
    p = subprocess.Popen(['bash', '-c', 'sleep 30 >/dev/null 2>&1 & echo $!'], start_new_session=True, stdout=subprocess.PIPE, text=True)
    child = int(p.stdout.readline()); p.wait()
    try:
        pgid = p.pid; info = m.proc_info(child); assert info['pgid'] == pgid
        s.groups[pgid] = {'leader': {'pid': pgid, 'starttime': info['starttime'] - 3, 'uid': os.getuid(), 'comm': 'bash'},
                          'inner': None, 'popen': p}
        assert s.kill_group(pgid, time.monotonic() + 3) is False
        assert m.proc_info(child)['starttime'] == info['starttime'], 'an unverified process was signalled'
        assert not s.signals_sent and s.refused and s.refused[-1]['reason'] == 'unverified_identity'
    finally:
        try: os.kill(child, signal.SIGKILL)
        except ProcessLookupError: pass


def test_hold3_cleanup_claims_group_when_recorded_inner_survives(tmp_path):
    m = _module()
    s = m.Supervisor(m.parse(['--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'd')]))
    p = subprocess.Popen(['bash', '-c', 'sleep 30 >/dev/null 2>&1 & echo $!'], start_new_session=True, stdout=subprocess.PIPE, text=True)
    child = int(p.stdout.readline()); p.wait()
    try:
        info = m.proc_info(child)
        s.groups[p.pid] = {'leader': {'pid': p.pid, 'starttime': info['starttime'] - 3, 'uid': os.getuid(), 'comm': 'bash'},
                           'inner': m.ident(info), 'popen': p}
        assert s.kill_group(p.pid, time.monotonic() + 5) is True and s.signals_sent
    finally:
        try: os.kill(child, signal.SIGKILL)
        except ProcessLookupError: pass


# ---- 4. wrapper never lets extra args override pins ----------------------------------------
def _fake_chain(tmp_path, args):
    root = tmp_path / 'scope'; study = root / 'study'; cache = root / 'cache'
    for d in (study, cache): d.mkdir(parents=True)
    (study / 'study_bundle_v2.pt').write_bytes(b'x')
    rec = root / 'rec'; rec.mkdir(); (rec / 'supervisor_receipt.json').write_text('{}')
    text = CHAIN.read_text(); name = re.search(r'^SUPERVISOR_NAME=(\S+)$', text, re.M).group(1)
    marker = root / 'supervisor_argv.json'
    (study / name).write_text(
        'import json,sys\n'
        f'if "--post-run-probe" in sys.argv: open({str(rec / "post_run_probe_observation.json")!r}, "w").write("{{}}"); sys.exit(0)\n'
        'if "--finalize-probe" in sys.argv: print(json.dumps({"verdict": "clean"})); sys.exit(0)\n'
        f'open("{marker}", "w").write(json.dumps(sys.argv))\n'
        f'print(json.dumps({{"receipt": "{rec / "supervisor_receipt.json"}"}})); sys.exit(0)\n')
    for k, v in (('SCOPE', root), ('S', study), ('PY', sys.executable), ('CACHE', cache)):
        text = re.sub(rf'^{k}=/ml2/altgan\S*$', f'{k}={v}', text, count=1, flags=re.M)
    text = text.replace('@PINNED_COMMIT@', 'a' * 40)
    sh = tmp_path / 'chain.sh'; sh.write_text(text)
    p = subprocess.run(['bash', str(sh), *args], capture_output=True, text=True, timeout=60, start_new_session=True)
    return p, (json.loads(marker.read_text()) if marker.exists() else None)

REAL = ['auth-ref', 'owner-ref', '2026-10-14T12:00:00-06:00']

@pytest.mark.parametrize('extra', [['--harness-commit', 'b' * 40], ['--bundle', '/ml2/altgan/x.pt'], ['--bundle-sha256', 'c' * 64],
                                   ['--protocol-sha256', 'd' * 64], ['--seeds', '42'], ['--harness-commit=' + 'b' * 40],
                                   ['--arm-timeout', '300', '--arm-timeout', '30'], ['--arm-timeout', 'abc'], ['--dry-run'],
                                   ['--lease-valid-until', '2099-01-01T00:00:00+00:00'], ['--unknown']])
def test_hold4_wrapper_rejects_pin_overrides_and_unlisted_args(tmp_path, extra):
    p, argv = _fake_chain(tmp_path, REAL + extra)
    assert p.returncode == 2 and argv is None, (p.returncode, argv, p.stderr)

def test_hold4_wrapper_allows_listed_timeouts_once(tmp_path):
    p, argv = _fake_chain(tmp_path, REAL + ['--arm-timeout', '300', '--teardown-timeout', '90'])
    assert p.returncode == 0 and argv.count('--harness-commit') == 1 and argv[argv.index('--harness-commit') + 1] == 'a' * 40
    assert argv[argv.index('--arm-timeout') + 1] == '300'
    p2, argv2 = _fake_chain(tmp_path / 'dry', ['--dry-run', '--arm-timeout', '5'])
    assert p2.returncode == 2 and argv2 is None


# ---- 5. receipt written after the final wait; backstop kill classified from the schedule ------
def test_hold5_backstop_schedule_for_600s_slot():
    m = _module()
    sched = m.backstop_schedule(600.0, 1.0)
    assert sched == {'term_after_s': 585, 'kill_after_s': 590}
    assert m.classify_exit(124, elapsed=585.2, schedule=sched) == 'timeout'
    assert m.classify_exit(137, elapsed=590.1, schedule=sched) == 'timeout'
    assert m.classify_exit(124, elapsed=100.0, schedule=sched) == 'error_rc124'
    assert m.classify_exit(0, elapsed=10.0, schedule=sched) == 'ok'

def test_hold5_backstop_kill_is_labelled_timeout_end_to_end(tmp_path):
    # arm-timeout 20 s: backstop TERM at 5 s, supervisor's own kill at 10 s -> the backstop fires first
    rc, out, R = _dry(tmp_path, '--time-scale', '1', '--arm-timeout', '20', '--dry-arm-seconds', '60')
    a = R['arms'][0]
    assert a['status'] == 'timeout' and a.get('killed_by') == 'timeout_wrapper' and a['returncode'] in (124, 137)

def test_hold5_receipt_is_written_after_final_wait(tmp_path):
    m = _module(); real_alive = m.alive_same; state = {}
    def alive(rec):   # simulate an owned survivor that disappears 2 s after teardown starts
        if rec is None or rec.get('pid') is None: return False
        state.setdefault('dead_at', time.time() + 2.0)
        return time.time() < state['dead_at'] or real_alive(rec)
    m.alive_same = alive
    a = m.parse(['--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'd'), '--seeds', '42',
                 '--time-scale', '0.001', '--dry-arm-seconds', '0.1', '--teardown-timeout', '20000'])
    m.Supervisor(a).main()
    rec = next((tmp_path / 'd').glob('dry-v4-*/supervisor_receipt.json'))
    R = json.loads(rec.read_text())
    assert rec.stat().st_mtime >= state['dead_at'] - 0.05, 'receipt was written before the final wait ended'
    assert 'final_wait' in R and R['final_wait']['survivors_after'] == {} and R['cleanup']['all_children_gone']
