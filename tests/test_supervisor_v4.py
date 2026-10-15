"""CPU tests for study/supervise_v4.py and study/cal_chain_v4.sh (dry mode only; test locks under tmp)."""
import importlib.util, json, os, signal, subprocess, sys, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / 'study' / 'supervise_v4.py'
CHAIN = REPO / 'study' / 'cal_chain_v4.sh'


def _env(tmp_path):
    return dict(os.environ, ALZ_SCOPE_ROOT=str(tmp_path), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


def _run(tmp_path, *extra, timeout=120):
    cmd = [sys.executable, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_env(tmp_path))
    line = [l for l in p.stdout.splitlines() if l.startswith('{')][-1]
    out = json.loads(line)
    return p.returncode, out, json.loads(Path(out['receipt']).read_text())


def _probe(tmp_path, receipt):
    import hashlib
    h = hashlib.sha256(Path(receipt).read_bytes()).hexdigest()
    p = subprocess.run([sys.executable, str(SUP), '--post-run-probe', '--supervisor-receipt', receipt, '--sha256', h, '--probe-timeout', '20'],
                       capture_output=True, text=True, timeout=60, env=_env(tmp_path))
    return p.returncode, json.loads([l for l in p.stdout.splitlines() if l.startswith('{')][-1])


def _alive(rec):
    try: st = open(f"/proc/{rec['pid']}/stat").read()
    except OSError: return False
    f = st.rsplit(')', 1)[1].split()
    return int(f[19]) == rec['starttime'] and f[0] != 'Z'


def test_no_aggregate_cap_in_source():
    text = SUP.read_text()
    assert '3600' not in text and 'APPROVED_TOTAL_CAP' not in text and 'total_cap' not in text
    assert '3600' not in CHAIN.read_text() and 'total-cap' not in CHAIN.read_text()


def test_run_past_3600_seconds_scaled_completes_without_cap(tmp_path):
    rc, out, R = _run(tmp_path, '--time-scale', '0.001', '--dry-arm-seconds', '0.25')
    assert rc == 0 and out['cleanup_ok'] and all(s == 'ok' for _, _, s in out['arms']) and len(out['arms']) == 18
    assert R['elapsed_s'] > 3600 and R['aggregate_time_cap'] is None and R['operator_deadline_epoch'] is None
    assert R['total_timeout_s'] is None and not R['past_operator_deadline']
    grant = json.loads((Path(R['runs']['42']['path']) / 'grant.json').read_text())
    assert grant['aggregate_time_cap'] is None and 'max_total_wall_seconds' not in grant
    assert all(a['inner']['pid'] and a['leader']['pid'] == a['pgid'] for a in R['arms'])
    prc, P = _probe(tmp_path, out['receipt'])
    assert prc == 0 and P['verdict'] == 'clean' and all(s == 'free' for _, s, _ in P['locks'])


def test_cancellation_mid_arm_tears_down_by_identity(tmp_path):
    cmd = [sys.executable, str(SUP), '--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'dry'),
           '--time-scale', '1', '--dry-arm-seconds', '60', '--arm-timeout', '120']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(tmp_path))
    end = time.monotonic() + 30
    while time.monotonic() < end and not list((tmp_path / 'dry').glob('dry-v4-*/seed42/B0.log')): time.sleep(0.1)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    stdout, _ = p.communicate(timeout=60)
    out = json.loads([l for l in stdout.splitlines() if l.startswith('{')][-1]); R = json.loads(Path(out['receipt']).read_text())
    assert p.returncode == 3 and R['stop_reason'] == 'sigterm'
    g = R['arms'][0]
    assert g['status'] == 'interrupted_sigterm' and [a['arm'] for a in R['arms']] == ['B0']
    assert R['cleanup']['all_children_gone'] and not R['cleanup']['refused_signals']
    sent = [s for s in R['signals_sent'] if s['pgid'] == g['pgid']]
    assert sent and sent[0]['signal'] == 'SIGTERM' and sent[0]['leader'] == g['leader']
    assert not _alive(g['leader']) and not _alive(g['inner'])
    prc, P = _probe(tmp_path, out['receipt'])
    assert prc == 0 and P['verdict'] == 'clean'


def test_operator_deadline_bounds_arm_then_teardown_and_probe_run_after_it(tmp_path):
    # optional operator deadline (10 s real at scale 0.05): the running arm's slot is cut to it, the arm is killed,
    # the cohort stops, teardown completes, and the probe still runs after the deadline has passed.
    rc, out, R = _run(tmp_path, '--time-scale', '0.05', '--dry-arm-seconds', '60', '--total-timeout', '200')
    assert rc == 3 and R['stop_reason'] == 'timeout' and [a['status'] for a in R['arms']] == ['timeout']
    assert R['arms'][0]['slot_s'] <= 200 and R['arms'][0]['group_killed_clean']
    assert R['cleanup']['all_children_gone'] and R['cleanup']['teardown_within_budget'] and not _alive(R['arms'][0]['inner'])
    time.sleep(max(0, R['operator_deadline_epoch'] - time.time()) + 0.5)
    prc, P = _probe(tmp_path, out['receipt'])
    assert prc == 0 and P['verdict'] == 'clean'


def test_arm_error_propagates_and_stops(tmp_path):
    rc, out, R = _run(tmp_path, '--time-scale', '0.001', '--dry-arm-seconds', '0.1', '--dry-fail-arm', 'H')
    assert rc == 3 and R['stop_reason'] == 'error_rc7' and R['rc'] == 3
    assert [a['arm'] for a in R['arms']] == ['B0', 'G', 'H'] and R['arms'][-1]['status'] == 'error_rc7'


def test_real_mode_refuses_dry_options_and_missing_lease(tmp_path):
    for extra in (['--dry-skip-static-checks'], ['--dry-fail-arm', 'G'], []):
        p = subprocess.run([sys.executable, str(SUP), *extra, '--authorization-reference', 'x', '--ownership-reference', 'y']
                           + ([] if extra else ['--harness-commit', 'a', '--bundle', '/x', '--bundle-sha256', 'b', '--protocol-sha256', 'c']),
                           capture_output=True, text=True, timeout=30, env=_env(tmp_path))
        assert p.returncode == 2 and 'ABORT' in p.stdout


def _load_module():
    spec = importlib.util.spec_from_file_location('supervise_v4_under_test', SUP)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_cleanup_refuses_group_whose_identity_changed(tmp_path):
    m = _load_module()
    a = m.parse(['--dry-run', '--dry-skip-static-checks', '--dry-dir', str(tmp_path / 'd')])
    s = m.Supervisor(a)
    victim = subprocess.Popen(['sleep', '30'], start_new_session=True)
    try:
        real = m.ident(m.proc_info(victim.pid))
        s.groups[victim.pid] = {'leader': dict(real, starttime=real['starttime'] - 1), 'popen': victim}
        assert s.kill_group(victim.pid, time.monotonic() + 2) is False
        assert s.refused and s.refused[0]['reason'] == 'identity_mismatch' and victim.poll() is None and not s.signals_sent
        assert s.kill_group(999999, time.monotonic() + 1) is False and s.refused[-1]['reason'] == 'not_created_by_us'
        s.groups[victim.pid]['leader'] = real
        assert s.kill_group(victim.pid, time.monotonic() + 5) is True
    finally:
        if victim.poll() is None: victim.kill()
        victim.wait()


def test_grant_carries_lease_receipt_and_capacity_valid_until(tmp_path, monkeypatch):
    import datetime as dt
    from alz_perturb_gan.gene_protocol import validate_grant
    from alz_perturb_gan.gene_models import ModelConfig
    from test_v4 import cal_study
    m = _load_module(); study = cal_study()
    a = m.parse(['--dry-run', '--dry-skip-static-checks', '--protocol-sha256', study.protocol.fingerprint])
    s = m.Supervisor(a); s.lease_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    run = tmp_path / 'run'; run.mkdir()
    statics = {'bundle': 'b', 'harness_commit': 'c', 'script_sha256': 'd', **{k: 'x' for k in m.FEATURES}}
    p, _ = s.write_grant(run, tmp_path, 42, statics, tmp_path / 'lease_receipt.json', 'lsha', 'csha')
    g = json.loads(Path(p).read_text())
    assert g['lease_receipt'] == str(tmp_path / 'lease_receipt.json') and g['capacity_valid_until'] == s.lease_until.isoformat()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', m.GPU_UUID)
    validate_grant(g, study, ModelConfig(4, 12, seed=42, steps=10, arm='G'), 'cuda:0', 'train', 100)


def _fake_chain(tmp_path, sup_rc, probe_rc, pinned=True):
    root = tmp_path / 'scope'; study = root / 'study'; cache = root / 'cache'
    for d in (study, cache): d.mkdir(parents=True)
    (study / 'study_bundle_v2.pt').write_bytes(b'x')
    rec = root / 'rec'; rec.mkdir(); (rec / 'supervisor_receipt.json').write_text('{}')
    import re
    name = re.search(r'^SUPERVISOR_NAME=(\S+)$', CHAIN.read_text(), re.M).group(1)
    (study / name).write_text(
        'import json,sys\n'
        f'if "--post-run-probe" in sys.argv: print(json.dumps({{"verdict": "x"}})); sys.exit({probe_rc})\n'
        f'print(json.dumps({{"receipt": "{rec / "supervisor_receipt.json"}"}})); sys.exit({sup_rc})\n')
    text = CHAIN.read_text()
    for k, v in (('SCOPE', root), ('S', study), ('PY', sys.executable), ('CACHE', cache)):
        text = text.replace(f'\n{k}=/ml2/altgan' + {'SCOPE': '', 'S': '/studies/gene-feature-20261014', 'PY': '/.venv/bin/python', 'CACHE': '/.cache'}[k] + '\n', f'\n{k}={v}\n')
    if pinned: text = text.replace('@PINNED_COMMIT@', 'a' * 40)
    sh = tmp_path / 'chain.sh'; sh.write_text(text); sh.chmod(0o755)
    return subprocess.run(['bash', str(sh), '--dry-run'], capture_output=True, text=True, timeout=60, start_new_session=True)


@pytest.mark.parametrize('sup_rc,probe_rc,want', [(0, 0, 0), (3, 0, 3), (5, 0, 5), (0, 6, 6), (3, 6, 3)])
def test_wrapper_exits_nonzero_when_supervisor_or_probe_fails(tmp_path, sup_rc, probe_rc, want):
    p = _fake_chain(tmp_path, sup_rc, probe_rc)
    assert p.returncode == want, p.stdout + p.stderr
    assert ('CHAIN_DONE' in p.stdout) == (want == 0)


def test_wrapper_refuses_unpinned_commit_and_takes_no_commit_argument(tmp_path):
    p = _fake_chain(tmp_path, 0, 0, pinned=False)
    assert p.returncode == 2 and 'not pinned' in p.stderr
    text = CHAIN.read_text()
    assert '--harness-commit "$PINNED_COMMIT"' in text and 'COMMIT=$1' not in text
