"""Documented exit codes (0 ok, 2 usage, 3 already done, 4 stranded, 75 busy) and explicit-fit namespaces.

(f) scripts/microglia.py and scripts/microglia_runtime.py: --help documents the codes; a complete fit
    whose output is COMMITTED exits 3 before any subprocess; other existing outputs are usage errors (2);
    runtime 4/75 pass through; an explicit --fit without --out gets its own namespace
    generated/explicit/<sha12(abs fit path)>/fold{f}/seed{s}/, while selected and trained fits keep their
    paths; under --fit-source auto, selected and trained fits at the selected fold/seed require a choice.
CPU only; no data, GPU or torch. Paths that would land in the checkout are redirected through M.ROOT.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_microglia_output_state as H  # noqa: E402  (shared stub host)

R, M, ds = H.R, H.M, H.ds
CFG = M.load_config()
MODE = CFG['generation']['mode']


def plan(capsys, *argv):
    assert M.main([*argv, '--dry-run']) == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Redirect the CLI's checkout-relative fit, output and cache paths into tmp_path."""
    monkeypatch.setattr(M, 'ROOT', tmp_path)
    (tmp_path/'runs/improve_20261109/caches/hvg2000/fold0').mkdir(parents=True)
    (tmp_path/'runs/improve_20261109/caches/hvg2000/fold0/manifest.json').write_text('{"inner_fold": 0}')
    monkeypatch.setattr(M, 'check_fit', lambda *a, **kw: None)
    return tmp_path


def commit_pair(out, payload=b'npz bytes'):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    M.receipt_path(out).write_text(json.dumps({'generated_SHA256': hashlib.sha256(payload).hexdigest()}))


def no_subprocess(monkeypatch):
    monkeypatch.setattr(M.subprocess, 'run', lambda *a, **kw: pytest.fail(f'subprocess started: {a}'))


# --------------------------------------------------------------------------------------------------
# --help

@pytest.mark.parametrize('script', ['scripts/microglia.py', 'scripts/microglia_runtime.py'])
def test_help_documents_the_exit_codes(script):
    out = subprocess.run([sys.executable, '-B', str(ROOT/script), '--help'], capture_output=True, text=True,
                         timeout=60, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert 'exit codes:' in text
    for code, word in (('0', 'done'), ('2', 'usage'), ('3', 'already done'), ('4', 'STRANDED'), ('75', 'busy')):
        assert any(line.strip().startswith(code + ' ') and word in line for line in text.splitlines()), (code, text)


def test_exit_code_constants_agree():
    assert (M.ALREADY_DONE, M.EX_STRANDED, M.EX_BUSY, M.EX_USAGE) == (R.ALREADY_DONE, R.EX_STRANDED, R.EX_TEMPFAIL,
                                                                      R.EX_USAGE) == (3, 4, 75, 2)
    assert M.TRAIN_DAMAGED == R.EX_DAMAGED == R.EX_STRANDED


# --------------------------------------------------------------------------------------------------
# explicit-fit namespaces

def test_explicit_fit_without_out_gets_its_own_namespace(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = tmp_path/'fits/a', tmp_path/'fits/b'
    pa = plan(capsys, 'generate', '--fit', str(a), '--fold', '2', '--seed', '1')
    pb = plan(capsys, 'generate', '--fit', str(b), '--fold', '2', '--seed', '1')
    relative = plan(capsys, 'generate', '--fit', 'fits/a', '--fold', '2', '--seed', '1')
    root = ROOT/CFG['runtime']['generation_output_root']
    sha12 = hashlib.sha256(str(a.resolve()).encode()).hexdigest()[:12]
    assert pa['fit_source'] == 'explicit'
    assert pa['out'] == str(root/'explicit'/sha12/'fold2/seed1'/f'generated_{MODE}.npz')
    assert pb['out'] != pa['out'] and relative['out'] == pa['out']
    command = pa['commands'][-1]
    assert command[command.index('--out')+1] == pa['out']
    # Selected and trained fits keep the #19 default paths.
    trained = plan(capsys, 'generate', '--fold', '2', '--seed', '1')
    assert trained['fit_source'] == 'trained' and trained['out'] == str(root/'fold2/seed1'/f'generated_{MODE}.npz')
    selected = plan(capsys, 'generate')
    assert selected['out'] == str(root/'selected/fold0/seed0'/f'generated_{MODE}.npz')
    # --out still wins.
    assert plan(capsys, 'generate', '--fit', str(a), '--out', str(tmp_path/'x.npz'))['out'] == str(tmp_path/'x.npz')


def test_lone_legacy_output_of_an_explicit_fit_is_reported_not_moved(capsys, fake_root):
    legacy = M.default_out(CFG, 0, 0, 'trained')
    commit_pair(legacy)
    before = legacy.read_bytes()
    p = plan(capsys, 'generate', '--fit', str(fake_root/'elsewhere/fit'))
    assert str(legacy) in p['legacy_output'] and 'not moved' in p['legacy_output']
    assert p['out'] != str(legacy) and legacy.read_bytes() == before


def test_auto_requires_a_choice_when_selected_and_trained_both_exist(capsys, fake_root):
    trained, _ = M.choose_fit(CFG, 'train', 0, 0)
    trained.mkdir(parents=True)
    (trained/'final.pt').write_bytes(b'')
    with pytest.raises(SystemExit) as usage:
        M.main(['generate', '--dry-run'])
    assert usage.value.code == M.EX_USAGE and '--fit-source' in capsys.readouterr().err
    assert plan(capsys, 'generate', '--fit-source', 'selected')['fit_source'] == 'selected'
    assert plan(capsys, 'generate', '--fit-source', 'trained')['fit_source'] == 'trained'
    assert plan(capsys, 'train', '--fold', '0', '--seed', '0')['fit_source'] == 'trained'  # train never guesses


# --------------------------------------------------------------------------------------------------
# microglia.py exit codes

def test_complete_fit_with_committed_output_exits_3_without_a_subprocess(capsys, fake_root, monkeypatch):
    no_subprocess(monkeypatch)
    out = fake_root/'g.npz'
    commit_pair(out)
    fit = fake_root/'fit'
    fit.mkdir()
    (fit/'no_skip_audit.json').write_text('{}')
    monkeypatch.setattr(M, 'choose_fit', lambda *a, **kw: (fit, 'trained'))
    assert M.main(['generate', '--out', str(out)]) == M.ALREADY_DONE
    assert M.main(['train', '--fold', '0', '--seed', '0', '--out', str(out)]) == M.ALREADY_DONE
    assert 'already generated' in capsys.readouterr().out


@pytest.mark.parametrize('kind', ['npz', 'receipt', 'mismatch'])
def test_other_existing_outputs_are_usage_errors(capsys, fake_root, monkeypatch, kind):
    no_subprocess(monkeypatch)
    out = fake_root/'g.npz'
    if kind in ('npz', 'mismatch'):
        out.write_bytes(b'npz')
    if kind in ('receipt', 'mismatch'):
        M.receipt_path(out).write_text(json.dumps({'generated_SHA256': 'a'*64}))
    with pytest.raises(SystemExit) as usage:
        M.main(['generate', '--fold', '3', '--seed', '1', '--out', str(out)])
    assert usage.value.code == M.EX_USAGE
    state = {'npz': 'UNRECEIPTED_NPZ', 'receipt': 'ORPHAN_RECEIPT', 'mismatch': 'INCONSISTENT'}[kind]
    assert state in capsys.readouterr().err


def test_committed_output_of_an_incomplete_fit_is_a_usage_error(fake_root, monkeypatch):
    no_subprocess(monkeypatch)
    out = fake_root/'g.npz'
    commit_pair(out)
    with pytest.raises(SystemExit) as usage:  # the fit still needs training: never report it done
        M.main(['train', '--fold', '3', '--seed', '1', '--out', str(out)])
    assert usage.value.code == M.EX_USAGE


@pytest.mark.parametrize('train_rc,gen_rc,expected', [(75, None, 75), (4, None, 4), (3, 0, 0), (0, 75, 75),
                                                       (0, 3, 3), (3, 4, 4)])
def test_runtime_codes_pass_through(fake_root, monkeypatch, train_rc, gen_rc, expected):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[3])
        return subprocess.CompletedProcess(cmd, train_rc if cmd[3] == 'train' else gen_rc)
    monkeypatch.setattr(M.subprocess, 'run', run)
    assert M.main(['train', '--fold', '0', '--seed', '1', '--out', str(fake_root/'g.npz')]) == expected
    assert calls == (['train'] if gen_rc is None else ['train', 'generate'])


def test_unexpected_runtime_failure_raises(fake_root, monkeypatch):
    monkeypatch.setattr(M.subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(subprocess.CalledProcessError):
        M.main(['generate', '--fold', '0', '--seed', '1', '--out', str(fake_root/'g.npz')])


# --------------------------------------------------------------------------------------------------
# microglia_runtime.py exit codes (cli(), the __main__ path)

def test_runtime_generate_codes(tmp_path, monkeypatch, capsys):
    work = H.prepare(tmp_path/'w')
    H.install(monkeypatch.setattr, work, H.Stub())
    assert R.cli(H.gen_argv(work)) == 0
    assert R.cli(H.gen_argv(work)) == R.ALREADY_DONE
    lock = ds.FileLock(H.out_of(work), blocking=False).acquire()  # another process generating it
    try:
        assert R.cli(H.gen_argv(work, '--force')) == R.EX_TEMPFAIL
        assert 'is busy' in capsys.readouterr().err
    finally:
        lock.release()
    H.out_of(work).unlink()  # a receipt without an NPZ: an error (1) without --force, not "done"
    assert R.cli(H.gen_argv(work)) == 1 and 'ORPHAN_RECEIPT' in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:
        R.cli(['generate', '--cache', str(work/'cache')])
    assert usage.value.code == R.EX_USAGE


def test_runtime_train_busy_and_done_codes(tmp_path, monkeypatch):
    work = H.prepare(tmp_path/'w')
    H.install(monkeypatch.setattr, work, H.Stub())
    root = tmp_path/'fits'
    argv = ['train', '--cache', str(work/'cache'), '--out-root', str(root)]
    fit = root/'pbulk/hvg2000/fold0/seed0'
    fit.mkdir(parents=True)
    lock = ds.FileLock(fit/R.FIT_LOCK, blocking=False).acquire()
    try:
        assert R.cli(argv) == R.EX_TEMPFAIL
    finally:
        lock.release()
    assert R.cli(argv) == 0
    assert R.cli(argv) == R.ALREADY_DONE
