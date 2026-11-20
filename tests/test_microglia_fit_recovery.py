"""Regression tests for the adversarial review of the explicit fit lifecycle (CPU only, no data).

Each test turns one review repro into a check, using the harness in tests/test_microglia_fit_state.py (the
REAL derived trainer.run() with stub model collaborators, driven by scripts/microglia_runtime.main):

1. a crash after no_skip_audit.json, before fit_state.json says AUDITED, dead-ended scripts/microglia.py
2. a failure recording final_written after final.pt landed made a valid final.pt unreceiptable
3. an AUDITED fit whose files were damaged reported "complete" (exit 3) forever
4. scripts/microglia.py treated the runtime's exit codes 3 and 75 as hard failures
5. crash debris (recovery.pt beside final.pt, pid-named temps) was never removed
6. "read-only" generate and train --dry-run wrote into fit directories
7. pre-pinning fits with final.pt were audited under the live manifest without --adopt-epoch
8. damaged state files had no documented or working reset; a state ahead of the files looped
"""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import test_microglia_fit_state as H  # noqa: E402

R, M, ds, Fit, CFG = H.R, H.M, H.ds, H.Fit, H.CFG
JOURNAL = R.FIT_STATE + '.journal.jsonl'


def train_argv(work, *extra):
    return ['train', '--cache', str(work/'cache'), '--out-root', str(work/'fits'), *extra]


def runtime(work, *extra):
    """microglia_runtime main() in-process, with the CLI's error handling: the exit code."""
    try:
        return R.main(train_argv(work, *extra))
    except (ValueError, OSError, RuntimeError, KeyError, ds.DurableStateError) as exc:
        print(f'microglia runtime: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


def listing(fit):
    return {p.name: p.read_bytes() for p in sorted(Path(fit).iterdir()) if p.is_file()}


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A work tree with the runtime stubbed in-process; generation writes a small stand-in NPZ."""
    work = H.prepare_work(tmp_path/'w')
    T, G = H.install(lambda obj, name, value: monkeypatch.setattr(obj, name, value), work)

    def generate_fit(fit, cache, mode=None, out=None, device=None, **kw):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b'generated from ' + str(fit).encode())
        return None, {'generated_SHA256': R.sha(out), 'fit': str(fit)}
    monkeypatch.setattr(G, 'generate_fit', generate_fit)
    return work


def generate(work, fit, out=None):
    out = out or work/'gen'/'g.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    return R.main(['generate', '--cache', str(work/'cache'), '--fit', str(fit), '--out', str(out)])


# --------------------------------------------------------------------------------------------------
# scripts/microglia.py driven in-process: its runtime subprocesses call microglia_runtime.main here

def microglia(work, monkeypatch, argv, calls=None):
    cfg = json.loads(json.dumps(M.load_config()))
    cfg['runtime']['fit_output_root'] = str(work/'fits')
    cfg['runtime']['generation_output_root'] = str(work/'gen')
    monkeypatch.setattr(M, 'load_config', lambda path=None: json.loads(json.dumps(cfg)))
    monkeypatch.setattr(M, 'cache_path', lambda cfg, fold: work/'cache')

    def run(cmd, env=None, check=False, **kw):
        if calls is not None:
            calls.append(cmd[3])
        try:
            rc = R.main(cmd[3:])
        except (ValueError, OSError, RuntimeError, KeyError, ds.DurableStateError) as exc:
            print(f'microglia runtime: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            rc = 1
        if check and rc:
            raise subprocess.CalledProcessError(rc, cmd)
        return subprocess.CompletedProcess(cmd, rc)
    monkeypatch.setattr(M, 'subprocess', SimpleNamespace(run=run, CalledProcessError=subprocess.CalledProcessError))
    return M.main(argv)


# --------------------------------------------------------------------------------------------------
# 1. crash after the audit landed, before AUDITED was recorded

@pytest.mark.parametrize('how', ['crash_after_audit', 'audit_failed_after_a_pass'])
def test_audit_landed_state_lagging_reconciles_through_microglia_main(how, work, monkeypatch, capsys):
    assert H.run_child(work, crash='replace:no_skip_audit.json')[0] == 137
    fit = H.fit_path(work)
    sf = R.fit_state_file(fit)
    if how == 'audit_failed_after_a_pass':  # a transient audit failure recorded after an earlier pass
        sf.transition('audit_failed', update=R._noted('audit_failed', reason='OSError: transient'))
    assert sf.read().state is Fit.RECEIPTED
    M.check_fit(fit, CFG, 0, 0)  # the files are a complete audit: microglia.py used to skip training here
    with pytest.raises(ValueError, match='only an AUDITED fit generates'):
        generate(work, fit)  # ... and then the runtime refused to generate: the dead end
    calls = []
    assert microglia(work, monkeypatch, ['train', '--fold', '0', '--seed', '0'], calls) == 0
    assert calls == ['train', 'generate']  # routed through the runtime, which reconciled and exited 3
    snap = sf.read()
    assert snap.state is Fit.AUDITED and snap.data['last']['reconciled'] is True
    assert [e['event'] for e in sf.journal()][-1] == 'audit_passed'
    assert list((work/'gen'/'fold0'/'seed0').glob('generated_*.npz'))
    assert 'nothing to train (exit 3)' in capsys.readouterr().out


def test_reconcile_does_not_promote_an_audit_that_no_longer_verifies(work):
    assert H.run_child(work, crash='replace:no_skip_audit.json')[0] == 137
    fit = H.fit_path(work)
    H.FakeTorch.save({'step': 19000, 'tampered': True}, fit/'snap_19000.pt')  # check_weights now fails
    assert runtime(work) == 1
    snap = R.fit_state_file(fit).read()
    assert snap.state is Fit.RECEIPTED and 'audit_failed' in [e['event'] for e in R.fit_state_file(fit).journal()]


def test_reconcile_does_not_promote_an_audit_from_another_epoch(work, capsys):
    assert H.run_child(work, crash='replace:no_skip_audit.json')[0] == 137
    (work/'manifest.json').write_text('{"sources": {"epoch": 2}}')
    assert runtime(work) == 1
    assert R.fit_state_file(H.fit_path(work)).read().state is Fit.STRANDED
    assert 'refusing to relabel' in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------
# 2. a failure recording final_written must not strand a valid final.pt

@pytest.mark.parametrize('event', ['final_written', 'receipt_written', 'checkpoint'])
def test_enospc_recording_an_event_is_reconciled_in_the_same_run(event, work):
    rc, out, err = H.run_child(work, fail_event=event)
    assert rc == 0, err
    assert 'was not recorded' in out
    fit = H.fit_path(work)
    assert R.fit_state_file(fit).read().state is Fit.AUDITED
    assert not (fit/'recovery.pt').exists()
    assert json.loads((fit/'checkpoint_receipt.json').read_text())['path'] == 'final.pt'
    assert H.run_child(work)[0] == R.ALREADY_DONE


def test_interrupt_after_final_landed_keeps_final_receipt_and_converges(work):
    rc, out, err = H.run_child(work, fail_event='final_written:interrupt')
    assert rc not in (0, 137), err
    fit = H.fit_path(work)
    assert R.fit_state_file(fit).read().state is Fit.FAILED_RESUMABLE
    record = json.loads((fit/'checkpoint_receipt.json').read_text())
    # Before the fix the except branch wrote recovery.pt and pointed checkpoint_receipt.json at it.
    assert record['path'] == 'final.pt' and record['SHA256'] == R.sha(fit/'final.pt')
    assert pickle.loads((fit/'recovery.pt').read_bytes())['completed_updates'] < H.UPDATES  # the older one
    rc, out, err = H.run_child(work)
    assert rc == 0, err
    events = [e['event'] for e in R.fit_state_file(fit).journal()]
    assert events[-4:] == ['error', 'final_written', 'reconstruct_receipt', 'audit_passed']
    assert R.fit_state_file(fit).read().state is Fit.AUDITED
    assert not (fit/'recovery.pt').exists() and not (fit/'failure.json').exists()
    assert (fit/'failure.resolved.json').exists()


def test_superseded_checkpoint_receipt_reconstructs_from_the_final_header(work):
    """The layout the old except branch left: final.pt, a newer recovery.pt receipt, FAILED_RESUMABLE."""
    assert H.run_child(work, crash='replace:final.pt')[0] == 137
    fit = H.fit_path(work)
    cfg = json.loads((fit/'config.json').read_text())
    H.FakeTorch.save(H.checkpoint(cfg, 'incomplete', H.UPDATES), fit/'recovery.pt')
    (fit/'checkpoint_receipt.json').write_text(json.dumps({'path': 'recovery.pt', 'SHA256': R.sha(fit/'recovery.pt'),
                                                           'status': 'incomplete', 'completed_updates': H.UPDATES}))
    (fit/'failure.json').write_text('{"error": "OSError(28)"}')
    R.fit_state_file(fit).transition('error', update=R._noted('error', error='OSError(28)'))
    rc, out, err = H.run_child(work)  # the old code failed here on every rerun: 'does not describe final.pt'
    assert rc == 0, err
    receipt = json.loads((fit/'fit_receipt.json').read_text())
    assert receipt['reconstructed'] is True and receipt['reconstructed_from']['basis'] == 'final_header'
    assert R.fit_state_file(fit).read().state is Fit.AUDITED
    assert not (fit/'recovery.pt').exists() and (fit/'failure.resolved.json').exists()


def test_final_header_fallback_still_refuses_an_incomplete_final(work):
    assert H.run_child(work, crash='replace:final.pt')[0] == 137
    fit = H.fit_path(work)
    cfg = json.loads((fit/'config.json').read_text())
    (fit/'checkpoint_receipt.json').write_text(json.dumps({'path': 'recovery.pt', 'SHA256': '0'*64,
                                                           'status': 'incomplete', 'completed_updates': 18000}))
    q = H.checkpoint(cfg, 'completed', H.UPDATES)
    q['completed_updates'] = H.UPDATES - 1
    H.FakeTorch.save(q, fit/'final.pt')
    rc, out, err = H.run_child(work)
    assert rc == 1 and 'does not match the request' in err
    assert not (fit/'fit_receipt.json').exists()


# --------------------------------------------------------------------------------------------------
# 3. damage after AUDITED (or a state ahead of its files) is DAMAGED, left only by --repair

@pytest.mark.parametrize('damage', ['no_skip_audit.json', 'fit_receipt.json', 'final.pt'])
def test_damaged_audited_fit_is_reported_and_repaired_explicitly(damage, work, monkeypatch, capsys):
    assert H.run_child(work)[0] == 0
    fit = H.fit_path(work)
    (fit/damage).unlink()
    for _ in range(2):  # recorded once, then reported again without another transition
        assert runtime(work) == R.EX_DAMAGED
    snap = R.fit_state_file(fit).read()
    assert snap.state is Fit.DAMAGED and snap.data['damaged_from'] == 'AUDITED'
    assert [e['event'] for e in R.fit_state_file(fit).journal()].count('damaged') == 1
    assert '--repair' in capsys.readouterr().err
    with pytest.raises(ValueError, match='records DAMAGED'):
        generate(work, fit)
    with pytest.raises(ValueError, match='DAMAGED'):  # scripts/microglia.py reports it, no CalledProcessError
        microglia(work, monkeypatch, ['train', '--fold', '0', '--seed', '0'])
    assert runtime(work, '--repair') == 0
    snap = R.fit_state_file(fit).read()
    expected = {'no_skip_audit.json': 'RECEIPTED', 'fit_receipt.json': 'TRAINED', 'final.pt': 'CONFIGURED'}[damage]
    assert snap.state is Fit.AUDITED and snap.data['repairs'][-1]['to'] == expected
    M.check_fit(fit, CFG, 0, 0)
    assert runtime(work) == R.ALREADY_DONE


def test_damaged_snapshot_is_not_reaudited_into_audited(work):
    assert H.run_child(work)[0] == 0
    fit = H.fit_path(work)
    H.FakeTorch.save({'step': 17000, 'tampered': True}, fit/'snap_17000.pt')
    assert runtime(work) == R.EX_DAMAGED
    assert runtime(work, '--repair') == 1  # re-audit: check_weights fails, journaled as audit_failed
    assert R.fit_state_file(fit).read().state is Fit.RECEIPTED


def test_state_behind_a_final_it_cannot_explain_is_damaged_then_receipted(work):
    """t_corrupt 'before:fit_receipt.json forged CONFIGURED' failed on every rerun (no edge from CONFIGURED)."""
    assert H.run_child(work, crash='before:fit_receipt.json')[0] == 137
    fit = H.fit_path(work)
    doc = json.loads((fit/R.FIT_STATE).read_text())
    doc['state'] = 'CONFIGURED'
    (fit/R.FIT_STATE).write_text(json.dumps(doc))
    assert runtime(work) == R.EX_DAMAGED
    assert runtime(work, '--repair') == 0
    snap = R.fit_state_file(fit).read()
    assert snap.state is Fit.AUDITED and snap.data['repairs'][-1]['to'] == 'TRAINED'


@pytest.mark.parametrize('forged', ['TRAINED', 'RECEIPTED'])
def test_state_ahead_of_its_files_is_damaged_then_resumed(forged, work):
    """t_corrupt 'forged TRAINED/RECEIPTED' on a fit with only recovery.pt looped (IllegalTransition/FileNotFoundError)."""
    assert H.run_child(work, crash='replace:recovery.pt:3')[0] == 137
    fit = H.fit_path(work)
    doc = json.loads((fit/R.FIT_STATE).read_text())
    doc['state'] = forged
    (fit/R.FIT_STATE).write_text(json.dumps(doc))
    assert runtime(work) == R.EX_DAMAGED
    assert R.fit_state_file(fit).read().state is Fit.DAMAGED
    assert runtime(work, '--repair') == 0
    assert R.fit_state_file(fit).read().data['repairs'][-1]['to'] == 'FAILED_RESUMABLE'
    assert json.loads((fit/'fit_receipt.json').read_text())['resumed_from'] == 6000


# --------------------------------------------------------------------------------------------------
# 4. scripts/microglia.py and the runtime's exit codes

def test_microglia_main_reports_busy_75_instead_of_a_traceback(work, monkeypatch, capsys):
    fit = H.fit_path(work)
    fit.mkdir(parents=True)
    script = (f'import sys, time; sys.path.insert(0, {str(H.ROOT/"src")!r})\n'
              'from alz_perturb_gan.durable_state import FileLock\n'
              f'FileLock({str(fit/"fit")!r}).acquire(); print("held", flush=True); time.sleep(30)\n')
    holder = subprocess.Popen([sys.executable, '-B', '-c', script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'held'
        assert microglia(work, monkeypatch, ['train', '--fold', '0', '--seed', '0']) == M.TRAIN_BUSY == R.EX_TEMPFAIL
        assert 'is busy' in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait()


def test_microglia_main_accepts_already_done(work, monkeypatch):
    assert H.run_child(work)[0] == 0
    monkeypatch.setattr(M, 'recorded_fit_state', lambda fit: 'RECEIPTED')  # force the train call
    calls = []
    assert microglia(work, monkeypatch, ['train', '--fold', '0', '--seed', '0'], calls) == 0
    assert calls == ['train', 'generate']
    assert M.TRAIN_ALREADY_DONE == R.ALREADY_DONE and M.TRAIN_DAMAGED == R.EX_DAMAGED


def test_microglia_dry_run_reports_the_recorded_fit_state(work, monkeypatch, capsys):
    assert H.run_child(work)[0] == 0
    assert microglia(work, monkeypatch, ['train', '--fold', '0', '--seed', '0', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['preflight']['fit_state'] == 'AUDITED' and plan['training'].startswith('skipped')


# --------------------------------------------------------------------------------------------------
# 5. crash debris

def _dead_pid():
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def test_dead_writers_temps_are_swept_under_the_fit_lock(work):
    assert H.run_child(work, crash='replace:recovery.pt:2')[0] == 137
    fit = H.fit_path(work)
    pid = _dead_pid()
    debris = [f'.fit_state.json.abcdefgh.{pid}.tmp', f'.final.partial.{pid}.pt', f'.recovery.partial.{pid}.pt',
              f'.snap_5000.partial.{pid}.pt', f'.progress.json.{pid}.tmp']
    for name in debris:
        (fit/name).write_bytes(b'debris')
    assert H.run_child(work)[0] == 0
    assert not [n for n in debris if (fit/n).exists()]


def test_recovery_beside_a_receipted_final_is_removed(work):
    assert H.run_child(work, crash='state:RECEIPTED')[0] == 137
    fit = H.fit_path(work)
    H.FakeTorch.save({'stale': True}, fit/'recovery.pt')  # left by a kill between the rename and the unlink
    assert runtime(work) == 0
    assert not (fit/'recovery.pt').exists()


# --------------------------------------------------------------------------------------------------
# 6. read-only paths stay read-only

def test_generate_writes_nothing_into_the_fit(work):
    assert H.run_child(work)[0] == 0
    fit = H.fit_path(work)
    journal = fit/JOURNAL
    lines = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b''.join(lines[:-1]))  # the journal lags the state by one (a kill after the replace)
    for name in ('fit_state.json.lock', 'fit_state.json.lock.owner'):
        (fit/name).unlink(missing_ok=True)
    before = listing(fit)
    assert generate(work, fit) == 0
    assert listing(fit) == before


def test_dry_run_on_a_missing_fit_creates_nothing(work, capsys):
    assert R.main(train_argv(work, '--dry-run')) == 0
    assert not (work/'fits').exists()
    report = json.loads(capsys.readouterr().out)
    assert report['state'] is None and report['would_adopt_as'] == 'ABSENT'


# --------------------------------------------------------------------------------------------------
# 7. epoch policy for pre-pinning fits

@pytest.mark.parametrize('layout', ['final_only', 'receipted'])
def test_pre_pinning_final_needs_adopt_epoch_to_be_audited(layout, work):
    fit = H.fit_path(work)
    H.build_legacy(fit, work/'cache', layout)
    if layout == 'final_only':
        (fit/'checkpoint_receipt.json').write_text(json.dumps({'path': 'final.pt', 'SHA256': R.sha(fit/'final.pt'),
                                                               'status': 'completed', 'completed_updates': H.UPDATES}))
    with pytest.raises(ValueError, match='--adopt-epoch to audit'):
        R.main(train_argv(work))
    assert not (fit/'no_skip_audit.json').exists()
    assert R.main(train_argv(work, '--adopt-epoch')) == 0
    audit = json.loads((fit/'no_skip_audit.json').read_text())
    assert audit['epoch_adoptions'][0]['previous_start_epoch'] is None
    assert audit['training_start_note'].startswith('fit predates train_start.json')
    events = [e['event'] for e in R.fit_state_file(fit).journal()]
    assert 'adopt_epoch' in events and events[-1] == 'audit_passed'


# --------------------------------------------------------------------------------------------------
# 8. damaged state files: --readopt archives them and adopts again from the files

def _truncate(fit):
    (fit/R.FIT_STATE).write_bytes((fit/R.FIT_STATE).read_bytes()[:40])


def _delete_state(fit):
    (fit/R.FIT_STATE).unlink()


def _journal_ahead(fit):
    doc = json.loads((fit/R.FIT_STATE).read_text())
    doc['version'] -= 2
    (fit/R.FIT_STATE).write_text(json.dumps(doc))


@pytest.mark.parametrize('corrupt', [_truncate, _delete_state, _journal_ahead])
def test_damaged_state_file_needs_readopt(corrupt, work, capsys):
    assert H.run_child(work, crash='replace:recovery.pt:4')[0] == 137
    fit = H.fit_path(work)
    corrupt(fit)
    assert runtime(work) == 1
    assert '--readopt' in capsys.readouterr().err
    assert runtime(work, '--readopt') == 0
    snap = R.fit_state_file(fit).read()
    assert snap.state is Fit.AUDITED and snap.data['readopted']['archived']
    assert [p.name for p in fit.iterdir() if '.superseded.' in p.name]
    assert json.loads((fit/'fit_receipt.json').read_text())['resumed_from'] == 8000
