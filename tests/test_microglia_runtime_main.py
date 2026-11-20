"""Runtime and CLI main() paths with a stubbed host, trainer and generator (no torch, data or GPU)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('microglia_runtime_main', ROOT/'scripts/microglia_runtime.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
M = R.application()
DEFAULT_MODE = M.load_config()['generation']['mode']
NATIVE_RECIPE = json.loads(M.RECIPE.read_text())['native']['recipe']


class Stub:
    def __init__(self, tmp):
        self.tmp = tmp
        self.activated = 0
        self.generated = []
        self.post_check_failure = False
        self.crash_in_savez = False

    def activate(self, device, memory):
        self.activated += 1
        stub = self

        def generate_fit(fit, cache, *, mode=None, out=None, device='cpu', **kw):
            # Mirrors analysis/improve_20261109/r5/generate.py: np.savez is staged through a
            # hidden `.partial.npz`, and with a mode the generator dumps its own receipt
            # next to `out`.
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name('.' + out.stem + '.partial.npz')
            with open(tmp, 'wb') as fh:
                np.savez_compressed(fh, x=np.arange(3))
            if stub.crash_in_savez:
                raise OSError('simulated crash while writing the partial NPZ')
            tmp.replace(out)
            stub.generated.append(out)
            stats = {'generated_SHA256': R.sha(out), 'fit': str(fit), 'mode': mode}
            if mode is not None:
                out.with_name(out.stem + '.receipt.json').write_text(json.dumps(stats))
            return None, stats

        def run(args):
            # Emits the derived trainer's FIT_EVENTS (see vendored_host.FIT_LIFECYCLE_PATCHES).
            fit = Path(args.out_root)/'pbulk/hvg2000/fold0'/f'seed{args.seed}'
            fit.mkdir(parents=True, exist_ok=True)
            hook = getattr(T, 'FIT_EVENTS', None) or (lambda *a: None)
            if not (fit/'config.json').exists():
                (fit/'config.json').write_text('{}')
                hook('configure', fit, {})
            if not (fit/'final.pt').exists():
                hook('begin', fit, {'recovery': (fit/'recovery.pt').exists()})
                (fit/'final.pt').write_text('{}')
                hook('final_written', fit, {'step': 20000})
            if not (fit/'fit_receipt.json').exists():
                (fit/'fit_receipt.json').write_text(json.dumps({'status': 'training_completed'}))
                hook('receipt_written', fit, {'step': 20000})
            return fit

        record = {'calls': 20000, 'observed_steps': 20000}
        q = {'completed_updates': 20000, 'policy': {'optimizers': [{}, {'regularizer': {'record': record}}]}}
        torch = SimpleNamespace(load=lambda *a, **kw: q)
        T = SimpleNamespace(torch=torch, np=np, recipe=lambda: SimpleNamespace(to_dict=lambda: NATIVE_RECIPE),
                            parser=lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(
                                out_root=argv[argv.index('--out-root')+1], seed=int(argv[argv.index('--seed')+1]))),
                            run=run)
        G = SimpleNamespace(generate_fit=generate_fit, GEN_STREAMS=('a', 'b', 'c', 'd', 'e'))
        return T, G, {'stub': True}, {'cpus': [3, 4, 5], 'source': 'slot_lock'}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    s = Stub(tmp_path)
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{"sources": {}}')
    monkeypatch.setattr(R, 'SOURCE_MANIFEST', manifest)
    monkeypatch.setattr(R, 'check_sources', lambda: {'sources': {}})
    held_checks = []

    def check_cache(cache, fold, held=False):
        held_checks.append(held)
        # Fail only as a post-check: after this run's generator wrote its pending file.
        if s.post_check_failure and s.generated and s.generated[-1].exists():
            raise ValueError('development cache payload changed: heldout/counts.npz')
        return {'anchor': fold}
    monkeypatch.setattr(R, 'check_cache', check_cache)
    monkeypatch.setattr(R, 'check_weights', lambda fit, receipt: {})
    monkeypatch.setattr(R, 'check_checkpoint', lambda q, saved: None)
    monkeypatch.setattr(R, 'activate', s.activate)
    monkeypatch.setattr(M, 'check_fit', lambda *a, **kw: None)
    monkeypatch.setattr(R, 'application', lambda: M)
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'manifest.json').write_text('{"inner_fold": 0}')
    fit = tmp_path/'fit'
    fit.mkdir()
    (fit/'config.json').write_text(json.dumps({'cache_manifest_SHA256': R.sha(cache/'manifest.json')}))
    (fit/'fit_receipt.json').write_text('{}')
    s.cache, s.fit, s.out = cache, fit, tmp_path/'generated'/'out.npz'
    s.digest = R.sha(manifest)
    return s


def generate(s, *extra):
    return R.main(['generate', '--cache', str(s.cache), '--fit', str(s.fit), '--out', str(s.out), *extra])


def receipt(s):
    return json.loads(R.receipt_path(s.out).read_text())


def leftovers(s):
    """Hidden staging files the runtime or generator could leave beside the output."""
    return sorted(p.name for p in s.out.parent.glob('.*')) if s.out.parent.exists() else []


def test_generation_receipt_names_the_resolved_default_profile(stub):
    assert generate(stub, '--mode', DEFAULT_MODE) == 0
    app = receipt(stub)['application_default']
    assert app['name'] == 'microglia-empf-countshape-v1' and app['profile'] == 'default'
    assert app['mode'] == DEFAULT_MODE and app['profile_inferred_from_mode'] is True
    assert receipt(stub)['cpu_affinity']['cpus'] == [3, 4, 5]
    assert receipt(stub)['generated_SHA256'] == R.sha(stub.out)
    assert not R.pending_path(stub.out).exists()
    assert leftovers(stub) == []
    assert sorted(p.name for p in stub.out.parent.iterdir()) == ['out.npz', 'out.receipt.json']


@pytest.mark.parametrize('argv', [['--profile', 'profile-rmse'], ['--mode', 'biascal@stagehalf'],
                                  ['--profile', 'profile-rmse', '--mode', 'biascal@stagehalf']])
def test_profile_rmse_receipt_is_not_labelled_as_the_default(stub, argv):
    generate(stub, *argv)
    cfg = M.load_config()
    app = receipt(stub)['application_default']
    assert app['name'] == cfg['name'] + '-profile-rmse' and app['profile'] == 'profile-rmse'
    assert app['status'] == cfg['reference_profiles']['profile-rmse']['status']
    assert app['mode'] == 'biascal@stagehalf'
    assert app['calibration'] == cfg['reference_profiles']['profile-rmse']['calibration']


@pytest.mark.parametrize('argv', [['--profile', 'profile-rmse', '--mode', DEFAULT_MODE],
                                  ['--mode', 'snapema@5000:0.9995'], ['--profile', 'unknown']])
def test_mode_outside_the_profile_is_rejected_before_activation(stub, argv):
    with pytest.raises(ValueError):
        generate(stub, *argv)
    assert stub.activated == 0


def test_existing_output_is_refused_before_activation_and_force_replaces_it(stub):
    stub.out.parent.mkdir(parents=True)
    stub.out.write_bytes(b'old')
    with pytest.raises(FileExistsError):
        generate(stub)
    assert stub.activated == 0 and stub.out.read_bytes() == b'old'
    stub.out.unlink()
    R.receipt_path(stub.out).write_text('{}')  # a receipt alone also blocks
    with pytest.raises(FileExistsError):
        generate(stub)
    assert generate(stub, '--force') == 0
    assert receipt(stub)['generated_SHA256'] == R.sha(stub.out)


def test_failed_post_check_leaves_no_output_or_receipt(stub):
    stub.post_check_failure = True
    with pytest.raises(ValueError, match='payload changed'):
        generate(stub)
    assert stub.generated and not stub.out.exists() and not R.receipt_path(stub.out).exists()
    assert not R.pending_path(stub.out).exists()
    assert leftovers(stub) == []
    stub.post_check_failure = False
    assert generate(stub) == 0  # nothing left behind blocks the rerun
    assert leftovers(stub) == []


def test_crash_inside_the_generator_leaves_no_staging_files(stub):
    stub.crash_in_savez = True
    with pytest.raises(OSError, match='simulated crash'):
        generate(stub)
    assert not stub.out.exists() and not R.receipt_path(stub.out).exists()
    assert leftovers(stub) == []


def test_failed_post_check_with_force_keeps_the_previous_output(stub):
    assert generate(stub) == 0
    before = stub.out.read_bytes(), R.receipt_path(stub.out).read_text()
    stub.post_check_failure = True
    with pytest.raises(ValueError):
        generate(stub, '--force')
    assert (stub.out.read_bytes(), R.receipt_path(stub.out).read_text()) == before
    assert leftovers(stub) == []


def test_force_keeps_the_previous_output_if_the_receipt_write_fails(stub, monkeypatch):
    assert generate(stub) == 0
    before = stub.out.read_bytes(), R.receipt_path(stub.out).read_text()
    real_write = R.write

    def failing_write(path, value):
        if Path(path) == R.receipt_path(stub.out):
            raise OSError('simulated failure writing the receipt')
        return real_write(path, value)
    monkeypatch.setattr(R, 'write', failing_write)
    with pytest.raises(OSError, match='simulated failure'):
        generate(stub, '--force')
    # No unlink happens before the atomic replacements, so the old pair survives intact.
    assert (stub.out.read_bytes(), R.receipt_path(stub.out).read_text()) == before
    assert leftovers(stub) == []


def train(s, root):
    return R.main(['train', '--cache', str(s.cache), '--out-root', str(root)])


def test_fresh_training_pins_the_starting_manifest(stub, tmp_path):
    root = tmp_path/'fits'
    assert train(stub, root) == 0
    fit = root/'pbulk/hvg2000/fold0/seed0'
    start = json.loads((fit/'train_start.json').read_text())
    audit = json.loads((fit/'no_skip_audit.json').read_text())
    assert start['application_source_manifest_sha256'] == stub.digest
    assert start['cpu_affinity']['cpus'] == [3, 4, 5]
    assert audit['application_source_manifest_sha256'] == stub.digest
    assert audit['training_start_source_manifest_sha256'] == stub.digest


def test_reaudit_under_a_different_manifest_is_refused_and_audit_kept(stub, tmp_path):
    fit = tmp_path/'fits/pbulk/hvg2000/fold0/seed0'
    fit.mkdir(parents=True)
    old = {'application_source_manifest_sha256': 'a'*64, 'penalty_calls': 20000}
    (fit/'no_skip_audit.json').write_text(json.dumps(old))
    (fit/'final.pt').write_text('{}')
    with pytest.raises(ValueError, match='refusing to relabel'):
        train(stub, tmp_path/'fits')
    assert stub.activated == 0
    assert json.loads((fit/'no_skip_audit.json').read_text()) == old


def test_resume_under_a_different_manifest_is_refused(stub, tmp_path):
    fit = tmp_path/'fits/pbulk/hvg2000/fold0/seed0'
    fit.mkdir(parents=True)
    (fit/'train_start.json').write_text(json.dumps({'application_source_manifest_sha256': 'b'*64}))
    (fit/'recovery.pt').write_text('')
    with pytest.raises(ValueError, match='refusing to resume'):
        train(stub, tmp_path/'fits')
    assert stub.activated == 0


def test_fit_that_predates_train_start_resumes_only_with_an_explicit_epoch_adoption(stub, tmp_path):
    fit = tmp_path/'fits/pbulk/hvg2000/fold0/seed0'
    fit.mkdir(parents=True)
    (fit/'recovery.pt').write_text('')
    with pytest.raises(ValueError, match='--adopt-epoch'):
        train(stub, tmp_path/'fits')
    assert stub.activated == 0
    assert R.main(['train', '--cache', str(stub.cache), '--out-root', str(tmp_path/'fits'), '--adopt-epoch']) == 0
    audit = json.loads((fit/'no_skip_audit.json').read_text())
    assert audit['training_start_source_manifest_sha256'] is None and 'training_start_note' in audit
    assert [a['digest'] for a in audit['epoch_adoptions']] == [stub.digest]
    assert not (fit/'train_start.json').exists()
    events = [e['event'] for e in R.fit_state_file(fit).journal()]
    assert events[:2] == ['__adopt__', 'adopt_epoch'] and events[-1] == 'audit_passed'


# --- public CLI (scripts/microglia.py) ---

def plan(*argv, capsys):
    assert M.main([*argv, '--dry-run']) == 0
    return json.loads(capsys.readouterr().out)


def test_selected_and_trained_fold0_seed0_do_not_share_fit_or_output(capsys):
    generated = plan('generate', capsys=capsys)
    trained = plan('train', '--fold', '0', '--seed', '0', capsys=capsys)
    assert generated['fit_source'] == 'selected' and trained['fit_source'] == 'trained'
    g, t = generated['commands'][-1], trained['commands'][-1]
    assert g[g.index('--fit')+1] != t[t.index('--fit')+1]
    assert g[g.index('--out')+1] != t[t.index('--out')+1]
    assert generated['fit'] == g[g.index('--fit')+1] and generated['out'] == g[g.index('--out')+1]
    explicit = plan('generate', '--fit-source', 'trained', capsys=capsys)
    assert explicit['fit'] == trained['fit'] and explicit['out'] == trained['out']
    assert 'preflight' in generated and 'sources' in generated['preflight']


def test_selected_fit_source_requires_the_selected_fold():
    with pytest.raises(SystemExit):
        M.main(['generate', '--fold', '1', '--fit-source', 'selected', '--dry-run'])


def test_profile_and_force_reach_the_runtime(capsys):
    command = plan('generate', '--profile', 'profile-rmse', '--force', capsys=capsys)['commands'][-1]
    assert command[command.index('--profile')+1] == 'profile-rmse' and command[-1] == '--force'
    command = plan('generate', capsys=capsys)['commands'][-1]
    assert '--force' not in command


def test_cli_refuses_existing_output_before_any_subprocess(tmp_path, monkeypatch):
    out = tmp_path/'generated.npz'
    out.write_bytes(b'old')
    monkeypatch.setattr(M.subprocess, 'run', lambda *a, **kw: pytest.fail('subprocess started'))
    with pytest.raises(SystemExit):
        M.main(['train', '--fold', '3', '--seed', '1', '--out', str(out)])
    assert out.read_bytes() == b'old'


def test_complete_audited_fit_is_not_retrained_or_reaudited(tmp_path, monkeypatch, capsys):
    fit = tmp_path/'fit'
    fit.mkdir()
    (fit/'no_skip_audit.json').write_text('{}')
    monkeypatch.setattr(M, 'choose_fit', lambda *a, **kw: (fit, 'trained'))
    monkeypatch.setattr(M, 'check_fit', lambda *a, **kw: None)
    p = plan('train', '--fold', '3', '--seed', '1', '--out', str(tmp_path/'g.npz'), capsys=capsys)
    assert len(p['commands']) == 1 and p['commands'][0][3] == 'generate'
    assert p['training'].startswith('skipped')
