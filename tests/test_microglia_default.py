"""The served microglia defaults must reach training and inference consistently."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('microglia_cli', Path(__file__).resolve().parents[1] / 'scripts/microglia.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_dry_run_routes_training_and_served_readout(capsys):
    assert M.main(['train', '--fold', '2', '--seed', '7', '--device', 'cuda', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    train, generation = plan['commands']
    cfg = M.load_config()
    assert train[train.index('--arm') + 1] == cfg['training']['arm']
    assert train[train.index('--updates') + 1] == '20000'
    assert generation[generation.index('--mode') + 1] == cfg['generation']['mode']
    assert 'fold2/seed7' in generation[generation.index('--fit') + 1]
    assert generation[generation.index('--out') + 1].endswith('.npz')
    assert float(train[train.index('--gpu-mem-fraction') + 1]) == .04


@pytest.mark.parametrize('changes', [
    ('training', 'generator_lr', .02), ('training', 'updates', 5000),
    ('data', 'external_donors', True), ('runtime', 'gpu_memory_fraction', .5),
    ('generation', 'generated_donors_per_target', 64),
    ('generation', 'mode', 'snapema@5000:0.9995'),
])
def test_config_rejects_unvalidated_changes(tmp_path, changes):
    cfg = copy.deepcopy(M.load_config())
    section, key, value = changes
    cfg[section][key] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        M.load_config(path)


def test_fit_must_match_default_and_development_split(tmp_path):
    cfg = M.load_config()
    saved = {'arm': cfg['training']['arm'], 'panel': 'hvg2000', 'updates': 20000,
             'fold': 0, 'seed': 0, 'protocol_split': True}
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(saved))
    M.check_fit(tmp_path, cfg, 0, 0)
    saved['arm'] = 'pbulk,pzero,t.kappa@0.19'
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        M.check_fit(tmp_path, cfg, 0, 0)


def test_generate_dry_run_does_not_require_machine_data(tmp_path, capsys):
    assert M.main(['generate', '--fit', str(tmp_path / 'absent'), '--out', str(tmp_path / 'output.npz'), '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['commands']) == 1


def test_original_fit_directory_is_read_only(tmp_path):
    fit = tmp_path / 'fit'
    with pytest.raises(SystemExit):
        M.main(['generate', '--fit', str(fit), '--out', str(fit / 'generated.npz'), '--dry-run'])


def test_resource_override_cannot_exceed_cap():
    with pytest.raises(SystemExit):
        M.main(['train', '--gpu-mem-fraction', '.5', '--dry-run'])


def test_count_shape_is_explicit_and_does_not_mutate_defaults(capsys):
    M.main(['generate', '--profile', 'count-shape', '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    command = plan['commands'][0]
    assert command[command.index('--mode') + 1] == 'cellshape@0.5:biascal:stagehalf'
    assert command[2].endswith('mode_cellshape.py')
    assert M.load_config()['generation']['mode'] == 'biascal@stagehalf'
