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
    assert train[2].endswith('microglia_runtime.py') and train[3] == 'train'
    assert generation[2].endswith('microglia_runtime.py') and generation[3] == 'generate'
    assert generation[generation.index('--mode') + 1] == cfg['generation']['mode']
    assert 'fold2/seed7' in generation[generation.index('--fit') + 1]
    assert generation[generation.index('--out') + 1].endswith('.npz')
    assert generation[generation.index('--seed') + 1] == '7'
    assert float(train[train.index('--gpu-mem-fraction') + 1]) == .04


@pytest.mark.parametrize('changes', [
    ('training', 'generator_lr', .02), ('training', 'updates', 5000),
    ('data', 'external_donors', True), ('runtime', 'gpu_memory_fraction', .5),
    ('generation', 'generated_donors_per_target', 64),
    ('generation', 'mode', 'snapema@5000:0.9995'),
    ('training', 'lazy_k', 2), ('training', 'recipe', 'rhbi_optimized'),
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
    completed = M.ROOT / 'analysis/empf_comparison_20261117/completion/fits/native/pbulk/hvg2000/fold0/seed0'
    saved = json.loads((completed / 'config.json').read_text())
    for name in ('fit_receipt.json', 'no_skip_audit.json'):
        (tmp_path/name).write_text((completed/name).read_text())
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
    original = M.load_config()
    M.main(['generate', '--profile', 'count-shape', '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    command = plan['commands'][0]
    assert command[command.index('--mode') + 1] == 'cellshape@0.5:biascal:stagehalf'
    assert command[2].endswith('microglia_runtime.py') and command[3] == 'generate'
    assert M.load_config() == original


def test_profile_rmse_reference_remains_available_after_distribution_promotion(monkeypatch, capsys):
    cfg = M.load_config()
    cfg['generation']['mode'] = 'cellshape@0.5:biascal:stagehalf'
    monkeypatch.setattr(M, 'load_config', lambda: copy.deepcopy(cfg))
    M.main(['generate', '--dry-run'])
    default = json.loads(capsys.readouterr().out)['commands'][0]
    assert default[2].endswith('microglia_runtime.py')
    M.main(['generate', '--profile', 'profile-rmse', '--dry-run'])
    reference = json.loads(capsys.readouterr().out)['commands'][0]
    assert reference[2].endswith('microglia_runtime.py')
    assert reference[reference.index('--mode') + 1] == 'biascal@stagehalf'
    assert M.load_config()['generation']['mode'] == 'cellshape@0.5:biascal:stagehalf'


def test_reference_configuration_reports_its_actual_calibration_and_evidence(monkeypatch, capsys):
    selected = M.load_config()
    expected_evidence = selected['reference_profiles']['profile-rmse']['evidence']
    selected['evidence'] = 'different-distribution-selection.json'
    monkeypatch.setattr(M, 'load_config', lambda: copy.deepcopy(selected))
    M.main(['config', '--profile', 'profile-rmse'])
    cfg = json.loads(capsys.readouterr().out)
    assert cfg['generation']['calibration'] == 'training donors only; output bias'
    assert cfg['generation']['mode'] == 'biascal@stagehalf'
    assert cfg['evidence'] == expected_evidence


def test_default_generation_selects_the_completed_native_model(capsys):
    M.main(['generate', '--dry-run'])
    command = json.loads(capsys.readouterr().out)['commands'][0]
    assert command[command.index('--fit')+1] == str(M.ROOT/M.load_config()['runtime']['selected_fit'])
    assert M.load_config()['training']['particlegan_version'] == '0.10.0'
    assert M.load_config()['training']['particle_lr'] == .03
    assert M.load_config()['selection']['population_qualified'] is False


@pytest.mark.parametrize('damage', ['old_release', 'matched_variant', 'skipped_regularizer'])
def test_generation_rejects_wrong_recipe_or_skipped_regularization(tmp_path, damage):
    completed = M.ROOT / 'analysis/empf_comparison_20261117/completion/fits/native/pbulk/hvg2000/fold0/seed0'
    values = {name:json.loads((completed/name).read_text()) for name in
              ('config.json','fit_receipt.json','no_skip_audit.json')}
    if damage == 'old_release':values['config.json']['particlegan_version'] = '0.9.7'
    elif damage == 'matched_variant':
        values['config.json']['recipe'] = json.loads(M.RECIPE.read_text())['matched']['recipe']
    else:values['no_skip_audit.json']['penalty_calls'] = 10000
    for name,value in values.items():(tmp_path/name).write_text(json.dumps(value))
    with pytest.raises(ValueError):M.check_fit(tmp_path,M.load_config(),0,0)
