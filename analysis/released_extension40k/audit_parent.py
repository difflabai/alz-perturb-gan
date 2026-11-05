"""Archive the actual completed20k parent and audit CPU components without CUDA relabelling."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/ml2/alz')
HERE = Path(__file__).resolve().parent
PARENT = 'CAL_GPU0_V1_released_rate_g050d050p050_resume1_f0_pg_s0'
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from particlegan.policy import _validate_optimizer_state
    import released_adapter as model
    import released_update as update
    from state_fingerprint import fingerprint
    result_path = ROOT / 'analysis' / (PARENT + '-result.json')
    result = read(result_path)
    candidate = result['candidate']
    for flag in ('independent_saved_scores_checkpoint_and_count_exemplars_verified',
                 'predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                 'owned_release_and_clean_settlement_verified'):
        assert result[flag] is True
    assert candidate['completed_updates'] == candidate['fixed_endpoint'] == 20000
    hyper = {'generator_lr': .006, 'critic_lr': .009, 'particle_lr': .015, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
    cfg = {'z_dim': 8, 'stage_balance': False, 'cell_z_dim': 8, 'theta_init_scale': 1., 'fourier_feature_scale': 1.}
    assert candidate['released_hyperparameters'] == hyper and candidate['variant_config'] == cfg
    update.configure(cfg);update.configure_hyperparameters(hyper)
    auth_path = ROOT / 'artifacts/allocations' / PARENT / 'authorization.json'
    assert sha(auth_path) == result['authorization_SHA256']
    auth = read(auth_path)
    phases = []
    for path in (Path(candidate['continuation']['parent_authorization_path']), auth_path):
        a = read(path);ledger = Path(a['ledger_path']);settlement = ledger.with_name(a['attempt_id'] + '-settlement.json')
        job = read(ledger)['jobs'][0];s = read(settlement)
        assert job['reservation_released'] is True and s['uncertain'] is None and s['settlement']['status'] == 'settled'
        phases.append({'authorization_path': str(path), 'authorization_SHA256': sha(path),
                       'ledger_path': str(ledger), 'ledger_SHA256': sha(ledger),
                       'settlement_path': str(settlement), 'settlement_SHA256': sha(settlement),
                       'charged_GPU_seconds': s['settlement']['charged_GPU_seconds']})
    assert sum(phase['charged_GPU_seconds'] for phase in phases) == result['charged_GPU_seconds_total_scientific_trial']
    archive = ROOT / 'artifacts/diagnostics/released_best_rate_20k_to40k_parent'
    archive.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(candidate['checkpoint_path'])
    assert sha(checkpoint) == candidate['checkpoint_SHA256']
    immutable = archive / 'parent20000.pt';shutil.copyfile(checkpoint, immutable)
    assert sha(immutable) == candidate['checkpoint_SHA256']
    log = Path(candidate['learning_curve_path']);assert sha(log) == candidate['learning_curve_SHA256']
    log_copy = archive / 'learning_curve.jsonl';shutil.copyfile(log, log_copy)
    assert sha(log_copy) == candidate['learning_curve_SHA256']
    readout = ROOT / 'analysis/released_tuning_search_readout.json'
    snapshot = archive / 'selection_readout_snapshot.json';shutil.copyfile(readout, snapshot)
    prior = read(snapshot)
    completed = [row for row in prior['rows'] if row['state'] == 'audited_complete']
    assert len(completed) == 6
    reference = next(row for row in completed if row['id'] == candidate['tuning_candidate_id'])
    assert all(reference['energy_score'] <= row['energy_score'] and reference['cell_energy_distance'] <= row['cell_energy_distance'] for row in completed)
    q = torch.load(immutable, map_location='cpu', weights_only=False)
    assert q['status'] == 'completed' and q['completed_updates'] == q['fixed_endpoint'] == q['policy']['completed_steps'] == q['usage']['steps'] == 20000
    assert q['policy']['device'] == 'cuda:0' and q['native_recipe'] == 'bispf_optimized'
    assert q['recipe'] == q['policy']['recipe'] == model.recipe().to_dict() and q['recipe']['total_steps'] == 20000
    assert q['released_hyperparameters'] == hyper and q['variant_config'] == cfg and q['training_streams'] == model.training_streams(0, 0)
    bundle, *_ = model.build(torch.ones(2000), torch.ones(2000), 'cpu', 0, 0)
    models = {'generator': bundle[1], 'critic': bundle[2], 'prior': bundle[3]}
    finite_values = 0
    for name, module in models.items():
        saved = q['policy']['models'][name];expected = module.state_dict();assert saved.keys() == expected.keys()
        for key, value in saved.items():
            assert value.shape == expected[key].shape and value.dtype == expected[key].dtype
            if value.is_floating_point():
                assert torch.isfinite(value).all();finite_values += value.numel()
        module.load_state_dict(saved, strict=True)
    assert finite_values == 371177
    for optimizer, saved in zip(bundle[4:6], q['policy']['optimizers']):
        _validate_optimizer_state(optimizer, saved);optimizer.load_state_dict(saved)
        assert fingerprint(optimizer.state_dict()) == fingerprint(saved)
    assert q['prior_regularizer'] == {} and int(q['usage']['draws'].sum()) == 640000
    assert bool((q['usage']['draws'] > 0).all()) and int(q['usage']['unsampled_moved_steps'].sum()) == 0
    assert not torch.cuda.is_initialized()
    record = {'status': 'ACTUAL_COMPLETED20K_RELEASED_PARENT_CPU_COMPONENT_ARCHIVE_AUDIT_PASSED',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'parent_attempt': PARENT,
              'parent_result_path': str(result_path), 'parent_result_SHA256': sha(result_path),
              'parent_authorization_path': str(auth_path), 'parent_authorization_SHA256': sha(auth_path),
              'checkpoint_path': str(immutable), 'checkpoint_SHA256': sha(immutable),
              'whole_parent_scientific_fingerprint': fingerprint(q), 'parent_policy_fingerprint': fingerprint(q['policy']),
              'model_floating_values': finite_values, 'completed_updates': 20000, 'native_recipe_schedule_horizon': 20000,
              'parent_charged_GPU_seconds': result['charged_GPU_seconds_total_scientific_trial'],
              'charged_prior_phases': phases, 'released_hyperparameters': hyper, 'variant_config': cfg,
              'training_log': {'immutable_copy_path': str(log_copy), 'SHA256': sha(log_copy)},
              'selection_readout_snapshot_path': str(snapshot), 'selection_readout_snapshot_SHA256': sha(snapshot),
              'completed_rate_settings_when_extension_declared': 6,
              'reference_dominates_all_six_completed_rate_endpoints_on_both_metrics': True,
              'CUDA_initialized': False, 'native_CUDA_full_restore_still_required_before_extension': True,
              'native_device_metadata_relabelled': False,
              'source_SHA256': {str(path): sha(path) for path in (Path(__file__).resolve(), Path(model.__file__).resolve(), Path(update.__file__).resolve())},
              'scope': 'Actual frozen20k CUDA checkpoint deserialized onCPU; models/optimizers, clocks, streams, usage and immutable source/log/charge lineage verified. No CUDA restore claimed or metadata relabelled. Full exact CUDA/scientific snapshot restoration remains mandatory inside a separately admitted extension worker before update20001.'}
    output = HERE / 'parent20k_CPU_component_audit.json';assert not output.exists();output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in ('status', 'model_floating_values', 'whole_parent_scientific_fingerprint', 'parent_charged_GPU_seconds', 'CUDA_initialized')}))


if __name__ == '__main__':
    main()
