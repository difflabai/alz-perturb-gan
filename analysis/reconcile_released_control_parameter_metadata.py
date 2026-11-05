"""Reconcile the two equivalent parameter encodings, preserving every real state."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from state_fingerprint import fingerprint
    old_path = ROOT / 'analysis/CAL_GPU0_V1_network_rate_half_v1_f0_pg_s0-result.json'
    new_path = ROOT / 'analysis/CAL_GPU0_V1_released_rate_g050d050p100_f0_pg_s0-result.json'
    initial_path = ROOT / 'analysis/released_particle_full_control_20k_reproduction.json'
    old, new = [read(path)['candidate'] for path in (old_path, new_path)]
    a, b = [torch.load(candidate['checkpoint_path'], map_location='cpu', weights_only=False) for candidate in (old, new)]
    assert a['native_recipe'] == b['native_recipe'] == 'bispf_optimized'
    assert a['network_rate_config'] == {'lr': .006, 'prior_lr_mult': 5.0}
    assert b['released_hyperparameters'] == {'generator_lr': .006, 'critic_lr': .009, 'particle_lr': .03,
                                            'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
    assert a['recipe'] == b['recipe'] and a['policy']['recipe'] == b['policy']['recipe']
    assert a['recipe']['lr'] == .006 and a['recipe']['d_lr_mult'] == 1.5 and a['recipe']['prior_lr_mult'] == 5.
    assert set(a) - set(b) == {'network_rate_config'} and set(b) - set(a) == {'released_hyperparameters'}
    common_differences = [key for key in sorted(set(a) & set(b)) if fingerprint(a[key]) != fingerprint(b[key])]
    old_parameters = a.pop('network_rate_config')
    new_parameters = b.pop('released_hyperparameters')
    equal = fingerprint(a) == fingerprint(b)
    assert not torch.cuda.is_initialized()
    result = {'status': 'EXACT_20K_NATIVE_AND_ALL_SCIENTIFIC_STATE_CONTROL_REPRODUCTION_PARAMETER_ENCODINGS_RECONCILED',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'native_policy_byte_exact': fingerprint(a['policy']) == fingerprint(b['policy']),
              'all_native_and_scientific_state_byte_exact': equal,
              'different_common_state_fields': common_differences,
              'old_parameter_wrapper': old_parameters, 'new_parameter_wrapper': new_parameters,
              'old_normalized_full_state_fingerprint': fingerprint(a), 'new_normalized_full_state_fingerprint': fingerprint(b),
              'only_declared_equivalent_parameter_wrappers_removed': True,
              'native_recipe_and_device_metadata_retained_unchanged': True,
              'metrics_difference_new_minus_old': {key: new[key] - old[key] for key in ('energy_score', 'cell_energy_distance', 'spread_ratio')},
              'initial_comparison_path': str(initial_path), 'initial_comparison_SHA256': sha(initial_path),
              'initial_comparison_explanation': 'The first comparison removed native_recipe only from the new checkpoint, even though both contain it, and left the old network_rate_config. Its false full-wrapper equality flag is preserved. This addendum keeps native_recipe in both and removes only the two explicitly validated equivalent parameter encodings. No model/optimizer/RNG/usage/clock/cache/scientific state is removed or rewritten.',
              'CUDA_initialized': False, 'new_training_or_generation': False, 'incumbent_changed': False,
              'source_SHA256': {str(path): sha(path) for path in (Path(__file__).resolve(), initial_path, old_path, new_path,
                                Path(old['checkpoint_path']), Path(new['checkpoint_path']))}}
    out = ROOT / 'analysis/released_particle_full_control_20k_reproduction_metadata_reconciled.json'
    assert not out.exists();out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'native_policy_byte_exact', 'all_native_and_scientific_state_byte_exact', 'different_common_state_fields', 'metrics_difference_new_minus_old')}))


if __name__ == '__main__':
    main()
