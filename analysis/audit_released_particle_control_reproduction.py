"""Compare the declared historical particle-rate control at its audited20k endpoint."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from state_fingerprint import fingerprint
    old_path = ROOT / 'analysis/CAL_GPU0_V1_network_rate_half_v1_f0_pg_s0-result.json'
    new_path = ROOT / 'analysis/CAL_GPU0_V1_released_rate_g050d050p100_f0_pg_s0-result.json'
    old_result, new_result = read(old_path), read(new_path)
    assert new_result['predeclared_all_three_checkpoint_saved_scores_and_counts_verified']
    assert new_result['owned_release_and_clean_settlement_verified']
    old, new = old_result['candidate'], new_result['candidate']
    for key in ('donors', 'training_seed', 'generation_replicates', 'calibration_manifest_SHA256',
                'trace_manifest_SHA256', 'sampling_contract', 'variant_config', 'fixed_endpoint', 'completed_updates'):
        assert old[key] == new[key], key
    assert new['fixed_endpoint'] == 20000 and new['training_seed'] == 0
    assert old['status'] == new['status'] == 'completed'
    assert old['network_rate_config'] == {'lr': .006, 'prior_lr_mult': 5.0}
    assert new['released_hyperparameters'] == {'generator_lr': .006, 'critic_lr': .009, 'particle_lr': .03,
                                              'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
    for candidate in (old, new):
        assert sha(candidate['checkpoint_path']) == candidate['checkpoint_SHA256']
    before, after = [torch.load(candidate['checkpoint_path'], map_location='cpu', weights_only=False) for candidate in (old, new)]
    assert before['recipe'] == after['recipe'] and before['policy']['device'] == after['policy']['device'] == 'cuda:0'
    policy_equal = fingerprint(before['policy']) == fingerprint(after['policy'])
    policy_fields = {key: fingerprint(before['policy'].get(key)) == fingerprint(after['policy'].get(key))
                     for key in sorted(set(before['policy']) | set(after['policy']))}
    normalized = dict(after)
    removed = {}
    for key in ('native_recipe', 'released_hyperparameters'):
        removed[key] = normalized.pop(key)
    scientific_equal = fingerprint(before) == fingerprint(normalized)
    metric_deltas = {key: new[key] - old[key] for key in ('energy_score', 'cell_energy_distance', 'spread_ratio')}
    assert not torch.cuda.is_initialized()
    result = {'status': 'AUDITED_20K_RELEASED_API_HISTORICAL_CONTROL_REPRODUCTION_COMPARISON',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'old_result_path': str(old_path), 'new_result_path': str(new_path),
              'native_policy_state_byte_exact': policy_equal, 'native_policy_fields_equal': policy_fields,
              'all_scientific_state_byte_exact_after_two_declared_wrapper_fields_removed': scientific_equal,
              'removed_new_wrapper_fields': removed, 'old_native_policy_fingerprint': fingerprint(before['policy']),
              'new_native_policy_fingerprint': fingerprint(after['policy']),
              'metrics_difference_new_minus_old': metric_deltas,
              'native_device_metadata_relabelled': False, 'CUDA_initialized': False,
              'new_training_or_generation': False, 'incumbent_changed': False, 'stable_joint_ATH_claimed': False,
              'source_SHA256': {str(path): sha(path) for path in (Path(__file__).resolve(), old_path, new_path,
                    Path(old['checkpoint_path']), Path(new['checkpoint_path']))},
              'scope': 'Actual completed20k fixedseed0 same-rate same-data control, as prospectively declared in the rate grid. Native policies and scientific state compared onCPU without relabelling CUDA metadata; equality or differences are reported, never used to discard a completed configuration. Four adaptive calibration donors; no biological replication/evaluation or jointATH claim.'}
    path = ROOT / 'analysis/released_particle_full_control_20k_reproduction.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'native_policy_state_byte_exact',
        'all_scientific_state_byte_exact_after_two_declared_wrapper_fields_removed', 'metrics_difference_new_minus_old')}))


if __name__ == '__main__':
    main()
