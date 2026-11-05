"""Keep the declared duration endpoint separate from the 27-point rate grid."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
sys.path.insert(0, str(ROOT))
from selection import decide


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    protocol_path = ROOT / 'analysis/released_best_rate_extension40k_protocol.json'
    protocol = read(protocol_path)
    attempt = 'CAL_GPU0_V1_released_best_extend40k_f0_pg_s0'
    result_path = ROOT / 'analysis' / (attempt + '-result.json')
    result = read(result_path)  # No partial result or live loss substitutes for this audit.
    parent_path = Path(protocol['parent_result_path'])
    assert sha(parent_path) == protocol['parent_result_SHA256']
    parent_result = read(parent_path)
    parent, candidate = parent_result['candidate'], result['candidate']
    assert result['status'] == 'VERIFIED_RELEASED_PUBLIC_API_40K_EXTENSION_ENDPOINT_AND_30K_DIAGNOSTIC_RESULT'
    for key in (
        'independent_saved_scores_checkpoint_and_count_exemplars_verified',
        'full_native_recipe_model_and_seed_streams_verified',
        'native_optimizer_and_EMA_checkpoint_state_verified',
        'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified',
        'predeclared_both_extension_checkpoint_saved_scores_and_counts_verified',
        'sampled_raw_row_ownership_and_all512rows_used_verified',
        'owned_release_and_clean_settlement_verified',
        'native_completed20k_parent_and_all_scientific_streams_byte_exact_before_extension_verified',
        'public_execution_limit_extended_without_native_recipe_or_device_metadata_rewrite_verified',
    ):
        assert result[key] is True, key
    assert candidate['tuning_candidate_id'] == protocol['candidate_id']
    assert candidate['status'] == parent['status'] == 'completed'
    assert candidate['fixed_endpoint'] == candidate['completed_updates'] == 40000
    assert candidate['initial_completed_updates'] == candidate['new_model_updates'] == 20000
    assert parent['completed_updates'] == parent['fixed_endpoint'] == 20000
    assert candidate['native_recipe_schedule_horizon'] == 20000
    assert candidate['released_hyperparameters'] == parent['released_hyperparameters'] == protocol['released_hyperparameters']
    assert candidate['parent_checkpoint_SHA256'] == protocol['checkpoint_SHA256']
    assert sha(candidate['checkpoint_path']) == candidate['checkpoint_SHA256']
    for key in ('donors', 'generation_replicates', 'training_seed', 'metric_role',
                'calibration_manifest_SHA256', 'trace_manifest_SHA256', 'sampling_contract'):
        assert candidate[key] == parent[key], key
    assert candidate['training_seed'] == 0 and candidate['generation_replicates'] == 2
    assert candidate['integer_count_contract_passed'] and candidate['real_bag_replacements'] == 0
    curve_path = Path(candidate['convergence_curve_path'])
    assert sha(curve_path) == candidate['convergence_curve_SHA256']
    curve = read(curve_path)
    assert curve['extension_training_finished_before_new_calibration'] is True
    assert curve['parent_calibration_informed_extension'] is True
    assert curve['best_checkpoint_selection'] is False
    assert [e['completed_updates'] for e in curve['entries']] == [30000, 40000]
    assert [(e['diagnostic_only'], e['eligible_for_selection']) for e in curve['entries']] == [(True, False), (False, True)]
    incumbent_path = ROOT / 'pure_incumbent.json'
    incumbent = read(incumbent_path)
    assert sha(incumbent_path) == '869c9a55c3f3bb5628a9e0ab44357f83297a5b74b8cb7d67faf420794de487fa'
    assert result['decision'] == decide(candidate, incumbent)
    baseline_path = ROOT / 'analysis/baseline_calibration_seed0.json'
    baseline = read(baseline_path)
    baseline_comparison = {
        'ES_improvement_percent': 100 * (1 - candidate['energy_score'] / baseline['energy_score']),
        'cell_ED_difference': candidate['cell_energy_distance'] - baseline['cell_energy_distance'],
    }
    parent_seconds = parent_result.get('charged_GPU_seconds_total_scientific_trial', parent_result['charged_GPU_seconds'])
    assert parent_seconds == protocol['parent_charged_GPU_seconds'] == result['parent_charged_GPU_seconds']
    assert result['charged_GPU_seconds_total_scientific_trial'] == parent_seconds + result['charged_GPU_seconds']
    metrics = ('energy_score', 'cell_energy_distance', 'spread_ratio')
    previous_path = ROOT / 'analysis/pure_formulations_leaderboard_after_eighth_released_rate.json'
    previous = read(previous_path)
    assert previous['completed_seed0_comparisons'] == 32
    assert previous['completed_released_rate_configurations'] == 8
    new_row = dict(candidate, formulation=protocol['candidate_id'],
                   decision_evidence=str(result_path), decision_evidence_SHA256=sha(result_path),
                   accepted=result['decision']['accept'], complete_clean_settlement_verified=True,
                   same_model_duration_extension=True, parent_attempt=protocol['parent_attempt'],
                   charged_GPU_seconds=result['charged_GPU_seconds'],
                   charged_GPU_seconds_total_model=result['charged_GPU_seconds_total_scientific_trial'])
    leaderboard = dict(previous, status='PURE_COUNT_COMPARISONS_WITH_DECLARED_SAME_MODEL_40K_DURATION_ENDPOINT',
                       utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       prior_leaderboard_path=str(previous_path), prior_leaderboard_SHA256=sha(previous_path),
                       rows=sorted([*previous['rows'], new_row], key=lambda r: r['energy_score']),
                       completed_seed0_comparisons=33, completed_released_duration_extensions=1,
                       completed_released_rate_configurations=8, declared_released_rate_configurations=27,
                       additional_training_updates_since_prior_leaderboard=20000,
                       additional_charged_GPU_seconds=result['charged_GPU_seconds'],
                       incumbent_changed=False, stable_joint_ATH_claimed=False,
                       scope='Retains all32 previous endpoint comparisons plus one separately declared same-model 40k endpoint. The parent20k remains visible. This adds20k updates and only the new phase charge; it is not a ninth rate configuration, new independent seed or biological replicate. Four adaptive calibration donors; evaluation excluded.')
    summary = {
        'status': 'INDEPENDENTLY_AUDITED_RELEASED_20K_TO40K_DURATION_COMPARISON',
        'utc': leaderboard['utc'], 'attempt': attempt, 'released_hyperparameters': candidate['released_hyperparameters'],
        'parent20k': {k: parent[k] for k in metrics},
        'fixed40k': {k: candidate[k] for k in metrics},
        'difference40k_minus20k': {k: candidate[k] - parent[k] for k in metrics},
        'difference40k_minus_frozen_incumbent': {k: candidate[k] - incumbent[k] for k in metrics},
        'decision': result['decision'], 'baseline_comparison': baseline_comparison,
        'new_phase_charged_GPU_seconds': result['charged_GPU_seconds'],
        'parent_charged_GPU_seconds': parent_seconds,
        'total_model_charged_GPU_seconds': result['charged_GPU_seconds_total_scientific_trial'],
        'sources_SHA256': {str(p): sha(p) for p in (Path(__file__), protocol_path, result_path, parent_path, curve_path, incumbent_path, baseline_path, previous_path)},
        'parent_calibration_informed_duration_hypothesis': True,
        'only40k_extension_endpoint_selectable': True, 'rate_grid_completed': 8,
        'regularization_stage2_anchor_selected': False, 'incumbent_changed': False,
        'stable_joint_ATH_claimed': False, 'evaluation_used_for_tuning': False,
        'finisher_scope_prose_correction': 'The finisher retained an obsolete original20k scope sentence. Its executable assertions and prospective protocol require40k; only40k is the extension selection endpoint.30k is diagnostic.',
    }
    for name, value in (
        ('released_best_40k_duration_comparison.json', summary),
        ('pure_formulations_leaderboard_after_released_40k_extension.json', leaderboard),
    ):
        destination = ROOT / 'analysis' / name
        assert not destination.exists(), destination
        destination.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    build()
