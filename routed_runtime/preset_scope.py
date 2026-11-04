"""One fixed native routed E22 count-adapter trial; no implicit experiment grid."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/ml2/alz')
HERE = ROOT / 'routed_runtime'
MODEL_DIR = ROOT / 'routed_model'
PROTOCOL = ROOT / 'analysis/routed_e22_protocol.json'
MODEL_PROOF = ROOT / 'analysis/routed_e22_model_CPU_qualification.json'
RUNTIME_PROOF = ROOT / 'analysis/routed_e22_runtime_CPU_qualification.json'
ENGINE_PROOF = ROOT / 'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW = ROOT / 'analysis/research_window_20261104_five_days.json'
WINDOW_SHA = '2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN = HERE / 'plan.json'
CFG = {'z_dim': 8, 'stage_balance': False, 'cell_z_dim': 8,
       'theta_init_scale': 1., 'fourier_feature_scale': 1.}
ROUTED = {'native_recipe': 'e22_routed', 'row_semantics': 'dense_soft', 'probe_interval': 20, 'protected_fraction': 'fixed_per_donor_hash_rank_floor_one_fifth', 'context': {'query_dimension': 8, 'condition_dimension': 7, 'seed_streams': 3, 'words_per_seed': 4, 'word_bits': 16, 'effective_seed_bits': 64, 'output': 'raw_counts_and_existing_transformed_rate_observation'}, 'output_noise_coordinates': 'fake_150D_critic_observation_only', 'paired_fit_contexts': 'same_fit_context_and_targets_for_D_and_G_substeps'}
CELL = {'experiment': 'routed_e22_v1', 'fold': 0, 'method': 'pg', 'seed': 0}
ENABLED = {'enabled': True, 'native_recipe': 'e22_routed', 'fixed_endpoint': 20000,
           'pure_only': True, 'training_seed': 0, 'experiment': 'routed_e22_v1'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def protocol_paths(p):
    return [PROTOCOL, MODEL_PROOF, ENGINE_PROOF, WINDOW, PLAN,
            ROOT / 'pure_incumbent.json', ROOT / 'incumbent.json',
            *[Path(p[k]) for k in ('original_failed_confirmation_path',
                                  'uniform_ensemble_negative_screen_path',
                                  'hypothesis_path', 'previous_authorization_path',
                                  'failed_halloween_result_path', 'reverted_halloween_plan_path')]]


def validate_authorization(auth, *, require_runtime_proof=True):
    assert auth['operation'] == 'train_routed_e22' and auth['cell'] == CELL
    assert type(auth['cap_seconds']) is int and auth['cap_seconds'] == 21600
    assert type(auth['fixed_endpoint']) is int and auth['fixed_endpoint'] == 20000
    assert auth['variant_config'] == CFG and auth['hybrid_config'] is None
    assert auth['routed_config'] == ROUTED
    assert auth['pure_formulation'] == 'e22_routed_count' and auth['qualification_only'] is False
    assert auth['initial_completed_updates'] == 0 and auth['new_model_updates'] == 20000
    for key in ('checkpoint_path', 'checkpoint_SHA256', 'prior_update', 'stage_prior_config', 'halloween_native_config'):
        assert key not in auth
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type'] == 'OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['native_recipe'] == 'e22_routed' and auth['dynamic_shared_gpu0'] is True
    p, w, pins, research = read(PROTOCOL), read(WINDOW), auth['files_SHA256'], auth['research_source_SHA256']
    assert read(PLAN) == ENABLED, 'native routed E22 plan remains disabled or changed'
    assert p['native_API'] == "ParticleGAN0.9.4 get_recipe('e22_routed',total_steps=None,batch_size=32,num_particles=512,z_dim=8); publicE22Policy_RoutedRows_RoutedBatch, probe_interval20, external20kcap"
    assert p['routed_config'] == ROUTED
    assert p['task'] == {'fold': 0, 'training_seed': 0, 'updates': 20000, 'cap_seconds': 21600, 'variant_config': CFG}
    assert p['source_model_directory'] == str(MODEL_DIR)
    assert sha(WINDOW) == WINDOW_SHA and w['expires_unix'] - w['effective_unix'] == 432000
    assert w['status'] == 'OWNER_EXPLICITLY_EXTENDED_RESEARCH_WINDOW' and w['owner_instruction'] == 'extend it to 5 days'
    assert w['project_GPU_hour_cap'] == 1024 and w['finite_training_phase_cap_seconds'] == 21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix'] <= w['expires_unix'] and auth['expires_unix'] <= auth['issued_unix'] + 21630
    assert sha(ROOT / 'pure_incumbent.json') == p['current_pure_incumbent_SHA256'] == auth['frozen_pure_incumbent_SHA256']
    proof = read(MODEL_PROOF)
    assert proof['status'] == 'PASSED_FINAL_ROUTED_E22_COUNT_MODEL_CPU_QUALIFICATION'
    assert proof['tests_passed'] == 19 and proof['CUDA_initialized'] is False
    assert proof['model_source_directory'] == str(MODEL_DIR) and proof['full_model_floating_values'] == 371617
    assert proof['native_recipe'] == 'e22_routed' and proof['native_horizon'] is None
    assert proof['external_update_cap'] == 20000 and proof['fitting_cells'] == 11825 and proof['protected_cells'] == 2937
    assert proof['final_callback_native_proposal_and_guard_rejection_exercised'] is True
    for key in ('native_recipe_seed_prefix_and_controls_verified', 'fitting_protection_partition_sampling_stats_and_gradient_isolation_verified', 'complete_callback_raw_count_original_gradient_and_candidate_mass_codes_verified', 'full_unsigned64_context_seed_domain_and_refusals_verified', 'complete_dynamic_sampler_native_probe_checkpoint_replay_byte_exact_verified', 'full2000gene_native_optimizer_and_clean400cell_iterator_verified', 'independent_coupled_served_model_readout_verified'):
        assert proof[key] is True
    engine = read(ENGINE_PROOF)
    assert engine['status'] == 'PASSED_CPU_SHARED_GPU0_ENGINE_CANDIDATE_ONLY_WITH_CURRENT_READ_ONLY_FORECAST'
    assert engine['tests_passed'] == 10 and not engine['native_policy_deployed']
    assert not engine['canonical_reservation_created'] and not engine['GPU_context_created']
    for path, pin in {**proof['source_SHA256'], **proof['official_wheel_source_SHA256'], **engine['source_SHA256']}.items():
        assert pins[path] == pin == sha(path)
    for key in ('original_failed_confirmation', 'uniform_ensemble_negative_screen', 'hypothesis', 'previous_authorization',
                'failed_halloween_result', 'reverted_halloween_plan'):
        assert sha(p[key + '_path']) == p[key + '_SHA256']
    assert read(p['reverted_halloween_plan_path'])['enabled'] is False
    assert read(p['failed_halloween_result_path'])['decision']['accept'] is False
    assert read(p['original_failed_confirmation_path'])['stable_joint_target_passed'] is False
    assert read(p['uniform_ensemble_negative_screen_path'])['joint_analytic_screen_passed'] is False
    for path in protocol_paths(p):
        assert pins[str(path)] == sha(path)
    assert auth['preset_protocol_SHA256'] == sha(PROTOCOL) and auth['model_qualification_SHA256'] == sha(MODEL_PROOF)
    for folder in (ROOT, ROOT / 'model', HERE, MODEL_DIR):
        for path in folder.glob('*.py'):
            assert research[str(path)] == pins[str(path)] == sha(path)
    if require_runtime_proof:
        proof = read(RUNTIME_PROOF)
        assert auth['runtime_qualification_SHA256'] == pins[str(RUNTIME_PROOF)] == sha(RUNTIME_PROOF)
        assert proof['status'] == 'PASSED_ISOLATED_ROUTED_E22_RUNTIME_CPU_QUALIFICATION'
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged']
        assert proof['GPU_allocation_created'] is False and proof['prior_active_trial_native_launch_refusal_verified'] is True
        assert proof['tests_passed']==27
        assert proof['native_checkpoint_cpu_tensor_control_and_served_selection_audit_verified'] is True
        assert proof['fingerprint_tensor_numpy_shape_type_and_float_bit_boundaries_verified'] is True
        for path, pin in proof['source_SHA256'].items():
            assert pins[path] == pin == sha(path)
    return p
