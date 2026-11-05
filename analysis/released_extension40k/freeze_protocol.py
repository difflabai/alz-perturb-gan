"""Save the completed CPU evidence and a separate prospective40k duration test."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')
HERE = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit_path = HERE / 'parent20k_CPU_component_audit.json'
    audit = read(audit_path)
    assert audit['status'] == 'ACTUAL_COMPLETED20K_RELEASED_PARENT_CPU_COMPONENT_ARCHIVE_AUDIT_PASSED'
    site = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/lib/python3.12/site-packages')
    policy = site / 'particlegan/policy.py'
    native_proof = ROOT / 'analysis/routed_e22_model_CPU_qualification.json'
    assert read(native_proof)['official_wheel_source_SHA256'][str(policy)] == sha(policy)
    source_files = [Path(__file__).resolve(), HERE / 'extension.py', HERE / 'test_extension.py',
                    ROOT / 'analysis/released_tuning/released_adapter.py', ROOT / 'analysis/released_tuning/released_update.py',
                    ROOT / 'analysis/released_tuning/qualify_cpu.py', ROOT / 'routed_resume_runtime/state_fingerprint.py', policy]
    proof_path = HERE / 'CPU_qualification.json'
    assert not proof_path.exists()
    proof = {
        'status': 'PASSED_RELEASED_PUBLIC_EXECUTION_CAP_EXTENSION_FULLWIDTH_CPU_REPLAY_AND_REFUSALS',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tests_passed': 4,
        'actual_test_command': 'Native isolated Python; runpy test_extension.py; CUDA_VISIBLE_DEVICES empty; OMP/OPENBLAS/MKL threads1',
        'actual_test_result': 'Ran4tests in24.453s;OK',
        'actual_public_begin_step_overruns_saved_recipe_horizon': True,
        'full2000gene_same_model_2_to4_pair_replay_byte_exact': True,
        'saved_native_policy_above_schedule_horizon_public_reload_byte_exact': True,
        'all_scientific_streams_next_pair_replay_above_schedule_horizon_byte_exact': True,
        'metadata_parent_identity_clock_recipe_cache_seed_and_hyperparameter_refusals': True,
        'native_endpoint_and_diagnostic_overrides_refused_before_GPU': True,
        'real_CPU_fitting_refused_before_checkpoint_or_output': True,
        'CUDA_initialized': False, 'GPU_job_or_allocation_started': False,
        'native_runtime_scope_and_resource_qualification_still_pending': True,
        'source_SHA256': {str(path): sha(path) for path in source_files},
        'scope': 'Actual finite synthetic CPU fullwidth model/public0.9.4 API tests. Native recipe and policy metadata stay unchanged while public execution_limit controls an external cap. No actual ALZ/CUDA continuation claimed. Separate40k declaration, runtime source qualification, guarded admission and full exact CUDA parent restore remain required.',
    }
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    copied_keys = ('parent_attempt', 'parent_result_path', 'parent_result_SHA256', 'parent_authorization_path',
                   'parent_authorization_SHA256', 'checkpoint_path', 'checkpoint_SHA256',
                   'whole_parent_scientific_fingerprint', 'parent_policy_fingerprint',
                   'parent_charged_GPU_seconds', 'charged_prior_phases', 'training_log',
                   'selection_readout_snapshot_path', 'selection_readout_snapshot_SHA256',
                   'released_hyperparameters', 'variant_config')
    value = {key: audit[key] for key in copied_keys}
    source_files = [ROOT / 'analysis/research_window_20261104_five_days.json',
                    ROOT / 'analysis/released_tuning/protocol.json', ROOT / 'pure_incumbent.json',
                    ROOT / 'selection.py', policy, native_proof, HERE / 'extension.py', audit_path, proof_path]
    value.update(
        status='PROSPECTIVE_RELEASED_BEST_RATE_20K_TO40K_DURATION_EXTENSION_NOT_LAUNCHED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        candidate_id='extend40k_lr_g050_d050_p050', reference_rate_candidate_id='lr_g050_d050_p050',
        initial_completed_updates=20000, fixed_endpoint=40000, new_model_updates=20000,
        training_seed=0, fold=0, native_recipe='bispf_optimized', native_recipe_schedule_horizon=20000,
        public_execution_limit=40000, parent_audit_path=str(audit_path), parent_audit_SHA256=sha(audit_path),
        completed_rate_endpoints_when_reference_frozen=6, reference_dominates_all_six_on_both_metrics=True,
        scientific_hypothesis='For the best completed rate setting, pooledcellED improved5k->10k->20k while final20kES remained below the incumbent. Test a fixed40k endpoint through public execution_limit, preserving the completed20k native/scientific state. A separate duration experiment, not a replacement20k result or premature stage2 anchor.',
        diagnostic_checkpoints_after_parent=[30000], new_calibration_checkpoints_after_extension_training=[30000, 40000],
        eligible_for_selection_updates=[40000], generation_replicates=2,
        parent_calibration_informed_extension_declaration=True,
        new_extension_calibration_only_after_full40k_training=True,
        no_best_intermediate_checkpoint_selection=True, no_new_training_model_or_seed=True,
        no_native_recipe_or_device_metadata_rewrite=True,
        new_scientific_extension_phase_cap_seconds=21600,
        original_completed20k_phase_deadlines_not_reset_or_reopened=True,
        new_phase_expires_within_existing_five_day_window=True,
        original27_rate_settings_retained=True, regularization_stage2_anchor_selected=False,
        incumbent_promoted=False, evaluation_used_for_tuning=False, stable_joint_ATH_claimed=False,
        model_CPU_proof_path=str(proof_path), model_CPU_proof_SHA256=sha(proof_path),
        native_runtime_scope_and_resource_qualification_still_pending=True,
        source_SHA256={str(path): sha(path) for path in source_files},
        scope='Four adaptive calibration donors,33fitting donors,20evaluation donors excluded. Fixed40k alone eligible under unchanged selection.decide. Observed parent calibration informed this duration hypothesis; new30k/40k scores occur only after extension fitting. Not an independent biological replicate, completed rate-grid endpoint or stableATH. Separate finite qualified allocation/admission required; no automatic launch/promotion.',
    )
    out = ROOT / 'analysis/released_best_rate_extension40k_protocol.json'
    assert not out.exists()
    out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'status': value['status'], 'whole_parent_state_FP': value['whole_parent_scientific_fingerprint'],
                      'public_API_CPU_tests': 4, 'native_GPU_extension_started': False}))


if __name__ == '__main__':
    main()
