"""One explicit released-API configuration from the registered rate grid."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'released_rate_resume_runtime'
MODEL_DIR=ROOT/'analysis/released_tuning'
PROTOCOL=MODEL_DIR/'protocol.json'
MODEL_PROOF=MODEL_DIR/'CPU_qualification.json'
NATIVE_PROOF=ROOT/'analysis/routed_e22_model_CPU_qualification.json'
PARAMETERIZATION=MODEL_DIR/'rate_parameterization_and_controls.json'
COMPAT_PROOF=MODEL_DIR/'historical_native_checkpoint_CPU_audit.json'
RESUME_PROTOCOL=ROOT/'analysis/released_rate_resume_protocol.json'
CONTINUATION=json.loads(RESUME_PROTOCOL.read_bytes())
INITIAL=10842
RUNTIME_PROOF=ROOT/'analysis/released_rate_resume_runtime_CPU_qualification.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
CANDIDATE_ID='lr_g050_d050_p050'
HYPER={'generator_lr': 0.006, 'critic_lr': 0.009, 'particle_lr': 0.015, 'reg_coeff': 1.0, 'reg_kappa': 1.0, 'prior_reg': 0.01}
CELL={'experiment':'released_rate_g050d050p050_resume1','fold':0,'method':'pg','seed':0}
ENABLED={'enabled':True,'native_recipe':'bispf_optimized','fixed_endpoint':20000,'pure_only':True,
    'training_seed':0,'experiment':CELL['experiment'],'candidate_id':CANDIDATE_ID}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol_paths(p):
    return [RESUME_PROTOCOL,*[Path(CONTINUATION[k]) for k in ('parent_authorization_path','checkpoint_path','parent_stop_path','parent_audit_path','parent_reverted_plan_path')],*[Path(x['immutable_copy_path']) for x in CONTINUATION['diagnostic_checkpoints']],Path(CONTINUATION['training_log']['immutable_copy_path']),PROTOCOL,MODEL_PROOF,NATIVE_PROOF,PARAMETERIZATION,COMPAT_PROOF,ENGINE_PROOF,WINDOW,PLAN,
        ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'selection.py',
        Path(p['frozen_science']['fixed_confirmation_path']),Path(p['native_latest_verification_path']),
        Path('/ml2/alz/analysis/released_rate_g050d050p050_initial_resource_observation.json')]
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['operation']=='resume_released_hyperparameters' and auth['cell']==CELL
    assert type(auth['cap_seconds']) is int and auth['cap_seconds']==21600
    assert type(auth['fixed_endpoint']) is int and auth['fixed_endpoint']==20000
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='released_hyperparameters' and auth['qualification_only'] is False
    assert auth['native_recipe']=='bispf_optimized' and auth['released_hyperparameters']==HYPER
    assert auth['tuning_candidate_id']==CANDIDATE_ID and auth['initial_completed_updates']==INITIAL and auth['new_model_updates']==20000-INITIAL
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
    for k in ('checkpoint_path','checkpoint_SHA256','prior_update','stage_prior_config','halloween_native_config','routed_config'):
        assert k not in auth
    assert auth['continuation']==CONTINUATION and auth['continuation_protocol_SHA256']==sha(RESUME_PROTOCOL)
    assert auth['expires_unix']<=CONTINUATION['original_absolute_deadline_unix']
    assert sha(CONTINUATION['checkpoint_path'])==CONTINUATION['checkpoint_SHA256']
    stop=read(CONTINUATION['parent_stop_path']);audit=read(CONTINUATION['parent_audit_path'])
    assert stop['completed_updates']==INITIAL and stop['quality_comparison_completed'] is False
    assert stop['owned_processes_absent'] and stop['owned_registry_empty'] and stop['canonical_reservation_released']
    assert stop['original_absolute_training_deadline_unix']==CONTINUATION['original_absolute_deadline_unix']
    assert audit['status']=='ACTUAL_RELEASED_RATE_10842_RECOVERY_NATIVE_CPU_COMPONENT_AND_SAMPLER_AUDIT_PASSED'
    assert audit['completed_updates']==INITIAL and audit['model_floating_values']==371177 and audit['CUDA_initialized'] is False
    assert audit['whole_scientific_checkpoint_fingerprint']==CONTINUATION['whole_parent_scientific_fingerprint']
    for key in ('parent_authorization','parent_stop','parent_audit','parent_reverted_plan'):
        assert sha(CONTINUATION[key+'_path'])==CONTINUATION[key+'_SHA256']
    assert read(CONTINUATION['parent_reverted_plan_path'])['enabled'] is False
    for source in [*CONTINUATION['diagnostic_checkpoints'],CONTINUATION['training_log']]:assert sha(source['immutable_copy_path'])==source['SHA256']
    assert read(PLAN)==ENABLED,'released rate plan remains disabled or changed'
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert p['status']=='PROSPECTIVE_RELEASED_PARTICLEGAN_HYPERPARAMETER_SEARCH_NOT_LAUNCHED'
    assert [s for s in p['stage1']['configs'] if s['id']==CANDIDATE_ID]==[{'id':CANDIDATE_ID,'parameters':HYPER}]
    assert len(p['stage1']['configs'])==27 and p['frozen_science']['evaluation_used_for_tuning'] is False
    assert p['convergence_evidence']['checkpoints_at_updates']==[5000,10000,20000]
    assert sha(WINDOW)==WINDOW_SHA and w['expires_unix']-w['effective_unix']==432000
    assert w['project_GPU_hour_cap']==1024 and w['finite_training_phase_cap_seconds']==21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+21630)
    assert sha(ROOT/'pure_incumbent.json')==p['frozen_science']['pure_incumbent_SHA256']==auth['frozen_pure_incumbent_SHA256']
    proof=read(MODEL_PROOF)
    assert proof['status']=='PASSED_ACTUAL_RELEASED_PARAMETER_SYNTHETIC_CPU_QUALIFICATION'
    assert proof['tests_passed']==6 and proof['CUDA_initialized'] is False and proof['GPU_job_created'] is False
    for k in ('all27_declared_native_recipes_and_optimizer_rates_verified','baseline_PG8_two_update_complete_state_byte_parity',
            'same_native_update_observation_context_and_seed_streams_AST','rate_and_regularization_extremes_connected_gradients_and_full_state_replay',
            'full2000gene_connected_gradients_and_exact_next_pair_replay','wrong_hyperparameter_clock_status_recipe_cache_configuration_and_CPU_fit_refused'):
        assert proof[k] is True
    assert read(PARAMETERIZATION)['CPU_qualification_SHA256']==sha(MODEL_PROOF)
    compatibility=read(COMPAT_PROOF)
    assert compatibility['status']=='PASSED_ACTUAL_HISTORICAL_NATIVE_CUDA_CHECKPOINT_CPU_MODEL_OPTIMIZER_COMPATIBILITY_AUDIT'
    assert compatibility['fullwidth_model_floating_values']==371177 and compatibility['CUDA_initialized'] is False
    for path,pin in compatibility['source_SHA256'].items():assert pins[path]==pin==sha(path)
    native=read(NATIVE_PROOF)['official_wheel_source_SHA256'];assert len(native)==51
    engine=read(ENGINE_PROOF)
    assert engine['status']=='PASSED_CPU_SHARED_GPU0_ENGINE_CANDIDATE_ONLY_WITH_CURRENT_READ_ONLY_FORECAST'
    assert engine['tests_passed']==10 and not engine['native_policy_deployed'] and not engine['GPU_context_created']
    for path,pin in {**proof['source_SHA256'],**native,**engine['source_SHA256']}.items():
        assert pins[path]==pin==sha(path)
    for path in protocol_paths(p):assert pins[str(path)]==sha(path)
    assert auth['preset_protocol_SHA256']==sha(PROTOCOL) and auth['model_qualification_SHA256']==sha(MODEL_PROOF)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        proof=read(RUNTIME_PROOF)
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        assert proof['status']=='PASSED_EXACT_RELEASED_PARAMETER_RESUME_RUNTIME_CPU_QUALIFICATION'
        assert proof['tests_passed']==27 and proof['exact_scientific_restore_replay_and_CPU_refusal_verified'] is True
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged']
        assert proof['GPU_allocation_created'] is False and proof['prior_active_trial_native_launch_refusal_verified'] is True
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
