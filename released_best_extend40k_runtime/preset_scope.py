"""One explicit released-API configuration from the registered rate grid."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'released_best_extend40k_runtime'
MODEL_DIR=ROOT/'analysis/released_tuning'
PROTOCOL=MODEL_DIR/'protocol.json'
MODEL_PROOF=MODEL_DIR/'CPU_qualification.json'
NATIVE_PROOF=ROOT/'analysis/routed_e22_model_CPU_qualification.json'
PARAMETERIZATION=MODEL_DIR/'rate_parameterization_and_controls.json'
COMPAT_PROOF=MODEL_DIR/'historical_native_checkpoint_CPU_audit.json'
EXT_DIR=ROOT/'analysis/released_extension40k'
EXT_PROTOCOL=ROOT/'analysis/released_best_rate_extension40k_protocol.json'
EXT= json.loads(EXT_PROTOCOL.read_bytes())
EXT_PROOF=EXT_DIR/'CPU_qualification.json'
PARENT_AUDIT=EXT_DIR/'parent20k_CPU_component_audit.json'
RUNTIME_PROOF=ROOT/'analysis/released_best_extend40k_runtime_CPU_qualification.json'
ENGINE_PROOF=ROOT/'analysis/routed_e22_resume_shared_GPU1_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
CANDIDATE_ID='extend40k_lr_g050_d050_p050'
HYPER={'generator_lr': 0.006, 'critic_lr': 0.009, 'particle_lr': 0.015, 'reg_coeff': 1.0, 'reg_kappa': 1.0, 'prior_reg': 0.01}
CELL={'experiment':'released_best_extend40k','fold':0,'method':'pg','seed':0}
ENABLED={'enabled':True,'native_recipe':'bispf_optimized','fixed_endpoint':40000,'pure_only':True,
    'training_seed':0,'experiment':CELL['experiment'],'candidate_id':CANDIDATE_ID,
    'initial_completed_updates':20000,'new_model_updates':20000,'native_recipe_schedule_horizon':20000}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol_paths(p):
    return [*parent_plan_paths(),EXT_PROTOCOL,EXT_PROOF,PARENT_AUDIT,EXT_DIR/'extension.py',*[Path(phase[key]) for phase in EXT['charged_prior_phases'] for key in ('authorization_path','ledger_path','settlement_path')],*[Path(EXT[key]) for key in ('parent_result_path','parent_authorization_path','checkpoint_path','selection_readout_snapshot_path')],Path(EXT['training_log']['immutable_copy_path']),PROTOCOL,MODEL_PROOF,NATIVE_PROOF,PARAMETERIZATION,COMPAT_PROOF,ENGINE_PROOF,WINDOW,PLAN,
        ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'selection.py',
        Path(p['frozen_science']['fixed_confirmation_path']),Path(p['native_latest_verification_path']),
        Path('/ml2/alz/analysis/released_best_extend40k_initial_resource_observation.json')]
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['operation']=='extend_released_hyperparameters' and auth['cell']==CELL
    assert type(auth['cap_seconds']) is int and auth['cap_seconds']==21600
    assert type(auth['fixed_endpoint']) is int and auth['fixed_endpoint']==40000
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='released_horizon40k' and auth['qualification_only'] is False
    assert auth['native_recipe']=='bispf_optimized' and auth['released_hyperparameters']==HYPER
    assert auth['tuning_candidate_id']==CANDIDATE_ID and auth['initial_completed_updates']==20000 and auth['new_model_updates']==20000
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and auth['dynamic_shared_gpu0'] is False
    for k in ('checkpoint_path','checkpoint_SHA256','prior_update','stage_prior_config','halloween_native_config','continuation','routed_config'):
        assert k not in auth
    validate_extension_inputs(auth)
    assert read(PLAN)==ENABLED,'released rate plan remains disabled or changed'
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert p['status']=='PROSPECTIVE_RELEASED_PARTICLEGAN_HYPERPARAMETER_SEARCH_NOT_LAUNCHED'
    assert [s for s in p['stage1']['configs'] if s['id']==EXT['reference_rate_candidate_id']]==[{'id':EXT['reference_rate_candidate_id'],'parameters':HYPER}]
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
    assert engine['status']=='PASSED_CPU_UNCHANGED_SHARED_GPU1_ENGINE_FOR_ROUTED_CONTINUATION'
    assert engine['tests_passed']==10 and not engine['native_policy_deployed'] and not engine['GPU_context_created']
    for path,pin in {**proof['source_SHA256'],**native,**engine['source_SHA256']}.items():
        assert pins[path]==pin==sha(path)
    for path in protocol_paths(p):assert pins[str(path)]==sha(path)
    assert auth['preset_protocol_SHA256']==sha(PROTOCOL) and auth['model_qualification_SHA256']==sha(MODEL_PROOF)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,EXT_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        proof=read(RUNTIME_PROOF)
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        assert proof['status']=='PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION'
        assert proof['tests_passed']==27 and proof['extension_parent_source_and_external_cap_refusals_verified'] is True
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged']
        assert proof['GPU_allocation_created'] is False and proof['prior_active_trial_native_launch_refusal_verified'] is True
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p


def validate_extension_inputs(auth):
    assert EXT['status']=='PROSPECTIVE_RELEASED_BEST_RATE_20K_TO40K_DURATION_EXTENSION_NOT_LAUNCHED'
    assert auth['extension']==EXT and auth['extension_protocol_SHA256']==sha(EXT_PROTOCOL)
    assert auth['native_recipe_schedule_horizon']==20000
    assert auth['extension_qualification_SHA256']==sha(EXT_PROOF)
    assert EXT['initial_completed_updates']==20000 and EXT['fixed_endpoint']==40000 and EXT['new_model_updates']==20000
    assert EXT['native_recipe_schedule_horizon']==20000 and EXT['public_execution_limit']==40000
    assert EXT['released_hyperparameters']==HYPER and EXT['variant_config']==CFG
    assert EXT['new_calibration_checkpoints_after_extension_training']==[30000,40000] and EXT['eligible_for_selection_updates']==[40000]
    assert EXT['parent_calibration_informed_extension_declaration'] is True and EXT['new_extension_calibration_only_after_full40k_training'] is True
    for key in ('parent_result','parent_authorization','parent_audit'):
        assert sha(EXT[key+'_path'])==EXT[key+'_SHA256']
    assert sha(EXT['checkpoint_path'])==EXT['checkpoint_SHA256']
    assert sha(EXT['selection_readout_snapshot_path'])==EXT['selection_readout_snapshot_SHA256']
    assert sha(EXT['training_log']['immutable_copy_path'])==EXT['training_log']['SHA256']
    audit=read(PARENT_AUDIT);result=read(EXT['parent_result_path']);candidate=result['candidate']
    assert audit['status']=='ACTUAL_COMPLETED20K_RELEASED_PARENT_CPU_COMPONENT_ARCHIVE_AUDIT_PASSED'
    assert audit['CUDA_initialized'] is False and audit['model_floating_values']==371177
    assert audit['whole_parent_scientific_fingerprint']==EXT['whole_parent_scientific_fingerprint']
    assert result['owned_release_and_clean_settlement_verified'] is True and result['predeclared_all_three_checkpoint_saved_scores_and_counts_verified'] is True
    assert candidate['status']=='completed' and candidate['completed_updates']==candidate['fixed_endpoint']==20000
    assert candidate['released_hyperparameters']==HYPER and candidate['variant_config']==CFG
    assert result['charged_GPU_seconds_total_scientific_trial']==EXT['parent_charged_GPU_seconds']
    phases=EXT['charged_prior_phases'];assert sum(phase['charged_GPU_seconds'] for phase in phases)==EXT['parent_charged_GPU_seconds']
    for phase in phases:
        for key in ('authorization','ledger','settlement'):assert sha(phase[key+'_path'])==phase[key+'_SHA256']
        parent_auth=read(phase['authorization_path']);job=read(phase['ledger_path'])['jobs'][0];settlement=read(phase['settlement_path'])
        assert job['reservation_released'] is True and settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
        assert settlement['settlement']['charged_GPU_seconds']==phase['charged_GPU_seconds']
        assert not read(Path(parent_auth['registry_path'])/'registry.json')['jobs']
        plans=[Path(arg).resolve().parent/'plan.json' for arg in parent_auth['command'] if Path(arg).name=='worker_gpu0.py']
        assert len(plans)==1 and plans[0].is_relative_to(ROOT) and str(plans[0]) in parent_auth['files_SHA256']
        assert read(plans[0])['enabled'] is False and auth['files_SHA256'][str(plans[0])]==sha(plans[0])
    proof=read(EXT_PROOF)
    assert proof['status']=='PASSED_RELEASED_PUBLIC_EXECUTION_CAP_EXTENSION_FULLWIDTH_CPU_REPLAY_AND_REFUSALS'
    assert proof['tests_passed']==4 and proof['CUDA_initialized'] is False and proof['GPU_job_or_allocation_started'] is False
    pins=auth['files_SHA256']
    for group in (proof['source_SHA256'],audit['source_SHA256'],EXT['source_SHA256']):
        for path,pin in group.items():assert pins[path]==pin==sha(path)


def parent_plan_paths():
    plans=[]
    for phase in EXT['charged_prior_phases']:
        parent=read(phase['authorization_path'])
        paths=[Path(arg).resolve().parent/'plan.json' for arg in parent['command'] if Path(arg).name=='worker_gpu0.py']
        assert len(paths)==1 and paths[0].is_relative_to(ROOT) and str(paths[0]) in parent['files_SHA256']
        plans.extend(paths)
    return plans
