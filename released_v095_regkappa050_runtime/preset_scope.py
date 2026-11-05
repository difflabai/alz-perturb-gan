"""One explicit released-API configuration from the registered rate grid."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'released_v095_regkappa050_runtime'
MODEL_DIR=ROOT/'analysis/released_tuning_v095'
PROTOCOL=ROOT/'analysis/released_penalty_v095/protocol.json'
MODEL_PROOF=MODEL_DIR/'CPU_qualification.json'
NATIVE_PROOF=MODEL_DIR/'native_release.json'
PARAMETERIZATION=MODEL_DIR/'rate_parameterization_and_controls.json'
COMPAT_PROOF=ROOT/'analysis/released_tuning/historical_native_checkpoint_CPU_audit.json'
RUNTIME_PROOF=ROOT/'analysis/released_v095_regkappa050_runtime_CPU_qualification.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
PENALTY_PROOF=ROOT/'analysis/released_penalty_v095/CPU_qualification.json'
REFERENCE=ROOT/'analysis/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0-result.json'
REFERENCE_PLAN=ROOT/'released_v095_bridge_runtime/plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
CANDIDATE_ID='v095_penalty_reg_kappa_m050'
HYPER={'generator_lr': 0.006, 'critic_lr': 0.009, 'particle_lr': 0.015, 'reg_coeff': 1.0, 'reg_kappa': 0.5, 'prior_reg': 0.01}
CELL={'experiment':'released_v095_regkappa050','fold':0,'method':'pg','seed':0}
ENABLED={'enabled':True,'native_recipe':'bispf_optimized','fixed_endpoint':20000,'pure_only':True,
    'training_seed':0,'experiment':CELL['experiment'],'candidate_id':CANDIDATE_ID}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol_paths(p):
    return [PENALTY_PROOF,REFERENCE,REFERENCE_PLAN,PROTOCOL,MODEL_PROOF,NATIVE_PROOF,PARAMETERIZATION,COMPAT_PROOF,ENGINE_PROOF,WINDOW,PLAN,
        ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'selection.py',
        Path(p['frozen_science']['fixed_confirmation_path']),Path(p['native_latest_verification_path']),
        Path('/ml2/alz/analysis/released_v095_regkappa050_initial_resource_observation.json')]
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['particlegan_version']=='0.9.5' and auth['penalty_options']=={'input_domain':'continuous'}
    assert auth['operation']=='train_released_hyperparameters' and auth['cell']==CELL
    assert type(auth['cap_seconds']) is int and auth['cap_seconds']==21600
    assert type(auth['fixed_endpoint']) is int and auth['fixed_endpoint']==20000
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='released_hyperparameters' and auth['qualification_only'] is False
    assert auth['native_recipe']=='bispf_optimized' and auth['released_hyperparameters']==HYPER
    assert auth['tuning_candidate_id']==CANDIDATE_ID and auth['initial_completed_updates']==0 and auth['new_model_updates']==20000
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
    for k in ('checkpoint_path','checkpoint_SHA256','prior_update','stage_prior_config','halloween_native_config','continuation','routed_config'):
        assert k not in auth
    assert read(PLAN)==ENABLED,'released rate plan remains disabled or changed'
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert p['status']=='PROSPECTIVE_OFFICIAL095_TARGETED_PENALTY_FOLLOWUP_NOT_LAUNCHED'
    settings=[s for s in p['stage1']['configs'] if s['id']==CANDIDATE_ID];assert len(settings)==1 and settings[0]['parameters']==HYPER
    assert len(p['stage1']['configs'])==12 and p['frozen_science']['evaluation_used_for_tuning'] is False
    assert p['convergence_evidence']['checkpoints_at_updates']==[5000,10000,20000]
    assert sha(WINDOW)==WINDOW_SHA and w['expires_unix']-w['effective_unix']==432000
    assert w['project_GPU_hour_cap']==1024 and w['finite_training_phase_cap_seconds']==21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+21630)
    assert sha(ROOT/'pure_incumbent.json')==p['frozen_science']['pure_incumbent_SHA256']==auth['frozen_pure_incumbent_SHA256']
    ref=read(REFERENCE)
    assert ref['owned_release_and_clean_settlement_verified'] is True and ref['official095_origin_version_and_continuous_domain_verified'] is True
    assert ref['decision']['accept'] is False and read(REFERENCE_PLAN)['enabled'] is False
    assert ref['candidate']['completed_updates']==ref['candidate']['fixed_endpoint']==20000
    assert ref['candidate']['particlegan_version']=='0.9.5' and ref['candidate']['released_hyperparameters']==p['baseline_parameters']
    penalty=read(PENALTY_PROOF)
    assert penalty['status']=='ALL12_OFFICIAL095_ONE_FIELD_PENALTY_CONTROLS_FULLWIDTH_CPU_REPLAY_PASS' and penalty['configurations_checked']==12
    assert penalty['actual_optimizer_pairs_executed']==36 and penalty['CUDA_initialized'] is False and penalty['GPU_job_created'] is False
    tested=[x for x in penalty['checks'] if x['candidate_id']==CANDIDATE_ID]
    assert len(tested)==1 and tested[0]['parameters']==HYPER and tested[0]['complete_next_pair_state_byte_exact'] is True
    for path,pin in penalty['source_SHA256'].items():assert pins[path]==pin==sha(path)
    proof=read(MODEL_PROOF)
    assert proof['status']=='PASSED_ACTUAL_RELEASED_PARAMETER_SYNTHETIC_CPU_QUALIFICATION'
    assert proof['native_version']=='0.9.5' and proof['explicit_continuous_domain_arithmetic_and_version_bound_restore_verified'] is True
    assert proof['tests_passed']==8 and proof['CUDA_initialized'] is False and proof['GPU_job_created'] is False
    for k in ('all27_declared_native_recipes_and_optimizer_rates_verified','baseline_PG8_two_update_complete_state_byte_parity',
            'same_native_update_observation_context_and_seed_streams_AST','rate_and_regularization_extremes_connected_gradients_and_full_state_replay',
            'full2000gene_connected_gradients_and_exact_next_pair_replay','wrong_hyperparameter_clock_status_recipe_cache_configuration_and_CPU_fit_refused'):
        assert proof[k] is True
    assert read(PARAMETERIZATION)['CPU_qualification_SHA256']==sha(MODEL_PROOF)
    compatibility=read(COMPAT_PROOF)
    assert compatibility['status']=='PASSED_ACTUAL_HISTORICAL_NATIVE_CUDA_CHECKPOINT_CPU_MODEL_OPTIMIZER_COMPATIBILITY_AUDIT'
    assert compatibility['fullwidth_model_floating_values']==371177 and compatibility['CUDA_initialized'] is False
    for path,pin in compatibility['source_SHA256'].items():assert pins[path]==pin==sha(path)
    native=read(NATIVE_PROOF)['official_wheel_source_SHA256'];assert len(native)==68
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
        assert proof['status']=='PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION'
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged']
        assert proof['GPU_allocation_created'] is False and proof['prior_active_trial_native_launch_refusal_verified'] is True
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
