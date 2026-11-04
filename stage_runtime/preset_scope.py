"""One bounded native finite fixed-particle control; no implicit experiment grid."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'stage_runtime';MODEL_DIR=ROOT/'stage_model'
PROTOCOL=ROOT/'analysis/stage_prior_protocol.json'
MODEL_PROOF=ROOT/'analysis/stage_prior_model_CPU_qualification.json'
RUNTIME_PROOF=ROOT/'analysis/stage_prior_runtime_CPU_qualification.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
STAGE={'conditioning': 'known_ADNC_stage', 'stages': 4, 'particles_per_stage': 128, 'num_particles': 512, 'partition': 'contiguous_fixed_equal_groups', 'sampling': 'uniform_with_replacement_within_stage'}
CELL={'experiment':'stage_prior_v1','fold':0,'method':'pg','seed':0}
ENABLED={'enabled':True,'native_recipe':'bispf_optimized','fixed_endpoint':20000,'pure_only':True,'training_seed':0,'experiment':'stage_prior_v1','stage_prior_config':STAGE}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def protocol_paths(p):
    return [PROTOCOL,MODEL_PROOF,ENGINE_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',*[Path(p[k]) for k in ('original_failed_confirmation_path','uniform_ensemble_negative_screen_path','hypothesis_path','failed_ema_result_path','reverted_ema_plan_path')]]
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['operation']=='train_stage_prior' and auth['cell']==CELL
    assert auth['cap_seconds']==21600 and type(auth['cap_seconds']) is int
    assert auth['fixed_endpoint']==20000 and type(auth['fixed_endpoint']) is int
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='stage_conditioned_particle_prior' and auth['qualification_only'] is False
    assert auth['initial_completed_updates']==0 and auth['new_model_updates']==20000
    assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['native_recipe']=='bispf_optimized' and auth['stage_prior_config']==STAGE and 'prior_update' not in auth
    assert auth['dynamic_shared_gpu0'] is True
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert read(PLAN)==ENABLED,'native stage-conditioned prior plan remains disabled or changed'
    assert p['native_API']=="ParticleGAN0.9.4 get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8); taskstageindexlaw only"
    assert p['task']=={'fold':0,'training_seed':0,'updates':20000,'cap_seconds':21600,'variant_config':CFG}
    assert p['stage_prior_config']==STAGE and p['source_model_directory']==str(MODEL_DIR)
    assert sha(WINDOW)==WINDOW_SHA and w['expires_unix']-w['effective_unix']==432000
    assert w['status']=='OWNER_EXPLICITLY_EXTENDED_RESEARCH_WINDOW' and w['owner_instruction']=='extend it to 5 days'
    assert w['project_GPU_hour_cap']==1024 and w['finite_training_phase_cap_seconds']==21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix']<=w['expires_unix'] and auth['expires_unix']<=auth['issued_unix']+21630
    assert sha(ROOT/'pure_incumbent.json')==p['current_pure_incumbent_SHA256']==auth['frozen_pure_incumbent_SHA256']
    proof=read(MODEL_PROOF)
    assert proof['status']=='PASSED_SYNTHETIC_CPU_NATIVE_STAGE_PRIOR_MODEL_QUALIFICATION' and proof['tests_passed']==7 and proof['CUDA_initialized'] is False
    assert proof['model_source_directory']==str(MODEL_DIR) and proof['stage_prior_config']==STAGE and proof['full_model_floating_values']==371177
    for key in ('initial_G_D_rawPz_recipe_old11streams_and_capacity_match_verified','all4_stage_support_replacement_native_forward_and_RNG_advance_verified','native_G_D_dispersion_P_gradients_and_sampled_row_ownership_verified','complete_serialized_next_update_bitwise_and_config_cache_refusals_verified','Context_fit_streams_AST_and_only_two_sampling_calls_in_update_change_verified','invalid_condition_config_and_CPU_fitting_refusals_verified','full2000gene_update_group_count_and_unchanged_generation_view_contract_verified'):assert proof[key] is True
    engine=read(ENGINE_PROOF)
    assert engine['status']=='PASSED_CPU_SHARED_GPU0_ENGINE_CANDIDATE_ONLY_WITH_CURRENT_READ_ONLY_FORECAST' and engine['tests_passed']==10
    assert not engine['native_policy_deployed'] and not engine['canonical_reservation_created'] and not engine['GPU_context_created']
    for path,pin in {**proof['source_SHA256'],**proof['official_wheel_source_SHA256'],**engine['source_SHA256']}.items():assert pins[path]==pin==sha(path)
    for key in ('original_failed_confirmation','uniform_ensemble_negative_screen','hypothesis','failed_ema_result','reverted_ema_plan'):
        assert sha(p[key+'_path'])==p[key+'_SHA256']
    assert read(p['original_failed_confirmation_path'])['stable_joint_target_passed'] is False
    assert read(p['uniform_ensemble_negative_screen_path'])['joint_analytic_screen_passed'] is False
    negative=read(p['failed_ema_result_path'])
    assert negative['status']=='VERIFIED_NATIVE_KA2_SAVED_EMA_READOUT_RESULT' and negative['decision']['accept'] is False
    assert negative['owned_release_and_clean_settlement_verified'] and read(p['reverted_ema_plan_path'])['enabled'] is False
    for path in protocol_paths(p):assert pins[str(path)]==sha(path)
    assert auth['preset_protocol_SHA256']==sha(PROTOCOL) and auth['model_qualification_SHA256']==sha(MODEL_PROOF)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        proof=read(RUNTIME_PROOF)
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        assert proof['status']=='PASSED_ISOLATED_STAGE_PRIOR_RUNTIME_CPU_QUALIFICATION'
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged'] and proof['GPU_allocation_created'] is False
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
