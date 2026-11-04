"""Single saved native KA2 coupled EMA readout; generation only, no training."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'ema_runtime';MODEL_DIR=ROOT/'ema_readout_model'
PROTOCOL=ROOT/'analysis/ka2_ema_readout_protocol.json';MODEL_PROOF=ROOT/'analysis/ka2_ema_readout_CPU_qualification.json'
RUNTIME_PROOF=ROOT/'analysis/ka2_ema_readout_runtime_CPU_qualification.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json';WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1';PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
READOUT={'weights':'native_saved_generator_and_prior_EMA','ema_decay':.995,'training_serve_average':0.,'output_noise':False}
CELL={'experiment':'ka2_ema_readout_v1','fold':0,'method':'pg','seed':0}
ENABLED={'enabled':True,'native_recipe':'ka2','readout_config':READOUT,'fixed_endpoint':20000,'pure_only':True,'training_seed':0,'experiment':'ka2_ema_readout_v1','new_model_updates':0}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def protocol_paths(p):
    keys=('original_failed_confirmation_path','uniform_ensemble_negative_screen_path','hypothesis_path','parent_checkpoint_path','parent_result_path','failed_fixed_control_result_path','reverted_fixed_control_plan_path')
    return [PROTOCOL,MODEL_PROOF,ENGINE_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',*[Path(p[k]) for k in keys]]
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['operation']=='evaluate_ka2_ema' and auth['cell']==CELL
    assert auth['cap_seconds']==900 and type(auth['cap_seconds']) is int
    assert auth['fixed_endpoint']==20000 and type(auth['fixed_endpoint']) is int
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='ka2_saved_native_ema_readout' and auth['qualification_only'] is False
    assert auth['initial_completed_updates']==20000 and auth['new_model_updates']==0
    assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['native_recipe']=='ka2' and auth['readout_config']==READOUT
    assert 'prior_update' not in auth and 'frozen_prior_config' not in auth
    assert auth['dynamic_shared_gpu0'] is True
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert read(PLAN)==ENABLED,'saved EMA readout plan remains disabled or changed'
    assert p['task']=={'fold':0,'training_seed':0,'parent_updates':20000,'new_training_updates':0,'cap_seconds':900,'variant_config':CFG}
    assert p['readout_config']==READOUT and p['source_model_directory']==str(MODEL_DIR)
    assert sha(WINDOW)==WINDOW_SHA and w['expires_unix']-w['effective_unix']==432000 and w['owner_instruction']=='extend it to 5 days'
    assert w['project_GPU_hour_cap']==1024 and w['finite_training_phase_cap_seconds']==21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix']<=w['expires_unix'] and auth['expires_unix']<=auth['issued_unix']+930
    assert sha(ROOT/'pure_incumbent.json')==p['current_pure_incumbent_SHA256']==auth['frozen_pure_incumbent_SHA256']
    proof=read(MODEL_PROOF)
    assert proof['status']=='PASSED_ACTUAL_PARENT_CPU_NATIVE_KA2_SAVED_EMA_READOUT_QUALIFICATION' and proof['tests_passed']==4 and proof['CUDA_initialized'] is False
    assert proof['readout_config']==READOUT and proof['coupled_EMA_model_floating_values']==316832 and proof['parent_raw_model_floating_values']==371105
    for key in ('exact_saved_EMA_modules_shape_dtype_finite_and_storage_independent_verified','both_EMA_modules_differ_from_fast_verified','full2000gene_400cell_native_integer_count_sampler_bitwise_replay_verified','incomplete_wrong_parent_cache_seed_recipe_contract_and_corrupt_averages_refused','original_parent_RAW_and_EMA_states_and_CUDA_device_metadata_unchanged_verified','CPU_policy_not_relabelled_or_restored_from_CUDA_policy'):assert proof[key] is True
    assert proof['new_training_updates']==0 and proof['native_default_served_source']=='fast'
    for name in ('parent_checkpoint','parent_result'):
        assert auth[name+'_path']==p[name+'_path']==proof[name+'_path'] and auth[name+'_SHA256']==p[name+'_SHA256']==proof[name+'_SHA256']==sha(p[name+'_path'])
    parent=read(p['parent_result_path']);assert parent['status']=='VERIFIED_PUBLISHED_KA2_NATIVE_RESULT' and parent['owned_release_and_clean_settlement_verified'] and parent['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    assert parent['candidate']['completed_updates']==20000 and parent['candidate']['checkpoint_SHA256']==auth['parent_checkpoint_SHA256']
    engine=read(ENGINE_PROOF);assert engine['tests_passed']==10 and not engine['native_policy_deployed'] and not engine['canonical_reservation_created']
    for path,pin in {**proof['source_SHA256'],**proof['official_wheel_source_SHA256'],**engine['source_SHA256']}.items():assert pins[path]==pin==sha(path)
    for key in ('original_failed_confirmation','uniform_ensemble_negative_screen','hypothesis','parent_checkpoint','parent_result','failed_fixed_control_result','reverted_fixed_control_plan'):assert sha(p[key+'_path'])==p[key+'_SHA256']
    negative=read(p['failed_fixed_control_result_path']);assert negative['decision']['accept'] is False and negative['owned_release_and_clean_settlement_verified']
    assert read(p['reverted_fixed_control_plan_path'])['enabled'] is False
    assert read(p['original_failed_confirmation_path'])['stable_joint_target_passed'] is False
    for path in protocol_paths(p):assert pins[str(path)]==sha(path)
    assert auth['preset_protocol_SHA256']==sha(PROTOCOL) and auth['model_qualification_SHA256']==sha(MODEL_PROOF)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        proof=read(RUNTIME_PROOF);assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        assert proof['status']=='PASSED_ISOLATED_KA2_EMA_READOUT_RUNTIME_CPU_QUALIFICATION'
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged'] and proof['GPU_allocation_created'] is False
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
