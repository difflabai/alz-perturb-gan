"""One explicitly bounded fixed Fourier128 observation trial; no implicit experiment grid."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'fourier128_runtime'
PROTOCOL=ROOT/'analysis/fourier128_protocol.json'
MODEL_PROOF=ROOT/'analysis/fourier128_model_CPU_qualification.json'
RUNTIME_PROOF=ROOT/'analysis/fourier128_runtime_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
OBS={'PC_dimensions': 50, 'moment_dimensions': 100, 'frequencies': 128, 'frequency_seed': 20261030, 'features': 356, 'hidden_width': 128}
CELL={'experiment':'fourier128_v1','fold':0,'method':'pg','seed':0}
ENABLED={'enabled': True, 'native_recipe': 'bispf_optimized', 'fixed_endpoint': 20000, 'pure_only': True, 'training_seed': 0, 'experiment': 'fourier128_v1', 'fourier_observation_config': {'PC_dimensions': 50, 'moment_dimensions': 100, 'frequencies': 128, 'frequency_seed': 20261030, 'features': 356, 'hidden_width': 128}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['operation']=='train_fourier128' and auth['cell']==CELL
    assert auth['cap_seconds']==21600 and type(auth['cap_seconds']) is int
    assert auth['fixed_endpoint']==20000 and type(auth['fixed_endpoint']) is int
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='fourier128' and auth['qualification_only'] is False
    assert auth['initial_completed_updates']==0 and auth['new_model_updates']==20000
    assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert read(PLAN)==ENABLED,'Fourier128 trial plan remains disabled or changed'
    assert p['native_API']=="ParticleGAN0.9.4 get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8); no other recipe overrides"
    assert p['task']=={'fold':0,'training_seed':0,'updates':20000,'cap_seconds':21600,'variant_config':CFG}
    assert sha(WINDOW)==WINDOW_SHA and w['expires_unix']-w['effective_unix']==432000
    assert w['status']=='OWNER_EXPLICITLY_EXTENDED_RESEARCH_WINDOW' and w['owner_instruction']=='extend it to 5 days'
    assert w['duration_seconds']==432000 and w['project_GPU_hour_cap']==1024 and w['finite_training_phase_cap_seconds']==21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix']<=w['expires_unix'] and auth['expires_unix']<=auth['issued_unix']+21630
    assert sha(ROOT/'pure_incumbent.json')==p['current_pure_incumbent_SHA256']==auth['frozen_pure_incumbent_SHA256']
    proof=read(MODEL_PROOF)
    assert proof['status']=='PASSED_SYNTHETIC_CPU_FOURIER128_MODEL_QUALIFICATION' and proof['tests_passed']==6 and proof['CUDA_initialized'] is False
    assert auth['fourier_observation_config']==p['fourier_observation_config']==proof['observation_config']==OBS
    assert sha(p['hypothesis_path'])==p['hypothesis_SHA256']
    for key in ('first100_moment_coordinates_byteidentical','permutation_invariance_and_unit_fourier_block_bound_verified','official_recipe_components_initial_G_prior_and_streams_verified','critic_input_capacity_change_declared','native_update_context_batch_and_fit_AST_unchanged','isolated_new_Fourier_block_G_dispersion_prior_gradients_verified','finite_nonzero_native_update_gradients_verified','full_serialized_next_update_bitwise_equal','wrong_observation_recipe_cache_architecture_refusals_verified'):assert proof[key] is True
    for path,pin in {**proof['source_SHA256'],**proof['official_wheel_source_SHA256']}.items():assert pins[path]==pin==sha(path)
    paths=[PROTOCOL,MODEL_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',Path(p['original_failed_confirmation_path']),Path(p['uniform_ensemble_negative_screen_path']),Path(p['hypothesis_path']),Path(p['failed_BCAP_Adam_result_path'])]
    assert sha(p['failed_BCAP_Adam_result_path'])==p['failed_BCAP_Adam_result_SHA256']
    assert read(p['failed_BCAP_Adam_result_path'])['decision']['accept'] is False
    assert sha(p['original_failed_confirmation_path'])==p['original_failed_confirmation_SHA256']
    assert sha(p['uniform_ensemble_negative_screen_path'])==p['uniform_ensemble_negative_screen_SHA256']
    assert read(p['original_failed_confirmation_path'])['stable_joint_target_passed'] is False
    assert read(p['uniform_ensemble_negative_screen_path'])['joint_analytic_screen_passed'] is False
    for path in paths:assert pins[str(path)]==sha(path)
    assert auth['preset_protocol_SHA256']==sha(PROTOCOL) and auth['model_qualification_SHA256']==sha(MODEL_PROOF)
    for folder in (ROOT,ROOT/'model',HERE):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        proof=read(RUNTIME_PROOF)
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        assert proof['status']=='PASSED_ISOLATED_FOURIER128_RUNTIME_CPU_QUALIFICATION'
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged'] and proof['GPU_allocation_created'] is False
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
