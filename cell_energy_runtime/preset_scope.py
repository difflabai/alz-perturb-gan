"""One explicitly bounded explicit ALZ cell-energy task-loss trial; no implicit experiment grid."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'cell_energy_runtime'
PROTOCOL=ROOT/'analysis/cell_energy_protocol.json'
MODEL_PROOF=ROOT/'analysis/cell_energy_model_CPU_qualification.json'
RUNTIME_PROOF=ROOT/'analysis/cell_energy_runtime_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
AUX={'cells_per_bag': 4, 'coefficient': 1.0, 'equal_stage_importance': True, 'exclude_same_bag_blocks': True, 'normalize_by_PC_dimensions_sqrt': True, 'PC_dimensions': 50}
OBJECTIVE='ALZ_stage_balanced_pooled_cell_energy_auxiliary'
CELL={'experiment':'cell_energy_aux_v1','fold':0,'method':'pg','seed':0}
ENABLED={'enabled':True,'native_recipe':'bispf_optimized','fixed_endpoint':20000,'pure_only':True,'training_seed':0,'experiment':'cell_energy_aux_v1','task_objective':OBJECTIVE,'cell_energy_config':AUX}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def validate_authorization(auth,*,require_runtime_proof=True):
    assert auth['operation']=='train_cell_energy' and auth['cell']==CELL
    assert auth['cap_seconds']==21600 and type(auth['cap_seconds']) is int
    assert auth['fixed_endpoint']==20000 and type(auth['fixed_endpoint']) is int
    assert auth['variant_config']==CFG and auth['hybrid_config'] is None
    assert auth['pure_formulation']=='cell_energy_task_extension' and auth['qualification_only'] is False
    assert auth['initial_completed_updates']==0 and auth['new_model_updates']==20000
    assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['native_recipe']=='bispf_optimized' and auth['task_objective']==OBJECTIVE and auth['cell_energy_config']==AUX
    p=read(PROTOCOL);w=read(WINDOW);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    assert read(PLAN)==ENABLED,'cell-energy task extension trial plan remains disabled or changed'
    assert p['native_API']=="ParticleGAN0.9.4 get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8); no other recipe overrides"
    assert p['task']=={'fold':0,'training_seed':0,'updates':20000,'cap_seconds':21600,'variant_config':CFG}
    assert sha(WINDOW)==WINDOW_SHA and w['expires_unix']-w['effective_unix']==432000
    assert w['status']=='OWNER_EXPLICITLY_EXTENDED_RESEARCH_WINDOW' and w['owner_instruction']=='extend it to 5 days'
    assert w['duration_seconds']==432000 and w['project_GPU_hour_cap']==1024 and w['finite_training_phase_cap_seconds']==21600
    assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
    assert auth['expires_unix']<=w['expires_unix'] and auth['expires_unix']<=auth['issued_unix']+21630
    assert sha(ROOT/'pure_incumbent.json')==p['current_pure_incumbent_SHA256']==auth['frozen_pure_incumbent_SHA256']
    proof=read(MODEL_PROOF)
    assert proof['status']=='PASSED_CPU_FINAL_PATH_CELL_ENERGY_TASK_EXTENSION_QUALIFICATION' and proof['tests_passed']==11 and proof['CUDA_initialized'] is False
    assert proof['math_tests_passed']==5 and proof['task_loop_tests_passed']==6 and proof['cell_energy_config']==p['cell_energy_config']==AUX
    assert p['task_objective']==OBJECTIVE and proof['final_import_paths_and_only_helper_import_rename_verified'] is True
    for key in ('unchanged_native_recipe_factories_and_initial_G_D_prior_tensors_verified','unchanged_fitting_draws_and_native_first_critic_update_verified','no_additional_random_draws_all_first_update_stream_states_equal','auxiliary_alone_finite_nonzero_G_dispersion_prior_gradients_verified','auxiliary_changes_G_update_verified','complete_serialized_next_update_bitwise_equal','task_objective_fitting_stage_counts_recipe_config_cache_refusals_verified','exact_population_energy_identity_and_cluster_bias_counterexample_verified'):assert proof[key] is True
    for path,pin in {**proof['source_SHA256'],**proof['official_wheel_source_SHA256']}.items():assert pins[path]==pin==sha(path)
    paths=[PROTOCOL,MODEL_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',Path(p['original_failed_confirmation_path']),Path(p['uniform_ensemble_negative_screen_path'])]
    assert sha(p['original_failed_confirmation_path'])==p['original_failed_confirmation_SHA256']
    assert sha(p['uniform_ensemble_negative_screen_path'])==p['uniform_ensemble_negative_screen_SHA256']
    assert read(p['original_failed_confirmation_path'])['stable_joint_target_passed'] is False
    assert read(p['uniform_ensemble_negative_screen_path'])['joint_analytic_screen_passed'] is False
    paths.extend([Path(p['hypothesis_path']),Path(p['failed_K3P_result_path'])])
    assert sha(p['hypothesis_path'])==p['hypothesis_SHA256'] and sha(p['failed_K3P_result_path'])==p['failed_K3P_result_SHA256']
    assert read(p['failed_K3P_result_path'])['decision']['accept'] is False
    for path in paths:assert pins[str(path)]==sha(path)
    assert auth['preset_protocol_SHA256']==sha(PROTOCOL) and auth['model_qualification_SHA256']==sha(MODEL_PROOF)
    for folder in (ROOT,ROOT/'model',HERE):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        proof=read(RUNTIME_PROOF)
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        assert proof['status']=='PASSED_ISOLATED_CELL_ENERGY_RUNTIME_CPU_QUALIFICATION'
        assert proof['native_profile_wire_and_refusals_verified'] and proof['old_execution_sources_unchanged'] and proof['GPU_allocation_created'] is False
        for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
