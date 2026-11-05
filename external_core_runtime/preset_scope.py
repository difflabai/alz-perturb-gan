"""Frozen external benchmark, with no evaluation-dependent configuration changes."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
MODEL_DIR=ROOT/'analysis/released_tuning_v096'
PROTOCOL=ROOT/'analysis/external_benchmark_frozen_candidate_v096_20261105.json'
PROTOCOL_SHA='7eb3efa1ed4d4dcc70de0fe08f94dfb18754b2d028da2b72d0dc5d59ca097856'
MODEL_PROOF=MODEL_DIR/'CPU_qualification.json';NATIVE_PROOF=MODEL_DIR/'native_release.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=ROOT/'analysis/external_core_runtime_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
PLAN=HERE/'plan.json'
ENABLED={'enabled':True,'benchmark':'external_b1','folds':[0,1,2],'training_seeds':[0,1,2,3,4],
         'methods':['pg','scvi'],'PG_updates':20000,'scVI_epochs':400}
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
HYPER={'generator_lr':.006,'critic_lr':.009,'particle_lr':.015,'reg_coeff':1.,'reg_kappa':.5,'prior_reg':.01}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(fold,method,seed):
    assert type(fold) is int and fold in range(3)
    assert method in ('pg','scvi') and type(seed) is int and seed in range(5)
    return dict(experiment='external_b1',fold=fold,method=method,seed=seed)
def reused(value):return value['method']=='scvi' and value['fold']==value['seed']==0
def model_input(value):
    if not reused(value):return None
    q=read(PROTOCOL)
    return Path(q['PG']['fold0_seed0_checkpoint']) if value['method']=='pg' else ROOT/q['scVI']['fold0_seed0_model']
def protocol_paths(p):
    return [PROTOCOL,MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,PLAN,
            ROOT/'analysis/external_core_pg_model_v096_CPU.json',ROOT/'analysis/external_core_scvi_model_v096_CPU.json',
            ROOT/'analysis/matched_scvi/CPU_model_qualification.json',
            ROOT/'analysis/CAL_GPU0_V1_released_v095_regkappa050_f0_pg_s0-result.json',
            ROOT/'analysis/CAL_GPU0_V1_matched_scvi_eval1_f0_scvi_s0-result.json']
def validate_authorization(auth,*,require_runtime_proof=True):
    value=auth['cell'];assert value==cell(value['fold'],value['method'],value['seed'])
    assert auth['operation']=='external_benchmark_cell' and auth['metric_role']=='final_evaluation'
    assert auth['fixed_endpoint']==(20000 if value['method']=='pg' else 400)
    assert auth['cap_seconds']==21600 and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)==PROTOCOL_SHA
    assert read(PLAN)==ENABLED and sha(WINDOW)==WINDOW_SHA
    assert auth['reuse_completed_model'] is reused(value)
    assert auth['new_model_updates']==(0 if reused(value) else auth['fixed_endpoint'])
    assert auth['particlegan_version']=='0.9.6' and auth['variant_config']==CFG and auth['released_hyperparameters']==HYPER
    assert auth['penalty_options']=={'input_domain':'continuous'}
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
    w=read(WINDOW)
    assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+21630)
    pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    if reused(value):
        path=model_input(value);expected=read(PROTOCOL)['PG' if value['method']=='pg' else 'scVI']['fold0_seed0_checkpoint_SHA256' if value['method']=='pg' else 'fold0_seed0_model_SHA256']
        assert auth['source_model_path']==str(path) and auth['source_model_SHA256']==sha(path)==expected
        assert pins[str(path)]==expected
    else:assert auth.get('source_model_path') is None and auth.get('source_model_SHA256') is None
    for path in protocol_paths(read(PROTOCOL)):assert pins[str(path)]==sha(path)
    proof=read(MODEL_PROOF);assert proof['tests_passed']==8 and proof['native_version']=='0.9.6'
    assert proof['full2000gene_connected_gradients_and_exact_next_pair_replay']
    assert len(read(NATIVE_PROOF)['official_wheel_source_SHA256'])==69
    for method in ('pg','scvi'):
        q=read(ROOT/f'analysis/external_core_{method}_model_v096_CPU.json')
        assert q['status']=='ACTUAL_NATIVE_CPU_RESTORE_AND_FULLWIDTH_COUNT_REPLAY_PASS'
        assert q['exact_repeated_native_count_and_PC_replay'] and q['count_shape']==[1,400,2000]
        assert not q['evaluation_target_values_read'] and not q['CUDA_initialized']
        for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
    for evidence in (proof['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256']):
        for path,pin in evidence.items():assert pins[path]==pin==sha(path)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        q=read(RUNTIME_PROOF);assert q['status']=='EXTERNAL_CORE_CPU_RUNTIME_AND_MODEL_CHECKS_PASS'
        assert q['GPU_allocation_created'] is False
        for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return read(PROTOCOL)
