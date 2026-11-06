"""One exploratory matched duration extension; no external benchmark retuning."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
MODEL_DIR=ROOT/'analysis/released_tuning_v096';EXT_DIR=ROOT/'analysis/duration096'
PROTOCOL=EXT_DIR/'protocol.json';MODEL_PROOF=MODEL_DIR/'CPU_qualification.json'
NATIVE_PROOF=MODEL_DIR/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
EXT_PROOF=EXT_DIR/'CPU_qualification.json';PARENT_PROOF=EXT_DIR/'parent_CPU_audit.json'
RUNTIME_PROOF=ROOT/'analysis/duration096_runtime_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json';PLAN=HERE/'plan.json'
WINDOW_SHA='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
HYPER={'generator_lr':.006,'critic_lr':.009,'particle_lr':.015,'reg_coeff':1.,'reg_kappa':.5,'prior_reg':.01}
ENABLED={'enabled':True,'experiment':'duration096','fold':0,'seed':0,'initial_updates':20000,'endpoint':40000,'calibration_only':True}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(fold,method,seed):
    assert type(fold) is int and fold==0 and method=='pg' and type(seed) is int and seed==0
    return dict(experiment='duration096',fold=fold,method=method,seed=seed)
def reused(value):return True
def model_input(value):return Path(read(PROTOCOL)['extension']['checkpoint_path'])
def protocol_paths(p):
    return [PROTOCOL,MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,EXT_PROOF,PARENT_PROOF,WINDOW,PLAN,
            Path(p['parent_audit_path']),model_input(None),Path(p['extension']['training_log']['immutable_copy_path'])]
def validate_authorization(auth,*,require_runtime_proof=True):
    value=auth['cell'];assert value==cell(value['fold'],value['method'],value['seed'])
    assert auth['operation']=='duration_calibration' and auth['metric_role']=='calibration_selection'
    assert auth['fixed_endpoint']==40000 and auth['initial_completed_updates']==auth['new_model_updates']==20000
    assert auth['cap_seconds']==21600 and read(PLAN)==ENABLED
    p=read(PROTOCOL);assert p['status']=='PROSPECTIVE_MATCHED_NATIVE096_DURATION_EXPERIMENT'
    assert auth['benchmark_protocol_SHA256']==sha(PROTOCOL) and auth['extension']==p['extension']
    assert p['training_endpoints_to_score']==[20000,30000,40000] and p['generation_replicates']==[0,1]
    assert p['ParticleGAN_version']=='0.9.6' and p['EMA_decay']==0 and p['hyperparameters']==HYPER and p['variant_config']==CFG
    assert auth['reuse_completed_model'] is True and auth['source_model_path']==str(model_input(value))
    assert auth['source_model_SHA256']==sha(model_input(value))==p['extension']['checkpoint_SHA256']
    assert sha(p['parent_audit_path'])==p['parent_audit_SHA256']
    parent=read(PARENT_PROOF);assert parent['status']=='ACTUAL_NATIVE096_COMPLETE20K_PARENT_CPU_COMPONENT_AUDIT_PASS'
    assert p['extension']['whole_parent_scientific_fingerprint']==parent['whole_parent_scientific_fingerprint']
    assert parent['native_device_metadata_relabelled'] is False and parent['CUDA_initialized'] is False
    assert auth['particlegan_version']=='0.9.6' and auth['variant_config']==CFG and auth['released_hyperparameters']==HYPER
    assert auth['penalty_options']=={'input_domain':'continuous'}
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
    w=read(WINDOW);assert sha(WINDOW)==WINDOW_SHA
    assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+21630)
    pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    for path in protocol_paths(p):assert pins[str(path)]==sha(path)
    proof=read(MODEL_PROOF);assert proof['tests_passed']==8 and proof['native_version']=='0.9.6'
    ext=read(EXT_PROOF);assert ext['tests_passed']==4 and ext['CUDA_initialized'] is False
    assert ext['full2000gene_native_over_horizon_complete_state_replay'] is True
    assert len(read(NATIVE_PROOF)['official_wheel_source_SHA256'])==69
    for evidence in (proof['source_SHA256'],ext['source_SHA256'],parent['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256']):
        for path,pin in evidence.items():assert pins[path]==pin==sha(path)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,EXT_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        q=read(RUNTIME_PROOF);assert q['status']=='DURATION096_CPU_RUNTIME_AND_NATIVE_EXTENSION_CHECKS_PASS'
        assert q['GPU_allocation_created'] is False
        for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return p
