"""Only the predeclared positive kappa values, same native500step formulation."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
BOUNDARY=ROOT/'analysis/hpo500_boundary';STUDY=BOUNDARY
PROTOCOL=BOUNDARY/'prospective_protocol.json';ANCHOR=BOUNDARY/'anchor.json'
ORIGINAL_PROTOCOL=ROOT/'analysis/hpo500/protocol.json'
MODEL_DIR=ROOT/'analysis/hpo500/model';MODEL_PROOF=ROOT/'analysis/hpo500/model_CPU_qualification.json'
NATIVE_PROOF=ROOT/'analysis/released_tuning_v096/native_release.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=BOUNDARY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
PLAN=HERE/'plan.json';FINALISTS=BOUNDARY/'frozen_selection.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
ENABLED={'enabled':True,'experiment':'hpo500_boundary','native_version':'0.9.6','formulation':'native096_bispf_optimized_nonEMA','endpoint':500,'maximum_selection_candidates':6,'confirmation_seeds':[1,2,3,4],'only_parameter_varied':'reg_kappa','owned_GPU_GiB':2,'calibration_only':True}
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def candidate_id(value):assert isinstance(value,str) and re.fullmatch(r'k00[0-5]',value);return value
def cell(candidate='k000',seed=0,phase='search'):
    candidate_id(candidate);assert type(seed) is int
    assert (phase=='search' and seed==0) or (phase=='confirm' and seed in (1,2,3,4))
    return dict(experiment=f'h5_{candidate}_{phase}',fold=0,method='pg',seed=seed)
def proposal_path(candidate):return BOUNDARY/'proposals'/(candidate_id(candidate)+'.json')
def validate_proposal(q):
    candidate_id(q['id']);assert q['native_version']=='0.9.6' and q['endpoint']==500
    assert q['protocol_SHA256']==sha(PROTOCOL) and q['anchor_SHA256']==sha(ANCHOR)
    assert not q['heldout_results_used'] and not q['fresh_confirmation_results_used'] and q['selection_seed']==0
    a=read(ANCHOR);assert a['trigger'] and a['confirmation_result_files_present_at_capture']==0
    h=q['hyperparameters'];base=a['anchor_hyperparameters'];assert set(h)==set(base)
    assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in h.values())
    assert all(h[k]==base[k] for k in base if k!='reg_kappa')
    assert base['reg_kappa']/64<=h['reg_kappa']<base['reg_kappa']
    if int(q['id'][1:])<4:assert h==a['initial_proposals'][int(q['id'][1:])]['hyperparameters']
    assert q['anchor_candidate']==a['anchor_candidate'] and q['only_parameter_changed']=='reg_kappa'
    return h
def protocol_paths(auth):
    a=read(ANCHOR)
    paths=[PROTOCOL,ANCHOR,ORIGINAL_PROTOCOL,MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,PLAN,proposal_path(auth['candidate_id'])]
    paths+=[ROOT/'analysis/hpo500/results'/(key+'.json') for key in a['original25_seed0_result_SHA256']]
    if auth['phase']=='confirm':paths.append(FINALISTS)
    return paths
def validate_authorization(auth,*,require_runtime_proof=True):
    cid=auth['candidate_id'];value=auth['cell'];assert value==cell(cid,value['seed'],auth['phase'])
    a=read(ANCHOR);rule=read(PROTOCOL)
    assert a['prospective_rule_SHA256']==sha(PROTOCOL) and rule['original_scientific_protocol_SHA256']==sha(ORIGINAL_PROTOCOL)
    assert auth['released_hyperparameters']==validate_proposal(read(proposal_path(cid))) and auth['variant_config']==CFG
    assert auth['proposal_SHA256']==sha(proposal_path(cid)) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
    assert auth['operation']=='hpo500_cell' and auth['metric_role']=='calibration_selection'
    assert auth['fixed_endpoint']==500 and auth['new_model_updates']==500 and auth['initial_completed_updates']==0
    assert auth['is_continuation'] is False and auth['reuse_completed_model'] is False
    assert auth['source_model_path'] is auth['source_model_SHA256'] is None
    assert auth['cap_seconds']==3600 and read(PLAN)==ENABLED
    assert auth['particlegan_version']=='0.9.6' and auth['penalty_options']=={'input_domain':'continuous'}
    if auth['phase']=='confirm':assert cid==read(FINALISTS)['candidate_id'] and read(FINALISTS)['frozen_before_new_confirmation']
    assert not auth['cpu_fixture'] and not auth['independent_current_device_review_claimed']
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and auth['dynamic_shared_gpu0'] is False
    assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
    assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+3630)
    pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    for path in protocol_paths(auth):assert pins[str(path)]==sha(path)
    for key,pin in a['original25_seed0_result_SHA256'].items():assert sha(ROOT/'analysis/hpo500/results'/(key+'.json'))==pin
    proof=read(MODEL_PROOF);assert proof['status']=='NATIVE096_FIXED_FORMULATION_500STEP_CPU_PASS' and proof['CUDA_initialized'] is False and proof['native_update_byte_identical']
    for evidence in (proof['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256']):
        for path,pin in evidence.items():assert pins[path]==pin==sha(path)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,ROOT/'analysis/released_tuning_v096',BOUNDARY):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        q=read(RUNTIME_PROOF);assert q['status']=='HPO500_BOUNDARY_NATIVE_SCOPE_ACCOUNTING_CPU_PASS' and not q['GPU_allocation_created']
        for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return rule
