"""Exact native096 formulation; only six public scalar parameters can vary."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
STUDY=ROOT/'analysis/hpo500';MODEL_DIR=STUDY/'model';PROTOCOL=STUDY/'protocol.json'
MODEL_PROOF=STUDY/'model_CPU_qualification.json'
NATIVE_PROOF=ROOT/'analysis/released_tuning_v096/native_release.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RECOVERY_DIR=ROOT/'analysis/hpo500_recovery_r4';RECOVERY_PROTOCOL=RECOVERY_DIR/'protocol.json';PARENT_PROOF=RECOVERY_DIR/'parent420_CPU_audit.json';RESUME_PROOF=RECOVERY_DIR/'CPU_qualification.json'
RUNTIME_PROOF=RECOVERY_DIR/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
PLAN=HERE/'plan.json';FINALISTS=STUDY/'finalists.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
HYPER={'generator_lr':.006,'critic_lr':.009,'particle_lr':.015,'reg_coeff':1.,'reg_kappa':.5,'prior_reg':.01}
ENABLED={'enabled':True,'experiment':'hpo500','formulation':'native096_bispf_optimized_nonEMA','endpoint':500,'maximum_search_candidates':25,'confirmation_seeds':[1,2,3,4],'calibration_only':True,'owned_GPU_GiB':2,'resource_recovery':'c008_420_to500_GPU1'}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidate_id(value):
    assert isinstance(value,str) and re.fullmatch(r'c(?:00[0-9]|01[0-9]|02[0-4])',value)
    return value
def cell(candidate='c000',seed=0,phase='search',resume=False):
    candidate_id(candidate);assert type(seed) is int
    assert (phase=='search' and seed==0) or (phase=='confirm' and seed in (1,2,3,4))
    assert type(resume) is bool and (not resume or (candidate=='c008' and seed==0 and phase=='search'))
    return dict(experiment=f'h5_{candidate}_{phase}'+('_r2' if resume else ''),fold=0,method='pg',seed=seed)
def proposal_path(candidate):return STUDY/'proposals'/(candidate_id(candidate)+'.json')
def validate_proposal(q):
    candidate_id(q['id']);assert q['endpoint']==500 and q['native_version']=='0.9.6'
    h=q['hyperparameters'];assert set(h)==set(HYPER)
    assert all(type(h[k]) in (int,float) and math.isfinite(h[k]) and HYPER[k]/4<=h[k]<=HYPER[k]*4 for k in HYPER)
    assert q['protocol_SHA256']==sha(PROTOCOL) and q['heldout_results_used'] is False
    return h
def protocol_paths(auth):
    parent=read(PARENT_PROOF)
    paths=[PROTOCOL,MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,PLAN,RECOVERY_PROTOCOL,PARENT_PROOF,RESUME_PROOF,proposal_path(auth['candidate_id'])]+[Path(parent[k]) for k in ('parent_authorization_path','parent_ledger_path','parent_settlement_path','checkpoint_path','learning_curve_path')]
    paths+=[Path(parent['previous_parent_proof_path'])]+[Path(row['settlement_path']) for row in parent['cumulative_parent_charge_lineage']]
    if auth['phase']=='confirm':paths.append(FINALISTS)
    return paths
def validate_authorization(auth,*,require_runtime_proof=True):
    cid=auth['candidate_id'];value=auth['cell'];resume=auth['is_continuation'];assert value==cell(cid,value['seed'],auth['phase'],resume)
    parent=read(PARENT_PROOF);assert auth['operational_protocol_SHA256']==sha(RECOVERY_PROTOCOL)
    assert read(RECOVERY_PROTOCOL)['original_scientific_protocol_SHA256']==sha(PROTOCOL)
    assert parent['status']=='ACTUAL_HPO500_RESOURCE_INTERRUPTION_NATIVE420_CPU_AUDIT_PASS'
    assert parent['owned_processes_and_lease_absent'] and parent['clean_settlement'] and parent['completed_updates']==420
    assert parent['parent_charged_GPU_seconds']==sum(row['charged_GPU_seconds'] for row in parent['cumulative_parent_charge_lineage'])
    assert parent['remaining_trial_cap_seconds']==3600-math.ceil(parent['parent_charged_GPU_seconds'])==3139
    assert read(RESUME_PROOF)['complete_fullwidth_native_state_next_pair_replay'] is True
    h=validate_proposal(read(proposal_path(cid)))
    assert auth['released_hyperparameters']==h and auth['variant_config']==CFG
    assert auth['proposal_SHA256']==sha(proposal_path(cid)) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
    assert auth['operation']==('resume_hpo500_cell' if resume else 'hpo500_cell') and auth['metric_role']=='calibration_selection'
    assert auth['fixed_endpoint']==500 and auth['new_model_updates']==(80 if resume else 500) and auth['initial_completed_updates']==(420 if resume else 0)
    assert auth['reuse_completed_model'] is False
    assert auth['source_model_path']==(parent['checkpoint_path'] if resume else None)
    assert auth['source_model_SHA256']==(parent['checkpoint_SHA256'] if resume else None)
    assert auth['cap_seconds']==(3139 if resume else 3600) and read(PLAN)==ENABLED
    assert auth['particlegan_version']=='0.9.6' and auth['penalty_options']=={'input_domain':'continuous'}
    if auth['phase']=='confirm':assert cid in read(FINALISTS)['candidates']
    assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
    assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
    assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and auth['dynamic_shared_gpu0'] is False
    assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
    assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+auth['cap_seconds']+30)
    if resume:assert auth['expires_unix']<=parent['parent_expiry_unix'] and auth['released_hyperparameters']==parent['hyperparameters']
    pins=auth['files_SHA256'];research=auth['research_source_SHA256']
    for path in protocol_paths(auth):assert pins[str(path)]==sha(path)
    proof=read(MODEL_PROOF);assert proof['status']=='NATIVE096_FIXED_FORMULATION_500STEP_CPU_PASS'
    assert proof['CUDA_initialized'] is False and proof['native_update_byte_identical'] is True
    assert len(read(NATIVE_PROOF)['official_wheel_source_SHA256'])==69
    for evidence in (proof['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256'],read(RESUME_PROOF)['source_SHA256']):
        for path,pin in evidence.items():assert pins[path]==pin==sha(path)
    for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,ROOT/'analysis/released_tuning_v096',RECOVERY_DIR):
        for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
    if require_runtime_proof:
        assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
        q=read(RUNTIME_PROOF);assert q['status']=='HPO500_R4_NATIVE_SCOPE_ACCOUNTING_CPU_PASS' and not q['GPU_allocation_created']
        for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
    return read(PROTOCOL)
