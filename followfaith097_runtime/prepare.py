"""Prepare one registered external benchmark cell under canonical accounting."""
import argparse,copy,datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope,resource_profile as profile

def build_authorization(*,candidate='t000',seed=0,phase='search',check_only=False,auth_path=None):
    base=b.read(ROOT/'artifacts/allocations/CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0/authorization.json')
    value=scope.cell(candidate,seed,phase);method='pg';folder=ROOT/'artifacts/allocations'/b.attempt_key(value)
    auth_path=Path(auth_path or folder/'authorization.json');now=time.time()
    pins={p:v for p,v in base['files_SHA256'].items() if not Path(p).is_relative_to(ROOT)}
    auth={k:copy.deepcopy(base[k]) for k in ('status','cpu_fixture','machine_id','source_review_passed','project_root','science_root','calibration_manifest_SHA256','core_GPU_hour_ceiling')}
    model_input=None;cap=21600;proposal=scope.read(scope.proposal_path(candidate));hyper=scope.validate_proposal(proposal)
    auth.update(cell=value,allocation_id=b.allocation_key(value),attempt_id=b.attempt_key(value),label=b.attempt_key(value),
      scheduler_allocation_id='ALZ_EXTERNAL_B1_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),issued_unix=now,
      expires_unix=min(now+cap+30,b.read(scope.WINDOW)['expires_unix']),cap_seconds=cap,
      role='alz_core_'+method,operation='followfaith097_cell',metric_role='calibration_selection',
      fixed_endpoint=20000,new_model_updates=20000,initial_completed_updates=0,is_continuation=False,operational_protocol_SHA256=scope.sha(scope.PROTOCOL),candidate_id=candidate,phase=phase,proposal_SHA256=scope.sha(scope.proposal_path(candidate)),
      reuse_completed_model=False,source_model_path=str(model_input) if model_input else None,
      source_model_SHA256=b.sha(model_input) if model_input else None,
      cache_dir=str(b.cache_path(b.NATIVE_LAYOUT,value)),output_dir=str(b.output_path(b.NATIVE_LAYOUT,value)),ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT,value)),
      particlegan_version='0.9.7',penalty_options={'input_domain':'continuous'},variant_config=scope.CFG,released_hyperparameters=hyper,
      benchmark_protocol_SHA256=b.sha(scope.PROTOCOL),files_SHA256=pins,
      candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      source_review_type='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS',independent_current_device_review_claimed=False)
    auth['command']=b.expected_command(b.NATIVE_LAYOUT,auth_path,value)
    profile.configure(auth,check_only=check_only)
    paths=[p for folder_source in (ROOT,ROOT/'model',HERE,scope.MODEL_DIR,ROOT/'analysis/released_tuning_v096',scope.STUDY) for p in folder_source.glob('*.py')]
    paths.extend(ROOT/'recovery_runtime'/n for n in ('parallel_runner.py','gpu1_admission.py','gpu1_runner.py','shared_gpu1_admission.py','shared_gpu1_runner.py'))
    paths.extend(ROOT/'analysis/shared_gpu0_candidate'/n for n in ('shared_gpu0_admission.py','shared_gpu0_runner.py'))
    research={str(p):b.sha(p) for p in paths};auth['research_source_SHA256']=research;pins.update(research)
    for proof in (scope.MODEL_PROOF,scope.HPO_PROOF,scope.ENGINE_PROOF,scope.BASELINE_PROOF,scope.SCALAR_PROOF):pins.update(b.read(proof)['source_SHA256'])
    pins.update(b.read(scope.BASELINE_PROOF)['files_SHA256'])
    native=b.read(scope.NATIVE_PROOF);pins.update(native['official_wheel_source_SHA256'])
    pins[native['package_metadata_path']]=native['package_metadata_SHA256']
    paths=scope.protocol_paths(auth)+[b.runtime_path(b.NATIVE_LAYOUT,value),b.runtime_path(b.NATIVE_LAYOUT,value).parent.parent/'pyvenv.cfg',b.PG_SHARED_PTH]
    if scope.RUNTIME_PROOF.exists():
        auth['runtime_qualification_SHA256']=b.sha(scope.RUNTIME_PROOF);paths.append(scope.RUNTIME_PROOF)
        for p,pin in b.read(scope.RUNTIME_PROOF)['source_SHA256'].items():pins.setdefault(p,pin)
    cache=Path(auth['cache_dir']);paths.append(cache/'manifest.json')
    pins.update({str(cache/name):pin for name,pin in b.read(cache/'manifest.json')['files_SHA256'].items()})
    data=ROOT/'data/calibration_fold0';paths.append(data/'manifest.json')
    pins.update({str(data/name):pin for name,pin in b.read(data/'manifest.json')['files_SHA256'].items()})
    if model_input:
        paths.append(model_input)
        if method=='scvi':paths.extend(p for p in model_input.parent.parent.rglob('*') if p.is_file())
    for p in paths:pins[str(p)]=b.sha(p)
    return auth,auth_path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='t000');parser.add_argument('--phase',choices=['search'],default='search');parser.add_argument('--seed',type=int,default=0);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if not args.check_only:
        profile.reviewed()
        b.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Commit qualified source/evidence before deployment')
    auth,path=build_authorization(candidate=args.candidate,phase=args.phase,seed=args.seed,check_only=args.check_only)
    b.require(auth['expires_unix']>time.time()+150,'Research window lacks cleanup interval')
    b.require(not path.parent.exists(),'One-use cell already exists; inspect rather than duplicate')
    if not args.check_only:
        profile.DEPLOYED.parent.mkdir(parents=True,exist_ok=True)
        if profile.DEPLOYED.exists():b.require(profile.DEPLOYED.read_bytes()==profile.deployed_bytes(),'Existing policy differs')
        else:
            with profile.DEPLOYED.open('xb') as f:f.write(profile.deployed_bytes())
    b.source_closure(auth,b.NATIVE_LAYOUT)
    controller,admission,_=b.modules(auth);observation=admission.snapshot()
    forecast=profile.capacity(observation,admission);profile.free_floor()
    reg=profile.REGISTRY/'registry.json';b.require(not reg.exists() or not b.read(reg)['jobs'],'Own ALZ namespace still active')
    if args.check_only:
        print(json.dumps(dict(status='EXTERNAL_CORE_CHECK_ONLY_SOURCE_AND_RESOURCE_PASS',cell=auth['cell'],GPU_fits=forecast['GPU_fits'],host_fits=forecast['host_fits'],new_GPU_allocation=False)));return
    pins=auth['files_SHA256'];backend=b.load_pinned(HERE/'project_runner_gpu0.py',pins[str(HERE/'project_runner_gpu0.py')],'project_runner_gpu0')
    metadata=dict(status='REVIEWED_FOR_BOUNDED_EXECUTION',path=str(HERE/'project_runner_gpu0.py'),source_SHA256={str(p):pins[str(p)] for p in [*HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']},files_SHA256=pins,policy_SHA256=auth['policy_SHA256'],**{k:auth[k] for k in ('allocation_id','attempt_id','scheduler_allocation_id','expires_unix')})
    control=controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'],'Canonical reservation still active')
        b.unseen(auth,b.NATIVE_LAYOUT,state);b.core_envelope(state,auth['cap_seconds']+30)
        b.require(control._status(state)['available_after_reservations_GPU_hours']>=(auth['cap_seconds']+30)/3600,'Insufficient project budget')
        state['future_accounting_policy']['current_launch_admission']=True;state['execution_controller']['status']='REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend']=metadata;backend.controller_gate(state,auth)
    path.parent.mkdir(parents=True,exist_ok=False);path.write_text(json.dumps(auth,indent=2)+'\n');b.verify(path)
    result=dict(status='EXTERNAL_CORE_ONE_USE_CELL_DEPLOYED_NO_JOB_STARTED',authorization_path=str(path),authorization_SHA256=b.sha(path),**{k:auth[k] for k in ('allocation_id','attempt_id','output_dir','ledger_path','cell','cap_seconds','expires_unix','candidate_commit')})
    (path.parent/'deployment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
