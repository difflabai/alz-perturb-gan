"""Prepare one exact native released parameter count-preset allocation; check-only creates no deployment."""
import argparse,copy,datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b
import preset_scope as scope
import resource_profile as profile

def build_authorization(*,check_only=False,auth_path=None):
    prior=b.read(ROOT/'calibration_deployment.json');base=b.read(prior['authorization_path'])
    value=dict(scope.CELL);folder=ROOT/'artifacts/allocations'/b.attempt_key(value)
    auth_path=Path(auth_path or folder/'authorization.json');now=time.time()
    # Retain the reviewed external native/runtime/data inventories. Owned
    # research and trial inputs are supplied explicitly from current sources.
    pins={p:v for p,v in base['files_SHA256'].items() if not Path(p).is_relative_to(ROOT)}
    keys=('status','cpu_fixture','machine_id','source_review_passed','project_root','science_root','cache_dir','calibration_manifest_SHA256','core_GPU_hour_ceiling','role')
    auth={key:copy.deepcopy(base[key]) for key in keys}
    auth.update(cell=value,allocation_id=b.allocation_key(value),attempt_id=b.attempt_key(value),label=b.attempt_key(value),
                scheduler_allocation_id='ALZ_RELEASED_RATE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),issued_unix=now,
                expires_unix=min(now+21630,b.read(scope.WINDOW)['expires_unix']),cap_seconds=21600,operation='train_released_hyperparameters',fixed_endpoint=20000,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT,value)),ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT,value)),
                variant_config=dict(scope.CFG),hybrid_config=None,pure_formulation='released_hyperparameters',qualification_only=False,native_recipe='bispf_optimized',
                initial_completed_updates=0,new_model_updates=20000,released_hyperparameters=dict(scope.HYPER),tuning_candidate_id=scope.CANDIDATE_ID,
                candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                preset_protocol_SHA256=b.sha(scope.PROTOCOL),model_qualification_SHA256=b.sha(scope.MODEL_PROOF),
                frozen_pure_incumbent_SHA256=b.sha(ROOT/'pure_incumbent.json'),files_SHA256=pins,
                source_review_type='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS',independent_current_device_review_claimed=False,
                source_review='One released ParticleGAN0.9.4 fixed bispf_optimized PG8 count model. G/D/prior rates=.006/.009/.015, native regcoeff1/kappa1/SBNP.01, original data/nets/512x8bank/11streams/counts. Six actual model CPU tests incl all27rate recipes, bit-exact baseline parity, connected fullwidth gradients and replay. GPU1 shared guard unchanged. Preserve5k/10k/20k checkpoints; only final20k can select. Exact native reload and independent saved score/cleanup/charge audit; no automatic successor or promotion')
    auth['command']=b.expected_command(b.NATIVE_LAYOUT,auth_path,value)
    profile.configure(auth,check_only=check_only)
    engine=ROOT/'recovery_runtime';paths=[p for folder_source in (ROOT,ROOT/'model',HERE,scope.MODEL_DIR) for p in folder_source.glob('*.py')]
    paths.extend(engine/name for name in ('parallel_runner.py','gpu1_admission.py','gpu1_runner.py','shared_gpu1_admission.py','shared_gpu1_runner.py'))
    paths.extend(ROOT/'analysis/shared_gpu0_candidate'/name for name in ('shared_gpu0_admission.py','shared_gpu0_runner.py'))
    research={str(path):b.sha(path) for path in paths};auth['research_source_SHA256']=research;pins.update(research)
    protocol=b.read(scope.PROTOCOL)
    paths=scope.protocol_paths(protocol)
    engine=b.read(scope.ENGINE_PROOF);pins.update(engine['source_SHA256'])
    model=b.read(scope.MODEL_PROOF);pins.update(model['source_SHA256']);pins.update(b.read(scope.NATIVE_PROOF)['official_wheel_source_SHA256'])
    if scope.RUNTIME_PROOF.exists():
        auth['runtime_qualification_SHA256']=b.sha(scope.RUNTIME_PROOF);paths.append(scope.RUNTIME_PROOF)
        for path,pin in b.read(scope.RUNTIME_PROOF)['source_SHA256'].items():pins.setdefault(path,pin)
    data=ROOT/'data/calibration_fold0';manifest=b.read(data/'manifest.json')
    b.require(b.sha(data/'manifest.json')==auth['calibration_manifest_SHA256'],'frozen calibration manifest changed')
    paths.append(data/'manifest.json');pins.update({str(data/name):pin for name,pin in manifest['files_SHA256'].items()})
    for path in paths:pins[str(path)]=b.sha(path)
    return auth,auth_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if not args.check_only:profile.reviewed()
    if not args.check_only:b.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Commit all qualified source before deployment')
    auth,auth_path=build_authorization(check_only=args.check_only)
    b.require(auth['expires_unix']>time.time()+150,'Owner window leaves no startup/cleanup interval')
    b.require(not auth_path.parent.exists(),'Exact trial already exists; inspect rather than duplicate/reset')
    if not args.check_only:
        profile.DEPLOYED.parent.mkdir(parents=True,exist_ok=True)
        if profile.DEPLOYED.exists():b.require(profile.DEPLOYED.read_bytes()==profile.deployed_bytes(),'Existing deployed policy differs; do not replace')
        else:
            with profile.DEPLOYED.open('xb') as stream:stream.write(profile.deployed_bytes())
    b.source_closure(auth,b.NATIVE_LAYOUT)
    controller,admission,_=b.modules(auth);observation=admission.snapshot()
    forecast=profile.capacity(observation,admission);profile.free_floor()
    reg=profile.REGISTRY/'registry.json';b.require(not reg.exists() or not b.read(reg)['jobs'],'ALZ resource namespace has active work')
    if args.check_only:
        print(json.dumps({'status':'CHECK_ONLY_RELEASED_PARAMETER_SOURCE_AND_CAPACITY_PASS_NO_ALLOCATION','attempt':auth['attempt_id'],'forecast':forecast},indent=2));return
    pins=auth['files_SHA256'];backend=b.load_pinned(HERE/'project_runner_gpu0.py',pins[str(HERE/'project_runner_gpu0.py')],'project_runner_gpu0')
    metadata={'status':'REVIEWED_FOR_BOUNDED_EXECUTION','path':str(HERE/'project_runner_gpu0.py'),
              'source_SHA256':{str(p):pins[str(p)] for p in [*HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']},
              'files_SHA256':pins,'policy_SHA256':auth['policy_SHA256'],**{k:auth[k] for k in ('allocation_id','attempt_id','scheduler_allocation_id','expires_unix')}}
    control=controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'],'Another canonical reservation is active')
        b.unseen(auth,b.NATIVE_LAYOUT,state);b.core_envelope(state,21630)
        b.require(control._status(state)['available_after_reservations_GPU_hours']>=21630/3600,'Insufficient canonical project hours')
        state['future_accounting_policy']['current_launch_admission']=True;state['execution_controller']['status']='REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend']=metadata;backend.controller_gate(state,auth)
    auth_path.parent.mkdir(parents=True,exist_ok=False)
    auth_path.write_text(json.dumps(auth,indent=2,allow_nan=False)+'\n');b.verify(auth_path)
    result={'status':'ONE_USE_RELEASED_PARAMETER_DEPLOYED_NO_JOB_STARTED','authorization_path':str(auth_path),'authorization_SHA256':b.sha(auth_path),
            **{k:auth[k] for k in ('allocation_id','attempt_id','candidate_commit','variant_config','hybrid_config','pure_formulation','qualification_only','output_dir','ledger_path','cap_seconds','expires_unix')},
            'endpoint':20000,'base_model_updates':0,'new_model_updates':20000,'source_pin_count':len(pins)}
    (auth_path.parent/'deployment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
