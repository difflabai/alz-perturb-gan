"""Prepare one source-bound scVI comparator. No GPU job is started here."""
import argparse,copy,datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b
import preset_scope as scope
import resource_profile as profile
def build_authorization(*,check_only=False,auth_path=None):
 base=b.read(ROOT/'artifacts/allocations/CAL_GPU0_V1_released_v095_regkappa050_f0_pg_s0/authorization.json')
 value=dict(scope.CELL);folder=ROOT/'artifacts/allocations'/b.attempt_key(value)
 auth_path=Path(auth_path or folder/'authorization.json');now=time.time();p=b.read(scope.PROTOCOL)
 pins={path:h for path,h in base['files_SHA256'].items() if not Path(path).is_relative_to(ROOT)}
 auth={key:copy.deepcopy(base[key]) for key in ('status','cpu_fixture','machine_id','source_review_passed','project_root','science_root','cache_dir','calibration_manifest_SHA256','core_GPU_hour_ceiling')}
 auth.update(cell=value,allocation_id=b.allocation_key(value),attempt_id=b.attempt_key(value),label=b.attempt_key(value),
  scheduler_allocation_id='ALZ_MATCHED_SCVI_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
  issued_unix=now,expires_unix=min(now+930,b.read(scope.WINDOW)['expires_unix'],b.read(scope.PARENT_AUTH)['expires_unix']),cap_seconds=900,
  operation='evaluate_matched_scvi_calibration',fixed_endpoint=400,role='alz_core_scvi',parent_fit_path=str(scope.PARENT_FIT),parent_fit_receipt_SHA256=b.sha(scope.PARENT_FIT/'fit_receipt.json'),parent_authorization_SHA256=b.sha(scope.PARENT_AUTH),parent_absolute_expiry_unix=b.read(scope.PARENT_AUTH)['expires_unix'],new_model_updates=0,
  output_dir=str(b.output_path(b.NATIVE_LAYOUT,value)),ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT,value)),
  candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  files_SHA256=pins,scvi_version='1.3.3',scvi_architecture=p['architecture'],scvi_training_plan=p['training_plan'],
  query_information=['stage'],evaluation_used=False,comparator_protocol_SHA256=b.sha(scope.PROTOCOL),
  source_review_type='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS',independent_current_device_review_claimed=False,
  source_review='Evaluation only from the exact preserved native scVI1.3.3 fold0seed0 final400epoch model. No training or parameter update. Correct public load device argument to auto, bind saved native state and empty Pyro sidecar, and generate first fixed calibration replicates. Same original absolute deadline and all charges retained;900second inference cap. No final evaluation or automatic promotion.')
 auth['command']=b.expected_command(b.NATIVE_LAYOUT,auth_path,value);profile.configure(auth,check_only=check_only)
 native_runtime=b.runtime_path(b.NATIVE_LAYOUT,value)
 pins[str(native_runtime)]=b.sha(native_runtime)
 config=native_runtime.parent.parent/'pyvenv.cfg';pins[str(config)]=b.sha(config)
 paths=[p for folder in (ROOT,ROOT/'model',HERE,scope.MODEL_DIR) for p in folder.glob('*.py')]
 paths.extend(ROOT/'recovery_runtime'/n for n in ('parallel_runner.py','gpu1_admission.py','gpu1_runner.py','shared_gpu1_admission.py','shared_gpu1_runner.py'))
 paths.extend(ROOT/'analysis/shared_gpu0_candidate'/n for n in ('shared_gpu0_admission.py','shared_gpu0_runner.py'))
 auth['research_source_SHA256']={str(p):b.sha(p) for p in paths};pins.update(auth['research_source_SHA256'])
 for proof in (b.read(scope.RELOAD_PROOF),p,b.read(scope.MODEL_PROOF),b.read(scope.PARITY_PROOF),b.read(scope.ENGINE_PROOF)):pins.update(proof['source_SHA256'])
 paths=scope.protocol_paths()
 receipt=b.read(scope.PARENT_FIT/'fit_receipt.json');pins.update({str(scope.PARENT_FIT/path):pin for path,pin in receipt['model_files_SHA256'].items()})
 if scope.RUNTIME_PROOF.exists():
  auth['runtime_qualification_SHA256']=b.sha(scope.RUNTIME_PROOF);paths.append(scope.RUNTIME_PROOF)
  for path,pin in b.read(scope.RUNTIME_PROOF)['source_SHA256'].items():pins.setdefault(path,pin)
 data=ROOT/'data/calibration_fold0';manifest=b.read(data/'manifest.json');paths.append(data/'manifest.json')
 pins.update({str(data/n):h for n,h in manifest['files_SHA256'].items()})
 pins.update({str(path):b.sha(path) for path in paths})
 return auth,auth_path
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
 if not args.check_only:
  profile.reviewed();b.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Commit qualified source before deployment')
 auth,path=build_authorization(check_only=args.check_only)
 b.require(not path.parent.exists(),'Exact trial exists; inspect instead of repeating')
 if not args.check_only:
  profile.DEPLOYED.parent.mkdir(parents=True,exist_ok=True)
  if profile.DEPLOYED.exists():b.require(profile.DEPLOYED.read_bytes()==profile.deployed_bytes(),'Deployed policy differs')
  else:profile.DEPLOYED.write_bytes(profile.deployed_bytes())
 b.source_closure(auth,b.NATIVE_LAYOUT)
 controller,admission,_=b.modules(auth);forecast=profile.capacity(admission.snapshot(),admission);profile.free_floor()
 reg=profile.REGISTRY/'registry.json';b.require(not reg.exists() or not b.read(reg)['jobs'],'Own namespace active')
 if args.check_only:
  print(json.dumps({'status':'MATCHED_SCVI_CHECK_ONLY_PASS_NO_ALLOCATION','attempt':auth['attempt_id'],'forecast':forecast},indent=2));return
 backend=b.load_pinned(HERE/'project_runner_gpu0.py',auth['files_SHA256'][str(HERE/'project_runner_gpu0.py')],'project_runner_gpu0')
 pins=auth['files_SHA256'];metadata={'status':'REVIEWED_FOR_BOUNDED_EXECUTION','path':str(HERE/'project_runner_gpu0.py'),'source_SHA256':{str(p):pins[str(p)] for p in [*HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']},'files_SHA256':pins,'policy_SHA256':auth['policy_SHA256'],**{k:auth[k] for k in ('allocation_id','attempt_id','scheduler_allocation_id','expires_unix')}}
 control=controller.Controller(b.PROJECT)
 with control.transaction() as state:
  b.require(not state['open_reservations'],'Another canonical reservation is active');b.unseen(auth,b.NATIVE_LAYOUT,state);b.core_envelope(state,930)
  b.require(control._status(state)['available_after_reservations_GPU_hours']>=930/3600,'Insufficient project hours')
  state['future_accounting_policy']['current_launch_admission']=True;state['execution_controller']['status']='REVIEWED_FOR_BOUNDED_EXECUTION';state['execution_controller']['calibration_backend']=metadata;backend.controller_gate(state,auth)
 path.parent.mkdir(parents=True,exist_ok=False);path.write_text(json.dumps(auth,indent=2)+'\n');b.verify(path)
 result={'status':'ONE_USE_MATCHED_SCVI_DEPLOYED_NO_JOB_STARTED','authorization_path':str(path),'authorization_SHA256':b.sha(path),**{k:auth[k] for k in ('allocation_id','attempt_id','candidate_commit','output_dir','ledger_path','cap_seconds','expires_unix')},'endpoint_epochs':400,'source_pin_count':len(pins)}
 (path.parent/'deployment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
