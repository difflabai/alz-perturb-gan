"""Read-only independent fresh released-objective baseline audit; no model fitting or GPU imports."""
import argparse,datetime,gc,hashlib,json,math,os,subprocess,sys
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/localfaith097';MODEL=ROOT/'analysis/faithful097/model'
MODEL_PROOF=ROOT/'analysis/faithful097/native_trainer_parity_CPU.json'
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def live(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split()
 return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def closed_preconditions():
 """Refuse incomplete confirmation before importing models or loading arrays."""
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 protocol=read(STUDY/'protocol.json');report=read(STUDY/'report.json')
 assert protocol['status']=='FROZEN_RELEASED097_FULL20K_LOCAL_COORDINATE_SEARCH'
 assert report['status']=='ALL_RELEASED097_PAIRED_LOCAL_COORDINATE_ATTEMPTS_AUDITED' and not report['SOTA_claim'] and not report['full_goal_complete']
 assert report['protocol_SHA256']==sha(STUDY/'protocol.json')
 assert not read(ROOT/'localfaith097_runtime/plan.json')['enabled']
 assert not live(read(ROOT/'artifacts/localfaith097-paired-search/identity.json'))
 progress=read(ROOT/'artifacts/localfaith097-paired-search/progress.json');assert progress['activation_reverted'] and progress['goal_still_active']
 assert not (ROOT/'artifacts/localfaith097-paired-search/failure.json').exists()
 assert not read(ROOT/'artifacts/gpu1-localfaith097-paired-search-20261107/live_registry/registry.json')['jobs']
 assert not read(P/'project_gpu_budget.json')['open_reservations']
 identity=read(ROOT/'artifacts/localfaith097-paired-search/identity.json');activation=identity['activation']
 changed=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',activation],cwd=ROOT,text=True).splitlines()
 assert changed==['localfaith097_runtime/plan.json']
 log=subprocess.check_output(['git','log','--format=%B'],cwd=ROOT,text=True)
 assert 'This reverts commit '+activation+'.' in log
 return protocol,report
def rows_by_candidate(summary):
 return [dict(candidate_id=c['candidate_id'],hyperparameters=c['hyperparameters'],finite_all_seeds=c['all_seeds_finite'],mean_metrics=c.get('mean')) for c in summary['candidates']]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
 protocol,report=closed_preconditions()
 sys.path[:0]=list(map(str,[MODEL,ROOT/'model',ROOT,ROOT/'analysis',SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
 from release_env import activate
 activate()
 import numpy as np,torch,released_update as update,released_adapter as model
 from calibration import score,average
 from released097_selection import summarize,proposed_confirmation,FORMULATION
 native=read(STUDY/'native_release.json');assert native['version']=='0.9.7' and len(native['official_wheel_source_SHA256'])==69
 for path,pin in native['official_wheel_source_SHA256'].items():assert sha(path)==pin
 assert sha(native['package_metadata_path'])==native['package_metadata_SHA256']
 proof=read(MODEL_PROOF);assert proof['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS'
 for path,pin in proof['source_SHA256'].items():assert sha(path)==pin
 for path,pin in protocol['design_evidence_SHA256'].items():assert sha(path)==pin
 expected={(c['id'],seed) for c in protocol['candidates'] for seed in (0,1)}
 files=sorted((STUDY/'results').glob('*.json'));records=[read(f) for f in files]
 assert len(records)==len(expected) and {(r['candidate_id'],r['seed']) for r in records}==expected
 assert set(report['result_SHA256'])=={r['attempt'] for r in records}
 rows=[];hashed=0;rescored=0;total=0
 for path,r in zip(files,records):
  assert sha(path)==report['result_SHA256'][r['attempt']]
  assert r['phase']=='search' and r['endpoint']==20000 and r['calibration_only'] and r['from_scratch'] and r['initial_completed_updates']==0
  assert r['clean_settlement'] and r['owned_processes_and_lease_absent'] and not r['resumed_same_trial']
  proposal=STUDY/'proposals'/(r['candidate_id']+'.json');assert sha(proposal)==r['proposal_SHA256'] and sha(STUDY/'protocol.json')==r['protocol_SHA256']
  assert r['hyperparameters']==read(proposal)['hyperparameters']
  auth_path=ROOT/'artifacts/allocations'/r['attempt']/'authorization.json';auth=read(auth_path)
  deployment=read(auth_path.parent/'deployment.json');assert sha(auth_path)==deployment['authorization_SHA256']
  assert auth['initial_completed_updates']==0 and auth['new_model_updates']==20000 and auth['fixed_endpoint']==20000
  assert auth['source_model_path'] is auth['source_model_SHA256'] is None and not auth['is_continuation'] and not auth['reuse_completed_model']
  assert auth['operation']=='localfaith097_cell' and auth['cell']['experiment']=='lf097_'+r['candidate_id']
  for name in ('released_update.py','released_adapter.py','release_env.py'):assert auth['files_SHA256'][str(MODEL/name)]==sha(MODEL/name)
  assert auth['cell']['seed']==r['seed'] and auth['cell']['fold']==0 and auth['released_hyperparameters']==r['hyperparameters']
  job=read(auth['ledger_path'])['jobs'][0];assert job['reservation_released'] and not live(job['child_identity']) and not live(job['guard_identity'])
  settlement=read(Path(auth['ledger_path']).with_name(r['attempt']+'-settlement.json'));assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
  assert math.isclose(r['charged_GPU_seconds'],settlement['settlement']['charged_GPU_seconds'],rel_tol=0,abs_tol=1e-8);total+=r['charged_GPU_seconds']
  if not r['finite']:
   assert r['failure']['error'].startswith('FloatingPointError(') and r['failed_candidate_not_retried'];rows.append(r);continue
  assert r['physical_start_update']==0 and r['new_phase_model_updates']==20000 and r['independent_saved_score_audit']
  for f,pin in r['files_SHA256'].items():assert sha(f)==pin;hashed+=1
  out=Path(auth['output_dir']);q=torch.load(out/'fit/final.pt',map_location='cpu',weights_only=False)
  assert q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==q['usage']['steps']==20000
  assert q['particlegan_version']=='0.9.7' and q['fold']==0 and q['seed']==r['seed'] and q['released_hyperparameters']==r['hyperparameters']
  assert q['training_streams']==model.training_streams(0,r['seed']) and q['cache_manifest_SHA256']==sha(SCIENCE/'caches/fold0/manifest.json')
  assert int(q['usage']['draws'].sum())==640000 and int(q['usage']['unsampled_moved_steps'].sum())==0
  assert all(torch.isfinite(v).all() for state in q['policy']['models'].values() for v in state.values())
  update.configure(q['variant_config']);update.configure_hyperparameters(r['hyperparameters']);recipe=update.recipe()
  assert q['recipe']==q['policy']['recipe']==recipe.to_dict() and recipe.total_steps==20000 and recipe.ema_decay==0 and recipe.generator_gradient_response=='none'
  completed=read(out/'completed.json');assert completed['from_scratch'] and not completed['evaluation_targets_read']
  for prefix,key,saved in [('r','metrics','replicate_metrics'),('resampling_r','baseline_metrics','baseline_replicate_metrics')]:
   reps=[]
   for rep in (0,1):
    folder=out/(prefix+str(rep));m=read(folder/'metrics.json')
    assert m['training_seed']==r['seed'] and m['generation_replicate']==rep and m['generated_cells']==320000 and m['arrays_SHA256']==sha(folder/'arrays.npz')
    with np.load(folder/'arrays.npz',allow_pickle=False) as a:measured=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
    for k,v in measured.items():
     if isinstance(v,(int,float,list)) and k!='donors':np.testing.assert_allclose(v,m[k],rtol=0,atol=1e-10)
    if prefix=='r':
     for stage in range(4):
      with np.load(folder/f'stage{stage}_exemplar.npz',allow_pickle=False) as a:
       counts=a['counts'];assert counts.shape==(1,400,2000) and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
    assert m==r[saved][rep]==completed[saved][rep];reps.append(m);rescored+=1
   assert average(reps)==r[key]==completed[key]
  assert r['baseline_metrics']==read(ROOT/'analysis/faithful097/resampling_controls_CPU.json')['metrics_by_seed'][str(r['seed'])]
  rows.append(r);del q;gc.collect()
 import importlib.util
 selector_path=STUDY/'selection.py';spec=importlib.util.spec_from_file_location('audited_local_selector',selector_path);selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
 baseline=read(protocol['baseline_audit_path']);assert baseline['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS' and baseline['formulation']==FORMULATION
 assert baseline['activation_actually_reverted'] and baseline['owned_processes_and_leases_absent'] and not baseline['heldout_values_read']
 for f,pin in baseline['source_SHA256'].items():assert sha(f)==pin
 for control in protocol['reference_controls']:
  original=read(control['source_result_path']);assert sha(control['source_result_path'])==control['source_result_SHA256']==baseline['result_SHA256'][control['attempt']]
  assert control==dict(original,read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0,source_result_path=control['source_result_path'],source_result_SHA256=control['source_result_SHA256'])
  for f,pin in original['files_SHA256'].items():assert sha(f)==pin;hashed+=1
 summary=selector.summarize_local(protocol['candidates'],rows,protocol['reference_controls']);assert report['summary']==summary
 runtime=read(STUDY/'runtime_CPU_qualification.json');assert runtime['status']=='RELEASED097_LOCAL_COORDINATE_SCOPE_ACCOUNTING_CPU_PASS' and runtime['actual_cells_checked']==len(expected)
 checks=runtime['actual_native_trainer_parity_rows'];assert len(checks)==len(expected) and {(q['candidate_id'],q['seed']) for q in checks}==expected
 for check in checks:assert check['optimizer_pairs_per_loop']==3 and check['exact_models_optimizers_rng_and_update_clock'] and check['exact_all_losses_and_penalty'] and check['from_scratch']
 for f,pin in runtime['source_SHA256'].items():assert sha(f)==pin
 assert math.isclose(total,report['actual_GPU_seconds'],rel_tol=0,abs_tol=1e-8)
 assert not torch.cuda.is_initialized()
 proof=dict(status='INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   required_attempts=len(expected)+4,required_new_attempts=len(expected),previously_measured_control_attempts=4,controls_refitted=False,all_planned_attempts_resolved=True,all_new_models_from_update_zero=True,training_seeds=[0,1],
   selection_seeds_are_exploratory=True,formulation=FORMULATION,model_parity_proof_path=str(MODEL_PROOF),model_parity_proof_SHA256=sha(MODEL_PROOF),summary=summary,result_SHA256=report['result_SHA256'],report_SHA256=sha(STUDY/'report.json'),
   protocol_SHA256=sha(STUDY/'protocol.json'),protocol_path=str(STUDY/'protocol.json'),report_path=str(STUDY/'report.json'),saved_artifacts_hashed=hashed,calibration_replicates_independently_rescored=rescored,
   total_actual_GPU_seconds=total,activation_actually_reverted=True,owned_processes_and_leases_absent=True,native_version='0.9.7',
   native_source_files_verified=69,EMA_decay=0,formulation_changed=False,heldout_values_read=False,CUDA_initialized=False,
   source_SHA256={str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'analysis/released097_selection.py',MODEL_PROOF,STUDY/'selection.py',STUDY/'runtime_CPU_qualification.json',Path(protocol['baseline_audit_path'])]},broader_goal_complete=False,SOTA_claim=False,
   limitation='Adaptive paired coordinate selection on the same four calibration donors, including four explicitly previously measured baseline/control endpoints. Fresh seed2..4 confirmation and45new external models remain.')
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:proof[k] for k in ('status','required_attempts','calibration_replicates_independently_rescored','total_actual_GPU_seconds')}))
if __name__=='__main__':main()
