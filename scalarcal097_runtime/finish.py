"""Independent saved calibration-curve, endpoint, native-state and cost audit."""
import argparse,datetime,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
from dual_coordination import identity

def finish(attempt):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 folder=scope.ROOT/'artifacts/allocations'/attempt;dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path'])
 assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(auth['cell'])
 scope.validate_authorization(auth)
 for path,pin in auth['files_SHA256'].items():assert b.sha(path)==pin,path
 job=b.read(auth['ledger_path'])['jobs'][0];settled=b.read(Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
 assert job['reservation_released'] and settled['uncertain'] is None and settled['settlement']['status']=='settled'
 for key in ('child_identity','guard_identity'):
  old=job[key];assert identity(old['pid'])!={k:old[k] for k in ('pid','birth','boot')}
 assert job['lease_token'] not in b.read(Path(auth['registry_path'])/'registry.json')['jobs']
 out=Path(auth['output_dir']);base=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=attempt,cell=auth['cell'],arm=auth['arm'],seed=auth['cell']['seed'],benchmark_protocol_SHA256=b.sha(scope.PROTOCOL),authorization_path=dep['authorization_path'],authorization_SHA256=dep['authorization_SHA256'],own_processes_absent=True,owned_registry_lease_absent=True,charged_GPU_seconds=settled['settlement']['charged_GPU_seconds'],all_new_training_from_scratch=True,initial_completed_updates=0,source_checkpoint=None,automatic_retry=False,calibration_only=True,SOTA_claim=False,full_goal_complete=False)
 if job['status']!='completed' or job['exit_code']!=0:
  base.update(status='SCALARCAL097_INCOMPLETE_OUTCOME_AND_CLEAN_SETTLEMENT_RETAINED',complete=False,failure=b.read(out/'failure.json') if (out/'failure.json').exists() else {},metrics=None)
 else:
  sys.path.extend(map(str,b.native_site_roots(auth['cell'],auth['files_SHA256'])))
  sys.path[:0]=list(map(str,[scope.MODEL_DIR,scope.CONTROLLER_DIR,scope.ROOT/'analysis/faithful097/model',scope.ROOT/'model',b.SCIENCE,scope.ROOT]))
  import numpy as np,torch
  import scalar_adapter as model,scalar_update as update
  from calibration import score,average,targets,cell_energy
  from calibration_scalar import array_hash
  from calibration_distance_lr import CalibrationDistanceLR,Config
  from worker_gpu0 import validate_pg
  from state_fingerprint import fingerprint
  from particlegan.policy import _validate_optimizer_state
  update.configure(scope.CFG);update.configure_hyperparameters(auth['released_hyperparameters'])
  completed=b.read(out/'completed.json');assert completed['from_scratch'] and not completed['heldout_values_used'] and completed['arm']==auth['arm'] and completed['seed']==auth['cell']['seed']
  assert completed['protocol_SHA256']==b.sha(scope.PROTOCOL) and completed['endpoint']==20000 and completed['source_checkpoint'] is None
  path=Path(completed['checkpoint_path']);assert b.sha(path)==completed['checkpoint_SHA256'];q=torch.load(path,map_location='cpu',weights_only=False);validate_pg(q,auth)
  assert q['usage']['steps']==20000 and int(q['usage']['draws'].sum())==640000 and int(q['usage']['unsampled_moved_steps'].sum())==0 and q['training_streams']==model.training_streams(0,auth['cell']['seed'])
  assert fingerprint(q['policy'])==completed['policy_fingerprint']
  restored=b.read(out/'native_restore.json');assert restored['status']=='EXACT_NATIVE_SCALAR_PG_CUDA_RESTORE_PASS' and restored['policy_fingerprint']==completed['policy_fingerprint'] and restored['completed_updates']==20000
  bundle,*_=model.build(torch.ones(2000),torch.ones(2000),'cpu',0,auth['cell']['seed'])
  assert q['recipe']==bundle[0].to_dict() and q['recipe'].get('generator_gradient_response','none')=='none'
  for name,module in zip(('generator','critic','prior'),bundle[1:4]):
   module.load_state_dict(q['policy']['models'][name],strict=True);assert all(torch.isfinite(t).all() for t in module.state_dict().values())
  for opt,saved in zip(bundle[4:6],q['policy']['optimizers']):
   _validate_optimizer_state(opt,saved);opt.load_state_dict(saved);assert fingerprint(opt.state_dict())==fingerprint(saved)
  curve=[json.loads(line) for line in (out/'fit/calibration_curve.jsonl').read_text().splitlines()];assert len(curve)==20
  control=CalibrationDistanceLR(Config()) if auth['arm']=='adaptive' else None
  _,ys,targets_cells=targets();real=targets_cells['equal_stage'].copy();ys.close();targets_cells.close()
  pins={str(path):b.sha(path),str(out/'native_restore.json'):b.sha(out/'native_restore.json'),str(out/'fit/calibration_curve.jsonl'):b.sha(out/'fit/calibration_curve.jsonl'),str(out/'completed.json'):b.sha(out/'completed.json')}
  for step,row in zip(range(1000,20001,1000),curve):
   ob=row['observation'];assert ob['completed_updates']==step and ob['metric']=='calibration_cell_energy_distance' and ob['split_role']=='calibration' and ob['cells_per_replicate']==320000 and ob['calibration_manifest_SHA256']==auth['calibration_manifest_SHA256']
   path=out/'fit/calibration'/f'update{step:05d}.npz';assert str(path)==ob['arrays_path'] and b.sha(path)==ob['arrays_SHA256']
   with np.load(path,allow_pickle=False) as a:
    cells=a['generated_cells_by_stage'];assert cells.shape==(2,4,375,50) and np.isfinite(cells).all() and np.array_equal(real,a['real_equal_stage_cells'])
    distances=[cell_energy(pool.reshape(-1,50),real) for pool in cells];assert array_hash(cells)==ob['pool_SHA256']
   np.testing.assert_allclose(distances,ob['replicate_values'],rtol=0,atol=1e-10);assert abs(float(np.mean(distances))-ob['value'])<=1e-10
   decision=None if control is None else control.observe(step,ob['value'],split_role=ob['split_role']);assert decision==row['decision'];pins[str(path)]=b.sha(path)
  assert (None if control is None else control.state_dict())==q['calibration_lr_controller']
  assert (None if control is None else control.decisions)==completed['adaptive_decisions']
  collections={}
  for prefix,key,pg in [('r','replicate_metrics',True),('resampling_r','baseline_replicate_metrics',False)]:
   records=[]
   for rep in (0,1):
    dest=out/(prefix+str(rep));m=b.read(dest/'metrics.json');assert m['status']=='completed' and m['training_seed']==auth['cell']['seed'] and m['generation_replicate']==rep and m['generated_cells']==320000 and m['trace_manifest_SHA256']==b.sha(scope.PROTOCOL)
    path=dest/'arrays.npz';assert b.sha(path)==m['arrays_SHA256']
    with np.load(path,allow_pickle=False) as arrays:recomputed=score(arrays['stage_synthetic_summaries'],arrays['generated_cells_by_stage'])
    for name,value in recomputed.items():
     if isinstance(value,(int,float,list)) and name!='donors':np.testing.assert_allclose(m[name],value,rtol=0,atol=1e-10)
    if pg:
     for stage in range(4):
      path=dest/f'stage{stage}_exemplar.npz'
      with np.load(path,allow_pickle=False) as a:
       counts=a['counts'];assert counts.shape==(1,400,2000) and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
      pins[str(path)]=b.sha(path)
    assert m==completed[key][rep];records.append(m)
    for path in (dest/'arrays.npz',dest/'metrics.json'):pins[str(path)]=b.sha(path)
   collections[prefix]=records
  assert average(collections['r'])==completed['metrics'] and average(collections['resampling_r'])==completed['baseline_metrics']
  np.testing.assert_allclose([m['cell_energy_distance'] for m in collections['r']],curve[-1]['observation']['replicate_values'],rtol=0,atol=1e-10)
  assert not torch.cuda.is_initialized()
  base.update(status='SCALARCAL097_NATIVE_STATE_ALL20_OBSERVATIONS_COUNTS_ENDPOINT_SCORES_CLEAN_AUDIT_PASS',complete=True,metrics=completed['metrics'],baseline_metrics=completed['baseline_metrics'],replicate_metrics=collections['r'],calibration_observations=curve,files_SHA256=pins,independent_CPU_saved_score_and_controller_audit=True,native_saved_state_matches_GPU_restore=True)
 path=scope.STUDY/'results'/(attempt+'.json');assert not path.exists();path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(base,indent=2)+'\n')
 print(json.dumps(dict(status=base['status'],cell=base['cell'],complete=base['complete'],charged_GPU_seconds=base['charged_GPU_seconds'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
