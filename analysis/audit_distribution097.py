"""Independently rescore completed native0.9.7 stage1 arms, without model fitting."""
import argparse,datetime,gc,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/distribution097';MODEL=STUDY/'model'
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[MODEL,ROOT/'model',ROOT,SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import numpy as np,torch,released_update as update,released_adapter as model
from calibration import score,average

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def live(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--complete',action='store_true');args=p.parse_args();assert not args.output.exists()
 native=read(STUDY/'native_release.json');assert native['version']=='0.9.7' and len(native['official_wheel_source_SHA256'])==69
 for f,pin in native['official_wheel_source_SHA256'].items():assert sha(f)==pin
 assert sha(native['package_metadata_path'])==native['package_metadata_SHA256']
 for name in ('released_update.py','released_adapter.py'):assert (MODEL/name).read_bytes()==(ROOT/'analysis/released_tuning_v096'/name).read_bytes()
 files=sorted((STUDY/'results').glob('*.json'));assert files;records=[read(f) for f in files];rows=[];hashed=0;rescored=0;total=0
 cfg={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
 for source,r in zip(files,records):
  assert r['candidate_id'] in ('d000','d001','d002') and r['seed']==0 and r['phase']=='search' and r['endpoint']==20000 and r['calibration_only']
  assert r['clean_settlement'] and r['owned_processes_and_lease_absent'] and not r['resumed_same_trial']
  proposal=STUDY/'proposals'/(r['candidate_id']+'.json');assert r['proposal_SHA256']==sha(proposal) and r['protocol_SHA256']==sha(STUDY/'protocol.json')
  assert r['hyperparameters']==read(proposal)['hyperparameters']
  auth=read(ROOT/'artifacts/allocations'/r['attempt']/'authorization.json');job=read(auth['ledger_path'])['jobs'][0]
  assert auth['initial_completed_updates']==0 and auth['new_model_updates']==20000 and auth['source_model_path'] is auth['source_model_SHA256'] is None and not auth['is_continuation'] and not auth['reuse_completed_model']
  assert not live(job['guard_identity']) and not live(job['child_identity']) and job['reservation_released']
  settlement=read(Path(auth['ledger_path']).with_name(r['attempt']+'-settlement.json'));assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
  assert math.isclose(r['charged_GPU_seconds'],settlement['settlement']['charged_GPU_seconds'],abs_tol=1e-8);total+=r['charged_GPU_seconds']
  row={k:r[k] for k in ('candidate_id','finite','metrics','hyperparameters','charged_GPU_seconds')};row['result_SHA256']=sha(source)
  if not r['finite']:
   assert r['failure']['error'].startswith('FloatingPointError(');row['failed_trial_retained']=True;rows.append(row);continue
  assert r['from_scratch'] and r['physical_start_update']==0 and r['new_phase_model_updates']==20000 and r['independent_saved_score_audit']
  for f,pin in r['files_SHA256'].items():assert sha(f)==pin;hashed+=1
  cp=next(f for f in r['files_SHA256'] if f.endswith('/fit/final.pt'));q=torch.load(cp,map_location='cpu',weights_only=False)
  assert q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==q['usage']['steps']==20000
  assert q['particlegan_version']=='0.9.7' and q['variant_config']==cfg and q['penalty_options']=={'input_domain':'continuous'}
  assert q['fold']==q['seed']==0 and q['training_streams']==model.training_streams(0,0) and q['released_hyperparameters']==r['hyperparameters']
  assert q['cache_manifest_SHA256']==sha(SCIENCE/'caches/fold0/manifest.json')
  assert int(q['usage']['draws'].sum())==640000 and int(q['usage']['unsampled_moved_steps'].sum())==0
  assert all(torch.isfinite(v).all() for state in q['policy']['models'].values() for v in state.values())
  update.configure(cfg);update.configure_hyperparameters(r['hyperparameters']);rec=update.recipe()
  assert q['recipe']==q['policy']['recipe']==rec.to_dict() and rec.total_steps==20000 and rec.ema_decay==0 and rec.generator_gradient_response=='none' and rec.lr_floor==rec.network_lr_floor==1
  out=Path(cp).parent.parent;c=read(out/'completed.json');assert not c['evaluation_targets_read'] and c['from_scratch'] and c['training_finished_before_calibration']
  reps=[]
  for rep in (0,1):
   folder=out/f'r{rep}';m=read(folder/'metrics.json');assert m['generated_cells']==320000 and m['training_seed']==0 and m['generation_replicate']==rep
   with np.load(folder/'arrays.npz',allow_pickle=False) as a:measured=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
   for k,v in measured.items():
    if isinstance(v,(int,float,list)) and k!='donors':np.testing.assert_allclose(v,m[k],rtol=0,atol=1e-10)
   for stage in range(4):
    with np.load(folder/f'stage{stage}_exemplar.npz',allow_pickle=False) as a:
     counts=a['counts'];assert counts.shape==(1,400,2000) and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
   assert m==r['replicate_metrics'][rep];reps.append(m);rescored+=1
  assert average(reps)==r['metrics'];rows.append(row);del q;gc.collect()
 ranked=sorted([r for r in rows if r['finite']],key=lambda r:(r['metrics']['cell_energy_distance'],r['metrics']['energy_score'],r['candidate_id']))
 if args.complete:
  assert {r['candidate_id'] for r in rows}=={'d000','d001','d002'}
  report=read(STUDY/'stage1_report.json');assert report['ranking_by_distribution']==[r['candidate_id'] for r in ranked] and not report['full_goal_complete'] and not report['SOTA_claim']
  assert math.isclose(total,report['total_actual_GPU_seconds'],abs_tol=1e-8)
  assert not read(ROOT/'distribution097_runtime/plan.json')['enabled']
  assert not read(ROOT/'artifacts/gpu1-distribution097-stage1-20261106/live_registry/registry.json')['jobs']
  assert not read(P/'project_gpu_budget.json')['open_reservations']
  assert not live(read(ROOT/'artifacts/distribution097-stage1/driver_identity.json'))
 assert not torch.cuda.is_initialized()
 proof=dict(status='INDEPENDENT_NATIVE097_FRESH20K_STAGE1_COUNTS_SCORES_AND_SETTLEMENT_AUDIT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_stage1=args.complete,rows=rows,ranked_candidate_ids=[r['candidate_id'] for r in ranked],saved_artifacts_hashed=hashed,full_calibration_replicates_independently_rescored=rescored,native_version='0.9.7',native_source_files_verified=69,all_new_models_from_update_zero=True,EMA_decay=0,formulation_changed=False,heldout_values_read=False,total_actual_GPU_seconds=total,source_SHA256=sha(__file__),broader_goal_complete=False,SOTA_claim=False,limitation='Stage1 selection on the same four adaptive calibration donors; fresh-seed confirmation and properly scoped external comparison remain required.')
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:proof[k] for k in ('status','complete_stage1','ranked_candidate_ids','full_calibration_replicates_independently_rescored','total_actual_GPU_seconds')},indent=2))
if __name__=='__main__':main()
