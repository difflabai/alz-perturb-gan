"""Read-only CPU sampling from frozen t005 checkpoints; no policy continuation."""
import datetime,gc,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
MODEL=ROOT/'analysis/faithful097/model';PROTOCOL=ROOT/'analysis/t005_checkpoint_CPU_calibration_protocol_v3_20261108.json';STATE=ROOT/'artifacts/t005-checkpoint-calibration-CPU-v3-20261108';RESULT=ROOT/'analysis/t005_checkpoint_CPU_calibration_result_v3_20261108.json'
sys.path[:0]=list(map(str,[ROOT,MODEL,SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np,torch
from released_adapter import Context
from particlegan import Recipe
from pg_model import CountGenerator
from generation import iter_pg,iter_resampling
from calibration import evaluate,average

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def fingerprint(module):
 h=hashlib.sha256()
 for k,v in sorted(module.state_dict().items()):
  x=v.detach().cpu().contiguous();h.update(k.encode()+str(x.dtype).encode()+repr(tuple(x.shape)).encode()+x.numpy().tobytes())
 return h.hexdigest()
def main():
 torch.set_num_threads(1);torch.set_grad_enabled(False);assert not torch.cuda.is_initialized()
 plan=read(PROTOCOL);assert plan['new_optimizer_updates']==0 and plan['saved_updates']==[5000,10000,20000] and plan['training_seed']==0
 for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
 assert plan['source_SHA256']==sha(__file__) and not RESULT.exists();STATE.mkdir(exist_ok=False)
 def timeout(signum,frame):raise TimeoutError('Finite CPU read-only calibration diagnostic allowance ended')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['CPU_wall_cap_seconds']);started=time.monotonic()
 try:
  ctx=Context(Path(plan['fitting_cache']),'cpu');rows=[];baseline=[]
  for rep in (0,1):baseline.append(evaluate(ctx,'hierarchical_resampling',0,rep,lambda schedule,c,s:iter_resampling(ctx,schedule,c,s),STATE/f'resampling_r{rep}'))
  control=average(baseline)
  for item in plan['checkpoints']:
   path=Path(item['path']);assert sha(path)==item['SHA256'];q=torch.load(path,map_location='cpu',weights_only=False);step=item['completed_updates']
   assert q['particlegan_version']=='0.9.7' and q['seed']==q['fold']==0 and q['completed_updates']==q['policy']['completed_steps']==step
   assert q['fixed_endpoint']==20000 and q['cache_manifest_SHA256']==sha(ctx.cache/'manifest.json') and q['released_hyperparameters']==plan['hyperparameters']
   assert q['variant_config']==dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
   assert q['native_recipe']=='bispf_optimized' and q['status']==('completed' if step==20000 else 'diagnostic')
   rec=Recipe.from_dict(q['recipe']);assert rec.to_dict()==q['recipe'] and rec.num_particles==512 and rec.ema_decay==0
   # Only generation modules are reconstructed. No optimizer, critic or update
   # policy is created, and saved CUDA RNG states are never relabeled as CPU RNGs.
   G=CountGenerator(ctx.mu.cpu(),ctx.theta.cpu(),zd=rec.z_dim,zc=8).to('cpu');P=rec.make_prior().to('cpu')
   rec.apply_prior_policy(P)
   for name,module in [('generator',G),('prior',P)]:
    module.load_state_dict(q['policy']['models'][name],strict=True)
    for key,value in module.state_dict().items():assert torch.equal(value,q['policy']['models'][name][key])
    module.eval()
   before=(fingerprint(G),fingerprint(P));replicas=[]
   for rep in (0,1):replicas.append(evaluate(ctx,'pg',0,rep,lambda schedule,c,s:iter_pg(ctx,G,P,schedule,c,s),STATE/f'update{step:05d}_r{rep}'))
   assert before==(fingerprint(G),fingerprint(P)) and sha(path)==item['SHA256'] and not torch.cuda.is_initialized()
   metrics=average(replicas);row=dict(completed_updates=step,checkpoint_SHA256=item['SHA256'],mean=metrics,replicates=replicas,generator_prior_state_unchanged=True)
   rows.append(row);write(STATE/'progress.json',dict(status='READ_ONLY_CPU_CHECKPOINT_CALIBRATION',completed_snapshot_updates=[r['completed_updates'] for r in rows],new_optimizer_updates=0,CUDA_initialized=False));print(json.dumps(dict(completed_snapshot_updates=step,cell_energy_distance=metrics['cell_energy_distance'],energy_score=metrics['energy_score'],elapsed_seconds=time.monotonic()-started)),flush=True)
   del G,P,q;gc.collect()
  for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
  assert sha(__file__)==plan['source_SHA256'];distance=[r['mean']['cell_energy_distance'] for r in rows]
  result=dict(status='COMPLETE_READ_ONLY_T005_SEED0_CPU_CALIBRATION_CHECKPOINT_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned_saved_updates=plan['saved_updates'],rows=rows,matched_CPU_resampling=control,ED_delta_20k_minus5k=distance[2]-distance[0],ED_delta_20k_minus10k=distance[2]-distance[1],training_seed=0,generation_replicates=2,cells_generated_per_snapshot=640000,total_native_generated_cells=1920000,CPU_backend='CPU float32 generation with original fitting-only depth traces and generation seeds; compare checkpoints within this diagnostic, not bitwise against CUDA sampling outcomes.',generator_prior_state_unchanged=True,new_optimizer_updates=0,new_training_fits=0,source_checkpoint_files_unchanged=True,only_four_calibration_donors=True,external_evaluation_target_values_read=False,no_checkpoint_or_parameter_reselection=True,original20k_endpoint_and_separate6fit_protocol_unchanged=True,CUDA_initialized=False,elapsed_seconds=time.monotonic()-started,protocol_path=str(PROTOCOL),protocol_SHA256=sha(PROTOCOL),source_SHA256={str(Path(__file__).resolve()):sha(__file__)},output_SHA256={str(p):sha(p) for p in STATE.rglob('*') if p.is_file()},limitations='One existing training initialization and four adaptively reused calibration donors. Matching prescribed seeds/traces does not create independent biological replication. Two generation replicates describe sampling variability only. No GPU-update replay, SOTA evidence or evidence of improvement beyond20000updates.',SOTA_claim=False,full_goal_complete=False)
  write(RESULT,result);write(STATE/'completed.json',dict(status=result['status'],result_path=str(RESULT)));print(json.dumps(dict(result_path=str(RESULT),ED_delta_20k_minus5k=result['ED_delta_20k_minus5k'],ED_delta_20k_minus10k=result['ED_delta_20k_minus10k'])),flush=True)
 except BaseException as exc:write(STATE/'failure.json',dict(error=repr(exc),new_training_fits=0,new_optimizer_updates=0,CUDA_initialized=torch.cuda.is_initialized()));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
