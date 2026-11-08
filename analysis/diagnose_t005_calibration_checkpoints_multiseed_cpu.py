"""Read-only additional CPU checkpoint diagnostics; fixed studies stay frozen."""
import datetime,gc,hashlib,importlib.util,json,os,signal,statistics,time
from pathlib import Path
ROOT=Path('/ml2/alz');PROTOCOL=ROOT/'analysis/t005_checkpoint_CPU_multiseed_protocol_20261108.json';STATE=ROOT/'artifacts/t005-checkpoint-calibration-CPU-multiseed-20261108';RESULT=ROOT/'analysis/t005_checkpoint_CPU_multiseed_result_20261108.json'
HELPER=ROOT/'analysis/diagnose_t005_calibration_checkpoints_cpu_v3.py'
spec=importlib.util.spec_from_file_location('qualified_read_only_generation_helper',HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
 p=Path(p);temp=p.with_suffix('.partial');temp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');temp.replace(p)
def generation_modules(ctx,q,seed,step,plan):
 assert q['particlegan_version']=='0.9.7' and q['seed']==seed and q['fold']==0 and q['completed_updates']==q['policy']['completed_steps']==step
 assert q['fixed_endpoint']==20000 and q['cache_manifest_SHA256']==sha(ctx.cache/'manifest.json') and q['released_hyperparameters']==plan['hyperparameters']
 assert q['variant_config']==dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
 assert q['native_recipe']=='bispf_optimized' and q['status']==('completed' if step==20000 else 'diagnostic')
 assert q['usage']['steps']==step and int(q['usage']['draws'].sum())==step*32
 rec=m.Recipe.from_dict(q['recipe']);assert rec.to_dict()==q['recipe'] and rec.num_particles==512 and rec.ema_decay==0
 G=m.CountGenerator(ctx.mu.cpu(),ctx.theta.cpu(),zd=rec.z_dim,zc=8).to('cpu');P=rec.make_prior().to('cpu');rec.apply_prior_policy(P)
 for name,module in [('generator',G),('prior',P)]:
  module.load_state_dict(q['policy']['models'][name],strict=True)
  for key,value in module.state_dict().items():assert m.torch.equal(value,q['policy']['models'][name][key])
  module.eval()
 return G,P

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';m.torch.set_num_threads(1);m.torch.set_grad_enabled(False);assert not m.torch.cuda.is_initialized()
 plan=read(PROTOCOL);assert plan['new_optimizer_updates']==0 and plan['additional_seeds']==[1,2,3,4] and plan['saved_updates']==[5000,10000,20000]
 assert plan['source_SHA256']==sha(__file__)
 for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
 assert not RESULT.exists();STATE.mkdir(exist_ok=False);stat=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()
 write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(stat[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),protocol_SHA256=sha(PROTOCOL),new_GPU_allocation=False,CPU_threads=1))
 def timeout(signum,frame):raise TimeoutError('Finite additional read-only CPU diagnostic allowance ended')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['CPU_wall_cap_seconds']);started=time.monotonic()
 try:
  ctx=m.Context(Path(plan['fitting_cache']),'cpu')
  # Qualify strict loading for every planned snapshot before any new quality
  # measurement. Optimizers/critics/policies are never constructed or advanced.
  for seedplan in plan['additional_checkpoint_groups']:
   for item in seedplan['checkpoints']:
    q=m.torch.load(item['path'],map_location='cpu',weights_only=False);G,P=generation_modules(ctx,q,seedplan['seed'],item['completed_updates'],plan);del G,P,q
  gc.collect();existing=read(plan['existing_seed0_result']);assert existing['training_seed']==0 and existing['planned_saved_updates']==plan['saved_updates'] and existing['new_optimizer_updates']==0
  allseeds={0:existing};seed_hashes={};write(STATE/'progress.json',dict(status='ALL12_ADDITIONAL_NATIVE_SNAPSHOTS_STRICT_CPU_LOAD_PASS',completed_additional_seeds=[],new_optimizer_updates=0))
  for group in plan['additional_checkpoint_groups']:
   seed=group['seed'];folder=STATE/f'seed{seed}';folder.mkdir();baseline=[];rows=[]
   for rep in (0,1):baseline.append(m.evaluate(ctx,'hierarchical_resampling',seed,rep,lambda schedule,c,s:m.iter_resampling(ctx,schedule,c,s),folder/f'resampling_r{rep}'))
   for item in group['checkpoints']:
    q=m.torch.load(item['path'],map_location='cpu',weights_only=False);step=item['completed_updates'];G,P=generation_modules(ctx,q,seed,step,plan);before=m.fingerprint(G),m.fingerprint(P);replicas=[]
    for rep in (0,1):replicas.append(m.evaluate(ctx,'pg',seed,rep,lambda schedule,c,s:m.iter_pg(ctx,G,P,schedule,c,s),folder/f'update{step:05d}_r{rep}'))
    assert before==(m.fingerprint(G),m.fingerprint(P)) and sha(item['path'])==item['SHA256'] and not m.torch.cuda.is_initialized()
    metrics=m.average(replicas);rows.append(dict(completed_updates=step,checkpoint_SHA256=item['SHA256'],mean=metrics,replicates=replicas,generator_prior_state_unchanged=True));print(json.dumps(dict(seed=seed,completed_snapshot_updates=step,cell_energy_distance=metrics['cell_energy_distance'],energy_score=metrics['energy_score'],elapsed_seconds=time.monotonic()-started)),flush=True);del G,P,q;gc.collect()
   seedrecord=dict(training_seed=seed,rows=rows,matched_CPU_resampling=m.average(baseline),new_optimizer_updates=0,source_checkpoint_files_unchanged=True,CPU_backend=existing['CPU_backend'],output_SHA256={str(p):sha(p) for p in folder.rglob('*') if p.is_file()});path=folder/'completed_seed.json';write(path,seedrecord);seed_hashes[str(path)]=sha(path);allseeds[seed]=seedrecord
   write(STATE/'progress.json',dict(status='READ_ONLY_ADDITIONAL_CPU_CALIBRATION_CHECKPOINT_DIAGNOSTICS',completed_additional_seeds=sorted(s for s in allseeds if s),new_optimizer_updates=0,new_training_fits=0,CUDA_initialized=False))
  assert set(allseeds)==set(range(5));fields=['cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80'];summary={}
  for index,step in enumerate(plan['saved_updates']):
   summary[str(step)]={k:dict(mean=statistics.mean(allseeds[s]['rows'][index]['mean'][k] for s in range(5)),initialization_and_sampling_SD=statistics.stdev(allseeds[s]['rows'][index]['mean'][k] for s in range(5))) for k in fields}
  contrasts={str(s):{k:allseeds[s]['rows'][2]['mean'][k]-allseeds[s]['rows'][1]['mean'][k] for k in fields} for s in range(5)}
  for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
  result=dict(status='COMPLETE_FIVE_EXISTING_INITIALIZATIONS_READ_ONLY_CPU_CHECKPOINT_CALIBRATION_DIAGNOSTICS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_seeds=[0,1,2,3,4],saved_updates=plan['saved_updates'],existing_seed0_reused_without_new_sampling=True,new_CPU_evaluated_training_seeds=plan['additional_seeds'],summary=summary,mean_matched_resampling={k:statistics.mean(allseeds[s]['matched_CPU_resampling'][k] for s in range(5)) for k in fields},per_seed_20k_minus10k=contrasts,seeds_with_worse20k_ED=sum(x['cell_energy_distance']>0 for x in contrasts.values()),seeds_with_better20k_ED=sum(x['cell_energy_distance']<0 for x in contrasts.values()),all_seed_records={str(s):allseeds[s] for s in range(5)},new_native_CPU_generated_cells=7680000,total_CPU_generated_cells_including_reused_seed0=9600000,generator_prior_state_unchanged=True,new_training_fits=0,new_optimizer_updates=0,new_GPU_allocations=0,CUDA_initialized=False,calibration_donors=4,reused_calibration_donors=True,external_evaluation_target_values_read=False,frozen45_and_separate6_protocols_unchanged=True,no_checkpoint_or_hyperparameter_reselection=True,protocol_path=str(PROTOCOL),protocol_SHA256=sha(PROTOCOL),source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(HELPER):sha(HELPER)},per_seed_output_SHA256=seed_hashes,elapsed_seconds=time.monotonic()-started,limitations='Retrospective diagnostics on a calibration-selected setting and existing training initializations0..4. Four adaptively reused calibration donors; five initializations and two sampling replicates per point are not biological replication. CPU float32 generation is compared within this diagnostic, not bitwise against CUDA endpoints. This does not identify an optimal new stopping point or show that the prespecified scalar controller helps; its six fresh20k fits remain separate.',SOTA_claim=False,full_goal_complete=False)
  write(RESULT,result);write(STATE/'completed.json',dict(status=result['status'],result_path=str(RESULT)));print(json.dumps(dict(result_path=str(RESULT),summary=summary,seeds_with_worse20k_ED=result['seeds_with_worse20k_ED'])),flush=True)
 except BaseException as exc:write(STATE/'failure.json',dict(error=repr(exc),new_training_fits=0,new_optimizer_updates=0,CUDA_initialized=m.torch.cuda.is_initialized()));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
