"""Read-only count-variance decomposition; no fitting or count-head modification.

For X|mu,theta ~ NB(mu,theta), Var(X)=Var(mu)+E(mu^2/theta)+E(mu).
The decomposition concerns raw per-gene marginal variances, not PCA energy
distance, donor-summary uncertainty, or a causal attribution of model error.
"""
import datetime,gc,hashlib,importlib.util,json,os,signal,statistics,time
from pathlib import Path
ROOT=Path('/ml2/alz');PLAN=ROOT/'analysis/t005_nb_variance_CPU_protocol_20261108.json';STATE=ROOT/'artifacts/t005-nb-variance-CPU-20261108';RESULT=ROOT/'analysis/t005_nb_variance_CPU_result_20261108.json'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def moments_class(np):
 class Moments:
  def __init__(self,width):self.n=0;self.mean=np.zeros(width,dtype=np.float64);self.m2=np.zeros(width,dtype=np.float64)
  def add(self,rows):
   x=np.asarray(rows,dtype=np.float64);assert x.ndim==2 and x.shape[1]==len(self.mean) and len(x)>0 and np.isfinite(x).all()
   n=len(x);mean=x.mean(0);m2=np.square(x-mean).sum(0);delta=mean-self.mean;total=self.n+n
   self.m2+=m2+delta**2*(self.n*n/total);self.mean+=delta*(n/total);self.n=total
  def variance(self):assert self.n>0;return self.m2/self.n
 return Moments
def components(np,mean,variance,theta):
 mean,variance,theta=map(lambda x:np.asarray(x,dtype=np.float64),(mean,variance,theta))
 assert mean.shape==variance.shape==theta.shape and np.isfinite(mean).all() and np.isfinite(variance).all() and np.isfinite(theta).all() and (mean>=0).all() and (variance>=0).all() and (theta>0).all()
 gamma=(variance+mean**2)/theta;poisson=mean;total=variance+gamma+poisson
 return dict(mean_variation=variance,gamma_excess=gamma,poisson=poisson,total=total)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';plan=read(PLAN);assert plan['status']=='PRESPECIFIED_READONLY_CURRENT_T005_RAW_COUNT_VARIANCE_DIAGNOSTIC' and plan['new_optimizer_updates']==0 and plan['saved_updates']==[10000,20000] and plan['bags_per_stage']==200 and plan['cells_per_bag']==400 and plan['source_SHA256']==sha(__file__)
 assert [g['seed'] for g in plan['checkpoint_groups']]==list(range(5)) and all([x['completed_updates'] for x in g['checkpoints']]==[10000,20000] for g in plan['checkpoint_groups'])
 for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
 assert not RESULT.exists();STATE.mkdir(exist_ok=False)
 spec=importlib.util.spec_from_file_location('strict_native_saved_generation',ROOT/'analysis/diagnose_t005_calibration_checkpoints_multiseed_cpu.py');loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);m=loader.m;np=m.np;torch=m.torch;torch.set_num_threads(1);torch.set_grad_enabled(False);assert not torch.cuda.is_initialized();Moments=moments_class(np)
 stat=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split();write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(stat[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),protocol_SHA256=sha(PLAN),CPU_threads=1,new_GPU_allocations=0,new_optimizer_updates=0))
 def timeout(signum,frame):raise TimeoutError('Finite read-only raw-count variance diagnostic allowance ended')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['CPU_wall_cap_seconds']);started=time.monotonic()
 try:
  ctx=m.Context(Path(plan['fitting_cache']),'cpu');raw=ctx.X.detach().numpy();assert np.equal(raw,np.floor(raw)).all() and (raw>=0).all();assert not set(ctx.ids.tolist())&set(ctx.manifest['calibration_donors'])
  for group in plan['checkpoint_groups']:
   for item in group['checkpoints']:
    q=torch.load(item['path'],map_location='cpu',weights_only=False);G,P=loader.generation_modules(ctx,q,group['seed'],item['completed_updates'],plan);del G,P,q
  schedules={};real_mean=np.empty((2,4,2000));real_var=np.empty_like(real_mean);rows_trace=[];lib_trace=[];donor_trace=[]
  for rep in (0,1):
   rrows=[];rlibs=[];rdonors=[]
   for stage in range(4):
    c=np.zeros(7,dtype=np.float32);c[3]=stage/3;seed=plan['schedule_seed_base']+10000*rep+stage;schedule=ctx.law.schedule(c,seed,m=200,k=400);schedules[rep,stage]=(c,schedule);moment=Moments(2000)
    for start in range(0,200,5):
     values=raw[schedule.rows[start:start+5]];np.testing.assert_array_equal(values.sum(-1),schedule.libraries[start:start+5]);moment.add(values.reshape(-1,2000))
    real_mean[rep,stage]=moment.mean;real_var[rep,stage]=moment.variance();rrows.append(schedule.rows);rlibs.append(schedule.libraries);rdonors.append(schedule.donors)
   rows_trace.append(rrows);lib_trace.append(rlibs);donor_trace.append(rdonors)
  traces=STATE/'fitting_traces_and_reference.npz';np.savez_compressed(traces,rows=np.asarray(rows_trace),libraries=np.asarray(lib_trace),donor_indices=np.asarray(donor_trace),real_mean=real_mean,real_variance=real_var)
  records=[]
  for group in plan['checkpoint_groups']:
   for item in group['checkpoints']:
    seed=group['seed'];step=item['completed_updates'];q=torch.load(item['path'],map_location='cpu',weights_only=False);G,P=loader.generation_modules(ctx,q,seed,step,plan);before=m.fingerprint(G),m.fingerprint(P);means=np.empty_like(real_mean);variances=np.empty_like(real_mean);theta=(torch.nn.functional.softplus(G.log_theta)+.05).cpu().numpy().astype(np.float64)
    for rep in (0,1):
     for stage in range(4):
      c,schedule=schedules[rep,stage];condition=torch.as_tensor(c)[None];draw=plan['latent_seed_base']+10000*rep+stage;lat=torch.Generator().manual_seed(draw);cell=torch.Generator().manual_seed(draw+100000);moment=Moments(2000)
      for start in range(0,200,5):
       z,_=P.sample(5,generator=lat);mu,returned_theta=G.parameters_nb(z,condition.expand(5,-1),400,torch.as_tensor(schedule.libraries[start:start+5]),cell);np.testing.assert_array_equal(returned_theta.cpu().numpy().astype(np.float64),theta);assert bool(torch.isfinite(mu).all()) and bool((mu>=0).all());moment.add(mu.cpu().numpy().reshape(-1,2000))
      means[rep,stage]=moment.mean;variances[rep,stage]=moment.variance()
    theta_grid=np.broadcast_to(theta,means.shape);parts=components(np,means,variances,theta_grid);path=STATE/f'seed{seed}-update{step:05d}.npz';np.savez_compressed(path,mean=means,mean_variation=variances,theta=theta,gamma_excess=parts['gamma_excess'],poisson=parts['poisson'],total_variance=parts['total']);replicas=[]
    for rep in (0,1):
     totals={name:float(x[rep].sum()) for name,x in parts.items()};denom=totals['total'];assert denom>0;fractions={name:totals[name]/denom for name in ('mean_variation','gamma_excess','poisson')};assert abs(sum(fractions.values())-1)<1e-12
     replicas.append(dict(replicate=rep,raw_marginal_variance_trace_components=totals,raw_marginal_variance_trace_fractions=fractions,model_to_matching_fitting_reference_variance_trace_ratio=denom/float(real_var[rep].sum()),per_stage_model_to_reference_variance_trace_ratio=[float(parts['total'][rep,s].sum()/real_var[rep,s].sum()) for s in range(4)]))
    assert before==(m.fingerprint(G),m.fingerprint(P)) and sha(item['path'])==item['SHA256'] and not torch.cuda.is_initialized()
    row=dict(training_seed=seed,saved_update=step,checkpoint_path=item['path'],checkpoint_SHA256=item['SHA256'],theta_quantiles=dict(zip(['min','q10','median','q90','max'],np.quantile(theta,[0,.1,.5,.9,1]).tolist())),theta_to_training_moment_initializer_ratio_quantiles=np.quantile(theta/ctx.theta.cpu().numpy(),[0,.1,.5,.9,1]).tolist(),replicates=replicas,mean_variance_trace_fractions={name:statistics.mean(x['raw_marginal_variance_trace_fractions'][name] for x in replicas) for name in ('mean_variation','gamma_excess','poisson')},mean_model_to_reference_variance_ratio=statistics.mean(x['model_to_matching_fitting_reference_variance_trace_ratio'] for x in replicas),retained_per_gene_arrays_path=str(path),retained_per_gene_arrays_SHA256=sha(path),generator_prior_fingerprints_before_after_equal=True)
    records.append(row);write(STATE/'progress.json',dict(status='READ_ONLY_RAW_COUNT_VARIANCE_DECOMPOSITION',completed_seed_step_pairs=[[x['training_seed'],x['saved_update']] for x in records],new_optimizer_updates=0));print(json.dumps(dict(seed=seed,step=step,fractions=row['mean_variance_trace_fractions'],variance_ratio=row['mean_model_to_reference_variance_ratio'],CPU_seconds=time.monotonic()-started)),flush=True);del G,P,q;gc.collect()
  assert len(records)==10
  summary={str(step):dict(mean_variance_trace_fractions={name:statistics.mean(x['mean_variance_trace_fractions'][name] for x in records if x['saved_update']==step) for name in ('mean_variation','gamma_excess','poisson')},mean_model_to_reference_variance_ratio=statistics.mean(x['mean_model_to_reference_variance_ratio'] for x in records if x['saved_update']==step)) for step in (10000,20000)}
  for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
  result=dict(status='COMPLETE_READONLY_CURRENT_T005_FIVE_SEED_RAW_COUNT_VARIANCE_DECOMPOSITION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_seeds=list(range(5)),saved_updates=[10000,20000],summary=summary,records=records,formula='Var(X)=Var(mu)+E(mu^2/theta)+E(mu); theta is per gene and fixed within each saved checkpoint.',reference_scope='Original fitting donors only, stage-balanced hierarchical fitting-donor/cell/library traces; each stage uses200bags*400cells, two common-random-number probes shared across checkpoints and initializations.',latent_and_library_variation_not_separated=True,statistic_scope='Equal-weight sum of four stage-conditional per-gene raw marginal variance traces across2000genes. Excludes between-stage mean variation, cross-gene covariance, normalized/PCA energy-distance decomposition and donor uncertainty.',conditional_means_evaluated=6400000,new_RNA_count_draws=0,new_optimizer_updates=0,new_training_fits=0,new_GPU_allocations=0,CUDA_initialized=False,frozen45_and_six_fresh_fit_protocols_unchanged=True,model_and_source_checkpoint_states_unchanged=True,heldout_values_loaded_for_scoring_or_selection=False,CPU_seconds=time.monotonic()-started,protocol_path=str(PLAN),protocol_SHA256=sha(PLAN),source_SHA256={str(Path(__file__).resolve()):sha(__file__)},fitting_trace_arrays_path=str(traces),fitting_trace_arrays_SHA256=sha(traces),limitations='A diagnostic of the current count application, not a missing-Poisson finding: native forward already includes Gamma and Poisson noise. Finite Monte Carlo conditional-mean ensembles and existing initialization seeds are not biological replicates. A large noise fraction alone does not establish excess technical noise or explain an energy-distance gap. No theta, objective, checkpoint or hyperparameter was changed; improvement requires a separate matched fresh training test.',full_goal_complete=False,SOTA_claim=False)
  assert not torch.cuda.is_initialized();write(RESULT,result);write(STATE/'completed.json',dict(status=result['status'],result_path=str(RESULT)));print(json.dumps(dict(status=result['status'],summary=summary,CPU_seconds=result['CPU_seconds'])),flush=True)
 except BaseException as exc:write(STATE/'failure.json',dict(error=repr(exc),new_optimizer_updates=0,new_GPU_allocations=0,CUDA_initialized=torch.cuda.is_initialized()));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
