"""Unmodified native count-FM sampler against analytic count-process oracles.

No learned model, ALZ arrays, fitting, GPU or benchmark recipe changes.
"""
import datetime,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');UPSTREAM=ROOT/'artifacts/comparator-source-review-20261108/count-flow-matching';OUT=ROOT/'artifacts/countfm-native-resolution-CPU-20261108'
SITE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/isolated/cfgen-cu124-py311/lib/python3.11/site-packages')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();OUT.mkdir()
 sys.path[:0]=[str(UPSTREAM),str(SITE)]
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from count_fm.experiments.scrna_countfm import sample_euler_batched
 import count_fm.experiments.scrna_countfm as native
 files=[UPSTREAM/'count_fm/sampling.py',UPSTREAM/'count_fm/experiments/scrna_countfm.py',UPSTREAM/'count_fm/utils/inference.py',Path(__file__).resolve()];pins={str(p):sha(p) for p in files}
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()=='7ce372db78581451103baf1afdb8fecd50e12230'
 class Oracle(torch.nn.Module):
  def __init__(self,birth,beta):super().__init__();self.register_buffer('birth',birth);self.register_buffer('beta',beta)
  def forward(self,xt):return torch.cat([self.birth.expand(len(xt),-1),self.beta.expand(len(xt),-1)],1)
 duration=1.-1e-4;birth_rates=(1.,10.,100.,500.,2000.);rates=torch.tensor(birth_rates).repeat_interleave(400);net=Oracle(rates,torch.zeros(2000));zero=torch.zeros((8,2000),dtype=torch.int64);records=[];started=time.monotonic()
 for steps in (500,2000,5000):
  torch.manual_seed(20261108+steps);start=time.monotonic();counts,_=sample_euler_batched(net,steps,zero,'cpu',batch_eval=8)
  assert counts.shape==(8,2000) and counts.dtype==torch.int64 and (counts>=0).all() and net.training
  assert (counts<=zero+steps).all();np_path=OUT/f'immigration-{steps}.pt';torch.save(counts,np_path);groups=[]
  for group,rate in enumerate(birth_rates):
   values=counts[:,group*400:(group+1)*400].to(torch.float64);p=-math.expm1(-rate*duration/steps);mean=steps*p;variance=steps*p*(1.-p);se=math.sqrt(variance/values.numel());empirical=float(values.mean());assert abs(empirical-mean)<6*se+.02
   exact=rate*duration;groups.append(dict(rate=rate,sample_coordinates=values.numel(),native_empirical_mean=empirical,native_exact_grid_mean=mean,native_exact_grid_variance=variance,empirical_mean_standard_error=se,continuous_time_exact_mean=exact,deterministic_relative_mean_bias=(mean/exact)-1.))
  row=dict(process='constant_immigration_from_zero',steps=steps,fullwidth_genes=2000,independent_samples=8,groups=groups,seconds=time.monotonic()-start,retained_counts_path=str(np_path),retained_counts_SHA256=sha(np_path));records.append(row);print(json.dumps(dict(process=row['process'],steps=steps,seconds=row['seconds'])),flush=True)
 # A unit-jump grid cannot remove more than one count per coordinate per step.
 death=Oracle(torch.zeros(2000),torch.full((2000,),10.));initial=torch.full((2,2000),1000,dtype=torch.int64);torch.manual_seed(20261108)
 counts,_=sample_euler_batched(death,500,initial,'cpu',batch_eval=2);assert counts.dtype==torch.int64 and (counts>=500).all() and (counts<=1000).all() and death.training
 path=OUT/'death-500.pt';torch.save(counts,path);records.append(dict(process='linear_death_from1000_beta10',steps=500,fullwidth_genes=2000,independent_samples=2,native_empirical_mean=float(counts.to(torch.float64).mean()),native_minimum_count=int(counts.min()),native_per_realized_path_lower_bound=500,continuous_time_exact_mean=1000*math.exp(-10*duration),retained_counts_path=str(path),retained_counts_SHA256=sha(path)))
 assert not torch.cuda.is_initialized()
 for p,pin in pins.items():assert sha(p)==pin,p
 result=dict(status='UNMODIFIED_NATIVE_COUNTFM_UNIT_JUMP_RESOLUTION_BIAS_ORACLES_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_repository_commit='7ce372db78581451103baf1afdb8fecd50e12230',torch_version=torch.__version__,source_SHA256=pins,records=records,total_CPU_seconds=time.monotonic()-started,fullwidth_genes=2000,training_updates=0,ALZ_data_read=False,GPU_allocations=0,CUDA_initialized=False,upstream_modified=False,PG_or_frozen45_protocol_changed=False,findings='Conditional on a realized initial count x0, each coordinate changes by at most n_step over n_step native steps. For constant immigration, its native terminal law is exactly Binomial(n_step,1-exp(-rate*(1-eps)/n_step)), while the continuous-time process is Poisson(rate*(1-eps)). Oracle bias demonstrates a need to check sampling resolution; it does not establish bias magnitude for a learned ALZ model.',next_requirement='Before a fair separately declared count-FM benchmark, validate stage/depth conditioning, fresh training and restored sampling, then select and justify converged inference resolution on fitting/calibration information only. The current frozen45 comparison is unchanged.',SOTA_claim=False,full_goal_complete=False)
 (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=result['status'],result_path=str(OUT/'result.json'),CPU_seconds=result['total_CPU_seconds'])),flush=True)
if __name__=='__main__':main()
