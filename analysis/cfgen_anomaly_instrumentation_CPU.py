"""Synthetic native CFGen debug-instrumentation timing and finite-state parity.

The frozen comparator explicitly declares detect_anomaly=True. This experiment
does not qualify changing that recipe in the current45-model comparison.
"""
import contextlib,copy,datetime,gc,hashlib,json,logging,os,signal,statistics,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
SITE=SCIENCE.parent/'isolated/cfgen-cu124-py311/lib/python3.11/site-packages'
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[ROOT/'analysis/benchmark097_cfgen',ROOT/'cfgen_gpu_qualification_runtime',SCIENCE,PROJECT,SITE]))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import fit_adapter as adapter
from fit_adapter import torch,np
from state_fingerprint import fingerprint
STATE=ROOT/'artifacts/cfgen-anomaly-instrumentation-CPU-20261108'
RESULT=ROOT/'analysis/cfgen_anomaly_instrumentation_CPU_result_20261108.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def main():
 assert not torch.cuda.is_initialized() and not STATE.exists() and not RESULT.exists()
 STATE.mkdir();native_trainer=adapter.Trainer;native_recipe=adapter.RECIPE
 assert native_recipe['detect_anomaly'] is True
 pins={str(Path(__file__).resolve()):sha(__file__),str(Path(adapter.__file__).resolve()):sha(adapter.__file__),**read(ROOT/'analysis/benchmark097_cfgen/protocol.json')['source_SHA256']}
 for path,pin in pins.items():assert sha(path)==pin,path
 logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
 def timeout(signum,frame):raise TimeoutError('Finite synthetic CPU diagnostic ended')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
 rng=np.random.default_rng(20261108);counts=rng.poisson(3,(512,2000)).astype(np.float32);stage=np.arange(4).repeat(128)
 names=np.array([f'SYNTH_CELL_{i}' for i in range(512)]);genes=np.array([f'SYNTH_GENE_{i}' for i in range(2000)])
 def run(seed,debug,warmup=False):
  def fixture_trainer(*args,**kwargs):
   assert kwargs['detect_anomaly'] is True and kwargs['accelerator']=='cpu'
   kwargs['detect_anomaly']=debug
   return native_trainer(*args,**kwargs)
  adapter.Trainer=fixture_trainer
  # Receipt metadata states the actual synthetic fixture setting truthfully.
  # The original native recipe object and all on-disk sources remain unchanged.
  adapter.RECIPE=dict(native_recipe,detect_anomaly=debug)
  out=STATE/(f'warmup' if warmup else f'seed{seed}-debug{int(debug)}')
  with (STATE/'fixture.log').open('a') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
   started=time.monotonic();model,receipt=adapter.fit_arrays(counts,stage,names,genes,out,seed,'cpu',time.monotonic()+120,synthetic=True);elapsed=time.monotonic()-started
  assert receipt['synthetic'] and receipt['AE_updates']==receipt['FM_updates']==4 and receipt['total_optimizer_updates']==8
  assert receipt['native_recipe']['detect_anomaly'] is debug and receipt['frozen_encoder_parameters_buffers_and_modes_preserved']
  ae=torch.load(out/'AE/final.ckpt',map_location='cpu',weights_only=False);fm=torch.load(out/'FM/final.ckpt',map_location='cpu',weights_only=False)
  state=dict(model=model.state_dict(),external_features={k:f.state_dict() for k,f in model.feature_embeddings.items()},AE_optimizer_states=ae['optimizer_states'],FM_optimizer_states=fm['optimizer_states'],AE_global_step=ae['global_step'],FM_global_step=fm['global_step'],AE_rng=ae['matched_CFGen_RNG'],FM_rng=fm['matched_CFGen_RNG'])
  fp=fingerprint(state);saved=out/'retained_finite_native_state.pt';torch.save(state,saved)
  assert fingerprint(torch.load(saved,map_location='cpu',weights_only=False))==fp
  curve=[json.loads(line) for line in (out/'learning_curve.jsonl').read_text().splitlines()]
  losses=[(row['phase'],row['phase_optimizer_updates'],row['native_loss']) for row in curve]
  result=dict(seed=seed,actual_detect_anomaly=debug,warmup=warmup,CPU_wall_seconds=elapsed,native_optimizer_updates=8,finite_native_state_fingerprint=fp,retained_state_path=str(saved),retained_state_SHA256=sha(saved),logged_native_losses=losses,synthetic=True,eligible_for_current_frozen_comparison=False)
  del model,ae,fm,state;gc.collect();assert not torch.cuda.is_initialized()
  return result
 try:
  warm=run(999,True,warmup=True);records=[];pairs=[]
  for seed,order in [(0,(True,False)),(1,(False,True)),(2,(True,False))]:
   group=[run(seed,debug) for debug in order];records.extend(group)
   on=next(r for r in group if r['actual_detect_anomaly']);off=next(r for r in group if not r['actual_detect_anomaly'])
   assert on['finite_native_state_fingerprint']==off['finite_native_state_fingerprint']
   assert on['logged_native_losses']==off['logged_native_losses']
   pairs.append(dict(seed=seed,exact_model_external_features_optimizer_rng_and_clock_parity=True,exact_logged_loss_parity=True,CPU_seconds_debug_on=on['CPU_wall_seconds'],CPU_seconds_debug_off=off['CPU_wall_seconds'],observed_CPU_speedup=on['CPU_wall_seconds']/off['CPU_wall_seconds']))
   print(json.dumps(pairs[-1]),flush=True)
  for path,pin in pins.items():assert sha(path)==pin,path
  assert adapter.RECIPE is not native_recipe and native_recipe['detect_anomaly'] is True
  q=dict(status='COMPLETE_SYNTHETIC_NATIVE_CFGEN_ANOMALY_INSTRUMENTATION_CPU_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),warmup=warm,records=records,paired_results=pairs,median_observed_CPU_speedup=statistics.median(p['observed_CPU_speedup'] for p in pairs),actual_synthetic_optimizer_updates_measured=48,additional_warmup_updates=8,training_cells_per_synthetic_fixture=512,genes=2000,original_explicit_gradient_and_loss_finiteness_callbacks_preserved=True,finite_trajectory_state_and_logged_loss_parity=True,frozen_recipe_requires_anomaly_detection=True,alternative_not_eligible_for_current45=True,current_frozen_or_active_sources_modified=False,not_a_GPU_throughput_measurement=True,no_equivalent_failure_semantics_claim=True,real_ALZ_data_read=False,new_ALZ_training_fits=0,new_GPU_allocations=0,CUDA_initialized=False,source_SHA256=pins,limitations='Three short paired synthetic CPU runs, not a production training or GPU speed estimate. Autograd anomaly instrumentation changes debugging/error-detection behavior; matching finite trajectories does not establish identical failure detection. Detect_anomaly=True is explicitly in the current frozen comparator recipe. No current or remaining scientific cell is altered or admitted by this diagnostic.',full_goal_complete=False,SOTA_claim=False)
  RESULT.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=q['status'],median_observed_CPU_speedup=q['median_observed_CPU_speedup'],new_ALZ_training_fits=0,current45_unchanged=True)),flush=True)
 finally:
  adapter.Trainer=native_trainer;adapter.RECIPE=native_recipe;signal.alarm(0)
if __name__=='__main__':main()
