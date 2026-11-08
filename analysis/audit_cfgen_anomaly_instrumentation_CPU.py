"""Independent retained-state and timing arithmetic audit of synthetic CPU runs."""
import datetime,hashlib,json,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path('/ml2/alz');SITE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/isolated/cfgen-cu124-py311/lib/python3.11/site-packages')
sys.path[:0]=[str(ROOT/'cfgen_gpu_qualification_runtime'),str(SITE)]
import torch
from state_fingerprint import fingerprint
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1)
 path=ROOT/'analysis/cfgen_anomaly_instrumentation_CPU_result_20261108.json';q=json.loads(path.read_bytes())
 for p,pin in q['source_SHA256'].items():assert sha(p)==pin,p
 records=[]
 for row in [q['warmup'],*q['records']]:
  assert sha(row['retained_state_path'])==row['retained_state_SHA256']
  state=torch.load(row['retained_state_path'],map_location='cpu',weights_only=False)
  assert fingerprint(state)==row['finite_native_state_fingerprint']
  assert state['AE_global_step']==state['FM_global_step']==4
  assert all(torch.isfinite(x).all() for x in state['model'].values())
  histograms={}
  for key in ('AE_optimizer_states','FM_optimizer_states'):
   steps=[float(v['step']) for saved in state[key] for v in saved['state'].values() if 'step' in v]
   # Native conditional paths may omit a parameter's gradient on some pairs.
   # Its Adam counter need not equal the phase's global optimizer clock.
   assert steps and all(0<x<=4 for x in steps)
   histograms[key]=dict(Counter(str(x) for x in steps))
  records.append(dict(seed=row['seed'],actual_detect_anomaly=row['actual_detect_anomaly'],warmup=row['warmup'],native_state_fingerprint=fingerprint(state),retained_state_SHA256=row['retained_state_SHA256'],per_parameter_Adam_step_histograms=histograms))
 for seed in range(3):
  pair=[r for r in records if not r['warmup'] and r['seed']==seed]
  assert len(pair)==2 and pair[0]['native_state_fingerprint']==pair[1]['native_state_fingerprint']
  assert pair[0]['per_parameter_Adam_step_histograms']==pair[1]['per_parameter_Adam_step_histograms']
 ratios=[]
 for seed in range(3):
  pair=[r for r in q['records'] if r['seed']==seed];on=next(r for r in pair if r['actual_detect_anomaly']);off=next(r for r in pair if not r['actual_detect_anomaly'])
  ratios.append(on['CPU_wall_seconds']/off['CPU_wall_seconds'])
 assert statistics.median(ratios)==q['median_observed_CPU_speedup'] and not torch.cuda.is_initialized()
 result=dict(status='PASSED_INDEPENDENT_RETAINED_CFGEN_CPU_STATES_CLOCKS_AND_TIMING_ARITHMETIC_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),result_path=str(path),result_SHA256=sha(path),retained_state_bundles_verified=7,matched_seed_pairs_exact=3,records=records,native_AE_and_FM_global_optimizer_clocks_verified=True,per_parameter_Adam_steps_positive_bounded_and_exactly_paired=True,initial_audit_assumption_corrected='Requiring every parameter Adam step to equal the global optimizer clock was too strong. Retained native conditional FM fixtures have some counters2 when phaseglobalclock4, identically in both debug arms. No model or saved state was changed.',median_CPU_speedup_recomputed=statistics.median(ratios),CPU_timings_not_GPU_predictions=True,current_declared_detect_anomaly_true_preserved=True,alternative_ineligible_for_current_frozen45=True,new_ALZ_training_fits=0,new_GPU_allocations=0,source_SHA256=sha(__file__),full_goal_complete=False,SOTA_claim=False)
 dest=ROOT/'analysis/cfgen_anomaly_instrumentation_CPU_root_audit_20261108.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('status','retained_state_bundles_verified','matched_seed_pairs_exact','median_CPU_speedup_recomputed')}))
if __name__=='__main__':main()
