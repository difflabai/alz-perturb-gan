"""Exploratory loss/quality association, not a new LR or checkpoint selector."""
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from scipy.stats import spearmanr

# Defined before loading the observations; no best-proxy selection is made.
PROXIES=('G_total','D_total','critic_gap','penalty_ema_residual')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def proxy(row,name):
 return row['penalty_diagnostics']['ema_residual'] if name=='penalty_ema_residual' else row[name]
def rho(x,y):
 value=float(spearmanr(x,y).statistic)
 return value if math.isfinite(value) else None

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 result_path=ROOT/'analysis/t005_checkpoint_CPU_multiseed_result_20261108.json'
 audit_path=ROOT/'analysis/t005_checkpoint_CPU_multiseed_root_audit_20261108.json'
 plan_path=ROOT/'analysis/t005_checkpoint_CPU_multiseed_protocol_20261108.json'
 q=read(result_path);audit=read(audit_path);plan=read(plan_path)
 assert audit['result_SHA256']==sha(result_path) and audit['all15_saved_snapshots_retained']
 assert q['training_seeds']==list(range(5)) and q['saved_updates']==[5000,10000,20000]
 inputs={str(p):sha(p) for p in (result_path,audit_path,plan_path)}
 seed0_plan_path=q['all_seed_records']['0']['protocol_path'];seed0_plan=read(seed0_plan_path)
 assert sha(seed0_plan_path)==q['all_seed_records']['0']['protocol_SHA256']
 inputs[seed0_plan_path]=sha(seed0_plan_path)
 groups=[dict(seed=0,existing_attempt=Path(seed0_plan['checkpoints'][0]['path']).parents[1].name,checkpoints=seed0_plan['checkpoints']),*plan['additional_checkpoint_groups']]
 records=[]
 for group in groups:
  seed=group['seed'];curve_path=ROOT/'runs'/group['existing_attempt']/'fit/learning_curve.jsonl'
  inputs[str(curve_path)]=sha(curve_path)
  curve=[json.loads(line) for line in curve_path.read_text().splitlines()]
  assert len(curve)==160 and curve[0]['step']==1 and curve[-1]['step']==20000
  assert len({row['step'] for row in curve})==len(curve)
  for row in curve:
   assert all(math.isfinite(proxy(row,k)) for k in PROXIES)
   np.testing.assert_allclose(row['G_prior_LRs'],[.012,.06],rtol=0,atol=1e-15)
   np.testing.assert_allclose(row['D_LRs'],[.018],rtol=0,atol=1e-15)
  snapshots=q['all_seed_records'][str(seed)]['rows']
  for snapshot in snapshots:
   step=snapshot['completed_updates'];window=[row for row in curve if step-1000<row['step']<=step]
   assert len(window)==8 and window[-1]['step']==step
   checkpoint=next(c for c in group['checkpoints'] if c['completed_updates']==step)
   assert snapshot['checkpoint_SHA256']==checkpoint['SHA256']==sha(checkpoint['path'])
   inputs[checkpoint['path']]=checkpoint['SHA256']
   means={k:statistics.mean(proxy(row,k) for row in window) for k in PROXIES}
   records.append(dict(training_seed=seed,saved_update=step,cell_energy_distance=snapshot['mean']['cell_energy_distance'],energy_score=snapshot['mean']['energy_score'],window_start_exclusive=step-1000,training_curve_observations=len(window),window_proxy_means=means,last_reported_proxy_values={k:proxy(window[-1],k) for k in PROXIES}))
 assert len(records)==15 and {(x['training_seed'],x['saved_update']) for x in records}=={(s,t) for s in range(5) for t in (5000,10000,20000)}
 paired=[]
 for seed in range(5):
  a=next(x for x in records if x['training_seed']==seed and x['saved_update']==10000)
  b=next(x for x in records if x['training_seed']==seed and x['saved_update']==20000)
  paired.append(dict(training_seed=seed,cell_energy_distance=b['cell_energy_distance']-a['cell_energy_distance'],energy_score=b['energy_score']-a['energy_score'],proxy_changes_20k_minus10k={k:b['window_proxy_means'][k]-a['window_proxy_means'][k] for k in PROXIES}))
 correlations={k:dict(pooled15_spearman_with_calibration_ED=rho([x['window_proxy_means'][k] for x in records],[x['cell_energy_distance'] for x in records]),within_seed_demeaned_spearman_with_ED=rho([x['window_proxy_means'][k]-statistics.mean(z['window_proxy_means'][k] for z in records if z['training_seed']==x['training_seed']) for x in records],[x['cell_energy_distance']-statistics.mean(z['cell_energy_distance'] for z in records if z['training_seed']==x['training_seed']) for x in records]),last_iteration15_spearman_with_ED=rho([x['last_reported_proxy_values'][k] for x in records],[x['cell_energy_distance'] for x in records]),same_sign_proxy_and_ED_10k_to20k=int(sum(np.sign(x['proxy_changes_20k_minus10k'][k])==np.sign(x['cell_energy_distance']) for x in paired))) for k in PROXIES}
 for path,pin in inputs.items():assert sha(path)==pin,path
 result=dict(status='COMPLETE_EXPLORATORY_CURRENT_T005_TRAINING_LOSS_CALIBRATION_RELATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),predeclared_descriptive_proxies=list(PROXIES),records=records,paired_20k_minus10k=paired,correlations=correlations,checkpoints=15,training_seeds=5,calibration_donors=4,ED_worse20k_than10k_seeds=sum(x['cell_energy_distance']>0 for x in paired),ES_worse20k_than10k_seeds=sum(x['energy_score']>0 for x in paired),window_definition='Arithmetic mean of8 actually logged training iterations in the1000update interval ending at the saved checkpoint; instantaneous values also retained.',training_loss_values_are_pre_generator_update_within_last_completed_pair=True,primary_LR_control_metric_unchanged='calibration_cell_energy_distance',no_new_metric_or_checkpoint_selection=True,no_hypothesis_test_or_generalization_claim=True,no_controller_change=True,all_existing_six_LR_and_theta_protocols_unchanged=True,input_SHA256=inputs,source_SHA256=sha(__file__),new_training_fits=0,new_optimizer_updates=0,new_generated_cells=0,new_GPU_allocations=0,external_evaluation_values_read=False,limitations='Posthoc description of one calibration-selected setting, five initialization seeds and three repeated checkpoints. Fifteen rows are not independent observations or biological replicates. Correlation across observed checkpoints does not establish a causal, invariant or prospective controller signal; critic losses have no universal lower-is-better quality interpretation. CPU checkpoint sampling differs from training CUDA draws. Actual fresh matched LR comparison is still needed.',full_goal_complete=False,SOTA_claim=False)
 path=ROOT/'analysis/t005_training_loss_calibration_relation_CPU_20261108.json';assert not path.exists() and 'torch' not in sys.modules
 path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('status','correlations','ED_worse20k_than10k_seeds','ES_worse20k_than10k_seeds')}))

if __name__=='__main__':main()
