"""Actual synthetic fullwidth native-update checks for conditional scalar arms.

CPU model qualification only. This neither activates the proposed search nor
satisfies its future runtime/source/resource/accounting qualification.
"""
import datetime,hashlib,importlib.util,json,os,sys,tempfile
from pathlib import Path
ROOT=Path('/ml2/alz');MODEL=ROOT/'analysis/distribution097/model'
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[MODEL,ROOT/'analysis/duration096',ROOT/'model',ROOT,SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,released_adapter as model,released_update as update
from native_fixture import context
spec=importlib.util.spec_from_file_location('state_fingerprint',ROOT/'confirmation097_runtime/state_fingerprint.py');finger=importlib.util.module_from_spec(spec);spec.loader.exec_module(finger)
spec=importlib.util.spec_from_file_location('unchanged_short_adapter',ROOT/'analysis/hpo500/model/released_adapter.py');short=importlib.util.module_from_spec(spec);spec.loader.exec_module(short)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=ROOT/'analysis/distribution097_conditional_next_search_design_20261107.json';p=read(plan)
 assert p['status']=='PROSPECTIVE_CONDITIONAL_CALIBRATION_CONTINUATION_DESIGN_NOT_ACTIVATED'
 assert len(p['candidates'])==10 and p['maximum_new_selection_fits']==20 and not p['activation']
 for path,pin in p['source_SHA256'].items():assert sha(path)==pin
 original=ROOT/'analysis/released_tuning_v096'
 for name in ('released_update.py','released_adapter.py'):assert (MODEL/name).read_bytes()==(original/name).read_bytes()
 native=read(ROOT/'analysis/distribution097/native_release.json')
 for path,pin in native['official_wheel_source_SHA256'].items():assert sha(path)==pin
 update.configure(dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.))
 rows=[];pairs=0
 for arm in p['candidates']:
  assert arm['training_seeds']==[0,1] and arm['endpoint']==20000 and arm['from_scratch'] and arm['source_checkpoint'] is None
  changed=[k for k,v in arm['hyperparameters'].items() if v!=p['reference_hyperparameters'][k]];assert changed==[arm['changed_field']]
  update.configure_hyperparameters(arm['hyperparameters']);recipe=update.recipe()
  assert recipe.name=='bispf_optimized' and recipe.total_steps==20000 and recipe.ema_decay==0 and recipe.generator_gradient_response=='none'
  assert recipe.lr_floor==recipe.network_lr_floor==1 and recipe.num_particles==512 and recipe.z_dim==8
  for seed in (0,1):
   with tempfile.TemporaryDirectory(prefix='conditional-native097-CPU-') as folder:
    cache=Path(folder);(cache/'manifest.json').write_text('{}');ctx=context(cache,fullwidth=True);states=[];logs=[]
    for adapter,horizon in ((short,500),(model,20000)):
     bundle,g,r,u,streams=adapter.build(ctx.mu,ctx.theta,'cpu',0,seed)
     assert bundle[-1].completed_steps==u.steps==0
     log=adapter.advance(ctx,bundle,g,r,u,execution_limit=horizon);pairs+=1
     state=adapter.snapshot(ctx,bundle,g,r,u,seed,streams,'synthetic');state.pop('fixed_endpoint')
     assert state['completed_updates']==state['policy']['completed_steps']==u.steps==1 and int(u.draws.sum())==32
     assert state['particlegan_version']=='0.9.7' and state['released_hyperparameters']==arm['hyperparameters']
     assert all(torch.isfinite(t).all() for m in bundle[1:4] for t in m.state_dict().values())
     states.append(finger.fingerprint(state));logs.append(log)
    assert states[0]==states[1] and logs[0]==logs[1]
    rows.append(dict(candidate_id=arm['id'],seed=seed,hyperparameters=arm['hyperparameters'],fullwidth2000gene_fresh_initialization=True,
       exact_next_pair_state_and_diagnostics_at500_and20000_limits=True,native_resolved_G_and_prior_LRs=logs[1]['G_prior_LRs'],native_resolved_D_LRs=logs[1]['D_LRs']))
 assert pairs==40 and len(rows)==20 and not torch.cuda.is_initialized()
 source=[Path(__file__),plan,ROOT/'analysis/duration096/native_fixture.py',ROOT/'confirmation097_runtime/state_fingerprint.py',ROOT/'analysis/hpo500/model/released_adapter.py',*MODEL.glob('*.py')]
 q=dict(status='CONDITIONAL_SCALAR_ARMS_NATIVE097_FULLWIDTH_SYNTHETIC_CPU_MODEL_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   candidates=10,matched_selection_seeds=[0,1],actual_full2000gene_optimizer_pairs=pairs,actual_full_configuration_seed_cases=20,
   rows=rows,source_SHA256={str(f):sha(f) for f in source},official_native_package_source_SHA256=native['official_wheel_source_SHA256'],
   original_native_model_source_unchanged=True,no_real_training_counts_used=True,no_real_calibration_or_heldout_values_read=True,
   CUDA_initialized=False,GPU_allocation_created=False,search_activated=False,live_runtime_deployment_qualified=False,
   limitation='Two fresh synthetic updates per configuration/seed across execution-limit fixtures establish native prefix equivalence and finite connected updates, not full training convergence or improved calibration quality. Actual search still conditional on all original fresh-seed outcomes, exact runtime/accounting qualification and current resource review.')
 output=ROOT/'analysis/distribution097_conditional_continuation_CPU_model_qualification_20261107.json';assert not output.exists();output.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:q[k] for k in ('status','candidates','actual_full2000gene_optimizer_pairs','actual_full_configuration_seed_cases','search_activated','live_runtime_deployment_qualified')}))
if __name__=='__main__':main()
