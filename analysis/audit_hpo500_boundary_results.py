"""Independent saved-count/endpoint audit and paired fresh-seed recomputation."""
import argparse,datetime,gc,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');NEW=ROOT/'analysis/hpo500_boundary';OLD=ROOT/'analysis/hpo500';MODEL=OLD/'model'
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[NEW,MODEL,ROOT/'model',ROOT,SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import numpy as np,torch,search,released_update as update,released_adapter as model
from calibration import score,average
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def live(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def main():
 p=argparse.ArgumentParser();p.add_argument('--original-audit',type=Path,required=True);p.add_argument('--name',required=True);args=p.parse_args();dest=ROOT/'analysis'/(args.name+'.json');assert not dest.exists()
 original_audit=read(args.original_audit);assert original_audit['completed_selection_candidates']==25 and len(original_audit['rows'])==37
 assert original_audit['full_calibration_replicates_independently_rescored']==2*sum(r['finite'] for r in original_audit['rows'])
 for row in original_audit['rows']:assert row['result_SHA256']==sha(OLD/'results'/(row['attempt']+'.json'))
 native=read(ROOT/'analysis/released_tuning_v096/native_release.json')
 for path,pin in native['official_wheel_source_SHA256'].items():assert sha(path)==pin
 assert sha(native['package_metadata_path'])==native['package_metadata_SHA256']
 assert (MODEL/'released_update.py').read_bytes()==(ROOT/'analysis/released_tuning_v096/released_update.py').read_bytes()
 anchor=read(NEW/'anchor.json');assert anchor['confirmation_result_files_present_at_capture']==0 and not anchor['confirmation_metric_values_read_for_selection']
 assert anchor['prospective_rule_SHA256']==sha(NEW/'prospective_protocol.json')
 for attempt,pin in anchor['original25_seed0_result_SHA256'].items():assert sha(OLD/'results'/(attempt+'.json'))==pin
 files=sorted((NEW/'results').glob('*.json'));rows=[read(p) for p in files];selection=sorted([r for r in rows if r['phase']=='search'],key=lambda r:r['candidate_id']);prior=[]
 assert 4<=len(selection)<=6
 for i,row in enumerate(selection):
  assert read(NEW/'proposals'/(row['candidate_id']+'.json'))==search.proposal(i,prior);prior.append(row)
 assert len(selection)==4+len(search.refinements(selection[:4]))
 frozen=read(NEW/'frozen_selection.json');winner=search.best(selection);assert frozen['candidate_id']==(winner['candidate_id'] if winner['candidate_id']!=anchor['anchor_candidate'] else None)
 hashed=0;rescored=0;native_finite=0
 cfg={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
 update.configure(cfg);update.configure_hyperparameters(anchor['anchor_hyperparameters']);anchor_recipe=update.recipe().to_dict()
 for r in rows:
  assert r['endpoint']==500 and r['calibration_only'] and r['clean_settlement'] and r['owned_processes_and_lease_absent']
  assert r['protocol_SHA256']==sha(NEW/'prospective_protocol.json') and r['proposal_SHA256']==sha(NEW/'proposals'/(r['candidate_id']+'.json'))
  assert r['hyperparameters']==read(NEW/'proposals'/(r['candidate_id']+'.json'))['hyperparameters']
  auth=read(ROOT/'artifacts/allocations'/r.get('physical_attempt',r['attempt'])/'authorization.json');job=read(auth['ledger_path'])['jobs'][0]
  assert not live(job['guard_identity']) and not live(job['child_identity'])
  if not r['finite']:
   assert r['failure']['error'].startswith('FloatingPointError(');continue
  native_finite+=1;assert r['from_scratch'] and r['independent_saved_score_audit'] and r['physical_start_update']==0 and r['new_phase_model_updates']==500
  for file,pin in r['files_SHA256'].items():assert sha(file)==pin;hashed+=1
  cp=next(file for file in r['files_SHA256'] if file.endswith('/fit/final.pt'));q=torch.load(cp,map_location='cpu',weights_only=False)
  assert q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==q['usage']['steps']==500
  assert q['particlegan_version']=='0.9.6' and q['variant_config']==cfg and q['training_streams']==model.training_streams(0,r['seed'])
  assert q['released_hyperparameters']==r['hyperparameters'] and q['penalty_options']=={'input_domain':'continuous'} and q['fold']==0 and q['seed']==r['seed']
  assert q['cache_manifest_SHA256']==sha(SCIENCE/'caches/fold0/manifest.json')
  assert all(torch.isfinite(value).all() for state in q['policy']['models'].values() for value in state.values())
  assert int(q['usage']['draws'].sum())==16000 and int(q['usage']['unsampled_moved_steps'].sum())==0
  update.configure_hyperparameters(r['hyperparameters']);rec=update.recipe();assert q['recipe']==q['policy']['recipe']==rec.to_dict()
  assert rec.ema_decay==0 and rec.generator_gradient_response=='none' and rec.total_steps==20000 and rec.lr_floor==rec.network_lr_floor==1
  assert {k for k in anchor_recipe if anchor_recipe[k]!=rec.to_dict()[k]}=={'reg_kappa'}
  out=Path(cp).parent.parent;c=read(out/'completed.json');assert not c['evaluation_targets_read'] and c['from_scratch'];reps=[]
  for rep in (0,1):
   folder=out/f'r{rep}';m=read(folder/'metrics.json');assert m['generated_cells']==320000 and m['training_seed']==r['seed'] and m['generation_replicate']==rep
   with np.load(folder/'arrays.npz',allow_pickle=False) as a:measured=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
   for k,v in measured.items():
    if isinstance(v,(int,float,list)) and k!='donors':np.testing.assert_allclose(v,m[k],rtol=0,atol=1e-10)
   for stage in range(4):
    with np.load(folder/f'stage{stage}_exemplar.npz',allow_pickle=False) as a:
     counts=a['counts'];assert counts.shape==(1,400,2000) and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
   assert m==r['replicate_metrics'][rep];reps.append(m);rescored+=1
  assert average(reps)==r['metrics'];del q;gc.collect()
 controls=[read(OLD/'results'/f'CAL_GPU0_V1_h5_c000_confirm_f0_pg_s{s}.json') for s in (1,2,3,4)];assert all(r['finite'] for r in controls)
 ids=[cid for cid in read(OLD/'finalists.json')['candidates'] if cid!='c000']+([frozen['candidate_id']] if frozen['candidate_id'] else [])
 report=read(NEW/'complete_report.json');declared={r['candidate']:r for r in report['comparisons']};comparisons=[];indices=np.random.default_rng(20261106).integers(0,4,size=(10000,4))
 for cid in ids:
  folder=NEW if cid.startswith('k') else OLD;candidate=[read(folder/'results'/f'CAL_GPU0_V1_h5_{cid}_confirm_f0_pg_s{s}.json') for s in (1,2,3,4)]
  if not all(r['finite'] for r in candidate):assert not declared[cid]['eligible'];continue
  es=np.array([r['metrics']['energy_score']-c['metrics']['energy_score'] for r,c in zip(candidate,controls)]);ed=np.array([r['metrics']['cell_energy_distance']-c['metrics']['cell_energy_distance'] for r,c in zip(candidate,controls)]);spread=np.mean([r['metrics']['spread_ratio'] for r in candidate]);eligible=bool(es.mean()<0 and ed.mean()<=0 and .7<=spread<=1.3 and (es<0).sum()>=3)
  mean_es=float(np.mean([r['metrics']['energy_score'] for r in candidate]));mean_ed=float(np.mean([r['metrics']['cell_energy_distance'] for r in candidate]));interval=np.percentile(es[indices].mean(1),[2.5,97.5]).tolist();d=declared[cid]
  assert d['eligible']==eligible;np.testing.assert_allclose([d['mean_energy_score'],d['mean_cell_energy_distance'],d['mean_spread_ratio']],[mean_es,mean_ed,spread],rtol=0,atol=1e-12);np.testing.assert_allclose(d['paired_seed_ES_deltas'],es,rtol=0,atol=1e-12);np.testing.assert_allclose(d['paired_seed_cellED_deltas'],ed,rtol=0,atol=1e-12);np.testing.assert_allclose(d['ES_delta95percent_seed_bootstrap_interval'],interval,rtol=0,atol=1e-12)
  comparisons.append(dict(candidate=cid,eligible=eligible,mean_ES=mean_es,mean_cellED=mean_ed,mean_spread=float(spread),paired_ES_deltas=es.tolist(),paired_cellED_deltas=ed.tolist(),seed_bootstrap95=interval))
 eligible=sorted([r for r in comparisons if r['eligible']],key=lambda r:(r['mean_ES'],r['mean_cellED'],r['candidate']));selected=eligible[0]['candidate'] if eligible else 'c000';assert selected==report['selected_candidate']
 folder=NEW if selected.startswith('k') else OLD;assert report['selected_hyperparameters']==read(folder/'proposals'/(selected+'.json'))['hyperparameters']
 total=sum(r['charged_GPU_seconds'] for r in rows)+sum(r['charged_GPU_seconds'] for r in original_audit['rows']);assert math.isclose(total,report['total_actual_GPU_seconds'],abs_tol=1e-8)
 for name in ('hpo500_runtime','hpo500_r2_runtime','hpo500_r3_runtime','hpo500_r4_runtime','hpo500_boundary_runtime'):assert not read(ROOT/name/'plan.json')['enabled']
 for name in ('gpu1-hpo500-r4-20261106','gpu1-hpo500-boundary-20261106'):assert not read(ROOT/'artifacts'/name/'live_registry/registry.json')['jobs']
 budget=read(P/'project_gpu_budget.json');assert not budget['open_reservations']
 assert not torch.cuda.is_initialized()
 proof=dict(status='INDEPENDENT_ORIGINAL_AND_BOUNDARY500STEP_NATIVE_COUNTS_AND_FRESH_SEED_AUDIT_PASS',original_audit_path=str(args.original_audit),original_audit_SHA256=sha(args.original_audit),boundary_report_SHA256=sha(NEW/'complete_report.json'),source_SHA256=sha(__file__),new_saved_artifacts_hashed=hashed,new_full_calibration_replicates_rescored=rescored,new_finite_models_at500=native_finite,new_failed_trials_retained=sum(not r['finite'] for r in rows),original_and_new_model_count=len(original_audit['rows'])+len(rows),comparisons=comparisons,selected_candidate=selected,selected_hyperparameters=report['selected_hyperparameters'],total_actual_GPU_seconds=total,all_activation_flags_reverted=True,owned_resource_registries_empty=True,canonical_open_reservations=0,heldout_used=False,formulation_changed=False,native_version='0.9.6',EMA_decay=0,calibration_donors=4,fresh_confirmation_seeds=[1,2,3,4],all_original_and_new_raw_result_hashes_verified=True,CUDA_initialized=False,limitation='Same four adaptive calibration donors; bootstrap is over training seeds and is not independent biological validation or a SOTA claim.')
 proof['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 dest.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:proof[k] for k in ('status','original_and_new_model_count','selected_candidate','total_actual_GPU_seconds','new_full_calibration_replicates_rescored')},indent=2))
if __name__=='__main__':main()
