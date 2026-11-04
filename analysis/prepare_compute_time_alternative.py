"""Source-only continuation proposal; no execution authorization, leases or CUDA."""
import argparse,datetime,hashlib,json,math,sys
from pathlib import Path
ROOT=Path('/ml2/alz');ACTIVE='CAL_GPU0_V1_pure_learned_a5b327c4c32e01fa8555_f0_pg_s1';QUAL='CAL_GPU0_V1_qual_learned_a5b327c4c32e01fa8555_f0_pg_s1'
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def budget(rows,expected,cap=21600,qualification_cap=900,cleanup=30):
 assert [r['attempt'] for r in rows]==expected and len(set(expected))==len(expected),'Missing/duplicate/out-of-order retained canonical charge'
 assert all(type(r['charged_GPU_seconds']) in (int,float) and math.isfinite(r['charged_GPU_seconds']) and r['charged_GPU_seconds']>=0 for r in rows),'Invalid canonical charge'
 used=math.fsum(r['charged_GPU_seconds'] for r in rows);remaining=cap-used
 assert remaining>qualification_cap+cleanup,'Cumulative six-hour compute allowance exhausted'
 train=math.floor(remaining-qualification_cap-cleanup)
 assert used+qualification_cap+train+cleanup<=cap
 return {'cumulative_limit_GPU_seconds':cap,'retained_canonical_charge_GPU_seconds':used,'remaining_GPU_seconds':remaining,'new_qualification_cap_GPU_seconds':qualification_cap,'safe_training_cap_before_actual_qualification_settlement_seconds':train,'cleanup_headroom_GPU_seconds':cleanup,'rule':'After new qualification settles, recompute training cap=floor21600-minusEVERYretainedcharge-minusACTUALnewqualificationcharge-minus30cleanup; never grant another21600seconds. Newcaps must also fit canonical project budget and actual shared resources.'}
def canonical_row(attempt,ledger,settlement):
 l=read(ledger);s=read(settlement);assert len(l['jobs'])==1;j=l['jobs'][0]
 assert j['label']==attempt and j['status'] in ('failed','completed','budget_exhausted','interrupted') and j['reservation_released']
 assert s['attempt_id']==attempt and s['uncertain'] is None and s['settlement']['status']=='settled'
 assert j['charged_seconds']==s['settlement']['charged_GPU_seconds']
 return {'attempt':attempt,'ledger_path':str(ledger),'ledger_SHA256':sha(ledger),'settlement_path':str(settlement),'settlement_SHA256':sha(settlement),'charged_GPU_seconds':j['charged_seconds'],'status':j['status']}
def build():
 sys.path.insert(0,str(ROOT/'recovery_runtime'));import recovery_scope as scope
 p,_=scope.lineage();rows=[];expected=[r['attempt'] for r in p['retained_charge_records']]+[QUAL,ACTIVE]
 for r in p['retained_charge_records']:
  row=canonical_row(r['attempt'],Path(r['ledger_path']),Path(r['settlement_path']));assert all(row[k]==r[k] for k in r);rows.append(row)
 dep=read(ROOT/'artifacts/allocations'/ACTIVE/'deployment.json');auth=read(dep['authorization_path']);assert sha(dep['authorization_path'])==dep['authorization_SHA256'];ledgers=Path(dep['ledger_path']).parent
 rows.append(canonical_row(QUAL,ledgers/(QUAL+'.json'),ledgers/(QUAL+'-settlement.json')))
 terminal_path=ledgers/(ACTIVE+'-settlement.json');terminal=terminal_path.exists()
 if terminal:rows.append(canonical_row(ACTIVE,Path(dep['ledger_path']),terminal_path))
 else:assert read(dep['ledger_path'])['jobs'][0]['status']=='active'
 fit=Path(dep['output_dir'])/'fit';receipt=read(fit/'learned/checkpoint_receipt.json');raw=(fit/'learned'/receipt['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['SHA256'] and receipt==read(fit/'learned/checkpoint_receipt.json')
 assert receipt['initial_completed_updates']==9237 and receipt['new_model_updates']==receipt['completed_updates']-9237 and receipt['resume_parent_checkpoint_SHA256']==p['checkpoint_SHA256'];assert receipt['native_recipe_schedule_horizon']==20000 and 9237<receipt['completed_updates']<=20000
 pg8=fit/'pg8/final.pt';assert sha(pg8)==p['completed_parent_checkpoint']['SHA256']
 assert not (Path(dep['output_dir'])/'completed.json').exists(),'Complete quality already exists; audit it instead of preparing another trajectory'
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=ROOT/'artifacts/compute_time_alternative'/stamp;folder.mkdir(parents=True,exist_ok=False)
 cp=folder/f"learned-step{receipt['completed_updates']}.pt";cp.write_bytes(raw)
 archived={}
 for path,pin in auth['research_source_SHA256'].items():
  assert sha(path)==pin,'Actually executed source changed';dst=folder/'executed_source'/path.lstrip('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(path).read_bytes());archived[str(dst)]=pin
 for path,pin in p['scientific_source_SHA256'].items():assert sha(path)==pin,'Scientific source changed'
 if terminal:
  import gpu1_admission as a
  job=read(dep['ledger_path'])['jobs'][0]
  assert a.identity(job['guard_pid']) is None and a.identity(job['child_pid']) is None,'Owned process remains alive'
  assert not read(Path(auth['registry_path'])/'registry.json')['jobs'],'Owned lease remains'
  calc=budget(rows,expected)
 else:calc={'status':'PENDING_CURRENT_CANONICAL_SETTLEMENT_NO_FINAL_BUDGET','settled_charge_so_far_GPU_seconds':math.fsum(r['charged_GPU_seconds'] for r in rows),'current_attempt_required':ACTIVE,'current_charge_not_estimated_or_omitted':True}
 result={'status':'SOURCE_ONLY_TERMINAL_ALTERNATIVE_AWAITING_PARENT_DIRECTION' if terminal else 'SOURCE_ONLY_PROVISIONAL_ALTERNATIVE_CURRENT_RUN_ACTIVE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'operational_authorization':False,'GPU_allocation_created':False,'native_successor_launched':False,'policy_or_active_source_mutation':False,'request_message':'task11-ALZ12500-wall-deadline-report-0615-20261104','original_protocol_path':str(scope.PROTOCOL),'original_protocol_SHA256':sha(scope.PROTOCOL),'original_absolute_deadline_unix':p['wall_clock_envelope']['absolute_recovery_deadline_unix'],'original_wall_clock_protocol_completed':False,'proposed_only_change':'Replace elapsed wall time including outages/review gaps with <=21600 cumulatively charged seed1 GPU-seconds, counting all past and future qualifications/training/cleanup. Requires explicit parent direction and fresh reviewed implementation before execution. Preserve original failed/capped result separately.','expected_all_retained_attempts':expected,'retained_canonical_charge_records':rows,'budget':calc,'checkpoint':{'path':str(cp),'SHA256':sha(cp),'completed_updates':receipt['completed_updates'],'receipt':receipt,'terminal_checkpoint':terminal},'completed_PG8':p['completed_parent_checkpoint'],'executed_source_archive_SHA256':archived,'scientific_source_SHA256':p['scientific_source_SHA256'],'frozen_science':{'fold':0,'seed':1,'parent_endpoints':[20000,20000],'remaining_learned_updates':20000-receipt['completed_updates'],'variant_config':auth['variant_config'],'ensemble_config':auth['ensemble_config'],'calibration_replicates':[0,1],'whole_bag_mixture':'75% PG8 +25% learned; no real resampling','data_cache':auth['cache_dir'],'calibration_manifest_SHA256':auth['calibration_manifest_SHA256'],'evaluation_donors_excluded':True,'fixed_confirmation_seeds':[0,1,2],'seed_selection_or_dropping':False},'required_before_any_execution':['Parent resolves explicit original-wall-deadline versus cumulative-compute protocol change.','Current allocation terminal checkpoint, owned cleanup, lease release and exact canonical settlement independently verified.','Fresh source-only implementation/CPU refusal tests and current exact shared peer/resource profile use>=41.25GiB foreign known-peak forecast; no foreign control.','Actual latest terminal checkpoint full native restoration, completedPG8 exact reuse and two discarded next-update branches bitwise equality; independent CPU proof and clean qualification settlement.','Recompute cumulative charge including new qualification before finite continuation admission under canonical Controller.','Original saved weights initialize training; never discarded replay weights. Complete fixed endpoints then unchanged two replicas, audit counts/checkpoints/arrays/cleanup before reporting any score.','Explicit revised reporting/aggregation protocol for all three fixed seeds; no drop/rank/retry based on quality; retain original wall-capped attempt as incomplete.'],'incumbents_changed':False}
 out=folder/'proposal.json';out.write_text(json.dumps(result,indent=2)+'\n');assert 'torch' not in sys.modules
 return out,result
if __name__=='__main__':
 path,q=build();print(json.dumps({'path':str(path),'SHA256':sha(path),'status':q['status'],'learned_updates':q['checkpoint']['completed_updates'],'budget':q['budget']},indent=2))
