"""Freeze an actual incomplete owned phase and its evidence; never launch a successor."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
R=Path('/ml2/alz')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def row(name,ledger):
 settle=ledger.with_name(name+'-settlement.json');j=read(ledger)['jobs'][0];s=read(settle);assert j['status'] in ('failed','completed') and j['reservation_released'] and s['uncertain'] is None and s['settlement']['status']=='settled';assert j['charged_seconds']==s['settlement']['charged_GPU_seconds']
 return {'attempt':name,'ledger_path':str(ledger),'ledger_SHA256':sha(ledger),'settlement_path':str(settle),'settlement_SHA256':sha(settle),'charged_GPU_seconds':j['charged_seconds'],'status':j['status']}
def freeze(attempt):
 sys.path.insert(0,str(R/'recovery_runtime'));import recovery_scope as scope;import gpu1_admission as a
 p,_=scope.lineage();dep=read(R/'artifacts/allocations'/attempt/'deployment.json');auth=read(dep['authorization_path']);assert sha(dep['authorization_path'])==dep['authorization_SHA256'];assert auth['attempt_id']==scope.attempt(p['recovery_cell'])==attempt and auth['operation']=='resume_pure_ensemble'
 ledger=Path(dep['ledger_path']);j=read(ledger)['jobs'][0];settle=ledger.with_name(attempt+'-settlement.json');s=read(settle);out=Path(dep['output_dir']);receipt=read(out/'fit/learned/checkpoint_receipt.json');failure=read(out/'fit/learned/failure.json')
 assert j['status']=='failed' and j['reservation_released'] and s['uncertain'] is None and s['settlement']['status']=='settled';assert a.identity(j['guard_pid']) is None and a.identity(j['child_pid']) is None and not read(Path(auth['registry_path'])/'registry.json')['jobs']
 assert not (out/'completed.json').exists() and failure['phase']=='ready' and failure['completed_updates']==receipt['completed_updates']<20000 and receipt['status']=='incomplete'
 assert receipt['initial_completed_updates']==scope.PARENT_INITIAL and receipt['new_model_updates']==receipt['completed_updates']-scope.PARENT_INITIAL and receipt['resume_parent_checkpoint_SHA256']==p['checkpoint_SHA256']
 terminal=R/'artifacts/confirmation_recovery_lineage'/attempt/'terminal';terminal.mkdir(parents=True,exist_ok=False);cp=terminal/f"learned-step{receipt['completed_updates']}.pt";cp.write_bytes((out/'fit/learned'/receipt['path']).read_bytes());assert sha(cp)==receipt['SHA256']
 pg8=terminal/'pg8-step20000.pt';pg8.write_bytes((out/'fit/pg8/final.pt').read_bytes());assert sha(pg8)==p['completed_parent_checkpoint']['SHA256']
 for source,pin in auth['research_source_SHA256'].items():
  assert sha(source)==pin;dst=terminal/'source'/source.lstrip('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(source).read_bytes())
 records=list(p['retained_charge_records']);qual=scope.attempt(p['qualification_cell']);records+=[row(qual,ledger.with_name(qual+'.json')),row(attempt,ledger)]
 refs={'PG8_completed':{'path':str(pg8),'SHA256':sha(pg8),'completed_updates':20000},'learned_interrupted':{'path':str(cp),'SHA256':sha(cp),'completed_updates':receipt['completed_updates']}}
 evidence_paths=[Path(dep['authorization_path']),ledger,settle,out/'fit/learned/checkpoint_receipt.json',out/'fit/learned/failure.json',Path(p['prior_qualification_proof']),scope.NATIVE_PROOF,scope.PROTOCOL]
 evidence={'status':'VERIFIED_SHARED_RESOURCE_INTERRUPTION_NO_QUALITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,'previous_initial_updates':scope.INITIAL,'previous_learned_initial_updates':scope.PARENT_INITIAL,'previous_checkpoint_SHA256':p['checkpoint_SHA256'],'new_updates_this_allocation':receipt['new_model_updates'],'total_completed_updates':20000+receipt['completed_updates'],'remaining_learned_updates':20000-receipt['completed_updates'],'checkpoints':refs,'receipt':receipt,'failure':failure,'guard_error':s['guard_error'],'authorized_absolute_deadline_unix':auth['expires_unix'],'deadline_changed':False,'quality_result_created':False,'owned_cleanup_and_settlement_verified':True,'execution_source_SHA256':auth['research_source_SHA256'],'actual_source_archive':str(terminal/'source'),'all_retained_charge_records':records,'all_retained_charges_GPU_seconds':sum(x['charged_GPU_seconds'] for x in records),'canonical_evidence_SHA256':{str(q):sha(q) for q in evidence_paths},'native_latest_state_restoration_verified':False,'successor_launched':False}
 path=R/'analysis'/(attempt+'-interruption.json');assert not path.exists();path.write_text(json.dumps(evidence,indent=2)+'\n');return path,evidence
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args();path,e=freeze(args.attempt);print(json.dumps({'path':str(path),'SHA256':sha(path),'checkpoints':e['checkpoints'],'charges':e['all_retained_charges_GPU_seconds'],'remaining':e['remaining_learned_updates']},indent=2))
