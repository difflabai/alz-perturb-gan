"""Freeze terminal infrastructure interruption without changing any native allocation.

Read-only while live; after clean terminal closure archive exact newest PG8 state,
canonical receipts and actual execution source bytes. No new model/run/seed choice.
"""
import argparse,datetime,hashlib,json
from pathlib import Path
ROOT=Path('/ml2/alz')
DEFAULT='CAL_GPU0_V1_pure_recover_bf27534c3a4250b2b4d2_f0_pg_s1'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',default=DEFAULT);parser.add_argument('--check-only',action='store_true');args=parser.parse_args();assert args.attempt==DEFAULT
 dep=read(ROOT/'artifacts/allocations'/args.attempt/'deployment.json');auth_path=Path(dep['authorization_path']);assert sha(auth_path)==dep['authorization_SHA256'];auth=read(auth_path)
 ledger=Path(dep['ledger_path']);record=read(ledger);assert len(record['jobs'])==1;job=record['jobs'][0]
 if job['status']=='active':
  print(json.dumps({'status':'LIVE_ALLOCATION_PRESERVE_NO_ARCHIVE_OR_NEW_JOB','attempt':args.attempt,'guardian_pid':job['guard_pid'],'worker_pid':job.get('child_pid'),'worker_handle_present':(Path('/proc')/str(job.get('child_pid'))/'stat').exists()}));return
 assert job['status']=='failed' and job['reservation_released'] and job['exit_code']!=0
 settlement_path=ledger.with_name(args.attempt+'-settlement.json');settlement=read(settlement_path);assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
 assert settlement['settlement']['charged_GPU_seconds']==job['charged_seconds']
 assert not (Path('/proc')/str(job['guard_pid'])/'stat').exists() and not (Path('/proc')/str(job['child_pid'])/'stat').exists(),'Terminal bookkeeping alone is insufficient; owned handles must be gone'
 assert not read(Path(auth['registry_path'])/'registry.json')['jobs'],'Owned resource registry must be empty'
 assert any(text in job.get('error','') for text in ['protected peer identity changed','Protected process identity changed','Protected compute/graphics membership changed','Unregistered GPU allocation(s)']),'Not the declared external-peer interruption; inspect actual failure'
 out=Path(dep['output_dir']);assert not (out/'completed.json').exists() and not (out/'r0').exists() and not (out/'r1').exists() and not (out/'fit/learned').exists()
 phase=read(out/'training_phase.json');assert phase['parent']=='PG8' and phase['training_seed']==1
 cp=out/'fit/pg8/recovery.pt';receipt_path=cp.with_name('checkpoint_receipt.json');receipt=read(receipt_path);payload=cp.read_bytes();pin=hashlib.sha256(payload).hexdigest()
 assert pin==receipt['SHA256'] and receipt['status']=='incomplete' and receipt['path']=='recovery.pt'
 n=receipt['completed_updates'];assert type(n) is int and 8942<n<20000 and receipt['initial_completed_updates']==8942 and receipt['new_model_updates']==n-8942
 assert receipt['resume_parent_checkpoint_SHA256']=='cf2aeb33b2d21fa3195cfc4c2e10a71d49e5bd38e03d7a9159cd0ee146e025b1' and receipt['native_recipe_schedule_horizon']==20000
 failure=read(cp.parent/'failure.json');assert failure['completed_updates']==n and failure['phase']=='ready' and failure['primary_endpoint_complete'] is False
 original=read(ROOT/'analysis/exact_seed1_recovery_protocol.json');assert auth['expires_unix']==original['wall_clock_envelope']['absolute_recovery_deadline_unix'] and (40000-n)==auth['new_model_updates']-(n-auth['initial_completed_updates'])
 for p,h in auth['research_source_SHA256'].items():assert sha(p)==h,'Archive actual original source bytes before any continuation-framework edit'
 source=auth['research_source_SHA256'];folder=ROOT/'artifacts/confirmation_recovery_lineage'/args.attempt/'terminal';assert not folder.exists()
 proof={'status':'FROZEN_CLEAN_EXTERNAL_PEER_INTERRUPTION_NEWEST_EXACT_PG8_STATE_NO_QUALITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':args.attempt,'initial_completed_updates':8942,'completed_updates':n,'new_model_updates':n-8942,'checkpoint_SHA256':pin,'original_parent_checkpoint_SHA256':receipt['resume_parent_checkpoint_SHA256'],'checkpoint_receipt':receipt,'new_training_updates_remaining':40000-n,'original_confirmation_freeze_SHA256':original['original_confirmation_freeze_SHA256'],'original_seed1_attempt':original['original_attempt'],'original_absolute_deadline_unix':auth['expires_unix'],'charged_GPU_seconds':job['charged_seconds'],'guard_error':job['error'],'quality_result_created':False,'learned_parent_started':False,'owned_cleanup_and_settlement_verified':True,'next_scientific_seed':1,'execution_source_SHA256':source,'canonical_input_SHA256':{str(p):sha(p) for p in [auth_path,ledger,settlement_path,receipt_path,cp.parent/'failure.json',ROOT/'analysis/exact_seed1_native_restore_verification.json',ROOT/'analysis/exact_seed1_recovery_protocol.json']},'scope':'Infrastructure interruption of the SAME frozen seed1 trajectory. Archive newest ready fullstate; preserve original failed8942 attempt and this phase charges. No replacement/ranking/restart or deadline reset. Any next native phase needs separately scoped source/profile review and actual fullCUDA restore/replay qualification of THIS newest state.','source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())}}
 if args.check_only:print(json.dumps({k:proof[k] for k in ['status','completed_updates','new_training_updates_remaining','checkpoint_SHA256']}));return
 folder.mkdir(parents=True,exist_ok=False);archive=folder/f'pg8-step{n}.pt';archive.write_bytes(payload);proof['archived_checkpoint']=str(archive)
 for p,h in source.items():
  target=folder/'source'/p.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 for label,p in [('authorization.json',auth_path),('ledger.json',ledger),('settlement.json',settlement_path),('checkpoint_receipt.json',receipt_path),('failure.json',cp.parent/'failure.json')]: (folder/label).write_bytes(Path(p).read_bytes())
 dest=ROOT/'analysis'/(args.attempt+'-turnover-interruption.json');assert not dest.exists();dest.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':proof['status'],'completed_updates':n,'checkpoint_SHA256':pin,'archived_checkpoint':str(archive),'proof_path':str(dest),'charged_GPU_seconds':job['charged_seconds']}))
if __name__=='__main__':main()
