"""Preserve both terminal seed1 parents and the exact executed v3 source bytes."""
import datetime,hashlib,json
from pathlib import Path
R=Path('/ml2/alz');A='CAL_GPU0_V1_pure_turnover_759f745011e764dd9490_f0_pg_s1'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=read(R/'artifacts/allocations'/A/'deployment.json');authp=Path(d['authorization_path']);auth=read(authp);assert sha(authp)==d['authorization_SHA256']
 lp=Path(d['ledger_path']);j=read(lp)['jobs'][0];sp=lp.with_name(A+'-settlement.json');s=read(sp)
 assert j['status']=='failed' and j['reservation_released'] and j['exit_code']==1 and j['error']=="Refusal('protected peer identity changed')"
 assert s['uncertain'] is None and s['settlement']['status']=='settled' and s['settlement']['charged_GPU_seconds']==j['charged_seconds'] and s['no_foreign_processes_signalled']
 assert all(not (Path('/proc')/str(j[k])).exists() for k in ('guard_pid','child_pid'))
 assert not read(Path(auth['registry_path'])/'registry.json')['jobs']
 out=Path(d['output_dir']);assert not any((out/n).exists() for n in ['completed.json','r0','r1'])
 assert read(out/'training_phase.json')['parent']=='learned_observation'
 base=out/'fit/pg8/final.pt';learned=out/'fit/learned/recovery.pt';br=read(base.with_name('checkpoint_receipt.json'));lr=read(learned.with_name('checkpoint_receipt.json'));failure=read(learned.with_name('failure.json'))
 assert br['status']=='completed' and br['completed_updates']==20000 and br['initial_completed_updates']==14410 and br['new_model_updates']==5590 and br['SHA256']==sha(base)
 assert lr['status']=='incomplete' and lr['completed_updates']==9237 and lr['SHA256']==sha(learned)
 assert failure['completed_updates']==9237 and failure['phase']=='ready' and failure['primary_endpoint_complete'] is False
 for path,pin in auth['research_source_SHA256'].items():assert sha(path)==pin,'Original executed source must be preserved before editing'
 dest=R/'artifacts/confirmation_recovery_lineage'/A/'terminal';assert not dest.exists();dest.mkdir(parents=True)
 paths={'PG8_completed':base,'learned_interrupted':learned};cp={}
 for key,path in paths.items():
  target=dest/('pg8-step20000.pt' if key=='PG8_completed' else 'learned-step9237.pt');target.write_bytes(path.read_bytes());assert sha(target)==sha(path);cp[key]={'path':str(target),'SHA256':sha(target),'completed_updates':20000 if key=='PG8_completed' else 9237}
 for path,pin in auth['research_source_SHA256'].items():
  target=dest/'source'/path.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==pin
 evidence=[authp,lp,sp,base.with_name('checkpoint_receipt.json'),learned.with_name('checkpoint_receipt.json'),learned.with_name('failure.json'),out/'failure.json',R/'analysis/exact_seed1_recovery_protocol_v2.json',R/'analysis/exact_seed1_native_restore_verification_v2.json',R/'analysis/gpu0_turnover_independent_review.json']
 for i,path in enumerate(evidence):(dest/(str(i)+'-'+path.name)).write_bytes(path.read_bytes())
 proof={'status':'FROZEN_CLEAN_DESKTOP_TURNOVER_PG8_COMPLETE_LEARNED9237_NO_QUALITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':A,'checkpoints':cp,'total_completed_updates':29237,'new_updates_this_allocation':14827,'remaining_learned_updates':10763,'charged_GPU_seconds':j['charged_seconds'],'original_absolute_deadline_unix':auth['expires_unix'],'previous_initial_updates':14410,'previous_PG8_checkpoint_SHA256':auth['checkpoint_SHA256'],'execution_source_SHA256':auth['research_source_SHA256'],'canonical_evidence_SHA256':{str(p):sha(p) for p in evidence},'quality_result_created':False,'owned_cleanup_and_settlement_verified':True,'terminal_old_boot':j['guard_identity']['boot'],'current_boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'source_SHA256':{str(Path(__file__).resolve()):sha(__file__)}}
 p=R/'analysis/learned9237_interruption.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k not in ('execution_source_SHA256','canonical_evidence_SHA256')},indent=2))
if __name__=='__main__':main()
