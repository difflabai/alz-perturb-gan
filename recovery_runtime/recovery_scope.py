"""Exact retained lineage under the owner's explicit five-day research extension."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'recovery_runtime'
PROTOCOL=ROOT/'analysis/exact_seed1_recovery_protocol_v4.json'
CPU_PROOF=ROOT/'analysis/recovery_runtime_cpu_qualification_v4.json'
NATIVE_PROOF=ROOT/'analysis/exact_seed1_native_restore_verification_v4.json'
INITIAL=33265
PARENT_INITIAL=13265
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen():
 sys.path.insert(0,str(ROOT/'confirmation_runtime'))
 from ensemble_confirmation_scope import frozen as original_frozen
 return original_frozen()
def _prior():
 import importlib.util
 p=read(PROTOCOL);path=Path(p['retained_scope_path']);assert sha(path)==p['retained_scope_SHA256']
 spec=importlib.util.spec_from_file_location('retained_wallcapped_scope_v3',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
def lineage():
 p=read(PROTOCOL);previous=_prior();old,f=previous.lineage()
 assert sha(p['lineage_parent_protocol'])==p['lineage_parent_protocol_SHA256'] and read(p['lineage_parent_protocol'])==old
 for key in ('original_attempt','original_confirmation_freeze','original_confirmation_freeze_SHA256','scientific_source_SHA256','frozen_baseline_pins','frozen_ensemble_config','completed_parent_checkpoint'):
  assert p[key]==old[key],'Retained scientific lineage differs: '+key
 assert sha(p['owner_extension_path'])==p['owner_extension_SHA256'];window=read(p['owner_extension_path'])
 assert window['status']=='OWNER_EXPLICITLY_EXTENDED_RESEARCH_WINDOW' and window['owner_instruction']=='extend it to 5 days'
 assert window['duration_seconds']==432000 and window['expires_unix']-window['effective_unix']==432000
 assert window['project_GPU_hour_cap']==1024 and window['finite_training_phase_cap_seconds']==21600 and window['qualification_phase_cap_seconds']==900
 wall=p['wall_clock_envelope'];assert wall['absolute_recovery_deadline_unix']==window['expires_unix'] and wall['owner_extended_start_unix']==window['effective_unix'] and wall['total_seed_wall_cap_seconds']==432000
 assert wall['original_absolute_deadline_unix']==p['previous_absolute_deadline_unix']==old['wall_clock_envelope']['absolute_recovery_deadline_unix']
 assert p['initial_parent_updates']==[20000,PARENT_INITIAL] and p['fixed_parent_endpoints']==[20000,20000]
 assert p['new_training_updates_required']==40000-INITIAL==20000-PARENT_INITIAL and p['training_seed']==1
 assert sha(p['latest_interruption_evidence'])==p['latest_interruption_evidence_SHA256'];latest=read(p['latest_interruption_evidence'])
 assert latest['status']=='VERIFIED_ORIGINAL_WALL_CAPPED_LEARNED13265_INCOMPLETE_NO_QUALITY'
 assert latest['total_completed_updates']==INITIAL and latest['learned_completed_updates']==PARENT_INITIAL and latest['remaining_learned_updates']==40000-INITIAL
 assert latest['new_learned_updates_this_allocation']==INITIAL-29237==4028
 assert latest['quality_result_created'] is False and latest['guard_and_worker_gone'] and latest['owned_lease_released'] and latest['clean_settlement_uncertain'] is None and latest['deadline_or_policy_changed'] is False
 assert p['checkpoint_path']==latest['checkpoint']['path'] and p['checkpoint_SHA256']==latest['checkpoint']['SHA256']==sha(p['checkpoint_path'])
 assert sha(p['completed_parent_checkpoint']['path'])==p['completed_parent_checkpoint']['SHA256']
 dep=read(ROOT/'artifacts/allocations'/latest['attempt']/'deployment.json');auth=read(dep['authorization_path'])
 assert sha(dep['authorization_path'])==dep['authorization_SHA256'] and auth['initial_completed_updates']==29237 and auth['initial_parent_updates']==[20000,9237]
 assert auth['checkpoint_SHA256']==old['checkpoint_SHA256'] and auth['expires_unix']==latest['original_absolute_deadline_unix']==p['previous_absolute_deadline_unix']
 archive=Path(p['actual_execution_source_archive'])
 for path,pin in auth['research_source_SHA256'].items():assert sha(archive/path.lstrip('/'))==pin
 for path,pin in latest['evidence_SHA256'].items():assert sha(path)==pin
 assert sha(p['terminal_proposal_path'])==p['terminal_proposal_SHA256'];proposal=read(p['terminal_proposal_path'])
 assert proposal['status']=='SOURCE_ONLY_TERMINAL_ALTERNATIVE_AWAITING_PARENT_DIRECTION' and proposal['checkpoint']==latest['checkpoint'] and proposal['operational_authorization'] is False
 proof=read(p['prior_qualification_proof']);assert sha(p['prior_qualification_proof'])==p['prior_qualification_proof_SHA256']
 assert proof['checkpoint_SHA256']==old['checkpoint_SHA256'] and proof['complete_initial_state_restored_exactly'] and proof['independent_saved_branch_states_bitwise_equal'] and proof['complete_clean_settlement_verified']
 assert proof['qualified_execution_source_SHA256']==auth['research_source_SHA256']
 expected=[r['attempt'] for r in old['retained_charge_records']]+[previous.attempt(old['qualification_cell']),latest['attempt']]
 assert [r['attempt'] for r in p['retained_charge_records']]==expected and p['retained_charge_records']==proposal['retained_canonical_charge_records']==latest['all_charge_records']
 for row in p['retained_charge_records']:
  assert sha(row['ledger_path'])==row['ledger_SHA256'] and sha(row['settlement_path'])==row['settlement_SHA256']
  job=read(row['ledger_path'])['jobs'][0];settlement=read(row['settlement_path'])
  assert job['status'] in ('failed','completed') and job['reservation_released'] and settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
  assert job['charged_seconds']==row['charged_GPU_seconds']==settlement['settlement']['charged_GPU_seconds']
 assert sum(r['charged_GPU_seconds'] for r in p['retained_charge_records'])==latest['all_seven_retained_seed1_charges_GPU_seconds']
 assert sum(r['charged_GPU_seconds'] for r in p['retained_charge_records'])<432000
 assert sha(p['terminal_CPU_audit_path'])==p['terminal_CPU_audit_SHA256'];audit=read(p['terminal_CPU_audit_path'])
 assert audit['status']=='CPU_FULL_STATE_VERIFIED_PG8_COMPLETED_AND_WALLCAPPED_LEARNED' and audit['total_completed_updates']==INITIAL and audit['remaining_learned_updates']==6735 and audit['CUDA_initialized'] is False
 assert [r['completed_updates'] for r in audit['parents']]==[20000,PARENT_INITIAL] and [r['finite_model_values'] for r in audit['parents']]==[371177,376066]
 assert [r['SHA256'] for r in audit['parents']]==[p['completed_parent_checkpoint']['SHA256'],p['checkpoint_SHA256']]
 return p,f
def continuation_paths(p):
 previous=_prior();old,_=previous.lineage();latest=read(p['latest_interruption_evidence']);dep=read(ROOT/'artifacts/allocations'/latest['attempt']/'deployment.json');auth=read(dep['authorization_path']);archive=Path(p['actual_execution_source_archive'])
 paths=previous.continuation_paths(old)+[Path(p[k]) for k in ('lineage_parent_protocol','retained_scope_path','latest_interruption_evidence','prior_qualification_proof','terminal_proposal_path','terminal_CPU_audit_path','owner_extension_path')]+[Path(p['completed_parent_checkpoint']['path'])]
 paths.extend(map(Path,latest['evidence_SHA256']));paths.extend(archive/path.lstrip('/') for path in auth['research_source_SHA256'])
 for row in p['retained_charge_records']:paths.extend([Path(row['ledger_path']),Path(row['settlement_path'])])
 paths.extend(map(Path,read(p['terminal_CPU_audit_path'])['source_SHA256']))
 return list(dict.fromkeys(paths))

def cell_for(mode):
 p,f=lineage()
 return p['qualification_cell'] if mode=='qualify' else p['recovery_cell'] if mode=='resume' else f['confirmation_identities']['2']['cell']
def attempt(cell):return f"CAL_GPU0_V1_{cell['experiment']}_f0_pg_s{cell['seed']}"
def allocation(cell):return f"ALZ_CAL_V1_{cell['experiment']}_f0_pg_s{cell['seed']}"
def validate_authorization(auth,*,require_proof=True):
 p,f=lineage();ops={'qualify_interrupted_pg8':'qualify','resume_pure_ensemble':'resume','confirm_frozen_ensemble_seed2':'seed2'};mode=ops[auth['operation']]
 assert all(type(auth[key]) is int for key in ['cap_seconds','fixed_endpoint','initial_completed_updates','new_model_updates'])
 c=cell_for(mode);assert type(auth['cell']['seed']) is int and type(auth['cell']['fold']) is int;assert auth['cell']==c and auth['attempt_id']==auth['label']==attempt(c) and auth['allocation_id']==allocation(c)
 assert auth['cpu_fixture'] is False and auth['source_review_passed'] is True and auth['variant_config']==f['seed0_candidate']['variant_config']
 assert auth['pure_formulation']=='pg8_learned_ensemble75' and auth['hybrid_config'] is None and auth['ensemble_config']==p['frozen_ensemble_config']
 assert auth['qualification_only']==(mode=='qualify') and auth['cap_seconds']==(900 if mode=='qualify' else 21600)
 assert auth['fixed_endpoint']==(20000 if mode=='qualify' else 40000)
 assert auth['initial_completed_updates']==(0 if mode=='seed2' else PARENT_INITIAL if mode=='qualify' else INITIAL)
 assert auth['new_model_updates']==({'qualify':2,'resume':40000-INITIAL,'seed2':40000}[mode])
 assert auth['initial_parent_updates']==([0,0] if mode=='seed2' else [20000,PARENT_INITIAL])
 assert auth['recovery_protocol_SHA256']==sha(PROTOCOL)
 if mode!='seed2':
  assert auth['checkpoint_path']==p['checkpoint_path'] and auth['checkpoint_SHA256']==p['checkpoint_SHA256']
  assert auth['expires_unix']==p['wall_clock_envelope']['absolute_recovery_deadline_unix'],'Owner-authorized five-day deadline differs'
 else:
  assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth
  assert auth['expires_unix']<=p['wall_clock_envelope']['absolute_recovery_deadline_unix']
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 paths=[PROTOCOL,Path(p['original_confirmation_freeze']),Path(p['original_failed_ledger']),Path(p['original_interruption_evidence']),Path(p['checkpoint_path']),ROOT/'analysis/interrupted_seed1_source_history.json',ROOT/'analysis/failed_seed1_full_state_audit.json',ROOT/'analysis/exact_interrupted_resume_cpu_qualification.json',ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json']
 paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
 history=read(ROOT/'analysis/interrupted_seed1_source_history.json');paths.append(Path(history['authorization_path']));paths.extend(map(Path,history['verification_source_SHA256']))
 for path in paths+continuation_paths(p):assert pins[str(path)]==sha(path)
 for path,pin in f['scientific_source_SHA256'].items():assert research[path]==pins[path]==pin==sha(path)
 for folder in [HERE,ROOT/'confirmation_runtime']:
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_proof:
  assert auth['recovery_CPU_proof_SHA256']==pins[str(CPU_PROOF)]==sha(CPU_PROOF)
  proof=read(CPU_PROOF);assert proof['status']=='PASSED_CPU_FROZEN_EXACT_RECOVERY_RUNTIME' and proof['native_CUDA_restore_verified'] is False
  assert proof['recovery_protocol_SHA256']==sha(PROTOCOL)
  for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path)
  if mode!='qualify':
   proof=read(NATIVE_PROOF);assert pins[str(NATIVE_PROOF)]==auth['native_restore_proof_SHA256']==sha(NATIVE_PROOF)
   assert proof['status']=='VERIFIED_ACTUAL_CUDA_INTERRUPTED_STATE_AND_NEXT_UPDATE' and proof['checkpoint_SHA256']==p['checkpoint_SHA256']
   assert proof['complete_initial_state_restored_exactly'] and proof['independent_saved_branch_states_bitwise_equal'] and proof['complete_clean_settlement_verified']
   assert proof['completed_parent_restored_exactly'] and proof['completed_parent_checkpoint_SHA256']==p['completed_parent_checkpoint']['SHA256']
   assert proof['qualified_execution_source_SHA256']==research,'Same execution sources required across qualification/recovery/fixed seed2'
   for path,pin in proof['verification_source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p,f,mode

def reference_payload(seed,parents):
 p,f=lineage();assert type(seed) is int and seed in (1,2) and len(parents)==2
 c=cell_for('resume' if seed==1 else 'seed2');name=attempt(c)
 for index,parent in enumerate(parents):
  path=ROOT/'runs'/name/'fit'/['pg8','learned'][index]/'final.pt'
  assert parent['name']==['PG8','learned_observation'][index] and parent['completed_updates']==20000
  assert parent['checkpoint_path']==str(path) and sha(path)==parent['checkpoint_SHA256']
  receipt=read(path.with_name('checkpoint_receipt.json'));assert receipt['status']=='completed' and receipt['completed_updates']==20000 and receipt['SHA256']==parent['checkpoint_SHA256']
  if seed==1 and index==0:
   assert receipt['retained_parent_checkpoint_SHA256']==p['completed_parent_checkpoint']['SHA256']==parent['checkpoint_SHA256']
   assert receipt['initial_completed_updates']==20000 and receipt['new_model_updates']==0
  elif seed==1 and index==1:assert receipt['resume_parent_checkpoint_SHA256']==p['checkpoint_SHA256'] and receipt['initial_completed_updates']==PARENT_INITIAL and receipt['new_model_updates']==20000-PARENT_INITIAL
  else:assert 'resume_parent_checkpoint_SHA256' not in receipt and receipt.get('initial_completed_updates',0)==0
 return {'schema':1,'type':'frozen_exact_recovery_or_seed2_complete_parent_references','status':'completed','fold':0,'seed':seed,'completed_updates':40000,'fixed_endpoint':40000,
         'parent_training_updates_total':40000,'new_model_updates':40000-INITIAL if seed==1 else 40000,'initial_completed_updates':INITIAL if seed==1 else 0,
         'ensemble_config':p['frozen_ensemble_config'],'recovery_protocol_SHA256':sha(PROTOCOL),'original_confirmation_freeze_SHA256':p['original_confirmation_freeze_SHA256'],
         'original_interrupted_attempt':p['original_attempt'] if seed==1 else None,'resume_parent_checkpoint_SHA256':p['checkpoint_SHA256'] if seed==1 else None,'parents':parents}
def verify_reference(candidate):
 path=Path(candidate['checkpoint_path']);assert path.name=='ensemble.json' and sha(path)==candidate['checkpoint_SHA256'];q=read(path)
 assert q==reference_payload(candidate['training_seed'],q['parents'])
 assert candidate['ensemble_config']==q['ensemble_config'] and candidate['new_model_updates']==q['new_model_updates'] and candidate['initial_completed_updates']==q['initial_completed_updates']
 return q
