"""Frozen exact recovery lineage; old scheduler policy retained, reviewed own parallel profile."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'recovery_runtime'
PROTOCOL=ROOT/'analysis/exact_seed1_recovery_protocol.json'
CPU_PROOF=ROOT/'analysis/recovery_runtime_cpu_qualification.json'
NATIVE_PROOF=ROOT/'analysis/exact_seed1_native_restore_verification.json'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen():
 sys.path.insert(0,str(ROOT/'confirmation_runtime'))
 from ensemble_confirmation_scope import frozen as original_frozen
 return original_frozen()
def lineage():
 p=read(PROTOCOL);f=frozen()
 assert sha(p['original_confirmation_freeze'])==p['original_confirmation_freeze_SHA256']
 assert p['scientific_source_SHA256']==f['scientific_source_SHA256'] and p['frozen_baseline_pins']==f['matched_baselines']
 assert p['initial_parent_updates']==[8942,0] and p['fixed_parent_endpoints']==[20000,20000] and p['new_training_updates_required']==31058
 assert p['training_seed']==1 and p['frozen_ensemble_config']==f['seed0_candidate']['ensemble_config']
 assert sha(p['original_failed_ledger'])==p['original_failed_ledger_SHA256']
 assert sha(p['original_interruption_evidence'])==p['original_interruption_evidence_SHA256']
 failure=read(p['original_interruption_evidence']);assert failure['completed_updates']==8942 and failure['quality_result_created'] is False and failure['learned_parent_started'] is False
 for path,pin in failure['source_SHA256'].items():assert sha(path)==pin
 assert p['checkpoint_path']==failure['archived_checkpoint'] and sha(p['checkpoint_path'])==p['checkpoint_SHA256']==failure['checkpoint_SHA256']
 job=read(p['original_failed_ledger'])['jobs'][0]
 assert job['status']=='failed' and job['reservation_released'] and job['exit_code']==1
 assert job['error']=="Conflict('Unregistered GPU allocation(s): [2156154]')"
 wall=p['wall_clock_envelope'];assert wall['original_started_unix']==job['started_unix'] and wall['absolute_recovery_deadline_unix']==job['started_unix']+21600
 assert wall['original_charged_GPU_seconds']==failure['charged_GPU_seconds']==job['charged_seconds']
 out=ROOT/'runs'/p['original_attempt'];assert not (out/'completed.json').exists() and not (out/'r0').exists() and not (out/'r1').exists(),'No quality result may precede infrastructure bridge'
 history=read(ROOT/'analysis/interrupted_seed1_source_history.json');assert history['status']=='VERIFIED_INTERRUPTED_ALLOCATION_COMMITTED_SOURCE_HISTORY'
 assert history['attempt']==p['original_attempt'] and history['original_freeze_SHA256']==p['original_confirmation_freeze_SHA256']
 assert sha(history['authorization_path'])==history['authorization_SHA256']
 return p,f

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
 assert auth['initial_completed_updates']==(0 if mode=='seed2' else 8942)
 assert auth['new_model_updates']==({'qualify':2,'resume':31058,'seed2':40000}[mode])
 assert auth['recovery_protocol_SHA256']==sha(PROTOCOL)
 if mode!='seed2':
  assert auth['checkpoint_path']==p['checkpoint_path'] and auth['checkpoint_SHA256']==p['checkpoint_SHA256']
  assert auth['expires_unix']==p['wall_clock_envelope']['absolute_recovery_deadline_unix'],'Original absolute sixhour deadline cannot reset'
 else:assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 paths=[PROTOCOL,Path(p['original_confirmation_freeze']),Path(p['original_failed_ledger']),Path(p['original_interruption_evidence']),Path(p['checkpoint_path']),ROOT/'analysis/interrupted_seed1_source_history.json',ROOT/'analysis/failed_seed1_full_state_audit.json',ROOT/'analysis/exact_interrupted_resume_cpu_qualification.json',ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json']
 paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
 history=read(ROOT/'analysis/interrupted_seed1_source_history.json');paths.append(Path(history['authorization_path']));paths.extend(map(Path,history['verification_source_SHA256']))
 for path in paths:assert pins[str(path)]==sha(path)
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
  if seed==1 and index==0:assert receipt['resume_parent_checkpoint_SHA256']==p['checkpoint_SHA256'] and receipt['initial_completed_updates']==8942 and receipt['new_model_updates']==11058
  else:assert 'resume_parent_checkpoint_SHA256' not in receipt and receipt.get('initial_completed_updates',0)==0
 return {'schema':1,'type':'frozen_exact_recovery_or_seed2_complete_parent_references','status':'completed','fold':0,'seed':seed,'completed_updates':40000,'fixed_endpoint':40000,
         'parent_training_updates_total':40000,'new_model_updates':31058 if seed==1 else 40000,'initial_completed_updates':8942 if seed==1 else 0,
         'ensemble_config':p['frozen_ensemble_config'],'recovery_protocol_SHA256':sha(PROTOCOL),'original_confirmation_freeze_SHA256':p['original_confirmation_freeze_SHA256'],
         'original_interrupted_attempt':p['original_attempt'] if seed==1 else None,'resume_parent_checkpoint_SHA256':p['checkpoint_SHA256'] if seed==1 else None,'parents':parents}
def verify_reference(candidate):
 path=Path(candidate['checkpoint_path']);assert path.name=='ensemble.json' and sha(path)==candidate['checkpoint_SHA256'];q=read(path)
 assert q==reference_payload(candidate['training_seed'],q['parents'])
 assert candidate['ensemble_config']==q['ensemble_config'] and candidate['new_model_updates']==q['new_model_updates'] and candidate['initial_completed_updates']==q['initial_completed_updates']
 return q
