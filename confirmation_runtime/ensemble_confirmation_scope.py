"""Exact frozen two-parent seed confirmation; standard-library admission only."""
import hashlib,json
from pathlib import Path
ROOT=Path('/ml2/alz')
HERE=ROOT/'confirmation_runtime'
FREEZE=ROOT/'analysis/pure_ensemble_confirmation_freeze.json'
PROOF=ROOT/'analysis/ensemble_confirmation_cpu_qualification.json'
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scientific_sources():return {str(p):sha(p) for p in [*ROOT.glob('*.py'),*(ROOT/'model').glob('*.py')]}
def frozen():
 f=read(FREEZE)
 assert f['status']=='FROZEN_PURE_THREE_SEED_CALIBRATION_CONFIRMATION' and f['training_seeds']==[0,1,2] and f['confirmation_training_seeds']==[1,2] and f['ranking_seeds_allowed'] is False
 assert f['seed0_candidate']['pure_formulation']=='pg8_learned_ensemble75' and f['seed0_candidate']['ensemble_config']=={'learned_weight':.25,'mask_seed':20261108,'bags_per_stage':200}
 assert sha(ROOT/'pure_incumbent.json')==f['pure_incumbent_SHA256'] and sha(ROOT/'analysis/pure_confirmation_protocol.json')==f['pure_confirmation_protocol_SHA256']
 assert scientific_sources()==f['scientific_source_SHA256'],'Frozen scientific sources changed; do not silently substitute models/scorer'
 for baseline in f['matched_baselines'].values():assert sha(baseline['path'])==baseline['SHA256']
 evidence=ROOT/'analysis'/(f['seed0_attempt']+'-result.json');assert sha(evidence)==f['seed0_decision_SHA256'] and read(evidence)['decision']['accept']
 return f

def validate_authorization(auth,*,require_proof=True):
 f=frozen();seed=auth['cell']['seed'];assert type(seed) is int and seed in (1,2)
 ident=f['confirmation_identities'][str(seed)]
 assert auth['cell']==ident['cell'] and auth['allocation_id']==ident['allocation_id'] and auth['attempt_id']==auth['label']==ident['attempt_id']
 assert auth['operation']=='confirm_pure_ensemble' and auth['qualification_only'] is False
 assert auth['cap_seconds']==21600 and auth['fixed_endpoint']==40000
 assert auth['initial_completed_updates']==0 and auth['new_model_updates']==40000 and auth['parent_updates']==[20000,20000]
 assert auth['hybrid_config'] is None and auth['variant_config']==f['seed0_candidate']['variant_config'] and auth['pure_formulation']=='pg8_learned_ensemble75'
 assert auth['ensemble_config']==f['seed0_candidate']['ensemble_config'] and auth['ensemble_weight']==.25
 assert 'checkpoint_path' not in auth and 'checkpoint_SHA256' not in auth,'No parent warm start for confirmation'
 assert auth['confirmation_freeze_SHA256']==sha(FREEZE)
 assert auth['cpu_fixture'] is False and auth['source_review_passed'] is True
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 paths=[FREEZE,ROOT/'pure_incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json',ROOT/'ensemble_parents.json',ROOT/'ensemble_protocol.json']
 paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
 for path in paths:assert pins[str(path)]==sha(path)
 for path,pin in f['scientific_source_SHA256'].items():assert research[path]==pins[path]==pin
 for path in HERE.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_proof:
  assert pins[str(PROOF)]==auth['confirmation_qualification_SHA256']==sha(PROOF)
  proof=read(PROOF);assert proof['status']=='PASSED_CPU_FROZEN_TWO_PARENT_CONFIRMATION' and proof['native_confirmation_completed'] is False
  assert proof['freeze_SHA256']==sha(FREEZE) and proof['GPU_allocation_created'] is False
  for path,pin in proof['source_SHA256'].items():assert pins[path]==sha(path)==pin
 return f

def reference_payload(seed,parents):
 f=frozen();assert type(seed) is int and seed in (1,2)
 assert len(parents)==2
 for index,parent in enumerate(parents):
  assert parent['name']==['PG8','learned_observation'][index] and parent['completed_updates']==20000
  path=ROOT/'runs'/f['confirmation_identities'][str(seed)]['attempt_id']/'fit'/['pg8','learned'][index]/'final.pt'
  assert parent['checkpoint_path']==str(path) and sha(path)==parent['checkpoint_SHA256']
  receipt=read(path.with_name('checkpoint_receipt.json'))
  assert receipt['status']=='completed' and receipt['path']=='final.pt' and receipt['completed_updates']==20000 and receipt['SHA256']==parent['checkpoint_SHA256']
 return {'schema':1,'type':'confirmed_pure_ensemble_complete_parent_references','status':'completed','fold':0,'seed':seed,
         'completed_updates':40000,'fixed_endpoint':40000,'parent_training_updates_total':40000,'new_model_updates':40000,
         'ensemble_config':f['seed0_candidate']['ensemble_config'],'confirmation_freeze_SHA256':sha(FREEZE),'parents':parents}

def verify_reference(candidate):
 assert candidate['training_seed'] in (1,2) and candidate['pure_formulation']=='pg8_learned_ensemble75'
 path=Path(candidate['checkpoint_path']);assert path.name=='ensemble.json' and sha(path)==candidate['checkpoint_SHA256']
 q=read(path);assert q==reference_payload(candidate['training_seed'],q['parents'])
 assert candidate['ensemble_config']==q['ensemble_config'] and candidate['new_model_updates']==40000 and candidate['parent_training_updates_total']==40000
 return q
