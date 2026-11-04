"""CPU-only exact-lineage source/refusal tests; private proof wires create no admission."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/ml2/alz');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'runtime'),str(ROOT),str(ROOT/'analysis')]
import binding as b
import recovery_scope as scope

class RecoveryScope(unittest.TestCase):
 def auth(self,mode):
  p,f=scope.lineage();c=scope.cell_for(mode);prior=b.read(ROOT/'calibration_deployment.json');auth=copy.deepcopy(b.read(prior['authorization_path']))
  auth.pop('checkpoint_path',None);auth.pop('checkpoint_SHA256',None)
  auth.update(cell=c,allocation_id=scope.allocation(c),attempt_id=scope.attempt(c),label=scope.attempt(c),operation={'qualify':'qualify_interrupted_pg8','resume':'resume_pure_ensemble','seed2':'confirm_frozen_ensemble_seed2'}[mode],
   qualification_only=mode=='qualify',cap_seconds=900 if mode=='qualify' else 21600,fixed_endpoint=20000 if mode=='qualify' else 40000,
   initial_completed_updates=0 if mode=='seed2' else 8942,new_model_updates={'qualify':2,'resume':31058,'seed2':40000}[mode],
   expires_unix=p['wall_clock_envelope']['absolute_recovery_deadline_unix'],hybrid_config=None,variant_config=f['seed0_candidate']['variant_config'],pure_formulation='pg8_learned_ensemble75',ensemble_config=p['frozen_ensemble_config'],recovery_protocol_SHA256=scope.sha(scope.PROTOCOL))
  if mode!='seed2':auth.update(checkpoint_path=p['checkpoint_path'],checkpoint_SHA256=p['checkpoint_SHA256'])
  research={str(path):b.sha(path) for folder in [ROOT,ROOT/'model',ROOT/'runtime',ROOT/'confirmation_runtime',scope.HERE] for path in folder.glob('*.py')};auth['research_source_SHA256']=research;pins=auth['files_SHA256'];pins.update(research)
  history=scope.read(ROOT/'analysis/interrupted_seed1_source_history.json')
  paths=[scope.PROTOCOL,Path(p['original_confirmation_freeze']),Path(p['original_failed_ledger']),Path(p['original_interruption_evidence']),Path(p['checkpoint_path']),ROOT/'analysis/interrupted_seed1_source_history.json',ROOT/'analysis/failed_seed1_full_state_audit.json',ROOT/'analysis/exact_interrupted_resume_cpu_qualification.json',ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json',Path(history['authorization_path']),*map(Path,history['verification_source_SHA256'])]
  paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
  for path in paths:pins[str(path)]=b.sha(path)
  return auth
 def test_exact_qualification_recovery_and_original_seed2_identity_source_scope(self):
  for mode in ('qualify','resume','seed2'):
   auth=self.auth(mode);scope.validate_authorization(auth,require_proof=False)
  self.assertEqual(scope.cell_for('seed2'),scope.frozen()['confirmation_identities']['2']['cell']);self.assertNotIn('torch',sys.modules)
 def test_scientific_seed_checkpoint_horizon_and_alpha_cannot_change(self):
  for change in [{'checkpoint_SHA256':'0'*64},{'initial_completed_updates':0},{'new_model_updates':40000},{'fixed_endpoint':40000},{'cap_seconds':21600},{'qualification_only':False},{'cell':{**scope.cell_for('qualify'),'seed':False}},{'ensemble_config':{'learned_weight':.5,'mask_seed':20261108,'bags_per_stage':200}},{'hybrid_config':{'pg_bag_fraction':.75}}]:
   with self.subTest(change=change),self.assertRaises((AssertionError,KeyError)):scope.validate_authorization({**self.auth('qualify'),**change},require_proof=False)
 def test_original_seed1_absolute_sixhour_deadline_cannot_reset(self):
  for mode in ('qualify','resume'):
   auth=self.auth(mode)
   with self.assertRaises(AssertionError):scope.validate_authorization({**auth,'expires_unix':auth['expires_unix']+1},require_proof=False)
 def test_seed2_cannot_warm_start_from_interrupted_or_qualification_weights(self):
  auth=self.auth('seed2')
  for change in [{'checkpoint_path':'qualification_step8943.pt'},{'initial_completed_updates':8942},{'new_model_updates':31058}]:
   with self.assertRaises(AssertionError):scope.validate_authorization({**auth,**change},require_proof=False)
 def test_changed_recovery_source_pin_refuses_before_model_import(self):
  auth=self.auth('qualify');auth['research_source_SHA256'][str(scope.HERE/'execution.py')]='0'*64
  with self.assertRaises(AssertionError):scope.validate_authorization(auth,require_proof=False)
  self.assertNotIn('torch',sys.modules)
 def test_full_native_source_closure_all_three_modes_private_wire_proofs(self):
  with tempfile.TemporaryDirectory(prefix='alz-exact-recovery-proof-wire-') as temp:
   cpu=Path(temp)/'CPU.json';native=Path(temp)/'native.json'
   for mode in ('qualify','resume','seed2'):
    auth=self.auth(mode)
    cpu.write_text(json.dumps({'status':'PASSED_CPU_FROZEN_EXACT_RECOVERY_RUNTIME','recovery_protocol_SHA256':scope.sha(scope.PROTOCOL),'native_CUDA_restore_verified':False,'source_SHA256':{},'fixture':'PRIVATE_PROOF_WIRE_NOT_A_NATIVE_PERMISSION'}))
    native.write_text(json.dumps({'status':'VERIFIED_ACTUAL_CUDA_INTERRUPTED_STATE_AND_NEXT_UPDATE','checkpoint_SHA256':scope.lineage()[0]['checkpoint_SHA256'],'complete_initial_state_restored_exactly':True,'independent_saved_branch_states_bitwise_equal':True,'complete_clean_settlement_verified':True,'qualified_execution_source_SHA256':auth['research_source_SHA256'],'verification_source_SHA256':{},'fixture':'COUNTERFACTUAL WIRE ONLY, NOT ACTUAL CUDA PROOF'}))
    auth.update(recovery_CPU_proof_SHA256=scope.sha(cpu),native_restore_proof_SHA256=scope.sha(native));auth['files_SHA256'].update({str(cpu):scope.sha(cpu),str(native):scope.sha(native)})
    with patch.object(scope,'CPU_PROOF',cpu),patch.object(scope,'NATIVE_PROOF',native):
     b.source_closure(auth,b.NATIVE_LAYOUT)
     changed=copy.deepcopy(auth);changed['files_SHA256'][str(scope.HERE/'execution.py')]='0'*64
     with self.assertRaises(b.Refusal):b.source_closure(changed,b.NATIVE_LAYOUT)
  self.assertNotIn('torch',sys.modules)
 def test_reference_binds_complete_resume_receipt_and_both_parents(self):
  p,f=scope.lineage()
  with tempfile.TemporaryDirectory(prefix='alz-recovery-reference-wire-') as temp:
   root=Path(temp);name=scope.attempt(scope.cell_for('resume'));parents=[]
   for index,label in enumerate(['pg8','learned']):
    path=root/'runs'/name/'fit'/label/'final.pt';path.parent.mkdir(parents=True);path.write_bytes(b'PRIVATE_WIRE_NOT_A_TORCH_MODEL')
    receipt={'status':'completed','path':'final.pt','completed_updates':20000,'SHA256':scope.sha(path)}
    if index==0:receipt.update(resume_parent_checkpoint_SHA256=p['checkpoint_SHA256'],initial_completed_updates=8942,new_model_updates=11058)
    path.with_name('checkpoint_receipt.json').write_text(json.dumps(receipt));parents.append({'name':['PG8','learned_observation'][index],'checkpoint_path':str(path),'checkpoint_SHA256':scope.sha(path),'completed_updates':20000})
   with patch.object(scope,'ROOT',root),patch.object(scope,'lineage',return_value=(p,f)):
    reference=scope.reference_payload(1,parents);self.assertEqual(reference['new_model_updates'],31058)
    bad=Path(parents[0]['checkpoint_path']).with_name('checkpoint_receipt.json');receipt=json.loads(bad.read_text());receipt['initial_completed_updates']=0;bad.write_text(json.dumps(receipt))
    with self.assertRaises(AssertionError):scope.reference_payload(1,parents)
 def test_seed2_reference_rejects_resume_tagged_parent(self):
  p,f=scope.lineage()
  with tempfile.TemporaryDirectory(prefix='alz-seed2-reference-wire-') as temp:
   root=Path(temp);name=scope.attempt(scope.cell_for('seed2'));parents=[]
   for index,label in enumerate(['pg8','learned']):
    path=root/'runs'/name/'fit'/label/'final.pt';path.parent.mkdir(parents=True);path.write_bytes(b'PRIVATE_WIRE_NOT_A_TORCH_MODEL')
    receipt={'status':'completed','path':'final.pt','completed_updates':20000,'SHA256':scope.sha(path)}
    if index==0:receipt['resume_parent_checkpoint_SHA256']=p['checkpoint_SHA256']
    path.with_name('checkpoint_receipt.json').write_text(json.dumps(receipt));parents.append({'name':['PG8','learned_observation'][index],'checkpoint_path':str(path),'checkpoint_SHA256':scope.sha(path),'completed_updates':20000})
   with patch.object(scope,'ROOT',root),patch.object(scope,'lineage',return_value=(p,f)),self.assertRaises(AssertionError):scope.reference_payload(2,parents)
 def test_real_failed_allocation_and_checkpoint_not_a_quality_result(self):
  p,f=scope.lineage();failure=scope.read(p['original_interruption_evidence']);self.assertFalse(failure['quality_result_created']);self.assertEqual(failure['completed_updates'],8942)
  self.assertTrue(failure['owned_release_and_clean_settlement_verified'])
if __name__=='__main__':unittest.main()
