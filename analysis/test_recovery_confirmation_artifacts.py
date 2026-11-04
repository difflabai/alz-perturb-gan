"""Isolated aggregate/reference wires; actual lineage is tested separately in test_scope.

Private fake bytes and private proof dictionaries never claim models or admission.
"""
import contextlib,copy,hashlib,importlib.util,io,json,sys
from pathlib import Path
from unittest.mock import patch
import unittest
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'recovery_runtime')]
import test_ensemble_confirmation_artifacts as fixture_base
import aggregate_validation as common
import aggregate_pure_confirmation as base_aggregate
import ensemble_confirmation_scope as original_scope
import recovery_scope as scope
spec=importlib.util.spec_from_file_location('recovery_aggregate_wire_target',ROOT/'recovery_runtime/aggregate.py');aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)

class RecoveryArtifacts(fixture_base.EnsembleArtifacts):
 def setUp(self):
  super().setUp();r=self.root;self.recovery=r/'recovery_runtime';self.recovery.mkdir()
  self.recovery_protocol=r/'analysis/exact_seed1_recovery_protocol.json';self.recovery_cpu=r/'analysis/PRIVATE-recovery-CPU-proof.json';self.native_restore=r/'analysis/PRIVATE-native-restore-wire.json'
  archived=r/'artifacts/learned9237.pt';archived.write_bytes(b'PRIVATE_NOT_A_MODEL_LEARNED9237');cp_sha=common.sha(archived)
  base=Path(self.deployments[1]['output_dir'])/'fit/pg8/final.pt';base_archive=r/'artifacts/pg8completed.pt';base_archive.write_bytes(base.read_bytes())
  failed=r/'analysis/FAILED-SEED1-interrupted.json';self.write(failed,{'attempt':'PRIVATE_PRIOR_4','fixture':'PRIVATE_INFRASTRUCTURE_HISTORY_WIRE'})
  records=[]
  for index,(status,charge) in enumerate([('failed',100.),('completed',1.),('failed',100.),('completed',1.),('failed',100.)]):
   name='PRIVATE_PRIOR_'+str(index);ledger=r/'ledgers'/(name+'.json');settlement=ledger.with_name(name+'-settlement.json')
   self.write(ledger,{'jobs':[{'status':status,'exit_code':0 if status=='completed' else 1,'reservation_released':True,'charged_seconds':charge}]});self.write(settlement,{'uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':charge}})
   records.append({'attempt':name,'ledger_path':str(ledger),'ledger_SHA256':common.sha(ledger),'settlement_path':str(settlement),'settlement_SHA256':common.sha(settlement),'charged_GPU_seconds':charge})
  self.original_ledger=Path(records[0]['ledger_path']);f=common.read(self.freeze)
  owner=r/'analysis/PRIVATE-owner-window-wire.json';self.write(owner,{'fixture':'COUNTERFACTUAL_FIVE_DAY_WINDOW_WIRE_ONLY','duration_seconds':432000})
  p={'original_confirmation_freeze':str(self.freeze),'original_confirmation_freeze_SHA256':common.sha(self.freeze),'original_attempt':records[0]['attempt'],
    'original_failed_ledger':str(self.original_ledger),'original_interruption_evidence':str(failed),'checkpoint_path':str(archived),'checkpoint_SHA256':cp_sha,
    'completed_parent_checkpoint':{'path':str(base_archive),'SHA256':common.sha(base_archive),'completed_updates':20000},'initial_parent_updates':[20000,scope.PARENT_INITIAL],
    'qualification_cell':{**self.ids['1']['cell'],'experiment':'private_qualification'},'recovery_cell':self.ids['1']['cell'],
    'frozen_ensemble_config':self.candidates[0]['ensemble_config'],'wall_clock_envelope':{'absolute_recovery_deadline_unix':22600.,'total_seed_wall_cap_seconds':432000},'retained_charge_records':records,
    'owner_extension_path':str(owner),'owner_extension_SHA256':common.sha(owner),'latest_interruption_evidence':str(failed),'scientific_source_SHA256':self.sources,'frozen_baseline_pins':f['matched_baselines']}
  self.write(self.recovery_protocol,p);self.bridge=p
  self.write(r/'analysis/interrupted_seed1_source_history.json',{'authorization_path':self.deployments[0]['authorization_path'],'verification_source_SHA256':{}})
  for path in [r/'analysis/failed_seed1_full_state_audit.json',r/'analysis/exact_interrupted_resume_cpu_qualification.json']:self.write(path,{'fixture':'PRIVATE_METADATA_NOT_CUDA'})
  self.write(self.recovery_cpu,{'status':'PASSED_CPU_FROZEN_EXACT_RECOVERY_RUNTIME','native_CUDA_restore_verified':False,'recovery_protocol_SHA256':common.sha(self.recovery_protocol),'source_SHA256':{},'fixture':'PRIVATE_WIRE_NOT_PERMISSION'})
  self.write(self.native_restore,{'status':'VERIFIED_ACTUAL_CUDA_INTERRUPTED_STATE_AND_NEXT_UPDATE','checkpoint_SHA256':cp_sha,'complete_initial_state_restored_exactly':True,'independent_saved_branch_states_bitwise_equal':True,'complete_clean_settlement_verified':True,
    'completed_parent_restored_exactly':True,'completed_parent_checkpoint_SHA256':common.sha(base_archive),'qualified_execution_source_SHA256':self.sources,'verification_source_SHA256':{},'charged_GPU_seconds':1.,'fixture':'COUNTERFACTUAL_PROOF_WIRE_NOT_CUDA'})
  for seed in (1,2):
   candidate=self.candidates[seed];dep=self.deployments[seed];auth=self.auths[seed];attempt=self.ids[str(seed)]['attempt_id']
   auth.update(operation='resume_pure_ensemble' if seed==1 else 'confirm_frozen_ensemble_seed2',qualification_only=False,
     cap_seconds=21600,fixed_endpoint=40000,initial_completed_updates=scope.INITIAL if seed==1 else 0,initial_parent_updates=[20000,scope.PARENT_INITIAL] if seed==1 else [0,0],new_model_updates=40000-scope.INITIAL if seed==1 else 40000,
     expires_unix=22600.,recovery_protocol_SHA256=common.sha(self.recovery_protocol),recovery_CPU_proof_SHA256=common.sha(self.recovery_cpu),native_restore_proof_SHA256=common.sha(self.native_restore))
   if seed==1:auth.update(checkpoint_path=str(archived),checkpoint_SHA256=cp_sha)
   else:auth.pop('checkpoint_path',None);auth.pop('checkpoint_SHA256',None)
   candidate.update(operation=auth['operation'],allocation_id=auth['allocation_id'],initial_completed_updates=auth['initial_completed_updates'],new_model_updates=auth['new_model_updates'],original_interrupted_attempt=p['original_attempt'] if seed==1 else None,resume_parent_checkpoint_SHA256=cp_sha if seed==1 else None)
   parents=[]
   for index,label in enumerate(['pg8','learned']):
    path=Path(dep['output_dir'])/'fit'/label/'final.pt';receipt=common.read(path.with_name('checkpoint_receipt.json'))
    if seed==1:
     if index==0:receipt.update(retained_parent_checkpoint_SHA256=common.sha(base_archive),initial_completed_updates=20000,new_model_updates=0)
     else:receipt.update(resume_parent_checkpoint_SHA256=cp_sha,initial_completed_updates=scope.PARENT_INITIAL,new_model_updates=20000-scope.PARENT_INITIAL)
    self.write(path.with_name('checkpoint_receipt.json'),receipt);parents.append({'name':['PG8','learned_observation'][index],'checkpoint_path':str(path),'checkpoint_SHA256':common.sha(path),'completed_updates':20000})
   with patch.object(scope,'ROOT',r),patch.object(scope,'PROTOCOL',self.recovery_protocol),patch.object(scope,'lineage',return_value=(p,f)):reference=scope.reference_payload(seed,parents)
   cp=Path(dep['output_dir'])/'fit/ensemble.json';self.write(cp,reference);candidate.update(checkpoint_path=str(cp),checkpoint_SHA256=common.sha(cp))
   paths=[self.recovery_protocol,self.freeze,self.original_ledger,failed,archived,r/'analysis/interrupted_seed1_source_history.json',r/'analysis/failed_seed1_full_state_audit.json',r/'analysis/exact_interrupted_resume_cpu_qualification.json',r/'pure_incumbent.json',r/'incumbent.json',self.protocol,Path(self.deployments[0]['authorization_path']),self.recovery_cpu,self.native_restore]
   paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
   for path in paths:auth['files_SHA256'][str(path)]=common.sha(path)
   self.write(dep['authorization_path'],auth);dep['authorization_SHA256']=common.sha(dep['authorization_path']);self.write(Path(dep['authorization_path']).with_name('deployment.json'),dep);self.write(Path(dep['output_dir'])/'completed.json',candidate)
   ledger=Path(dep['ledger_path']);self.write(ledger,{'jobs':[{'status':'completed','exit_code':0,'reservation_released':True,'ended_unix':1300.}]});self.write(ledger.with_name(attempt+'-settlement.json'),{'uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':100.}})
   audit=r/'analysis'/f'checkpoint-seed{seed}-fixture.json';q=common.read(audit);q.update(status='VERIFIED_EXACT_RECOVERED_OR_FROZEN_SEED2_CHECKPOINTS',checkpoint_SHA256=candidate['checkpoint_SHA256'],parents=parents,recovery_protocol_SHA256=common.sha(self.recovery_protocol));self.write(audit,q)
   pins={}
   for i in range(10):
    path=Path(dep['output_dir'])/f'PRIVATE_ARRAY_WIRE_{i}.bin';path.write_bytes(b'NOT_ACTUAL_ARRAY');pins[str(path)]=common.sha(path)
   self.write(r/'analysis'/(attempt+'-confirmation.json'),{'status':'EXACT_LINEAGE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION','candidate':candidate,'baseline':self.baselines[seed],'recovery_protocol_SHA256':common.sha(self.recovery_protocol),
     'independently_verified_saved_scores_counts_and_checkpoint':True,'full_checkpoint_and_native_recipe_verified':True,'owned_release_and_clean_settlement_verified':True,'training_seed_was_not_ranked':True,'incumbent_unchanged':True,'checkpoint_audit':{'path':str(audit),'SHA256':common.sha(audit)},'array_and_exemplar_SHA256':pins,'charged_GPU_seconds':100.,'fixture':'PRIVATE_WIRE_NOT_NATIVE_RESULT'})
 def wire_lineage(self):
  # Ancestor restoration is covered by the real-data read-only scope tests.
  # This fixture isolates current aggregate/reference and proof-wire behavior.
  p=common.read(self.recovery_protocol);f=scope.frozen();assert p['original_confirmation_freeze_SHA256']==common.sha(self.freeze)
  for row in p['retained_charge_records']:
   assert common.sha(row['ledger_path'])==row['ledger_SHA256'] and common.sha(row['settlement_path'])==row['settlement_SHA256']
  return p,f
 @contextlib.contextmanager
 def environment(self):
  with patch.object(aggregate,'ROOT',self.root),patch.object(common,'ROOT',self.root),patch.object(base_aggregate,'ROOT',self.root),patch.object(original_scope,'ROOT',self.root),patch.object(original_scope,'HERE',self.extension),patch.object(original_scope,'FREEZE',self.freeze),patch.object(original_scope,'PROOF',self.proof),patch.object(scope,'ROOT',self.root),patch.object(scope,'HERE',self.recovery),patch.object(scope,'PROTOCOL',self.recovery_protocol),patch.object(scope,'CPU_PROOF',self.recovery_cpu),patch.object(scope,'NATIVE_PROOF',self.native_restore),patch.object(scope,'lineage',side_effect=self.wire_lineage),patch.object(scope,'continuation_paths',return_value=[]):yield
 def run_aggregate(self):
  with self.environment(),contextlib.redirect_stdout(io.StringIO()):aggregate.aggregate(self.output)
 def test_failed_original_ledger_tampering_refuses(self):
  q=common.read(self.original_ledger);q['jobs'][0]['status']='completed';self.write(self.original_ledger,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_resumed_parent_cannot_reset_initial_step_or_drop_lineage(self):
  ref=common.read(self.candidates[1]['checkpoint_path']);p=Path(ref['parents'][1]['checkpoint_path']).with_name('checkpoint_receipt.json');q=common.read(p);q.pop('resume_parent_checkpoint_SHA256');self.write(p,q)
  with self.assertRaises((AssertionError,KeyError)):self.run_aggregate()
 def test_owner_extended_deadline_is_enforced_after_recovery(self):
  p=Path(self.deployments[1]['ledger_path']);q=common.read(p);q['jobs'][0]['ended_unix']=22601.;self.write(p,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
 def test_total_native_charge_includes_all_failures_and_qualifications(self):
  self.run_aggregate();q=common.read(self.output);self.assertEqual(q['seed1_total_charged_GPU_seconds_including_failure_and_qualification'],403.)
  self.assertTrue(all(isinstance(v,str) and len(v)==64 for v in q['evidence_SHA256'].values()))
 def test_confirmation_charge_cannot_disagree_with_canonical_settlement(self):
  p=self.root/'analysis'/(self.ids['1']['attempt_id']+'-confirmation.json');q=common.read(p);q['charged_GPU_seconds']=0.;self.write(p,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
 def test_missing_actual_native_restore_proof_refuses(self):
  self.native_restore.unlink()
  with self.assertRaises(FileNotFoundError):self.run_aggregate()
if __name__=='__main__':unittest.main()
