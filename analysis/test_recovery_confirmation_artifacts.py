"""Confined recovery wire fixtures. Fake bytes are NOT models, training or CUDA proofs."""
import contextlib,copy,hashlib,importlib.util,io,json,shutil,sys
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
  super().setUp()
  self.recovery=self.root/'recovery_runtime';self.recovery.mkdir()
  self.recovery_protocol=self.root/'analysis/exact_seed1_recovery_protocol.json'
  self.recovery_cpu=self.root/'analysis/PRIVATE-recovery-CPU-proof.json'
  self.native_restore=self.root/'analysis/PRIVATE-native-restore-wire.json'
  original=copy.deepcopy(self.deployments[1]);old_auth=copy.deepcopy(self.auths[1]);old_attempt=Path(original['output_dir']).name
  original_output=Path(original['output_dir']);(original_output/'completed.json').unlink()
  archived=self.root/'artifacts/archived8942.pt';archived.parent.mkdir(parents=True,exist_ok=True);archived.write_bytes(b'COUNTERFACTUAL_WIRE_NOT_A_MODEL_8942')
  cp_sha=common.sha(archived);freeze_pin=common.sha(self.freeze)
  suffix=hashlib.sha256(json.dumps({'interrupted_attempt':old_attempt,'checkpoint_SHA256':cp_sha,'freeze_SHA256':freeze_pin},sort_keys=True).encode()).hexdigest()[:20]
  recovery_cell={'fold':0,'method':'pg','seed':1,'experiment':'pure_recover_'+suffix};qualification_cell={**recovery_cell,'experiment':'qual_recover_'+suffix}
  name=scope.attempt(recovery_cell);new_output=self.root/'runs'/name;shutil.copytree(original_output,new_output)
  shutil.rmtree(original_output/'fit');(original_output/'fit/pg8').mkdir(parents=True)
  source_checkpoint=original_output/'fit/pg8/recovery.pt';source_checkpoint.write_bytes(archived.read_bytes())
  self.write(source_checkpoint.with_name('checkpoint_receipt.json'),{'status':'incomplete','path':'recovery.pt','completed_updates':8942,'SHA256':cp_sha})
  self.write(original_output/'failure.json',{'status':'incomplete','fixture':'COUNTERFACTUAL GUARD INTERRUPTION'})
  original_ledger=Path(original['ledger_path']);self.write(original_ledger,{'jobs':[{'status':'failed','exit_code':1,'reservation_released':True,'error':"Conflict('Unregistered GPU allocation(s): [2156154]')",'started_unix':1000.,'ended_unix':1100.,'charged_seconds':100.}]})
  original_settlement=original_ledger.with_name(old_attempt+'-settlement.json');self.write(original_settlement,{'uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':100.}})
  failure=self.root/'analysis'/(old_attempt+'-interrupted.json');self.write(failure,{'completed_updates':8942,'quality_result_created':False,'learned_parent_started':False,'archived_checkpoint':str(archived),'checkpoint_SHA256':cp_sha,'charged_GPU_seconds':100.,'owned_release_and_clean_settlement_verified':True,
     'source_SHA256':{str(path):common.sha(path) for path in [original_ledger,original_settlement,source_checkpoint.with_name('checkpoint_receipt.json'),original_output/'failure.json']},'fixture':'COUNTERFACTUAL_WIRE_NOT_AN_ACTUAL_INTERRUPTION'})
  p={'status':'PREDECLARED_EXACT_TRAJECTORY_RECOVERY_ONLY_NATIVE_NOT_ADMITTED','original_confirmation_freeze':str(self.freeze),'original_confirmation_freeze_SHA256':freeze_pin,
     'original_attempt':old_attempt,'original_allocation_id':original['allocation_id'],'original_failed_ledger':str(original_ledger),'original_failed_ledger_SHA256':common.sha(original_ledger),
     'original_interruption_evidence':str(failure),'original_interruption_evidence_SHA256':common.sha(failure),'checkpoint_path':str(archived),'checkpoint_SHA256':cp_sha,
     'training_seed':1,'initial_parent_updates':[8942,0],'fixed_parent_endpoints':[20000,20000],'new_training_updates_required':31058,
     'qualification_cell':qualification_cell,'recovery_cell':recovery_cell,'scientific_source_SHA256':self.sources,'frozen_baseline_pins':common.read(self.freeze)['matched_baselines'],
     'frozen_ensemble_config':self.candidates[0]['ensemble_config'],'wall_clock_envelope':{'original_started_unix':1000.,'absolute_recovery_deadline_unix':22600.,'original_charged_GPU_seconds':100.},'fixture':'PRIVATE COUNTERFACTUAL METADATA, NOT A RECOVERY APPROVAL'}
  # Add a separately retained intermediate infrastructure phase (wire metadata only).
  old_protocol=self.root/'analysis/PRIVATE-original-recovery-protocol.json';prior_p=copy.deepcopy(p);prior_p['recovery_cell']={'fold':0,'method':'pg','seed':1,'experiment':'private_first_resume'};prior_p['qualification_cell']={**prior_p['recovery_cell'],'experiment':'private_first_qual'};self.write(old_protocol,prior_p)
  root_cp_sha=cp_sha
  phase=scope.attempt(prior_p['recovery_cell']);phase_dir=self.root/'artifacts/allocations'/phase;phase_dir.mkdir(parents=True)
  phase_auth=copy.deepcopy(old_auth);phase_auth.update(initial_completed_updates=8942,expires_unix=22600.)
  phase_auth_path=phase_dir/'authorization.json';self.write(phase_auth_path,phase_auth)
  phase_ledger=self.root/'ledgers'/(phase+'.json');self.write(phase_ledger,{'jobs':[{'status':'failed','exit_code':1,'reservation_released':True,'error':"Refusal('protected peer identity changed')",'charged_seconds':100.}]})
  self.write(phase_ledger.with_name(phase+'-settlement.json'),{'uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':100.}})
  self.write(phase_dir/'deployment.json',{'authorization_path':str(phase_auth_path),'authorization_SHA256':common.sha(phase_auth_path),'ledger_path':str(phase_ledger)})
  oldqual=self.root/'analysis/PRIVATE-first-qualification.json';qualname=scope.attempt(prior_p['qualification_cell'])
  self.write(oldqual,{'status':'VERIFIED_ACTUAL_CUDA_INTERRUPTED_STATE_AND_NEXT_UPDATE','complete_initial_state_restored_exactly':True,'independent_saved_branch_states_bitwise_equal':True,'attempt':qualname,'checkpoint_SHA256':root_cp_sha,'complete_clean_settlement_verified':True,'qualified_execution_source_SHA256':phase_auth['research_source_SHA256'],'verification_source_SHA256':{},'charged_GPU_seconds':1.,'fixture':'COUNTERFACTUAL WIRE NOT CUDA'})
  self.write(phase_ledger.with_name(qualname+'.json'),{'jobs':[{'status':'completed','exit_code':0,'reservation_released':True,'charged_seconds':1.}]})
  self.write(phase_ledger.with_name(qualname+'-settlement.json'),{'uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':1.}})
  archived=self.root/'artifacts/phase14410/pg8-step14410.pt';archived.parent.mkdir(parents=True);archived.write_bytes(b'COUNTERFACTUAL_WIRE_NOT_A_MODEL_14410');cp_sha=common.sha(archived)
  for source,pin in phase_auth['research_source_SHA256'].items():
   dest=archived.parent/'source'/source.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(source).read_bytes());assert common.sha(dest)==pin
  latest=self.root/'analysis/PRIVATE-turnover-interruption.json';self.write(latest,{'status':'FROZEN_CLEAN_EXTERNAL_PEER_INTERRUPTION_NEWEST_EXACT_PG8_STATE_NO_QUALITY','attempt':phase,'initial_completed_updates':8942,'completed_updates':scope.INITIAL,'new_model_updates':scope.INITIAL-8942,'original_parent_checkpoint_SHA256':root_cp_sha,'archived_checkpoint':str(archived),'checkpoint_SHA256':cp_sha,'quality_result_created':False,'learned_parent_started':False,'owned_cleanup_and_settlement_verified':True,'charged_GPU_seconds':100.,'original_absolute_deadline_unix':22600.,'canonical_input_SHA256':{},'execution_source_SHA256':phase_auth['research_source_SHA256'],'fixture':'COUNTERFACTUAL LINEAGE WIRE'})
  self.write(self.root/'analysis/turnover_seed1_full_state_audit.json',{'status':'VERIFIED_TURNOVER_NATIVE_PG8_EXACT_STATE_CPU_ONLY','completed_updates':scope.INITIAL,'checkpoint_SHA256':cp_sha,'strict_complete_G_D_prior_load_verified':True,'finite_model_floating_values':371177,'CUDA_initialized':False,'source_SHA256':{},'fixture':'COUNTERFACTUAL MODEL-AUDIT WIRE, NO ACTUAL TENSORS'})
  p.update(checkpoint_path=str(archived),checkpoint_SHA256=cp_sha,initial_parent_updates=[scope.INITIAL,0],new_training_updates_required=40000-scope.INITIAL,latest_interruption_evidence=str(latest),latest_interruption_evidence_SHA256=common.sha(latest),previous_protocol_path=str(old_protocol),previous_protocol_SHA256=common.sha(old_protocol),previous_native_qualification_path=str(oldqual),previous_native_qualification_SHA256=common.sha(oldqual))
  self.previous_phase_ledger=phase_ledger;self.latest_interruption=latest
  self.write(self.recovery_protocol,p);self.bridge=p
  self.write(self.root/'analysis/interrupted_seed1_source_history.json',{'status':'VERIFIED_INTERRUPTED_ALLOCATION_COMMITTED_SOURCE_HISTORY','attempt':old_attempt,'original_freeze_SHA256':freeze_pin,
       'authorization_path':original['authorization_path'],'authorization_SHA256':common.sha(original['authorization_path']),'verification_source_SHA256':{str(self.root/'scorer.py'):common.sha(self.root/'scorer.py')},'fixture':'PRIVATE WIRE, NO GIT AUDIT'})
  for path in [self.root/'analysis/failed_seed1_full_state_audit.json',self.root/'analysis/exact_interrupted_resume_cpu_qualification.json']:self.write(path,{'fixture':'PRIVATE_METADATA_NO_MODEL_AUDIT_OR_CUDA'})
  self.write(self.recovery_cpu,{'status':'PASSED_CPU_FROZEN_EXACT_RECOVERY_RUNTIME','native_CUDA_restore_verified':False,'recovery_protocol_SHA256':common.sha(self.recovery_protocol),'source_SHA256':{},'fixture':'PRIVATE_PROOF_WIRE_NOT_PERMISSION'})
  self.write(self.native_restore,{'status':'VERIFIED_ACTUAL_CUDA_INTERRUPTED_STATE_AND_NEXT_UPDATE','checkpoint_SHA256':cp_sha,'complete_initial_state_restored_exactly':True,'independent_saved_branch_states_bitwise_equal':True,'complete_clean_settlement_verified':True,
    'qualified_execution_source_SHA256':self.sources,'verification_source_SHA256':{},'charged_GPU_seconds':1.,'fixture':'COUNTERFACTUAL WIRE ONLY. NO CUDA JOB OR RESTORE EXECUTED.'})
  self.original_ledger=original_ledger
  self.deployments[1]=copy.deepcopy(original);self.deployments[1].update(allocation_id=scope.allocation(recovery_cell),output_dir=str(new_output),ledger_path=str(self.root/'ledgers'/(name+'.json')),authorization_path=str(self.root/'artifacts/allocations'/name/'authorization.json'))
  self.ids['1']={'cell':recovery_cell,'attempt_id':name,'allocation_id':scope.allocation(recovery_cell)}
  for seed in (1,2):
   candidate=self.candidates[seed];dep=self.deployments[seed];auth=self.auths[seed];mode='resume' if seed==1 else 'seed2';c=recovery_cell if seed==1 else self.ids['2']['cell'];attempt=scope.attempt(c)
   auth.update(cell=c,allocation_id=scope.allocation(c),attempt_id=attempt,label=attempt,operation='resume_pure_ensemble' if seed==1 else 'confirm_frozen_ensemble_seed2',qualification_only=False,
      cap_seconds=21600,fixed_endpoint=40000,initial_completed_updates=scope.INITIAL if seed==1 else 0,new_model_updates=40000-scope.INITIAL if seed==1 else 40000,
      expires_unix=22600.,recovery_protocol_SHA256=common.sha(self.recovery_protocol),recovery_CPU_proof_SHA256=common.sha(self.recovery_cpu),native_restore_proof_SHA256=common.sha(self.native_restore))
   if seed==1:auth.update(checkpoint_path=str(archived),checkpoint_SHA256=cp_sha)
   else:auth.pop('checkpoint_path',None);auth.pop('checkpoint_SHA256',None)
   candidate.update(operation=auth['operation'],allocation_id=auth['allocation_id'],initial_completed_updates=auth['initial_completed_updates'],new_model_updates=auth['new_model_updates'],
        original_interrupted_attempt=old_attempt if seed==1 else None,resume_parent_checkpoint_SHA256=cp_sha if seed==1 else None)
   parents=[]
   for index,label in enumerate(['pg8','learned']):
    path=Path(dep['output_dir'])/'fit'/label/'final.pt';receipt=common.read(path.with_name('checkpoint_receipt.json'))
    if seed==1 and index==0:receipt.update(resume_parent_checkpoint_SHA256=cp_sha,initial_completed_updates=scope.INITIAL,new_model_updates=20000-scope.INITIAL)
    self.write(path.with_name('checkpoint_receipt.json'),receipt)
    parents.append({'name':['PG8','learned_observation'][index],'checkpoint_path':str(path),'checkpoint_SHA256':common.sha(path),'completed_updates':20000})
   reference={'schema':1,'type':'frozen_exact_recovery_or_seed2_complete_parent_references','status':'completed','fold':0,'seed':seed,'completed_updates':40000,'fixed_endpoint':40000,
       'parent_training_updates_total':40000,'new_model_updates':auth['new_model_updates'],'initial_completed_updates':auth['initial_completed_updates'],'ensemble_config':candidate['ensemble_config'],
       'recovery_protocol_SHA256':common.sha(self.recovery_protocol),'original_confirmation_freeze_SHA256':freeze_pin,'original_interrupted_attempt':old_attempt if seed==1 else None,'resume_parent_checkpoint_SHA256':cp_sha if seed==1 else None,'parents':parents}
   cp=Path(dep['output_dir'])/'fit/ensemble.json';self.write(cp,reference);candidate.update(checkpoint_path=str(cp),checkpoint_SHA256=common.sha(cp))
   paths=[self.recovery_protocol,self.freeze,original_ledger,failure,archived,self.root/'analysis/interrupted_seed1_source_history.json',self.root/'analysis/failed_seed1_full_state_audit.json',self.root/'analysis/exact_interrupted_resume_cpu_qualification.json',self.root/'pure_incumbent.json',self.root/'incumbent.json',self.protocol,Path(original['authorization_path']),self.root/'scorer.py',self.recovery_cpu,self.native_restore]
   paths.extend(Path(q['path']) for q in common.read(self.freeze)['matched_baselines'].values())
   paths.extend([latest,old_protocol,oldqual,self.root/'analysis/turnover_seed1_full_state_audit.json',phase_dir/'deployment.json',phase_ledger.with_name(qualname+'.json'),phase_ledger.with_name(qualname+'-settlement.json')])
   paths.extend(archived.parent/'source'/source.lstrip('/') for source in phase_auth['research_source_SHA256'])
   for path in paths:auth['files_SHA256'][str(path)]=common.sha(path)
   self.write(dep['authorization_path'],auth);dep['authorization_SHA256']=common.sha(dep['authorization_path']);self.write(Path(dep['authorization_path']).with_name('deployment.json'),dep);self.write(Path(dep['output_dir'])/'completed.json',candidate)
   ledger=Path(dep['ledger_path']);self.write(ledger,{'jobs':[{'status':'completed','exit_code':0,'reservation_released':True,'ended_unix':1300.}]});self.write(ledger.with_name(attempt+'-settlement.json'),{'uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':100.}})
   audit=self.root/'analysis'/f'checkpoint-seed{seed}-fixture.json';q=common.read(audit);q.update(status='VERIFIED_EXACT_RECOVERED_OR_FROZEN_SEED2_CHECKPOINTS',checkpoint_SHA256=candidate['checkpoint_SHA256'],parents=parents,recovery_protocol_SHA256=common.sha(self.recovery_protocol));self.write(audit,q)
   array_pins={}
   for index in range(10):
    path=Path(dep['output_dir'])/f'PRIVATE_ARRAY_WIRE_{index}.bin';path.write_bytes(b'NOT_A_REAL_NUMPY_OR_COUNT_ARRAY');array_pins[str(path)]=common.sha(path)
   proof={'status':'EXACT_LINEAGE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION','candidate':candidate,'baseline':self.baselines[seed],'recovery_protocol_SHA256':common.sha(self.recovery_protocol),
       'independently_verified_saved_scores_counts_and_checkpoint':True,'full_checkpoint_and_native_recipe_verified':True,'owned_release_and_clean_settlement_verified':True,'training_seed_was_not_ranked':True,'incumbent_unchanged':True,
       'checkpoint_audit':{'path':str(audit),'SHA256':common.sha(audit)},'array_and_exemplar_SHA256':array_pins,'charged_GPU_seconds':100.,'fixture':'COUNTERFACTUAL PROOF WIRE. NO SAVED ARRAYS OR MODEL AUDIT EXECUTED.'}
   self.write(self.root/'analysis'/(attempt+'-confirmation.json'),proof)
 def run_aggregate(self):
  with patch.object(aggregate,'ROOT',self.root),patch.object(common,'ROOT',self.root),patch.object(base_aggregate,'ROOT',self.root), \
       patch.object(original_scope,'ROOT',self.root),patch.object(original_scope,'HERE',self.extension),patch.object(original_scope,'FREEZE',self.freeze),patch.object(original_scope,'PROOF',self.proof), \
       patch.object(scope,'ROOT',self.root),patch.object(scope,'HERE',self.recovery),patch.object(scope,'PROTOCOL',self.recovery_protocol),patch.object(scope,'CPU_PROOF',self.recovery_cpu),patch.object(scope,'NATIVE_PROOF',self.native_restore),contextlib.redirect_stdout(io.StringIO()):
   aggregate.aggregate(self.output)
 def test_failed_original_ledger_tampering_refuses(self):
  q=common.read(self.original_ledger);q['jobs'][0]['status']='completed';self.write(self.original_ledger,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_resumed_parent_cannot_reset_initial_step_or_drop_lineage(self):
  reference=common.read(self.candidates[1]['checkpoint_path']);path=Path(reference['parents'][0]['checkpoint_path']).with_name('checkpoint_receipt.json');q=common.read(path);q.pop('resume_parent_checkpoint_SHA256');self.write(path,q)
  with self.assertRaises((AssertionError,KeyError)):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_original_sixhour_deadline_is_enforced_after_recovery(self):
  p=Path(self.deployments[1]['ledger_path']);q=common.read(p);q['jobs'][0]['ended_unix']=22601.;self.write(p,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_total_native_charge_includes_failure_and_qualification(self):
  path=self.root/'analysis'/(self.ids['1']['attempt_id']+'-confirmation.json');q=common.read(path);q['charged_GPU_seconds']=21500.;self.write(path,q)
  settlement=Path(self.deployments[1]['ledger_path']).with_name(self.ids['1']['attempt_id']+'-settlement.json');q=common.read(settlement);q['settlement']['charged_GPU_seconds']=21500.;self.write(settlement,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_digest_inventory_and_canonical_charges_are_separate(self):
  self.run_aggregate();q=common.read(self.output)
  self.assertEqual(q['seed1_total_charged_GPU_seconds_including_failure_and_qualification'],302.)
  self.assertTrue(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in q['evidence_SHA256'].values()))
 def test_confirmation_charge_cannot_disagree_with_canonical_settlement(self):
  path=self.root/'analysis'/(self.ids['1']['attempt_id']+'-confirmation.json');q=common.read(path);q['charged_GPU_seconds']=0.;self.write(path,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_CPU_fixture_flag_cannot_be_aggregated_as_native_model_evidence(self):
  q=self.auths[1];q['cpu_fixture']=True;self.write(self.deployments[1]['authorization_path'],q)
  self.deployments[1]['authorization_SHA256']=common.sha(self.deployments[1]['authorization_path']);self.write(Path(self.deployments[1]['authorization_path']).with_name('deployment.json'),self.deployments[1])
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_missing_actual_native_restore_proof_refuses(self):
  self.native_restore.unlink()
  with self.assertRaises(FileNotFoundError):self.run_aggregate()
  self.assertFalse(self.output.exists())
if __name__=='__main__':unittest.main()
