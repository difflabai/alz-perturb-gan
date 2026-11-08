"""Native prepared closures and refusal gates; no allocation or model imports."""
import ast,copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import binding as b,preset_scope as s,prepare,resource_profile as p,dual_coordination as d

def prepared_validate(a):return ORIGINAL_VALIDATE(a,require_runtime_proof=False,require_activation=False,require_handoff=False)
ORIGINAL_VALIDATE=s.validate_authorization

class Scope(unittest.TestCase):
 def test_six_fresh_cells_and_twelve_eligible_card_bindings(self):
  values=[s.cell(a,seed) for seed in (5,6,7) for a in ('fixed','adaptive')]
  self.assertEqual(len({b.allocation_key(c) for c in values}),6)
  self.assertEqual(len([(v,g) for v in values for g in d.GPUS]),12);s.protocol();d.validate_resource_protocol()
  for arm,seed in [('wrong',5),('fixed',0),('adaptive',4),('adaptive',8),('fixed',True)]:
   with self.assertRaises(AssertionError):s.cell(arm,seed)
 def test_all12_actual_prepared_source_closures_with_deployment_gates_held(self):
  for seed in (5,6,7):
   for arm in ('fixed','adaptive'):
    for gpu in d.GPUS:
     auth,_=prepare.build_authorization(arm=arm,seed=seed,gpu=gpu,check_only=True)
     prepared_validate(auth)
     with patch.object(s,'validate_authorization',side_effect=prepared_validate):b.source_closure(auth,b.NATIVE_LAYOUT)
     with self.assertRaises(AssertionError):ORIGINAL_VALIDATE(auth,require_runtime_proof=False,require_handoff=False)
     with self.assertRaises((AssertionError,FileNotFoundError)):s.prior_study_ready()
  self.assertNotIn('torch',sys.modules)
 def test_warmstart_changed_formulation_wrong_role_or_endpoint_refused(self):
  auth,_=prepare.build_authorization(arm='adaptive',seed=5,gpu=d.GPUS[0],check_only=True)
  for key,value in [('source_model_path','old.pt'),('is_continuation',True),('initial_completed_updates',500),('reuse_completed_model',True),('fixed_endpoint',500),('operation','external097_benchmark_cell'),('metric_role','final_evaluation'),('particlegan_version','0.9.6'),('arm','fixed')]:
   changed=copy.deepcopy(auth);changed[key]=value
   with self.subTest(key=key),self.assertRaises(AssertionError):prepared_validate(changed)
  changed=copy.deepcopy(auth);changed['released_hyperparameters']['particle_lr']=.03
  with self.assertRaises(AssertionError):prepared_validate(changed)
 def test_core_accounting_and_live_worker_binding_functions_unchanged(self):
  old=ast.parse((s.ROOT/'benchmark097_dual_runtime/binding.py').read_text());new=ast.parse((b.HERE/'binding.py').read_text())
  def node(tree,name):return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  for name in ('core_envelope','unseen','require_live','verify','modules'):self.assertEqual(node(old,name),node(new,name),name)
  self.assertEqual((b.HERE/'project_runner_gpu0.py').read_bytes(),(s.ROOT/'benchmark097_dual_runtime/project_runner_gpu0.py').read_bytes())
 def test_handoff_reads_metadata_and_hashes_reports_without_quality_values(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);paths=s.handoff_paths(root)
   for path in paths:path.parent.mkdir(exist_ok=True,parents=True)
   paths[0].write_text(json.dumps(dict(matrix_complete=True,complete_cells=45,required_cells=45,result_SHA256={str(i):'fixed' for i in range(45)})))
   paths[1].write_bytes(b'OPAQUE_TARGET_METRIC_REPORT_NEVER_PARSED_BY_HANDOFF')
   paths[2].write_bytes(b'OPAQUE_TARGET_SENSITIVITY_REPORT_NEVER_PARSED_BY_HANDOFF')
   paths[3].write_text(json.dumps(dict(status='FULL_RESEARCH_PIPELINE_COMPARISON_READY_FOR_FINAL_AGENT_AUDIT')))
   paths[4].write_text(json.dumps(dict(status='CALIBRATION_EXCLUDED53_SENSITIVITY_COMPLETE_FINAL_AGENT_AUDIT_REMAINS',original57_primary_preserved=True)))
   plan=root/'benchmark097_dual_runtime/plan.json';plan.parent.mkdir();plan.write_text('{"enabled":false}')
   self.assertEqual(len(s.prior_study_ready(root)),5)
   plan.write_text('{"enabled":true}')
   with self.assertRaises(AssertionError):s.prior_study_ready(root)
   plan.write_text('{"enabled":false}');paths[4].unlink()
   with self.assertRaises(AssertionError):s.prior_study_ready(root)
 def test_immutable_guard_resources_and_foreign_growth_reserve_on_both_cards(self):
  class Registry:
   def __init__(self,*args,**kwargs):pass
   def accounting(self,*args):pass
   def protected(self,*args):return [],0,0
  class Admission:pass
  Admission.Registry=Registry
  for gpu,policy in p.validate_proposal().items():
   self.assertEqual(policy['roles'],{'alz_core_pg':dict(gpu_bytes=3*2**30,ram_bytes=6*2**30)})
   obs=dict(used=0,gpu_memory={},total=48*2**30,available_ram=32*2**30)
   self.assertEqual(p.capacity(obs,Admission,gpu)['host_remaining_growth_required_bytes'],24*2**30)
   with self.assertRaises(AssertionError):p.capacity({**obs,'available_ram':23*2**30},Admission,gpu)
   with self.assertRaises(AssertionError):p.capacity({**obs,'total':5*2**30},Admission,gpu)
 def test_current_other_study_reservations_cannot_be_relabelled_as_new_peers(self):
  auth,_=prepare.build_authorization(arm='fixed',seed=5,gpu=d.GPUS[1],check_only=True)
  gpu=d.GPUS[1-d.GPUS.index(auth['gpu_uuid'])];key='ALZ_CAL_V1_external097_f0_pg_s0'
  reservation=dict(allocation_id=key,status='reserved',owner=d.identity(os.getpid()),gpu_uuids=[gpu])
  old=dict(open_reservations=[reservation],execution_controller=dict(calibration_parallel_backends={key:dict(cell=dict(experiment='external097',fold=0,method='pg',seed=0),gpu_uuid=gpu)}))
  with self.assertRaises((ValueError,AssertionError,KeyError)):d.validate_active_peers(old,auth)
  self.assertNotIn('torch',sys.modules)
 def test_scientific_worker_auditor_and_scoring_unchanged_from_qualified_v2(self):
  for name in ('worker_gpu0.py','finish.py','endpoint_evaluation.py'):
   old=subprocess.check_output(['git','show','3c49817:scalarcal097_runtime/'+name],cwd=s.ROOT)
   self.assertEqual(old,(b.HERE/name).read_bytes(),name)
 def test_balanced_lineage_barrier_hashes_reports_without_reading_quality(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);common=[root/'analysis/benchmark097/matrix_complete.json',root/'analysis/benchmark097/comparison.json',root/'analysis/benchmark097_calibration_excluded53_report.json']
   for path in common:path.parent.mkdir(parents=True,exist_ok=True)
   common[0].write_text(json.dumps(dict(matrix_complete=True,complete_cells=45,required_cells=45,result_SHA256={str(i):'fixed' for i in range(45)})))
   common[1].write_bytes(b'OPAQUE_PRIMARY_NEVER_PARSED');common[2].write_bytes(b'OPAQUE_SENSITIVITY_NEVER_PARSED')
   source=root/'analysis/run_released097_balanced_pipeline.py';source.write_text('CPU_FIXTURE_ONLY')
   aux=root/'analysis/benchmark097_balanced';aux.mkdir();proof=aux/'runtime_CPU_qualification_v2.json';handoff=aux/'handoff_completed.json';handoff.write_text('{}')
   proof.write_text(json.dumps(dict(status='EXTERNAL097_SAME45_BALANCED_RESOURCE_SCOPE_CPU_PASS',native_cell_device_bindings=90,source_SHA256={str(source):s.sha(source)})))
   barrier=aux/'scientific_reports_completed.json';value=dict(status='SAME_FROZEN45_ORIGINAL57_PRIMARY_AND_PRESPECIFIED53_COMPLETE',complete_fresh_models=45,original57_primary_preserved=True,prespecified53_complete=True,new_generated_counts_for53=0,source_pipeline_path=str(source),source_pipeline_SHA256=s.sha(source),runtime_qualification_path=str(proof),runtime_qualification_SHA256=s.sha(proof),handoff_SHA256=s.sha(handoff),report_SHA256={str(path):s.sha(path) for path in common});barrier.write_text(json.dumps(value))
   marker=root/'artifacts/released097-balanced-research-pipeline/completed.json';marker.parent.mkdir(parents=True);marker.write_text(json.dumps(dict(status='SAME45_SCIENTIFIC_REPORTS_READY_FINAL_AGENT_AUDIT_AND_SCALAR_CALIBRATION_REMAIN')))
   for name in ('benchmark097_dual_runtime','benchmark097_balanced_runtime'):
    path=root/name/'plan.json';path.parent.mkdir();path.write_text('{"enabled":false}')
   self.assertEqual(len(s.prior_study_ready(root)),8)
   common[1].write_bytes(b'TAMPERED_PRIMARY')
   with self.assertRaises(AssertionError):s.prior_study_ready(root)
 def test_authorization_pins_every_prior_barrier_dependency(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'extra-handoff-source.json';path.write_text('{"CPU_fixture":true}')
   with patch.object(s,'handoff_paths',return_value=[path]),patch.object(s,'prior_study_ready',return_value={str(path):s.sha(path)}):
    auth,_=prepare.build_authorization(arm='fixed',seed=5,gpu=d.GPUS[0],check_only=True)
   self.assertEqual(auth['files_SHA256'][str(path)],s.sha(path))

 def test_exiting_upstream_stat_race_is_missing_handle_not_success(self):
  with patch.object(Path,'read_text',side_effect=ProcessLookupError(3,'No such process')):
   self.assertIsNone(d.identity(os.getpid()))

 def test_recovered45_barrier_is_hash_only_and_rejects_wrong_mapping(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);common=[root/'analysis/benchmark097/matrix_complete.json',root/'analysis/benchmark097/comparison.json',root/'analysis/benchmark097_calibration_excluded53_report.json']
   for path in common:path.parent.mkdir(parents=True,exist_ok=True)
   common[1].write_bytes(b'OPAQUE_PRIMARY_NEVER_PARSED');common[2].write_bytes(b'OPAQUE_SENSITIVITY_NEVER_PARSED')
   failed=['CAL_GPU0_V1_external097_f1_cfgen_s2','CAL_GPU0_V1_external097_f2_cfgen_s1','CAL_GPU0_V1_external097_f2_pg_s1'];failedpins={}
   for key in failed:
    path=root/'analysis/benchmark097/results'/(key+'.json');path.parent.mkdir(exist_ok=True);path.write_bytes(b'OPAQUE_FAILED_RESULT');failedpins[key]=s.sha(path)
   mapping={f'f{f}_{method}_s{seed}':f"CAL_GPU0_V1_{'external097infra1' if (f,method,seed) in ((2,'pg',1),(2,'cfgen',1),(1,'cfgen',2)) else 'external097'}_f{f}_{method}_s{seed}" for f in range(3) for seed in range(5) for method in ('pg','scvi','cfgen')}
   m=dict(matrix_complete=True,complete_cells=45,required_cells=45,physical_attempts=48,result_SHA256={key:'fixed' for key in mapping.values()},logical_to_physical_attempt=mapping,all_new_training_from_scratch=True,failed_weights_resumed=False,scientific_protocol_unchanged=True,infrastructure_failed_attempts=failed,failed_result_SHA256=failedpins);common[0].write_text(json.dumps(m))
   aux=root/'analysis/benchmark097_infra1';aux.mkdir();source=root/'analysis/run_benchmark097_infra1_pipeline.py';source.write_text('CPU_FIXTURE_ONLY');proof=aux/'runtime_CPU_qualification_v5.json';proof.write_text(json.dumps(dict(status='ONE_FRESH_INFRASTRUCTURE_REPLACEMENT_NATIVE_SCOPE_CPU_PASS',actual_tests=22,native_cell_device_bindings=6,source_SHA256={str(source):s.sha(source)})))
   barrier=aux/'scientific_reports_completed.json';q=dict(status='RECOVERED_LOGICAL45_ORIGINAL57_PRIMARY_AND_PRESPECIFIED53_COMPLETE',complete_fresh_models=45,physical_attempts=48,failed_attempts=failed,original57_primary_preserved=True,prespecified53_complete=True,new_generated_counts_for53=0,source_pipeline_path=str(source),source_pipeline_SHA256=s.sha(source),runtime_qualification_path=str(proof),runtime_qualification_SHA256=s.sha(proof),report_SHA256={str(p):s.sha(p) for p in common});barrier.write_text(json.dumps(q))
   child=root/'artifacts/benchmark097-infra1-pipeline/completed.json';child.parent.mkdir(parents=True);child.write_text(json.dumps(dict(status='RECOVERED45_57_53_READY_FRESH_SCALAR_CALIBRATION_AND_FINAL_AUDIT_REMAIN')))
   for name in ('benchmark097_dual_runtime','benchmark097_balanced_runtime','benchmark097_infra1_runtime'):
    p=root/name/'plan.json';p.parent.mkdir();p.write_text('{"enabled":false}')
   self.assertEqual(len(s.prior_study_ready(root)),10)
   for filename in ('runtime_CPU_qualification_v5.json',):
    adopted=aux/filename;adopted.write_bytes(proof.read_bytes());q.update(runtime_qualification_path=str(adopted),runtime_qualification_SHA256=s.sha(adopted));barrier.write_text(json.dumps(q))
    self.assertEqual(len(s.prior_study_ready(root)),10)
   common[1].write_bytes(b'TAMPERED_PRIMARY')
   with self.assertRaises(AssertionError):s.prior_study_ready(root)
   common[1].write_bytes(b'OPAQUE_PRIMARY_NEVER_PARSED');m['logical_to_physical_attempt']['f2_pg_s1']='WRONG';common[0].write_text(json.dumps(m));q['report_SHA256'][str(common[0])]=s.sha(common[0]);barrier.write_text(json.dumps(q))
   with self.assertRaises(AssertionError):s.prior_study_ready(root)

if __name__=='__main__':unittest.main(verbosity=2)
