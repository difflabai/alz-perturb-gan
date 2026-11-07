"""Native prepared closures and refusal gates; no allocation or model imports."""
import ast,copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import binding as b,preset_scope as s,prepare,resource_profile as p,dual_coordination as d

def prepared_validate(a):return ORIGINAL_VALIDATE(a,require_runtime_proof=False,require_activation=False,require_handoff=False)
ORIGINAL_VALIDATE=s.validate_authorization

class Scope(unittest.TestCase):
 def test_six_fresh_namespace_cells_and_balanced_both_cards(self):
  values=[s.cell(a,seed) for seed in (5,6,7) for a in ('fixed','adaptive')]
  self.assertEqual(len({b.allocation_key(c) for c in values}),6)
  self.assertEqual([sum(d.assigned_gpu(c)==g for c in values) for g in d.GPUS],[3,3]);s.protocol();d.validate_resource_protocol()
  for arm,seed in [('wrong',5),('fixed',0),('adaptive',4),('adaptive',8),('fixed',True)]:
   with self.assertRaises(AssertionError):s.cell(arm,seed)
 def test_all_six_actual_prepared_source_closures_with_deployment_gates_explicitly_held(self):
  for seed in (5,6,7):
   for arm in ('fixed','adaptive'):
    auth,_=prepare.build_authorization(arm=arm,seed=seed,check_only=True)
    prepared_validate(auth)
    with patch.object(s,'validate_authorization',side_effect=prepared_validate):b.source_closure(auth,b.NATIVE_LAYOUT)
    with self.assertRaises(AssertionError):ORIGINAL_VALIDATE(auth,require_runtime_proof=False,require_handoff=False)
    with self.assertRaises((AssertionError,FileNotFoundError)):s.prior_study_ready()
  self.assertNotIn('torch',sys.modules)
 def test_warmstart_changed_formulation_wrong_role_or_endpoint_refused(self):
  auth,_=prepare.build_authorization(arm='adaptive',seed=5,check_only=True)
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
  auth,_=prepare.build_authorization(arm='fixed',seed=5,check_only=True)
  gpu=d.GPUS[1-d.GPUS.index(auth['gpu_uuid'])];key='ALZ_CAL_V1_external097_f0_pg_s0'
  reservation=dict(allocation_id=key,status='reserved',owner=d.identity(os.getpid()),gpu_uuids=[gpu])
  old=dict(open_reservations=[reservation],execution_controller=dict(calibration_parallel_backends={key:dict(cell=dict(experiment='external097',fold=0,method='pg',seed=0),gpu_uuid=gpu)}))
  with self.assertRaises((ValueError,AssertionError,KeyError)):d.validate_active_peers(old,auth)
  self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
