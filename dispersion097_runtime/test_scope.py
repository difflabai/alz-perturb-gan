"""Actual prepared native closures and scientifically meaningful refusal cases."""
import ast,copy,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import binding as b,preset_scope as s,prepare,resource_profile as p,dual_coordination as d,prior_handoff as handoff
ORIGINAL=s.validate_authorization
def prepared(a):return ORIGINAL(a,require_runtime_proof=False,require_activation=False,require_handoff=False)

class Scope(unittest.TestCase):
 def test_exact_six_fresh_cells_and_twelve_preimport_source_closures(self):
  cells=[s.cell(arm,seed) for seed in (8,9,10) for arm in ('theta1','theta5')]
  self.assertEqual(len({b.allocation_key(c) for c in cells}),6)
  for c in cells:
   for gpu in d.GPUS:
    a,_=prepare.build_authorization(arm=s.arm_of(c),seed=c['seed'],gpu=gpu,check_only=True);prepared(a)
    with patch.object(s,'validate_authorization',side_effect=prepared):b.source_closure(a,b.NATIVE_LAYOUT)
    with self.assertRaises(AssertionError):ORIGINAL(a,require_runtime_proof=False,require_handoff=False)
  self.assertNotIn('torch',sys.modules)
 def test_old_seeds_unknown_arms_and_coordinate_changes_refused(self):
  for arm,seed in [('fixed',8),('theta5',7),('theta1',11),('theta1',True)]:
   with self.assertRaises(AssertionError):s.cell(arm,seed)
  a,_=prepare.build_authorization(arm='theta5',seed=8,gpu=d.GPUS[0],check_only=True)
  for k,v in [('source_model_path','old.pt'),('is_continuation',True),('initial_completed_updates',500),('reuse_completed_model',True),('fixed_endpoint',500),('operation','scalarcal097_cell'),('metric_role','final_evaluation'),('particlegan_version','0.9.6'),('arm','theta1')]:
   other=copy.deepcopy(a);other[k]=v
   with self.subTest(k=k),self.assertRaises(AssertionError):prepared(other)
  for key,value in [('theta_init_scale',10.),('stage_balance',True)]:
   other=copy.deepcopy(a);other['variant_config'][key]=value
   with self.assertRaises(AssertionError):prepared(other)
  other=copy.deepcopy(a);other['released_hyperparameters']['particle_lr']=.03
  with self.assertRaises(AssertionError):prepared(other)
 def test_canonical_accounting_live_binding_and_physical_guard_unchanged(self):
  old=ast.parse((s.ROOT/'scalarcal097_runtime/binding.py').read_text());new=ast.parse((b.HERE/'binding.py').read_text())
  def node(tree,name):return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  for name in ('core_envelope','unseen','require_live','verify','modules'):self.assertEqual(node(old,name),node(new,name),name)
  self.assertEqual((b.HERE/'project_runner_gpu0.py').read_bytes(),(s.ROOT/'scalarcal097_runtime/project_runner_gpu0.py').read_bytes())
  self.assertEqual((b.HERE/'launch_core_gpu0.py').read_bytes(),(s.ROOT/'scalarcal097_runtime/launch_core_gpu0.py').read_bytes())
  for gpu,policy in p.validate_proposal().items():self.assertEqual(policy['roles'],{'alz_core_pg':dict(gpu_bytes=3*2**30,ram_bytes=6*2**30)})
 def test_other_study_reservations_cannot_be_relabelled_as_current_peers(self):
  a,_=prepare.build_authorization(arm='theta1',seed=8,gpu=d.GPUS[1],check_only=True)
  gpu=d.GPUS[0];key='ALZ_CAL_V1_scalarcal097_fixed_f0_pg_s5'
  reservation=dict(allocation_id=key,status='reserved',owner=d.identity(os.getpid()),gpu_uuids=[gpu])
  state=dict(open_reservations=[reservation],execution_controller=dict(calibration_parallel_backends={key:dict(cell=dict(experiment='scalarcal097_fixed',fold=0,method='pg',seed=5),gpu_uuid=gpu)}))
  with self.assertRaises((ValueError,AssertionError,KeyError)):d.validate_active_peers(state,a)
 def test_capacity_forecasts_preserve_foreign_growth_and_host_reserve(self):
  class Registry:
   def __init__(self,*args,**kwargs):pass
   def accounting(self,*args):pass
   def protected(self,*args):return [],0,0
  class Admission:pass
  Admission.Registry=Registry
  for gpu in d.GPUS:
   obs=dict(used=0,gpu_memory={},total=48*2**30,available_ram=32*2**30)
   self.assertEqual(p.capacity(obs,Admission,gpu)['host_remaining_growth_required_bytes'],24*2**30)
   with self.assertRaises(AssertionError):p.capacity({**obs,'available_ram':23*2**30},Admission,gpu)
   with self.assertRaises(AssertionError):p.capacity({**obs,'total':5*2**30},Admission,gpu)
 def test_handoff_requires_all_six_prior_models_and_committed_opaque_report(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);matrix,report,complete=handoff.paths(root)[-3:]
   for path in (matrix,report,complete):path.parent.mkdir(parents=True,exist_ok=True)
   pins={}
   for seed in (5,6,7):
    for arm in ('fixed','adaptive'):
     key=f'CAL_GPU0_V1_scalarcal097_{arm}_f0_pg_s{seed}';path=root/'analysis/scalarcal097/results'/(key+'.json');path.parent.mkdir(exist_ok=True);path.write_bytes(b'OPAQUE_CALIBRATION_RESULT');pins[key]=handoff.sha(path)
   value=dict(status='ALL_SIX_FRESH_SCALAR_LR_NATIVE_MODELS_AND_CALIBRATION_AUDITS_COMPLETE',matrix_complete=True,complete_cells=6,required_cells=6,result_SHA256=pins,all_new_training_from_scratch=True,initial_completed_updates=0,calibration_only=True)
   matrix.write_text(json.dumps(value));report.write_bytes(b'OPAQUE_CALIBRATION_REPORT_NEVER_DECODED')
   complete.write_text(json.dumps(dict(status='SCALARCAL097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS',report_path=str(report))))
   plan=root/'scalarcal097_runtime/plan.json';plan.parent.mkdir();plan.write_text('{"enabled":false}')
   proof=root/'analysis/scalarcal097/runtime_CPU_qualification_v10.json';proof.write_text(json.dumps(dict(status='SCALARCAL097_SIX_FRESH_CELLS_PREPARED_SCOPE_ACCOUNTING_CPU_PASS',actual_tests=39,native_cell_device_bindings=12,source_SHA256={})))
   committed=handoff.sha(report)
   with patch.object(handoff.recovered,'ready',return_value={'prior_benchmark_hashes':'CPU_fixture'}),patch.object(handoff,'committed_sha',return_value=committed):
    self.assertIn(str(report),handoff.ready(root))
    report.write_bytes(b'TAMPERED_REPORT')
    with self.assertRaises(AssertionError):handoff.ready(root)
    report.write_bytes(b'OPAQUE_CALIBRATION_REPORT_NEVER_DECODED');matrix.write_text(json.dumps({**value,'complete_cells':5}))
    with self.assertRaises(AssertionError):handoff.ready(root)
    matrix.write_text(json.dumps(value));complete.unlink()
    with self.assertRaises(AssertionError):handoff.ready(root)
 def test_exact_tie_retains_theta1_even_if_theta5_secondary_score_is_better(self):
  path=s.ROOT/'analysis/aggregate_dispersion097.py';spec=importlib.util.spec_from_file_location('theta_report',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  rows=[]
  for seed in (8,9,10):
   for arm in ('theta1','theta5'):
    metrics=dict.fromkeys(m.METRICS,1.);metrics['energy_score']=0. if arm=='theta5' else 1.
    rows.append(dict(arm=arm,seed=seed,complete=True,all_new_training_from_scratch=True,initial_completed_updates=0,metrics=metrics))
  self.assertEqual(m.summarize(rows)['calibration_preferred_arm'],'theta1')
  rows.pop()
  with self.assertRaises(AssertionError):m.summarize(rows)
 def test_fitting_trace_endpoint_math_identical_except_fresh_seed_domain(self):
  old=(s.ROOT/'scalarcal097_runtime/endpoint_evaluation.py').read_text().replace('(5,6,7)','(8,9,10)')
  self.assertEqual(old,(b.HERE/'endpoint_evaluation.py').read_text())
 def test_whole_study_window_uses_only_currently_eligible_cards(self):
  from window_fit import forecast
  both={g:dict(GPU_fits=True,host_fits=True) for g in d.GPUS};costs={'pg':[4000.,1000.]}
  self.assertTrue(forecast(costs,both,10000.)['all_six_predicted_to_fit'])
  slow={d.GPUS[0]:both[d.GPUS[0]]};q=forecast(costs,slow,10000.)
  self.assertFalse(q['all_six_predicted_to_fit']);self.assertEqual(q['forecast']['counts_per_lane']['pg'],[6,0])
  self.assertTrue(forecast(costs,{d.GPUS[1]:both[d.GPUS[1]]},10000.)['all_six_predicted_to_fit'])
 def test_capacity_wait_never_counts_unverified_or_host_blocked_card(self):
  from window_fit import forecast
  q=forecast({'pg':[4000.,1000.]},{d.GPUS[1]:dict(GPU_fits=True,host_fits=False)},10000.)
  self.assertFalse(q['all_six_predicted_to_fit']);self.assertFalse(q['eligible_cards']);self.assertIsNone(q['required_seconds'])
 def test_elapsed_wait_can_make_even_both_card_forecast_infeasible(self):
  from window_fit import forecast
  both={g:dict(GPU_fits=True,host_fits=True) for g in d.GPUS}
  self.assertFalse(forecast({'pg':[4000.,1000.]},both,7000.)['all_six_predicted_to_fit'])

if __name__=='__main__':unittest.main(verbosity=2)
