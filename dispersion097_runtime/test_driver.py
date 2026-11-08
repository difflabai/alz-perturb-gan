"""Actual six-fit dynamic queue, cost-only placement and capacity fallback."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import binding as b
from runtime_dispatch import choose_next,cost_table,settled_cost_rows
P=Path('/ml2/alz/analysis/run_dispersion097.py');spec=importlib.util.spec_from_file_location('scalar_dynamic_driver_test',P);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
COSTS={'pg':[4.,1.]}
class Scheduler(unittest.TestCase):
 def run_fake(self,*,failure=None,capacity_wait=0,expired=False):
  clock=[1000. if expired else 0.];started=[];finished=[];obs=[];tries={}
  def select(pending,active,gpu):
   timed={card:dict(cell=h['cell'],elapsed_seconds=clock[0]-h['start']) for card,h in active.items()}
   return choose_next(pending,timed,gpu,COSTS)[0]
  def start(v,gpu,n):
   key=b.allocation_key(v);tries[key]=tries.get(key,0)+1
   if capacity_wait and gpu==d.GPUS[0] and tries[key]<=capacity_wait:return None
   if failure=='admission' and gpu==d.GPUS[0]:raise RuntimeError('fixture admission failure')
   h=dict(cell=v,gpu=gpu,attempt=key,start=clock[0]);started.append(h);return h
  def ready(h):
   if clock[0]<h['start']+COSTS['pg'][d.GPUS.index(h['gpu'])]:return False
   if failure=='observation' and h['gpu']==d.GPUS[1]:raise TimeoutError('observation only; retain exact handle')
   return True
  def finish(h):finished.append(h);return dict(cell=h['cell'],complete=not(failure=='endpoint' and h['gpu']==d.GPUS[1]),charged_GPU_seconds=clock[0]-h['start'])
  def observe(rows,pending,active,errors,unsettled):obs.append(dict(active=list(active),errors=bool(errors)))
  def sleep(n):clock[0]+=n
  q=d.run_queue(d.matrix(),initial_rows=[],start=start,ready=ready,finish=finish,observe=observe,sleep=sleep,now=lambda:clock[0],training_expiry=1000,select=select)
  return q,started,finished,obs
 def test_six_models_once_and_faster_card_receives_more_work(self):
  q,started,_,obs=self.run_fake();self.assertFalse(q['errors']);self.assertEqual(len(q['rows']),6);self.assertEqual(len({h['attempt'] for h in started}),6)
  self.assertGreater(sum(h['gpu']==d.GPUS[1] for h in started),sum(h['gpu']==d.GPUS[0] for h in started));self.assertTrue(any(len(o['active'])==2 for o in obs))
 def test_capacity_wait_preserves_all_models(self):
  q,started,_,_=self.run_fake(capacity_wait=2);self.assertEqual(len(q['rows']),6);self.assertEqual(len(started),6)
 def test_failure_stops_admission_and_healthy_peer_drains(self):
  q,started,finished,_=self.run_fake(failure='endpoint');self.assertEqual(len(started),2);self.assertEqual(len(finished),2);self.assertTrue(q['errors'])
 def test_observation_timeout_never_restarts_a_model(self):
  q,started,_,_=self.run_fake(failure='observation');self.assertEqual(len(started),2);self.assertEqual(len(q['unsettled']),1)
 def test_failed_admission_keeps_admitted_peer(self):
  q,started,finished,_=self.run_fake(failure='admission');self.assertEqual(len(started),1);self.assertEqual(len(finished),1);self.assertTrue(q['errors'])
 def test_expired_window_refuses_every_new_model(self):
  q,started,_,_=self.run_fake(expired=True);self.assertFalse(started);self.assertEqual(len(q['pending']),6)
 def test_available_slow_card_is_used_when_fast_card_has_no_capacity(self):
  value=d.matrix()[0];selected,proposal=choose_next([value],{},d.GPUS[0],COSTS,eligible={d.GPUS[0]})
  self.assertEqual(selected,value);self.assertEqual(proposal['reason'],'only_card_with_current_capacity')
  self.assertIsNone(choose_next([value],{},d.GPUS[1],COSTS,eligible={d.GPUS[0]})[0])
 def test_no_capacity_does_not_allocate_or_drop_any_cell(self):
  values=d.matrix();self.assertIsNone(choose_next(values,{},d.GPUS[0],COSTS,eligible=set())[0]);self.assertEqual(len(values),6)
 def test_cost_window_scales_only_benchmark_estimates_not_actual_scalar_costs(self):
  rows=[]
  for gpu in d.GPUS:
   for i in range(4):rows.append(dict(gpu_uuid=gpu,effective_seconds=float(i+1),checked_unix=i,attempt=f'{gpu}_{i}'))
  costs,refs=cost_table(rows);self.assertEqual(costs['pg'],[3.,3.]);self.assertTrue(all(len(v)==3 for v in refs['pg']))
 def test_closed_cost_metadata_does_not_parse_quality_and_scales_only_external(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)
   for phase in ('benchmark097','scalarcal097'):
    attempt=phase+'_CPU_fixture';result=root/'analysis'/phase/'results'/(attempt+'.json');result.parent.mkdir(parents=True);result.write_bytes(b'OPAQUE_QUALITY_MUST_NOT_BE_PARSED')
    allocation=root/'artifacts/allocations'/attempt;allocation.mkdir(parents=True);authpath=allocation/'authorization.json';ledger=root/(attempt+'-ledger.json')
    auth=dict(cell=dict(method='pg'),fixed_endpoint=20000,new_model_updates=20000,initial_completed_updates=0,source_model_path=None,ledger_path=str(ledger),gpu_uuid=d.GPUS[0]);authpath.write_text(json.dumps(auth));(allocation/'deployment.json').write_text(json.dumps(dict(authorization_path=str(authpath))))
    ledger.write_text(json.dumps(dict(jobs=[dict(status='completed',exit_code=0,reservation_released=True)])))
    ledger.with_name(attempt+'-settlement.json').write_text(json.dumps(dict(uncertain=None,settlement=dict(status='settled',charged_GPU_seconds=100.),checked_unix=1.)))
   rows=settled_cost_rows(root);self.assertEqual(len(rows),2)
   for row in rows:self.assertAlmostEqual(row['effective_seconds'],100.,places=10)

 def recovered_cost_fixture(self,root):
  failed='CAL_GPU0_V1_external097_f2_pg_s1';replacement='CAL_GPU0_V1_external097infra1_f2_pg_s1';paths={}
  for phase,attempt,status in [('benchmark097',failed,'failed'),('benchmark097_infra1',replacement,'completed')]:
   path=root/'analysis'/phase/'results'/(attempt+'.json');path.parent.mkdir(parents=True);path.write_bytes(b'OPAQUE_RESULT_QUALITY_MUST_NOT_BE_PARSED');paths[attempt]=path
   folder=root/'artifacts/allocations'/attempt;folder.mkdir(parents=True);authpath=folder/'authorization.json';ledger=root/(attempt+'-ledger.json')
   authpath.write_text(json.dumps(dict(cell=dict(method='pg'),fixed_endpoint=20000,new_model_updates=20000,initial_completed_updates=0,source_model_path=None,ledger_path=str(ledger),gpu_uuid=d.GPUS[1])))
   (folder/'deployment.json').write_text(json.dumps(dict(authorization_path=str(authpath))))
   ledger.write_text(json.dumps(dict(jobs=[dict(status=status,exit_code=0 if status=='completed' else 1,reservation_released=True,error='Protected/registered GPU growth no longer fits')])))
   ledger.with_name(attempt+'-settlement.json').write_text(json.dumps(dict(uncertain=None,settlement=dict(status='settled',charged_GPU_seconds=100.),checked_unix=1.)))
  pin=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();matrix=root/'analysis/benchmark097/matrix_complete.json';value=dict(matrix_complete=True,complete_cells=45,required_cells=45,physical_attempts=47,result_SHA256={**{str(i):'fixture' for i in range(44)},replacement:pin(paths[replacement])},logical_to_physical_attempt={'f2_pg_s1':replacement},failed_result_SHA256={failed:pin(paths[failed])});matrix.write_text(json.dumps(value));(root/'analysis/benchmark097_infra1/protocol_v2.json').write_text(json.dumps(dict(known_failed_attempts={failed:pin(paths[failed])})))
  return failed,replacement,paths,matrix,value
 def test_declared_replaced_failure_is_retained_and_only_fresh_success_cost_used(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);failed,replacement,paths,matrix,value=self.recovered_cost_fixture(root);before=paths[failed].read_bytes();rows=settled_cost_rows(root);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['attempt'],replacement);self.assertEqual(rows[0]['source_phase'],'benchmark097_infra1');self.assertAlmostEqual(rows[0]['effective_seconds'],100.);self.assertEqual(paths[failed].read_bytes(),before)
 def test_missing_incomplete_or_tampered_replacement_never_skips_failure(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);failed,replacement,paths,matrix,value=self.recovered_cost_fixture(root)
   matrix.unlink()
   with self.assertRaises(FileNotFoundError):settled_cost_rows(root)
   matrix.write_text(json.dumps({**value,'complete_cells':44}))
   with self.assertRaises(AssertionError):settled_cost_rows(root)
   matrix.write_text(json.dumps(value));paths[replacement].write_bytes(b'TAMPERED_REPLACEMENT')
   with self.assertRaises(AssertionError):settled_cost_rows(root)
 def test_unknown_failed_pg_attempt_is_never_silently_skipped(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);failed,replacement,paths,matrix,value=self.recovered_cost_fixture(root);ledger=root/(replacement+'-ledger.json');job=json.loads(ledger.read_text());job['jobs'][0].update(status='failed',exit_code=1);ledger.write_text(json.dumps(job))
   with self.assertRaises(ValueError):settled_cost_rows(root)

if __name__=='__main__':unittest.main(verbosity=2)
