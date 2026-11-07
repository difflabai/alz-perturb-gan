"""Exercise the real queue's coverage, waiting and failure-drain behavior."""
import importlib.util,unittest
from pathlib import Path
P=Path('/ml2/alz/analysis/run_benchmark097_dual.py');spec=importlib.util.spec_from_file_location('dual_driver_under_test',P);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
class Scheduler(unittest.TestCase):
 def run_fake(self,*,failure=None,capacity_wait=0,expired=False):
  clock=[10000 if expired else 0];started=[];finished=[];observations=[];attempts={}
  def start(v,n):
   key=(v['fold'],v['method'],v['seed']);attempts[key]=attempts.get(key,0)+1;gpu=driver.assigned_gpu(v)
   if capacity_wait and gpu==driver.GPUS[0] and attempts[key]<=capacity_wait:return None
   if failure=='admission' and gpu==driver.GPUS[0]:raise RuntimeError('confined admission failure')
   h=dict(cell=v,gpu=gpu,attempt=str(key),started=clock[0]);started.append(h);return h
  def ready(h):
   if clock[0]<h['started']+2:return False
   if failure=='observation' and h['gpu']==driver.GPUS[1]:raise TimeoutError('observation only, original handle retained')
   if failure in ('endpoint','observation','admission') and h['gpu']==driver.GPUS[0] and clock[0]<h['started']+4:return False
   return True
  def finish(h):
   finished.append(h);return dict(cell=h['cell'],complete=not(failure=='endpoint' and h['gpu']==driver.GPUS[1]),charged_GPU_seconds=clock[0]-h['started'])
  def observe(rows,pending,active,errors,unsettled):observations.append(dict(active=list(active),errors=bool(errors),pending=list(pending)))
  def sleep(n):clock[0]+=n
  q=driver.run_queue(driver.matrix(),start=start,ready=ready,finish=finish,observe=observe,sleep=sleep,now=lambda:clock[0],training_expiry=10000)
  return q,started,finished,observations,attempts
 def test_all45_cells_run_once_at_most_one_per_card(self):
  q,s,f,o,_=self.run_fake();self.assertFalse(q['errors']);self.assertFalse(q['pending']);self.assertEqual(len(q['rows']),45)
  self.assertEqual(len({x['attempt'] for x in s}),45);self.assertTrue(any(len(x['active'])==2 for x in o));self.assertTrue(all(len(x['active'])==len(set(x['active']))<=2 for x in o))
 def test_capacity_wait_keeps_every_cell_and_other_card_progresses(self):
  q,s,f,o,a=self.run_fake(capacity_wait=3);self.assertEqual(len(q['rows']),45);self.assertEqual(len(s),45);self.assertTrue(any(n==4 for n in a.values()))
 def test_incomplete_cell_stops_new_admission_and_drains_other_card(self):
  q,s,f,o,_=self.run_fake(failure='endpoint');self.assertEqual(len(s),2);self.assertEqual(len(q['rows']),2);self.assertEqual(sum(x['complete'] for x in q['rows']),1);self.assertFalse(q['unsettled']);self.assertTrue(q['errors']);self.assertTrue(any(x['errors'] and x['active'] for x in o))
 def test_observation_timeout_retains_handle_and_never_restarts(self):
  q,s,f,o,_=self.run_fake(failure='observation');self.assertEqual(len(s),2);self.assertEqual(len(q['rows']),1);self.assertEqual(len(q['unsettled']),1);self.assertEqual(q['unsettled'][0]['attempt'],s[0]['attempt']);self.assertTrue(q['errors'])
 def test_admission_error_drains_already_admitted_cell(self):
  q,s,f,o,_=self.run_fake(failure='admission');self.assertEqual(len(s),1);self.assertEqual(len(q['rows']),1);self.assertTrue(q['errors']);self.assertFalse(q['unsettled'])
 def test_expired_training_window_never_admits(self):
  q,s,f,o,_=self.run_fake(expired=True);self.assertFalse(s);self.assertFalse(f);self.assertEqual(len(q['pending']),45);self.assertTrue(q['errors'])
if __name__=='__main__':unittest.main(verbosity=2)
