"""Actual dynamic queue coverage, prefix preservation and failure-drain tests."""
import copy,importlib.util,unittest
from pathlib import Path
import binding as b
from runtime_dispatch import choose_next,cost_table
P=Path('/ml2/alz/analysis/run_benchmark097_balanced.py');spec=importlib.util.spec_from_file_location('balanced_driver_test',P);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
COSTS={'pg':[4.,1.],'scvi':[.5,.2],'cfgen':[6.,3.]}
class Dispatch(unittest.TestCase):
 def run_fake(self,*,prefix=0,failure=None,capacity_wait=0,expired=False):
  clock=[1000. if expired else 0.];allcells=d.matrix();initial=[dict(cell=v,complete=True,charged_GPU_seconds=.5,immutable_reference=True) for v in allcells[:prefix]];started=[];finished=[];tries={};obs=[]
  def select(pending,active,gpu):
   active={g:dict(cell=h['cell'],elapsed_seconds=clock[0]-h['start']) for g,h in active.items()}
   return choose_next(pending,active,gpu,COSTS)[0]
  def start(v,gpu,n):
   key=b.allocation_key(v);tries[key]=tries.get(key,0)+1
   if capacity_wait and gpu==d.GPUS[0] and tries[key]<=capacity_wait:return None
   if failure=='admission' and gpu==d.GPUS[0]:raise RuntimeError('fixture admission error')
   h=dict(cell=v,gpu=gpu,attempt=key,start=clock[0]);started.append(h);return h
  def ready(h):
   if clock[0]<h['start']+COSTS[h['cell']['method']][d.GPUS.index(h['gpu'])]:return False
   if failure=='observation' and h['gpu']==d.GPUS[1]:raise TimeoutError('observation only; original handle retained')
   return True
  def finish(h):finished.append(h);return dict(cell=h['cell'],complete=not(failure=='endpoint' and h['gpu']==d.GPUS[1]),charged_GPU_seconds=clock[0]-h['start'])
  def observe(rows,pending,active,errors,unsettled):obs.append(dict(active=list(active),errors=bool(errors)))
  def sleep(n):clock[0]+=n
  result=d.run_queue(allcells[prefix:],initial_rows=initial,start=start,ready=ready,finish=finish,observe=observe,sleep=sleep,now=lambda:clock[0],training_expiry=1000,select=select)
  return result,started,finished,obs,initial
 def test_all45_once_with_real_prefix_references_and_both_cards(self):
  q,started,finished,obs,initial=self.run_fake(prefix=13)
  self.assertFalse(q['errors']);self.assertFalse(q['pending']);self.assertEqual(len(q['rows']),45);self.assertEqual(q['rows'][:13],initial);self.assertEqual(len(started),32)
  self.assertEqual(len({h['attempt'] for h in started}),32);self.assertTrue(any(len(o['active'])==2 for o in obs));self.assertTrue(all(len(o['active'])<=2 for o in obs))
 def test_capacity_wait_keeps_full_coverage_without_duplicate_models(self):
  q,started,_,_,_=self.run_fake(capacity_wait=2);self.assertEqual(len(q['rows']),45);self.assertEqual(len(started),45)
 def test_incomplete_native_endpoint_stops_admission_but_drains_healthy_peer(self):
  q,started,finished,_,_=self.run_fake(failure='endpoint');self.assertTrue(q['errors']);self.assertEqual(len(started),2);self.assertEqual(len(finished),2);self.assertFalse(q['unsettled'])
 def test_observation_timeout_preserves_handle_and_does_not_restart(self):
  q,started,finished,_,_=self.run_fake(failure='observation');self.assertEqual(len(started),2);self.assertEqual(len(q['unsettled']),1);self.assertTrue(q['errors'])
 def test_failed_admission_drains_existing_cell(self):
  q,started,finished,_,_=self.run_fake(failure='admission');self.assertEqual(len(started),1);self.assertEqual(len(finished),1);self.assertTrue(q['errors'])
 def test_expired_window_never_starts_any_model(self):
  q,started,_,_,_=self.run_fake(expired=True);self.assertFalse(started);self.assertEqual(len(q['pending']),45)
 def test_cost_window_uses_only_latest3_card_method_settlements(self):
  rows=[dict(method=m,gpu_uuid=g,seconds=float(i+1),checked_unix=i,attempt=f'{m}_{g}_{i}') for m in COSTS for g in d.GPUS for i in range(4)]
  costs,evidence=cost_table(rows)
  for m in COSTS:self.assertEqual(costs[m],[3.,3.]);self.assertTrue(all(len(p)==3 for p in evidence[m]))
 def test_slow_card_can_idle_when_it_would_delay_single_job(self):
  v=d.matrix()[0];selected,_=choose_next([v],{},d.GPUS[0],COSTS);self.assertIsNone(selected)
if __name__=='__main__':unittest.main(verbosity=2)
