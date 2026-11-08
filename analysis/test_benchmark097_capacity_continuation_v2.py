import importlib.util,pathlib,sys,unittest
root=pathlib.Path('/ml2/alz');spec=importlib.util.spec_from_file_location('continuation',root/'analysis/continue_benchmark097_after_capacity_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Checks(unittest.TestCase):
 def run_case(self,*,known=False,legacy=False,bad=False,start_error=False,capacity_wait=False,expiry=1000):
  t=[0];starts=[];finishes=[];seen=[];v=dict(id='new');old=dict(id='known');lh=dict(cell=dict(id='legacy'),gpu=m.GPUS[1],identity={},attempt='legacy');rows=[dict(cell=old,complete=False)] if known else [];active={m.GPUS[1]:lh} if legacy else {}
  def start(v,gpu,n):
   starts.append((v['id'],gpu))
   if start_error:raise RuntimeError('failure')
   if capacity_wait and len(starts)==1:return None
   return dict(cell=v,gpu=gpu,identity={},attempt=v['id'])
  def finish(h):finishes.append(h['attempt']);return dict(cell=h['cell'],complete=not bad)
  result=m.drive([v],rows,active,key=lambda v:v['id'],start=start,ready=lambda h:t[0]>0,finish=finish,select=lambda p,a,g:p[0] if p else None,observe=lambda *a:seen.append(a),sleep=lambda n:t.__setitem__(0,t[0]+1),now=lambda:t[0],expiry=expiry)
  return result,starts,finishes
 def test_known_failure_not_retried_and_legacy_drains(self):
  q,s,f=self.run_case(known=True,legacy=True);self.assertEqual(len(s),1);self.assertEqual(set(f),{'legacy','new'});self.assertEqual(sum(r['complete'] for r in q['rows']),2);self.assertFalse(q['errors'])
 def test_unexpected_failure_stops_without_retry(self):q,s,f=self.run_case(bad=True);self.assertEqual(len(s),1);self.assertTrue(q['errors'])
 def test_admission_error_drains_healthy_legacy(self):q,s,f=self.run_case(legacy=True,start_error=True);self.assertEqual(f,['legacy']);self.assertTrue(q['errors'])
 def test_capacity_wait_does_not_consume_key(self):q,s,f=self.run_case(capacity_wait=True);self.assertEqual(len(s),2);self.assertEqual(f,['new']);self.assertFalse(q['errors'])
 def test_expiry_stops_admissions_but_drains_legacy(self):q,s,f=self.run_case(legacy=True,expiry=1000);self.assertFalse(q['errors']);q,s,f=self.run_case(legacy=True,expiry=0);self.assertEqual(s,[]);self.assertEqual(f,['legacy']);self.assertTrue(q['errors'])
 def test_duplicate_original_key_refused(self):
  with self.assertRaises(AssertionError):m.drive([dict(id='x')],[dict(cell=dict(id='x'),complete=False)],{},key=lambda v:v['id'],start=None,ready=None,finish=None,select=None,observe=None,sleep=None,now=lambda:0,expiry=100)
class Stability(unittest.TestCase):
 def test_requires_contiguous_five_minutes(self):
  c={};self.assertFalse(m.stability_ready(c,'g',0,True,300));self.assertFalse(m.stability_ready(c,'g',299,True,300));self.assertTrue(m.stability_ready(c,'g',300,True,300))
 def test_capacity_loss_resets_stability(self):
  c={};m.stability_ready(c,'g',0,True,300);self.assertFalse(m.stability_ready(c,'g',299,False,300));self.assertFalse(m.stability_ready(c,'g',300,True,300));self.assertFalse(m.stability_ready(c,'g',599,True,300));self.assertTrue(m.stability_ready(c,'g',600,True,300))
 def test_cards_have_independent_stability(self):
  c={};m.stability_ready(c,'g0',0,True,300);m.stability_ready(c,'g1',200,True,300);self.assertTrue(m.stability_ready(c,'g0',300,True,300));self.assertFalse(m.stability_ready(c,'g1',300,True,300))
if __name__=='__main__':unittest.main()
