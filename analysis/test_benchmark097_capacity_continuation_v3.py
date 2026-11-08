import importlib.util,json,pathlib,sys,tempfile,unittest
root=pathlib.Path('/ml2/alz');spec=importlib.util.spec_from_file_location('continuation',root/'analysis/continue_benchmark097_after_capacity_v3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
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
class Adoption(unittest.TestCase):
 def test_live_exact_guard_is_observed_without_receipt_or_control(self):
  with tempfile.TemporaryDirectory() as folder:
   p=pathlib.Path(folder);i=dict(pid=1,birth=2,boot='test');o=m.ObservedOwner(i,p/'ledger',p/'settlement',get_identity=lambda pid:i);self.assertIsNone(o.poll());self.assertFalse(list(p.iterdir()))
 def test_absent_guard_requires_clean_terminal_receipts(self):
  with tempfile.TemporaryDirectory() as folder:
   p=pathlib.Path(folder);i=dict(pid=1,birth=2,boot='test');(p/'ledger').write_text(json.dumps(dict(jobs=[dict(status='completed',exit_code=0,reservation_released=True)])));(p/'settlement').write_text(json.dumps(dict(uncertain=None,settlement=dict(status='settled'))));o=m.ObservedOwner(i,p/'ledger',p/'settlement',get_identity=lambda pid:None);self.assertEqual(o.poll(),0);self.assertEqual(o.wait(),0)
 def test_missing_guard_without_receipts_has_grace_then_refuses_no_restart(self):
  with tempfile.TemporaryDirectory() as folder:
   p=pathlib.Path(folder);t=[0];o=m.ObservedOwner(dict(pid=1),p/'ledger',p/'settlement',get_identity=lambda pid:None,clock=lambda:t[0]);self.assertIsNone(o.poll());t[0]=31
   with self.assertRaises(RuntimeError):o.poll()
   self.assertFalse(list(p.iterdir()))
 def test_wait_timeout_does_not_imply_live_guard_terminal(self):
  with tempfile.TemporaryDirectory() as folder:
   p=pathlib.Path(folder);t=[0.];i=dict(pid=1);o=m.ObservedOwner(i,p/'ledger',p/'settlement',get_identity=lambda pid:i,clock=lambda:t[0],sleep=lambda seconds:t.__setitem__(0,t[0]+seconds))
   with self.assertRaises(TimeoutError):o.wait(timeout=.5)
   self.assertIsNone(o.poll())
class CapacityChoice(unittest.TestCase):
 def setUp(self):self.values=[m.scope.cell(1,x,1) for x in ('pg','scvi','cfgen')];self.costs=dict(pg=[4.,1.],scvi=[.5,.1],cfgen=[8.,2.])
 def test_only_slow_card_available_does_not_wait_on_unavailable_fast_card(self):
  v,q=m.choose_available(self.values,{},m.GPUS[0],self.costs,{m.GPUS[0]});self.assertIn(v,self.values);self.assertEqual(q['reason'],'only_eligible_card_uses_pending_work')
 def test_ineligible_card_never_admits(self):self.assertIsNone(m.choose_available(self.values,{},m.GPUS[0],self.costs,{m.GPUS[1]})[0])
 def test_one_short_untouched_scvi_bootstrap_keeps_native_endpoint(self):
  v,q=m.choose_available(self.values,{},m.GPUS[0],self.costs,set(m.GPUS),bootstrap_scvi=True);self.assertEqual(v['method'],'scvi');self.assertEqual(m.scope.endpoint(v['method']),400)
 def test_two_eligible_cards_retain_original_cost_only_choice(self):
  a=m.choose_available(self.values,{},m.GPUS[1],self.costs,set(m.GPUS));b=m.choose_next(self.values,{},m.GPUS[1],self.costs);self.assertEqual(a,b)
 def test_choice_error_still_drains_healthy_native_handle(self):
  old=dict(cell=dict(id='old'),gpu=m.GPUS[1],identity={},attempt='old')
  def bad(*args):raise RuntimeError('choice failure')
  q=m.drive([dict(id='new')],[],{m.GPUS[1]:old},key=lambda v:v['id'],start=None,ready=lambda h:True,finish=lambda h:dict(cell=h['cell'],complete=True),select=bad,observe=lambda *args:None,sleep=lambda seconds:None,now=lambda:0,expiry=1000);self.assertEqual(len(q['rows']),1);self.assertEqual(q['errors'][0]['stage'],'choice');self.assertFalse(q['unsettled'])

if __name__=='__main__':unittest.main()
