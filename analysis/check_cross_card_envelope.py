"""Private CPU regressions for symmetric postreserve accounting; never native leases."""
import argparse,ast,copy,datetime,hashlib,json,multiprocessing,os,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/ml2/alz');GPU0='GPU-72c1b506-891d-b8bc-b353-e020585e1c47';GPU1='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
p=argparse.ArgumentParser();p.add_argument('--card',choices=['0','1']);args=p.parse_args()
if args.card is None:
 rows=[]
 for card in ['0','1']:
  result=subprocess.run([sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--card',card],cwd=ROOT,capture_output=True,text=True,timeout=40)
  print(result.stderr,end='');print(result.stdout,end='');assert result.returncode==0
  rows.append(json.loads(result.stdout))
 paths=[Path(__file__),ROOT/'runtime/binding.py',ROOT/'runtime/project_runner_gpu0.py',ROOT/'gpu1_runtime/binding.py',ROOT/'gpu1_runtime/project_runner_gpu0.py']
 q={'status':'PASSED_PRIVATE_CPU_BOTH_CARD_ENVELOPE_REGRESSIONS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'Actual per-card owner/backend boundary/precheck/+1s refusal and forced postreserve race/cancel fixtures; simultaneous shared private immutableController reservations/postchecks/cancellation under barrier, not two complete simultaneous native owners. No canonical mutation or GPU/physical leases.','native_G1_qualified':False,'independent_integrated_review_complete':False}
 (ROOT/'analysis/cross_card_envelope_cpu_verification.json').write_text(json.dumps(q,indent=2)+'\n');raise SystemExit(0)
runtime=ROOT/('runtime' if args.card=='0' else 'gpu1_runtime');sys.path.insert(0,str(runtime))
import binding as b
import launch_core_gpu0 as owner
import test_integration as fixtures

class BothCardAccounting(unittest.TestCase):
 setUp=fixtures.Integration.setUp
 tearDown=fixtures.Integration.tearDown
 def prior(self,remaining):
  self.state['allocation_history']=[{'allocation_id':'ALZ_CAL_V1_private_prior','status':'settled','charged_GPU_seconds':180*3600-remaining}];fixtures.dump(self.layout.project/'project_gpu_budget.json',self.state)
 def no_execution(self):
  q=b.read(self.layout.project/'project_gpu_budget.json');self.assertFalse(q['open_reservations']);self.assertFalse(q['ledger_registry']);self.assertFalse(Path(self.auth['ledger_path']).exists());self.assertIsNone(self.peer.poll())
  self.assertTrue(all(r.get('charged_GPU_seconds',0)==0 for r in q['allocation_history'] if r['allocation_id']!='ALZ_CAL_V1_private_prior'))
 def test_exact_boundary_and_plus_one_refusal_before_reserve_no_charge(self):
  self.prior(44);before=b.sha(self.layout.project/'project_gpu_budget.json');_,_,guard=b.modules(self.auth);calls=[]
  with patch.object(guard,'run',lambda *a,**kw:calls.append(True)):
   with self.assertRaisesRegex(b.Refusal,'180GPU-hour'):owner.run(self.auth_path,self.layout,observe=lambda:{})
  self.assertFalse(calls);self.assertEqual(before,b.sha(self.layout.project/'project_gpu_budget.json'));self.no_execution()
  b.core_envelope(self.state,44)
  with self.assertRaisesRegex(b.Refusal,'180GPU-hour'):b.core_envelope(self.state,45)
 def test_exact_boundary_45_seconds_reserved_admitted_then_mock_guard_no_worker(self):
  self.prior(45);_,_,guard=b.modules(self.auth);calls=[]
  with patch.object(guard,'run',lambda *a,**kw:calls.append(True) or {'status':'CPU_GUARD_MOCK_NO_WORKER'}):owner.run(self.auth_path,self.layout,observe=lambda:{})
  self.assertEqual(calls,[True]);self.no_execution()
 def test_actual_backend_postreserve_competing_card_cancel_owned_only(self):
  self.prior(60);control,admission,guard=b.modules(self.auth);reserve=control.Controller.reserve;other=[];calls=[]
  def interleave(instance,**kw):
   own=reserve(instance,**kw)
   other.append(reserve(instance,allocation_id='ALZ_CAL_V1_private_competing',label='other_card',owner=admission.identity(os.getpid()),machine_id=self.auth['machine_id'],gpu_uuids=[GPU1 if args.card=='0' else GPU0],wall_seconds=15,job_cap_seconds=15,ledger_path='private_competing.json'));return own
  with patch.object(control.Controller,'reserve',interleave),patch.object(guard,'run',lambda *a,**kw:calls.append(True)):
   with self.assertRaisesRegex(b.Refusal,'180GPU-hour'):owner.run(self.auth_path,self.layout,observe=lambda:{})
  q=b.read(self.layout.project/'project_gpu_budget.json');self.assertFalse(calls);self.assertEqual([r['allocation_id'] for r in q['open_reservations']],['ALZ_CAL_V1_private_competing']);self.assertIsNone(self.peer.poll())
  own=[r for r in q['allocation_history'] if r['allocation_id']==self.auth['allocation_id']];self.assertEqual(len(own),1);self.assertEqual(own[0]['status'],'cancelled_before_execution');self.assertEqual(own[0]['charged_GPU_seconds'],0)
  control.Controller(self.layout.project).cancel_unstarted(other[0],resource_evidence={'no_bound_resource_lease':True,'no_worker_started':True,'basis':'Private CPU fixture only'});self.no_execution()
 def test_parallel_immutable_controller_postchecks_both_cancel_zero_charge(self):
  self.prior(60);control,admission,_=b.modules(self.auth);ctx=multiprocessing.get_context('fork');barrier=ctx.Barrier(2);queue=ctx.Queue()
  def contender(card):
   try:
    controller=control.Controller(self.layout.project)
    with controller.transaction() as state:b.core_envelope(state,45)
    barrier.wait(timeout=8)
    res=controller.reserve(allocation_id='ALZ_CAL_V1_parallel_'+str(card),label='parallel',owner=admission.identity(os.getpid()),machine_id=self.auth['machine_id'],gpu_uuids=[[GPU0,GPU1][card]],wall_seconds=15,job_cap_seconds=15,ledger_path='parallel_'+str(card)+'.json')
    barrier.wait(timeout=8);refused=False
    try:
     with controller.transaction() as state:b.core_envelope(state,0)
    except b.Refusal:refused=True
    barrier.wait(timeout=8)
    controller.cancel_unstarted(res,resource_evidence={'no_bound_resource_lease':True,'no_worker_started':True,'basis':'Parallel privateCPU reservation; no guard/worker'})
    queue.put({'card':card,'refused':refused})
   except BaseException as exc:queue.put({'card':card,'error':repr(exc)})
  children=[ctx.Process(target=contender,args=(i,)) for i in [0,1]]
  for child in children:child.start()
  results=[queue.get(timeout=15) for _ in children]
  for child in children:child.join(timeout=3);self.assertEqual(child.exitcode,0)
  self.assertEqual(sorted(results,key=lambda q:q['card']),[{'card':0,'refused':True},{'card':1,'refused':True}]);self.no_execution()

result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BothCardAccounting))
if not result.wasSuccessful():raise SystemExit(1)
print(json.dumps({'card':args.card,'tests_passed':result.testsRun,'actual_backend_postreserve_cancellation_before_guard_verified':True,'boundary_admission_and_plus_one_refusal_verified':True,'parallel_Controller_postchecks_both_refused_cancelled_zero_charge':True}))
