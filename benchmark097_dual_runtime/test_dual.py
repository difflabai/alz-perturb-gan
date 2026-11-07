"""Confined accounting, admission and real POSIX lock tests; no CUDA jobs."""
import copy,fcntl,hashlib,json,os,socket,tempfile,unittest
from pathlib import Path
import binding as b
import dual_coordination as d
import resource_profile as p
import test_integration as fixtures

class Dual(unittest.TestCase):
 def auth(self):
  return dict(allocation_id='ALZ_CAL_V1_external097_f0_pg_s0',gpu_uuid=d.GPUS[1],cap_seconds=21600,project_root='/tmp/project',benchmark_protocol_SHA256='frozen-scientific-protocol',files_SHA256={str(Path(d.__file__)):hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest()})
 def peer(self):
  a=self.auth();cell=dict(experiment='external097',fold=0,method='scvi',seed=0);gpu=d.assigned_gpu(cell);key='ALZ_CAL_V1_external097_f0_scvi_s0'
  r=dict(allocation_id=key,status='reserved',owner=d.identity(os.getpid()),gpu_uuids=[gpu],device_count=1,declared_wall_seconds=21600,job_cap_seconds=21600,ledger_path='peer.json')
  meta=dict(gpu_uuid=gpu,cell=cell,benchmark_protocol_SHA256=a['benchmark_protocol_SHA256'],resource_protocol_SHA256=hashlib.sha256(d.RESOURCE_PROTOCOL.read_bytes()).hexdigest(),path=str(b.HERE/'project_runner_gpu0.py'),ledger_path='/tmp/project/peer.json',source_SHA256=a['files_SHA256'])
  state=dict(open_reservations=[r],execution_controller=dict(calibration_parallel_backends={key:meta}))
  return a,state,meta
 def test_fixed45_assignment_is_balanced_without_missing_cells(self):
  q=d.validate_resource_protocol();self.assertEqual(len(q['cells']),45)
  self.assertEqual(sorted(sum(c['gpu_uuid']==g for c in q['cells']) for g in d.GPUS),[22,23])
  self.assertEqual([sum(c['method']==m for c in q['cells']) for m in ('pg','scvi','cfgen')],[15,15,15])
 def test_distinct_live_exact_peer_can_remain_during_admission(self):
  a,s,_=self.peer();self.assertTrue(d.validate_active_peers(s,a))
 def test_same_device_or_same_cell_cannot_be_admitted_twice(self):
  a,s,_=self.peer()
  for change in [dict(gpu_uuids=[a['gpu_uuid']]),dict(allocation_id=a['allocation_id'])]:
   t=copy.deepcopy(s);t['open_reservations'][0].update(change)
   with self.assertRaises(ValueError):d.validate_active_peers(t,a)
 def test_unscoped_legacy_or_third_reservation_cannot_bypass_phase(self):
  a,s,_=self.peer();t=copy.deepcopy(s);t['execution_controller']={}
  with self.assertRaises(ValueError):d.validate_active_peers(t,a)
  t=copy.deepcopy(s);t['open_reservations']*=2
  with self.assertRaises(ValueError):d.validate_active_peers(t,a)
 def test_peer_protocol_source_ledger_and_device_scope_cannot_drift(self):
  a,s,meta=self.peer();key=s['open_reservations'][0]['allocation_id']
  for field,value in [('benchmark_protocol_SHA256','changed'),('resource_protocol_SHA256','changed'),('ledger_path','/tmp/project/other.json'),('gpu_uuid',d.GPUS[1]),('source_SHA256',{})]:
   t=copy.deepcopy(s);t['execution_controller']['calibration_parallel_backends'][key][field]=value
   with self.assertRaises(ValueError):d.validate_active_peers(t,a)
 def test_uncertain_or_absent_owner_and_project_holds_refuse(self):
  a,s,_=self.peer()
  for change in [dict(status='uncertain'),dict(owner=dict(pid=99999999,birth=1,boot='absent'))]:
   t=copy.deepcopy(s);t['open_reservations'][0].update(change)
   with self.assertRaises(ValueError):d.validate_active_peers(t,a)
  t=copy.deepcopy(s);t['holds']=['actual-hold']
  with self.assertRaises(ValueError):d.validate_active_peers(t,a)
 def test_two_device_locks_coexist_and_exclude_legacy_exclusive_owner(self):
  with tempfile.TemporaryDirectory() as root:
   a=dict(gpu_uuid=d.GPUS[0],allocation_id='cell0');z=dict(gpu_uuid=d.GPUS[1],allocation_id='cell1')
   with d.execution_locks(root,a),d.execution_locks(root,z):
    with (Path(root)/'core_phase.lock').open('a+') as f:
     with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def test_duplicate_device_or_cell_lock_is_exclusive(self):
  with tempfile.TemporaryDirectory() as root:
   a=dict(gpu_uuid=d.GPUS[0],allocation_id='cell0')
   for z in [dict(gpu_uuid=d.GPUS[0],allocation_id='cell1'),dict(gpu_uuid=d.GPUS[1],allocation_id='cell0')]:
    with d.execution_locks(root,a):
     with self.assertRaises(BlockingIOError):
      with d.execution_locks(root,z):pass
 def test_unchanged_controller_two_live_reservations_are_accounted_and_cancel_independently(self):
  f=fixtures.Integration();f.setUp()
  try:
   controller,admission,_=b.modules(f.auth);c=controller.Controller(f.layout.project)
   rs=[]
   for index,pid in enumerate((os.getpid(),f.peer.pid)):
    rs.append(c.reserve(allocation_id=f'CPU_DUAL_{index}',label=f'cpu{index}',owner=admission.identity(pid),machine_id=socket.gethostname(),gpu_uuids=[d.GPUS[index]],wall_seconds=11,job_cap_seconds=11,ledger_path=f'lane{index}.json'))
   state=b.read(f.layout.project/'project_gpu_budget.json');self.assertEqual(len(state['open_reservations']),2);self.assertEqual(sum(r['remaining_wall_seconds']*r['device_count'] for r in state['open_reservations']),82)
   evidence=dict(no_bound_resource_lease=True,no_worker_started=True,basis='Confined CPU-only fixture; no worker or GPU registry created')
   c.cancel_unstarted(rs[0],resource_evidence=evidence);self.assertEqual([r['allocation_id'] for r in b.read(f.layout.project/'project_gpu_budget.json')['open_reservations']],['CPU_DUAL_1'])
   c.cancel_unstarted(rs[1],resource_evidence=evidence);self.assertFalse(b.read(f.layout.project/'project_gpu_budget.json')['open_reservations'])
  finally:f.tearDown()
 def test_both_physical_profiles_and_peer_host_reserve_enforce_capacity(self):
  class Registry:
   def __init__(self,*args,**kwargs):pass
   def accounting(self,*args):pass
   def protected(self,*args):return ([],0,0)
  class Admission:pass
  Admission.Registry=Registry
  for gpu,policy in p.validate_proposal().items():
   self.assertEqual(policy['gpu'],gpu);self.assertTrue(all(f'gpu-{d.GPUS.index(gpu)}.lock' in s for s in policy['locks']))
   obs=dict(used=0,gpu_memory={},total=48*2**30,available_ram=32*2**30)
   self.assertEqual(p.capacity(obs,Admission,gpu)['host_remaining_growth_required_bytes'],24*2**30)
   with self.assertRaisesRegex(AssertionError,'host growth'):p.capacity({**obs,'available_ram':23*2**30},Admission,gpu)
   with self.assertRaisesRegex(AssertionError,'GPU growth'):p.capacity({**obs,'total':5*2**30},Admission,gpu)
if __name__=='__main__':unittest.main(verbosity=2)
