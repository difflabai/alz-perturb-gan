"""Confined accounting, admission and real POSIX lock tests; no CUDA jobs."""
import ast,copy,fcntl,hashlib,json,os,socket,subprocess,sys,tempfile,unittest
from pathlib import Path
import binding as b
import dual_coordination as d
import resource_profile as p
import test_integration as fixtures

def run_full_fixture(auth_path,root):
 layout=b.Layout.fixture(root);auth=b.read(auth_path);peer=auth['protected_peers'][0]['identity']['pid']
 def observe():
  return dict(gpu=auth['gpu_uuid'],total=48*2**30,used=2**28,gpu_memory={peer:2**27},gpu_kinds={peer:['compute']},procs=fixtures.a.proc_table(),available_ram=32*2**30,total_ram=128*2**30,complete=True)
 return fixtures.owner.run(auth_path,layout,observe=observe)

class Dual(unittest.TestCase):
 def test_scientific_worker_and_native_sampling_wrappers_are_unchanged(self):
  original=b.HERE.parent/'benchmark097_runtime'
  def scientific(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scientific')
  self.assertEqual(ast.dump(scientific(original/'worker_gpu0.py')),ast.dump(scientific(b.HERE/'worker_gpu0.py')))
  for name in ('retained_pcs.py','native_scvi_reload.py','state_fingerprint.py','test_factory.py'):
   self.assertEqual((original/name).read_bytes(),(b.HERE/name).read_bytes(),name)
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
 def test_two_actual_confined_guard_worker_chains_overlap_and_settle_both_cells(self):
  f=fixtures.Integration();f.setUp();children=[]
  try:
   records={};paths=[]
   for lane,method in enumerate(('pg','scvi')):
    a=copy.deepcopy(f.auth);value=dict(fold=0,method=method,seed=0);a.update(cell=value,allocation_id=b.allocation_key(value),attempt_id=b.attempt_key(value),label=b.attempt_key(value),gpu_uuid=d.GPUS[lane],role='alz_core_'+method,fixed_endpoint=20000 if method=='pg' else 400,parallel_profile=True,dynamic_shared_foreign=True,dynamic_shared_gpu0=lane==0,fixture_dual_backend_map=True,fixture_dual_barrier=True)
    for key,fn in [('cache_dir',b.cache_path),('output_dir',b.output_path),('ledger_path',b.ledger_path)]:a[key]=str(fn(f.layout,value))
    policy=copy.deepcopy(b.read(f.policy));policy.update(gpu=d.GPUS[lane],registry_path=str(f.root/f'registry{lane}'),locks=[str(f.root/f'cpu-gpu{lane}-admission.lock'),str(f.root/f'cpu-gpu{lane}-worker.lock')],roles={'alz_core_'+method:dict(gpu_bytes=2**30,ram_bytes=2**30)})
    for folder in (b.HERE.parent/'recovery_runtime',b.HERE.parent/'analysis/shared_gpu0_candidate'):
     for path in folder.glob('*.py'):policy['source_sha256'][str(path)]=b.sha(path)
    policy_path=f.root/f'policy{lane}.json';fixtures.dump(policy_path,policy)
    for lock in policy['locks']:Path(lock).touch()
    a.update(policy_path=str(policy_path),policy_SHA256=b.sha(policy_path),registry_path=policy['registry_path'])
    for folder in (b.HERE.parent/'recovery_runtime',b.HERE.parent/'analysis/shared_gpu0_candidate'):
     for path in folder.glob('*.py'):a['files_SHA256'][str(path)]=b.sha(path)
    a['files_SHA256'][str(policy_path)]=b.sha(policy_path)
    auth_path=f.root/f'auth{lane}.json';a['command']=b.expected_command(f.layout,auth_path,value);fixtures.dump(auth_path,a);paths.append(auth_path)
    records[a['allocation_id']]=dict(status='REVIEWED_FOR_BOUNDED_EXECUTION',path=str(b.HERE/'project_runner_gpu0.py'),source_SHA256={str(path):a['files_SHA256'][str(path)] for path in [*b.HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']},files_SHA256=a['files_SHA256'],policy_SHA256=a['policy_SHA256'],**{k:a[k] for k in ('allocation_id','attempt_id','scheduler_allocation_id','expires_unix')})
   state=b.read(f.layout.project/'project_gpu_budget.json');state['execution_controller']['calibration_parallel_backends']=records;fixtures.dump(f.layout.project/'project_gpu_budget.json',state)
   code='import sys;sys.path.insert(0,sys.argv[1]);import test_dual;raise SystemExit(test_dual.run_full_fixture(sys.argv[2],sys.argv[3]))'
   for path in paths:children.append(subprocess.Popen([sys.executable,'-I','-S','-B','-c',code,str(b.HERE),str(path),str(f.root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
   for child in children:
    out,err=child.communicate(timeout=15);self.assertEqual(child.returncode,0,err)
   state=b.read(f.layout.project/'project_gpu_budget.json');self.assertFalse(state['open_reservations']);self.assertEqual(len(state['allocation_history']),2)
   self.assertEqual(len(list((f.root/'dual_barrier').glob('*.ready'))),2)
   for path in paths:
    a=b.read(path);ledger=b.read(a['ledger_path']);self.assertEqual(ledger['jobs'][0]['exit_code'],0);self.assertTrue(ledger['jobs'][0]['reservation_released']);self.assertGreater(ledger['aggregate_charged_seconds'],0)
    self.assertEqual(ledger['gpu_uuid'],a['gpu_uuid']);self.assertFalse(b.read(Path(a['registry_path'])/'registry.json')['jobs'])
    self.assertFalse(b.read(Path(a['output_dir'])/'CPU_FIXTURE_COMPLETE.json')['model_imported'])
  finally:
   for child in children:
    if child.poll() is None:child.terminate();child.wait(timeout=5)
    if child.stdout is not None:child.stdout.close()
    if child.stderr is not None:child.stderr.close()
   f.tearDown()
if __name__=='__main__':unittest.main(verbosity=2)
