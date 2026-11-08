import copy,json,os,subprocess,sys,unittest
from pathlib import Path
import binding as b,dual_coordination as d
import test_integration as fixtures
def run_full_fixture(auth_path,root):
 layout=b.Layout.fixture(root);auth=b.read(auth_path);peer=auth['protected_peers'][0]['identity']['pid']
 def observe():
  return dict(gpu=auth['gpu_uuid'],total=48*2**30,used=2**28,gpu_memory={peer:2**27},gpu_kinds={peer:['compute']},procs=fixtures.a.proc_table(),available_ram=32*2**30,total_ram=128*2**30,complete=True)
 return fixtures.owner.run(auth_path,layout,observe=observe)

class Pair(unittest.TestCase):
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
   code='import sys;sys.path.insert(0,sys.argv[1]);import test_pair;raise SystemExit(test_pair.run_full_fixture(sys.argv[2],sys.argv[3]))'
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
