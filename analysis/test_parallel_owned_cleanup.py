"""CPU fixtures for ALZ cleanup when exact foreign peer exits; never native GPUs."""
import ast,copy,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/ml2/alz');G=2**30
sys.path[:0]=[str(R/'recovery_runtime'),'/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/v2']
import parallel_runner as runner
import gpu0_admission as a
from test_gpu0_candidate import fixture,proc
class OwnedCleanup(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='alz-owned-cleanup-');self.root=Path(self.tmp.name)/'fixture';self.root.mkdir();self.locks=[self.root/'a.lock',self.root/'b.lock']
  for p in self.locks:p.touch()
  self.peers,self.obs=fixture();self.owner={k:self.obs['procs'][10][k] for k in ('pid','birth','boot')}
  self.reg=a.Registry(self.root/'registry',locks=self.locks,roles=['one'],protected_peers=self.peers,driver_headroom=G,host_free_floor=G,policy_sha256='f'*64)
  self.lease=self.reg.reserve('one',self.owner,{'gpu_bytes':G,'ram_bytes':G},self.obs);self.worker={k:proc(11)[k] for k in ('pid','birth','boot')};self.obs['procs'][11]=proc(11,ppid=10);self.lease.attach(self.worker,self.obs)
 def tearDown(self):
  # Fixtures only; simulate verified worker/session exit before release.
  self.obs['procs']={10:proc(10)};self.lease.release(self.obs);self.tmp.cleanup()
 def signals(self,obs,fresh=None):
  calls=[]
  def ident(pid):
   v=obs['procs'].get(pid);return {k:v[k] for k in ('pid','birth','boot')} if v else None
  with patch.object(runner,'identity',side_effect=ident),patch.object(runner,'proc_table',return_value=(fresh or obs)['procs']),patch.object(runner.os,'killpg',side_effect=lambda pid,sig:calls.append((pid,sig))),patch.object(runner.os,'kill',side_effect=lambda pid,sig:calls.append((pid,sig))):runner.signal_owned(self.lease,lambda:obs,signal.SIGTERM)
  return calls
 def test_live_foreign_peer_never_signalled(self):self.assertEqual(self.signals(self.obs),[(11,signal.SIGTERM)])
 def test_foreign_departure_strict_health_refuses_but_owned_cleanup_succeeds(self):
  del self.obs['procs'][1];del self.obs['gpu_memory'][1];del self.obs['gpu_kinds'][1]
  with self.assertRaises(a.Conflict):self.lease.health(self.obs)
  self.assertEqual(self.signals(self.obs),[(11,signal.SIGTERM)])
 def test_foreign_reuse_or_executable_change_cannot_be_adopted_or_signalled(self):
  for field,value in [('birth',999),('exe','/other'),('argv_sha256','b'*64),('uid',0)]:
   obs=copy.deepcopy(self.obs);obs['procs'][1][field]=value
   with self.assertRaises(a.Conflict):self.lease.health(obs)
   self.assertEqual(self.signals(obs),[(11,signal.SIGTERM)])
 def test_unknown_newCUDA_stops_health_not_owned_cleanup(self):
  self.obs['procs'][20]=proc(20);self.obs['gpu_memory'][20]=G;self.obs['gpu_kinds'][20]=['compute']
  with self.assertRaises(a.Conflict):self.lease.health(self.obs)
  self.assertEqual(self.signals(self.obs),[(11,signal.SIGTERM)])
 def test_protected_process_in_owned_group_or_tree_refuses(self):
  for key in ['pgid','ppid']:
   obs=copy.deepcopy(self.obs);obs['procs'][1][key]=11
   with self.assertRaises(a.Conflict):self.signals(obs)
 def test_unrelated_groupmember_refuses_even_if_not_protected(self):
  self.obs['procs'][20]=proc(20,ppid=1,pgid=11)
  with self.assertRaises(a.Conflict):self.signals(self.obs)
 def test_fresh_group_contamination_before_signal_refuses(self):
  fresh=copy.deepcopy(self.obs);fresh['procs'][20]=proc(20,ppid=1,pgid=11)
  with self.assertRaises(a.Conflict):self.signals(self.obs,fresh)
 def test_owned_leader_reuse_never_signalled(self):
  self.obs['procs'][11]['birth']=999;self.assertEqual(self.signals(self.obs),[])
 def test_owned_orphan_known_member_only_individual_target(self):
  self.obs['procs'][12]=proc(12,ppid=11,pgid=11);self.lease.health(self.obs);del self.obs['procs'][11];self.obs['procs'][12]['ppid']=1
  self.assertEqual(runner.cleanup_owned_targets(self.lease,self.obs),[{'pid':12,'identity':{k:self.obs['procs'][12][k] for k in ('pid','birth','boot')}}]);self.assertEqual(self.signals(self.obs),[(12,signal.SIGTERM)])
  del self.obs['procs'][12]
 def test_source_only_declared_runner_delta(self):
  base=Path(a.__file__).with_name('gpu0_runner.py');original=ast.parse(base.read_text());current=ast.parse(Path(runner.__file__).read_text())
  old={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};new={n.name:n for n in current.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  self.assertEqual(set(new)-set(old),{'cleanup_owned_targets'})
  for name in set(old)-{'signal_owned','run'}:self.assertEqual(ast.dump(old[name]),ast.dump(new[name]))
  expected=ast.unparse(old['run']).replace("required = {str(Path(__file__).resolve()), str(Path(__file__).with_name('gpu0_admission.py').resolve())}","admission_path = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/v2/gpu0_admission.py')\n    required = {str(Path(__file__).resolve()), str(admission_path)}").replace("    admission_path = Path(__file__).with_name('gpu0_admission.py').resolve()\n",'').replace('lease.cleanup_targets(o)','cleanup_owned_targets(lease, o)')
  self.assertEqual(ast.dump(ast.parse(expected).body[0]),ast.dump(new['run']))
 def test_native_guard_source_preflight_no_policy_activation_or_lease(self):
  import parallel_profile as profile
  policy_path=self.root/'native-proposal.json';policy_path.write_bytes(profile.deployed_bytes());pin=profile.sha(policy_path)
  class ReachedAfterSourceClosure(Exception):pass
  def stop():raise ReachedAfterSourceClosure()
  with self.assertRaises(ReachedAfterSourceClosure),patch.object(runner.Registry,'reserve',side_effect=AssertionError('must not reserve')):
   runner.run(self.root/'never-ledger.json',a.GPU0,'PRECHECK_ONLY',[],cap_seconds=900,policy_path=policy_path,role='alz_core_pg',absolute_deadline=time.time()+900,expected_policy_sha256=pin,protect=stop)
  self.assertFalse((self.root/'never-ledger.json').exists())
  policy=json.loads(policy_path.read_text());del policy['source_sha256'][str(Path(runner.__file__))];policy_path.write_text(json.dumps(policy))
  with self.assertRaisesRegex(a.Conflict,'Both active GPU0 source files'):
   runner.run(self.root/'never-ledger.json',a.GPU0,'PRECHECK_ONLY',[],cap_seconds=900,policy_path=policy_path,role='alz_core_pg',absolute_deadline=time.time()+900,expected_policy_sha256=profile.sha(policy_path),protect=stop)
 def test_actual_CPU_guard_peer_exit_terminates_owned_worker_and_releases(self):
  # Real CPU children; all files/locks/registry are confined fixtures, CUDA hidden.
  with tempfile.TemporaryDirectory(prefix='alz-parallel-run-') as t:
   root=Path(t)/'fixture';root.mkdir();locks=[root/'a',root/'b']
   for p in locks:p.touch()
   peer=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(1.5)'])
   try:
    ident=a.protected_identity(peer.pid);policy={'status':'PREPARED_FIXTURE_ONLY','gpu':a.GPU0,'registry_path':str(root/'registry'),'locks':list(map(str,locks)),'roles':{'one':{'gpu_bytes':G,'ram_bytes':G}},'driver_headroom_bytes':G,'host_free_floor_bytes':G,'protected_peers':[{'identity':ident,'gpu_kinds':['compute'],'gpu_budget_bytes':G,'ram_budget_bytes':G}],'source_sha256':{str(p):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [Path(runner.__file__),Path(a.__file__)]}}
    policy_path=root/'policy.json';policy_path.write_text(json.dumps(policy));pin=__import__('hashlib').sha256(policy_path.read_bytes()).hexdigest()
    def observe():
     procs=a.proc_table();live=peer.pid in procs
     return {'gpu':a.GPU0,'total':48*G,'used':G//4,'gpu_memory':{peer.pid:G//8} if live else {},'gpu_kinds':{peer.pid:['compute']} if live else {},'procs':procs,'available_ram':32*G,'total_ram':128*G,'complete':True}
    driver=root/'driver.py'
    driver.write_text("""import sys,json
from pathlib import Path
sys.path[:0]=['/ml2/alz/recovery_runtime','/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/v2']
import parallel_runner as runner
import gpu0_admission as a
root=Path(sys.argv[1]);p=root/'policy.json';pol=json.loads(p.read_text());peer=pol['protected_peers'][0]['identity']['pid'];G=2**30
def observe():
 procs=a.proc_table();live=peer in procs
 return {'gpu':a.GPU0,'total':48*G,'used':G//4,'gpu_memory':{peer:G//8} if live else {},'gpu_kinds':{peer:['compute']} if live else {},'procs':procs,'available_ram':32*G,'total_ram':128*G,'complete':True}
try:
 runner.run(root/'ledger.json',a.GPU0,'CPU_ONLY',[sys.executable,'-I','-S','-B','-c','import os,time;assert os.environ["CUDA_VISIBLE_DEVICES"]=="";time.sleep(30)'],cap_seconds=15,policy_path=p,role='one',observe=observe,poll_seconds=.05,test_mode=True,expected_policy_sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest())
except a.Conflict:
 assert (root/'ledger.json').exists()
else:raise AssertionError('Foreign exit should stop owned work')
""")
    result=subprocess.run([sys.executable,'-I','-S','-B',str(driver),str(root)],capture_output=True,text=True,timeout=15)
    self.assertEqual(result.returncode,0,result.stderr)
    job=json.loads((root/'ledger.json').read_text())['jobs'][0];self.assertEqual(job['status'],'failed');self.assertTrue(job['reservation_released']);self.assertIsNone(a.identity(job['child_pid']));self.assertGreater(job['charged_seconds'],0);self.assertEqual(json.loads((root/'registry/registry.json').read_text())['jobs'],{})
   finally:
    if peer.poll() is None:peer.terminate()
    peer.wait(timeout=3)
if __name__=='__main__':unittest.main()
