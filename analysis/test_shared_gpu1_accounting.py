"""Foreign turnover must not terminate ALZ unless observable capacity fails."""
import copy,hashlib,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/ml2/alz');sys.path[:0]=[str(R/'recovery_runtime'),'/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/v2']
import shared_gpu1_admission as a
import shared_gpu1_runner as runner
from test_gpu0_candidate import fixture,proc
G=2**30

class Sharing(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'fixture';self.root.mkdir()
  self.peers,self.obs=fixture();self.obs['gpu']=a.GPU0
  self.locks=[self.root/'a',self.root/'b']
  for p in self.locks:p.touch()
  self.owner={k:self.obs['procs'][10][k] for k in ('pid','birth','boot')}
  self.reg=a.Registry(self.root/'registry',locks=self.locks,roles=['one'],protected_peers=self.peers,driver_headroom=3*G,host_free_floor=12*G,policy_sha256='a'*64)
  with patch.object(a.os,'getpid',return_value=10):self.lease=self.reg.reserve('one',self.owner,{'gpu_bytes':3*G,'ram_bytes':6*G},self.obs)
  self.obs['procs'][11]=proc(11,ppid=10);self.worker={k:self.obs['procs'][11][k] for k in ('pid','birth','boot')}
  self.lease.attach(self.worker,self.obs)
 def tearDown(self):
  self.obs['procs']={10:proc(10)};self.lease.release(self.obs);self.tmp.cleanup()
 def add_peer(self,pid=20,used=G):
  self.obs['procs'][pid]=proc(pid);self.obs['gpu_memory'][pid]=used;self.obs['gpu_kinds'][pid]=['compute'];self.obs['used']=sum(self.obs['gpu_memory'].values())
 def test_new_foreign_is_observed_budgeted_and_never_owned(self):
  self.add_peer();health=self.lease.health(self.obs)
  self.assertEqual(health['own_gpu_bytes'],0)
  row=next(p for p in self.reg.observed_foreign if p['identity']['pid']==20)
  self.assertEqual(row['gpu_budget_bytes'],G+G//2);self.assertEqual(row['foreign_control'],'NONE')
  self.assertEqual(health['members'],[11]);self.assertEqual(runner.cleanup_owned_targets(self.lease,self.obs),[{'pgid':11,'leader':self.worker}])
 def test_departure_and_reused_foreign_identity_continue(self):
  for field,value in [('birth',10001),('exe','/changed'),('argv_sha256','b'*64)]:
   self.obs['procs'][1][field]=value;self.lease.health(self.obs)
  del self.obs['procs'][1];del self.obs['gpu_memory'][1];del self.obs['gpu_kinds'][1]
  self.lease.health(self.obs);self.assertNotIn(1,{p['identity']['pid'] for p in self.reg.observed_foreign})
 def test_unobservable_foreign_is_not_adopted(self):
  self.add_peer();del self.obs['procs'][20]
  with self.assertRaises(a.Uncertain):self.lease.health(self.obs)
 def test_other_uid_refuses_without_foreign_control(self):
  self.add_peer();self.obs['procs'][20]['uid']=0
  with self.assertRaises(a.Conflict):self.lease.health(self.obs)
 def test_foreign_growth_capacity_stop(self):
  self.add_peer(used=43*G)
  with self.assertRaises(a.Capacity):self.lease.health(self.obs)
 def test_own_gpu_usage_remains_enforced(self):
  self.obs['gpu_memory'][11]=3*G+1;self.obs['gpu_kinds'][11]=['compute'];self.obs['used']=sum(self.obs['gpu_memory'].values())
  self.assertTrue(self.lease.health(self.obs)['own_gpu_budget_exceeded'])
  self.assertNotIn(11,{p['identity']['pid'] for p in self.reg.observed_foreign})
 def test_known_owned_orphan_never_reclassified_foreign(self):
  self.obs['procs'][12]=proc(12,ppid=11,pgid=11);self.obs['gpu_memory'][12]=G;self.obs['gpu_kinds'][12]=['compute'];self.lease.health(self.obs)
  del self.obs['procs'][11];self.obs['procs'][12]['ppid']=1
  h=self.lease.health(self.obs);self.assertEqual(h['members'],[12]);self.assertEqual(h['own_gpu_bytes'],G)
  self.assertNotIn(12,{p['identity']['pid'] for p in self.reg.observed_foreign})
 def test_cleanup_rejects_unowned_group_contamination(self):
  self.add_peer();self.lease.health(self.obs);self.obs['procs'][20]['pgid']=11
  with self.assertRaises(a.Conflict):runner.cleanup_owned_targets(self.lease,self.obs)
 def test_source_origin_and_no_launch_preflight(self):
  import parallel_profile as p
  policy=self.root/'policy.json';policy.write_bytes(p.deployed_bytes())
  class Stop(Exception):pass
  with patch.object(runner.Registry,'reserve',side_effect=AssertionError('no allocation')):
   with self.assertRaises(Stop):runner.run(self.root/'never.json',a.GPU0,'PRECHECK',[],cap_seconds=900,policy_path=policy,role='alz_core_pg',absolute_deadline=time.time()+900,expected_policy_sha256=p.sha(policy),protect=lambda:(_ for _ in ()).throw(Stop()))
 def test_real_CPU_guard_continues_across_peer_turnover_and_releases(self):
  # Real child/guardian lifecycles; synthetic telemetry and temporary locks only.
  import parallel_profile as profile
  policy=json.loads(profile.deployed_bytes());policy.update(status='PREPARED_FIXTURE_ONLY',registry_path=str(self.root/'real-registry'),locks=list(map(str,self.locks)))
  peer=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(0.8)'])
  successor=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(8)'])
  try:
   policy['protected_peers']=[{'identity':a.protected_identity(peer.pid),'gpu_kinds':['compute'],'gpu_budget_bytes':G,'ram_budget_bytes':G}]
   path=self.root/'policy.json';path.write_text(json.dumps(policy));driver=self.root/'driver.py'
   driver.write_text('''import sys,time,json,hashlib
from pathlib import Path
sys.path.insert(0,'/ml2/alz/recovery_runtime')
import shared_gpu1_admission as a
import shared_gpu1_runner as r
root=Path(sys.argv[1]);peer=int(sys.argv[2]);successor=int(sys.argv[3]);started=time.monotonic();G=2**30
def observe():
 procs=a.proc_table();pids=([peer] if peer in procs else [])+([successor] if time.monotonic()-started>.4 else [])
 return {'gpu':a.GPU0,'total':48*G,'used':G,'gpu_memory':{p:G//8 for p in pids},'gpu_kinds':{p:['compute'] for p in pids},'procs':procs,'available_ram':64*G,'total_ram':128*G,'complete':True}
p=root/'policy.json'
assert r.run(root/'real-ledger.json',a.GPU0,'CPU_ONLY',[sys.executable,'-I','-S','-c','import os,time;assert os.environ["CUDA_VISIBLE_DEVICES"]=="";time.sleep(2)'],cap_seconds=15,policy_path=p,role='alz_core_pg',observe=observe,poll_seconds=.05,test_mode=True,expected_policy_sha256=hashlib.sha256(p.read_bytes()).hexdigest())==0
''')
   q=subprocess.run([sys.executable,'-I','-S','-B',str(driver),str(self.root),str(peer.pid),str(successor.pid)],capture_output=True,text=True,timeout=12)
   self.assertEqual(q.returncode,0,q.stderr)
   job=json.loads((self.root/'real-ledger.json').read_text())['jobs'][0]
   self.assertEqual(job['status'],'completed');self.assertTrue(job['reservation_released']);self.assertIsNone(successor.poll())
  finally:
   for child in (peer,successor):
    if child.poll() is None:child.terminate()
    child.wait()

if __name__=='__main__':unittest.main()
