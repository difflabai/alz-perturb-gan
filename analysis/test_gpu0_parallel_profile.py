"""CPU-only capacity/identity/refusal tests for exact ALZ parallel proposal."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/ml2/alz');sys.path[:0]=[str(R/'recovery_runtime'),str(R/'runtime'),'/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/v2']
import parallel_profile as p
import gpu0_admission as a
class Parallel(unittest.TestCase):
 def setUp(self):
  s=p.read(R/'analysis/learned_v5_profile_observation.json');peers=[v for v in s['peers'] if v['GPU_kinds']]
  self.obs={'gpu':s['gpu'],'total':s['total'],'used':s['used'],'available_ram':s['available_ram'],'total_ram':128*2**30,'complete':True,'procs':{v['pid']:{k:v[k] for k in a.IDENTITY_KEYS+('rss','hwm')} for v in peers},'gpu_memory':{v['pid']:v['GPU_bytes'] for v in peers},'gpu_kinds':{v['pid']:v['GPU_kinds'] for v in peers}}
  self.peer_pid=p.read(p.PREPARED)['protected_peers'][-1]['identity']['pid']
  for pid,v in self.obs['procs'].items():v.update(sid=pid,pgid=pid)
 def test_snapshot_capacity_fits_with_remaining_foreign_growth(self):
  q=p.capacity(self.obs,a);self.assertTrue(q['GPU_fits'] and q['host_fits']);self.assertFalse(q['allocation_or_lease_created']);self.assertGreater(q['GPU_required_bytes'],self.obs['used']+4*2**30)
 def test_unknown_cuda_worker_not_automatically_adopted(self):
  self.obs['gpu_memory'][999]=1
  with self.assertRaises(a.Conflict):p.capacity(self.obs,a)
 def test_missing_reused_changed_identity_or_kinds_refuses(self):
  for key,value in [('birth',0),('exe','/tmp/other'),('argv_sha256','0'*64),('uid',0)]:
   obs=copy.deepcopy(self.obs);obs['procs'][self.peer_pid][key]=value
   with self.assertRaises(a.Conflict):p.capacity(obs,a)
  self.obs['gpu_kinds'][self.peer_pid]=['compute'] if self.obs['gpu_kinds'][self.peer_pid]==['graphics'] else ['graphics']
  with self.assertRaises(a.Conflict):p.capacity(self.obs,a)
 def test_host_remaining_growth_and_hwm_protected(self):
  self.obs['available_ram']=1
  with self.assertRaises(AssertionError):p.capacity(self.obs,a)
 def test_device_growth_refuses_instead_of_reducing_foreign_usage(self):
  self.obs['gpu_memory'][self.peer_pid]=45*2**30;self.obs['used']=47*2**30
  with self.assertRaises(AssertionError):p.capacity(self.obs,a)
  self.assertEqual(self.obs['gpu_memory'][self.peer_pid],45*2**30)
 def test_original_guard_reserve_and_health_stop_own_only_cpu_fixture(self):
  with tempfile.TemporaryDirectory(prefix='alz-parallel-fixture-') as t:
   root=Path(t);locks=[root/'a.lock',root/'b.lock']
   for f in locks:f.touch()
   pol=p.validate_proposal();reg=a.Registry(root/'registry',locks=locks,roles=pol['roles'],driver_headroom=pol['driver_headroom_bytes'],host_free_floor=pol['host_free_floor_bytes'],protected_peers=pol['protected_peers'],policy_sha256=p.policy_sha())
   obs=copy.deepcopy(self.obs);owner={'pid':900,'birth':1,'boot':'fixture'};worker={'pid':901,'birth':2,'boot':'fixture'}
   obs['procs'][900]={**owner,'ppid':1,'sid':900,'pgid':900,'rss':1,'hwm':1}
   obs['procs'][901]={**worker,'ppid':900,'sid':901,'pgid':901,'rss':1,'hwm':1}
   lease=reg.reserve('alz_core_pg',owner,pol['roles']['alz_core_pg'],obs);lease.attach(worker,obs)
   self.assertEqual(lease.cleanup_targets(obs),[{'pgid':901,'leader':worker}])
   obs['gpu_memory'][999]=2**30
   with self.assertRaises(a.Conflict):lease.health(obs)
   self.assertEqual(lease.cleanup_targets(obs),[{'pgid':901,'leader':worker}])
   del obs['procs'][901];lease.release(obs)
   self.assertEqual(p.read(root/'registry/registry.json')['jobs'],{})
 def test_absent_independent_review_refuses(self):
  with tempfile.TemporaryDirectory() as t,patch.object(p,'REVIEW',Path(t)/'absent'):
   with self.assertRaises(FileNotFoundError):p.reviewed()
 def test_wrong_source_review_and_policy_cannot_activate(self):
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'review.json';q={'status':'INDEPENDENT_SOURCE_REVIEW_PASS_FOR_EXACT_ALZ_PARALLEL_PROFILE','authenticated_sender':'bits255.dots.ada_20261002','message_id':'COUNTERFACTUAL_CPU_WIRE_ONLY','policy_SHA256':p.policy_sha(),'source_SHA256':p.review_sources()};f.write_text(json.dumps(q))
   with patch.object(p,'REVIEW',f):
    self.assertEqual(p.reviewed(),q)
    q['source_SHA256'][str(p.HERE/'prepare.py')]='0'*64;f.write_text(json.dumps(q))
    with self.assertRaises(AssertionError):p.reviewed()
 def test_early_actual_GPU_floor_before_MOSS_stop(self):
  gpu=p.read(p.PREPARED)['gpu']
  with patch.object(p.subprocess,'check_output',return_value=gpu+', 3072'):
   free,host=p.free_floor();self.assertEqual(free,3*2**30)
  with patch.object(p.subprocess,'check_output',return_value=gpu+', 3071'):
   with self.assertRaises(AssertionError):p.free_floor()
 def test_early_host_floor_and_wrong_device_refuse(self):
  gpu=p.read(p.PREPARED)['gpu']
  original=Path.read_text
  def fake(path,*args,**kwargs):
   return 'MemAvailable: 12582911 kB\n' if str(path)=='/proc/meminfo' else original(path,*args,**kwargs)
  with patch.object(p.subprocess,'check_output',return_value=gpu+', 30000'),patch.object(Path,'read_text',fake):
   with self.assertRaises(AssertionError):p.free_floor()
  with patch.object(p.subprocess,'check_output',return_value='GPU-OTHER, 40000'):
   with self.assertRaises(AssertionError):p.free_floor()
 def test_complete_native_binding_branch_private_review_file_wire(self):
  # Exercise full native source/path checks without creating policy, lease or job.
  sys.path.insert(0,str(R/'analysis'))
  import binding as b
  import test_scope
  import recovery_scope as scope
  with tempfile.TemporaryDirectory(prefix='alz-parallel-native-wire-') as temp:
   root=Path(temp);review=root/'review.json';authpath=root/'auth.json'
   review.write_text(json.dumps({'status':'INDEPENDENT_SOURCE_REVIEW_PASS_FOR_EXACT_ALZ_PARALLEL_PROFILE','authenticated_sender':'bits255.dots.ada_20261002','message_id':'COUNTERFACTUAL_REVIEW_WIRE_NOT_AN_ACTUAL_MESSAGE','policy_SHA256':p.policy_sha(),'source_SHA256':p.review_sources()}))
   auth=test_scope.RecoveryScope().auth('qualify');auth['command']=b.expected_command(b.NATIVE_LAYOUT,authpath,auth['cell']);auth['output_dir']=str(b.output_path(b.NATIVE_LAYOUT,auth['cell']));auth['ledger_path']=str(b.ledger_path(b.NATIVE_LAYOUT,auth['cell']))
   proof=p.read(scope.CPU_PROOF);auth['recovery_CPU_proof_SHA256']=scope.sha(scope.CPU_PROOF)
   auth['files_SHA256'][str(scope.CPU_PROOF)]=scope.sha(scope.CPU_PROOF);auth['files_SHA256'].update(proof['source_SHA256'])
   bsha=b.sha;psha=p.sha;bread=b.read
   with patch.object(p,'REVIEW',review):
    p.configure(auth);authpath.write_text(json.dumps(auth))
    with patch.object(b,'read',side_effect=lambda path:json.loads(p.deployed_bytes()) if Path(path)==p.DEPLOYED else bread(path)),patch.object(p,'sha',side_effect=lambda path:p.policy_sha() if Path(path)==p.DEPLOYED else psha(path)),patch.object(b,'sha',side_effect=lambda path:p.policy_sha() if Path(path)==p.DEPLOYED else bsha(path)):
     self.assertEqual(b.verify(authpath)[0]['policy_path'],str(p.DEPLOYED))
     auth['operation']='train_variant';authpath.write_text(json.dumps(auth))
     with self.assertRaises(b.Refusal):b.verify(authpath)
   self.assertFalse(p.DEPLOYED.exists());self.assertFalse((p.REGISTRY/'registry.json').exists())
 def test_parallel_profile_only_exact_recovery_ops(self):
  with self.assertRaises(AssertionError):p.validate({'parallel_profile':True,'operation':'train_variant'})
if __name__=='__main__':unittest.main()
