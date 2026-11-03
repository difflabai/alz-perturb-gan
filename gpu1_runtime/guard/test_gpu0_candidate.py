"""Fixture-only admission and owned CPU lifecycle tests; no GPU or host leases."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import gpu0_admission as a
import gpu0_runner as runner

G=a.GiB

def proc(pid,ppid=0,pgid=None):
    return {'pid':pid,'birth':pid*10,'boot':'synthetic-boot','uid':1000,
            'exe':'/fixture/program','argv_sha256':'a'*64,'ppid':ppid,
            'pgid':pgid or pid,'sid':pgid or pid,'rss':G//4,'hwm':G//2}

def fixture():
    procs={1:proc(1),2:proc(2),10:proc(10)}
    peers=[{'identity':{k:procs[pid][k] for k in a.IDENTITY_KEYS},
            'gpu_kinds':[kind],'gpu_budget_bytes':4*G,'ram_budget_bytes':2*G}
           for pid,kind in [(1,'compute'),(2,'graphics')]]
    obs={'gpu':a.GPU1,'total':48*G,'used':G,'gpu_memory':{1:G//2,2:G//4},
         'gpu_kinds':{1:['compute'],2:['graphics']},'procs':procs,
         'available_ram':32*G,'total_ram':128*G,'complete':True,
         'utilization_percent_informational':'75'}
    return peers,obs

def run_fixture_driver(root):
    """Separate CPU guardian is a sibling of the protected CPU fixture peer."""
    root=Path(root);policy_path=root/'policy.json'
    policy=json.loads(policy_path.read_text());peer=policy['protected_peers'][0]['identity']['pid']
    def observe():
        return {'gpu':a.GPU1,'total':48*G,'used':G//4,
                'gpu_memory':{peer:G//8},'gpu_kinds':{peer:['compute']},
                'procs':a.proc_table(),'available_ram':32*G,'total_ram':128*G,
                'complete':True,'utilization_percent_informational':'31'}
    kwargs=dict(cap_seconds=15,policy_path=policy_path,role='fixture',observe=observe,
                poll_seconds=.05,expected_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),test_mode=True)
    return runner.run(root/'ledger.json',a.GPU1,'fixture-failure',[sys.executable,'-c','import os;assert os.environ["CUDA_VISIBLE_DEVICES"]=="";raise SystemExit(7)'],**kwargs)

class Admission(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root=Path(self.tmp.name)/'fixture';self.root.mkdir()
        self.paths=[self.root/'gpu0-admission.lock',self.root/'gpu0-worker.lock']
        for p in self.paths:p.touch()
        self.peers,self.obs=fixture()
        self.registry=a.Registry(self.root/'registry',locks=self.paths,gpu=a.GPU1,
                                 roles={'one','two'},protected_peers=self.peers,
                                 driver_headroom=G,host_free_floor=G,policy_sha256='f'*64)
        self.owner={k:self.obs['procs'][10][k] for k in ('pid','birth','boot')}
    def tearDown(self):self.tmp.cleanup()
    def reserve(self,obs=None,request=None):
        return self.registry.reserve('one',self.owner,request or {'gpu_bytes':4*G,'ram_bytes':4*G},obs or self.obs)
    def test_nonzero_protected_usage_admitted_and_not_adopted(self):
        lease=self.reserve()
        with self.registry.transaction() as state:
            self.assertEqual(set(state['jobs']),{lease.token})
            self.assertEqual(state['jobs'][lease.token]['owner'],self.owner)
            self.assertEqual(self.registry.accounting(state,self.obs)[0],{lease.token:0})
        lease.release(self.obs)
    def test_unknown_compute_and_graphics_refused_even_zero_allocation(self):
        for kind in ('compute','graphics'):
            obs=copy.deepcopy(self.obs);obs['procs'][3]=proc(3)
            obs['gpu_memory'][3]=0;obs['gpu_kinds'][3]=[kind]
            with self.assertRaisesRegex(a.Conflict,'Unregistered'):self.reserve(obs)
    def test_reused_changed_or_missing_protected_identity_refused(self):
        for field,value in [('birth',999),('exe','/changed'),('argv_sha256','b'*64),('uid',2000)]:
            obs=copy.deepcopy(self.obs);obs['procs'][1][field]=value
            with self.assertRaisesRegex(a.Conflict,'identity changed'):self.reserve(obs)
        obs=copy.deepcopy(self.obs);del obs['procs'][1]
        with self.assertRaisesRegex(a.Conflict,'identity changed'):self.reserve(obs)
    def test_changed_graphics_membership_refused(self):
        obs=copy.deepcopy(self.obs);obs['gpu_kinds'][2]=['compute']
        with self.assertRaisesRegex(a.Conflict,'membership changed'):self.reserve(obs)
    def test_declared_protected_growth_is_reserved(self):
        with self.assertRaisesRegex(a.Capacity,'physical capacity'):
            self.reserve(request={'gpu_bytes':41*G,'ram_bytes':G})
        obs=copy.deepcopy(self.obs);obs['available_ram']=7*G
        with self.assertRaisesRegex(a.Capacity,'available RAM'):self.reserve(obs)
    def test_inconsistent_global_snapshot_never_undercounts_pid_allocations(self):
        obs=copy.deepcopy(self.obs);obs['gpu_memory'][1]=20*G;obs['used']=G
        with self.assertRaisesRegex(a.Capacity,'physical capacity'):
            self.reserve(obs,request={'gpu_bytes':30*G,'ram_bytes':G})
        lease=self.reserve();obs['gpu_memory'][1]=44*G
        with self.assertRaisesRegex(a.Capacity,'stop owned worker only'):lease.health(obs)
        lease.release(self.obs)
    def test_registered_unconsumed_growth_reserved(self):
        lease=self.reserve(request={'gpu_bytes':30*G,'ram_bytes':G})
        other={k:self.obs['procs'][10][k] for k in ('pid','birth','boot')}
        with self.assertRaisesRegex(a.Capacity,'physical capacity'):
            self.registry.reserve('two',other,{'gpu_bytes':12*G,'ram_bytes':G},self.obs)
        lease.release(self.obs)
    def test_exclusive_physical_lock_conflict_not_bypassed(self):
        with self.paths[0].open('rb') as fd:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(a.Conflict,'exclusive guardian'):self.reserve()
    def test_cleanup_owns_only_bound_worker_group(self):
        lease=self.reserve();obs=copy.deepcopy(self.obs);obs['procs'][11]=proc(11,10,11)
        worker={k:obs['procs'][11][k] for k in ('pid','birth','boot')}
        lease.attach(worker,obs)
        targets=lease.cleanup_targets(obs)
        self.assertEqual(targets,[{'pgid':11,'leader':worker}])
        calls=[]
        identities={p['identity']['pid']:p['identity'] for p in self.peers}
        def current(pid):return worker if pid==11 else {k:identities[pid][k] for k in ('pid','birth','boot')}
        with patch.object(runner,'identity',side_effect=current),patch.object(runner,'protected_identity',side_effect=lambda pid:identities[pid]),patch.object(runner.os,'getpgid',side_effect=lambda pid:obs['procs'][pid]['pgid']),patch.object(runner.os,'killpg',side_effect=lambda p,s:calls.append(p)),patch.object(runner.os,'kill',side_effect=AssertionError('PID signal unexpected')):
            runner.signal_owned(lease,lambda:obs,15)
        self.assertEqual(calls,[11])
        contaminated=copy.deepcopy(obs);contaminated['procs'][1]['pgid']=11
        with self.assertRaisesRegex(a.Conflict,'Protected peer shares'):
            lease.cleanup_targets(contaminated)
        lease.release(self.obs)
    def test_process_only_cleanup_refuses_missing_or_changed_protected_peer(self):
        lease=self.reserve();obs=copy.deepcopy(self.obs);obs['procs'][11]=proc(11,10,11)
        worker={k:obs['procs'][11][k] for k in ('pid','birth','boot')};lease.attach(worker,obs)
        process_only={'procs':copy.deepcopy(obs['procs'])}
        self.assertEqual(lease.cleanup_targets(process_only),[{'pgid':11,'leader':worker}])
        del process_only['procs'][1]
        with self.assertRaisesRegex(a.Conflict,'identity changed'):lease.cleanup_targets(process_only)
        process_only['procs'][1]=proc(1);process_only['procs'][1]['birth']=999
        with self.assertRaisesRegex(a.Conflict,'identity changed'):lease.cleanup_targets(process_only)
        lease.release(self.obs)
    def test_group_signal_rechecks_fresh_protected_identity_without_gpu_query(self):
        lease=self.reserve();obs=copy.deepcopy(self.obs);obs['procs'][11]=proc(11,10,11)
        worker={k:obs['procs'][11][k] for k in ('pid','birth','boot')};lease.attach(worker,obs)
        with patch.object(runner,'identity',return_value=worker),patch.object(runner,'protected_identity',return_value=None),patch.object(runner.os,'killpg') as signal:
            with self.assertRaisesRegex(a.Conflict,'not exactly observable'):
                runner.signal_owned(lease,lambda:{'procs':obs['procs']},15)
            signal.assert_not_called()
        lease.release(self.obs)
    def test_replaced_or_symlink_physical_lock_refused(self):
        lease=self.reserve();old=self.paths[0].with_suffix('.original');self.paths[0].rename(old);self.paths[0].touch()
        with self.assertRaisesRegex(a.Conflict,'path replaced'):lease.health(self.obs)
        self.paths[0].unlink();old.rename(self.paths[0]);lease.release(self.obs)
        target=self.root/'alias-target.lock';target.touch();self.paths[0].unlink();self.paths[0].symlink_to(target)
        with self.assertRaises(OSError):self.reserve()
    def test_registry_lock_contention_refuses_without_blocking(self):
        self.registry.path.mkdir()
        with (self.registry.path/'registry.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(a.Uncertain,'transaction busy'):self.reserve()
    def test_ram_forecast_breach_is_telemetry_not_runtime_rss_kill(self):
        lease=self.reserve();obs=copy.deepcopy(self.obs);obs['available_ram']=G
        self.assertFalse(lease.health(obs)['host_growth_forecast_fits'])
        with self.assertRaisesRegex(a.Capacity,'available RAM'):
            self.registry.reserve('two',self.owner,{'gpu_bytes':G,'ram_bytes':G},obs)
        lease.release(self.obs)
    def test_pid_reuse_not_cleanup_target_and_live_owner_refuses_release(self):
        lease=self.reserve();obs=copy.deepcopy(self.obs);obs['procs'][11]=proc(11,10,11)
        worker={k:obs['procs'][11][k] for k in ('pid','birth','boot')};lease.attach(worker,obs)
        with self.assertRaisesRegex(a.Conflict,'still alive'):lease.release(obs)
        obs['procs'][11]['birth']=9999;obs['procs'][11]['sid']=123;obs['procs'][11]['pgid']=123
        self.assertEqual(lease.cleanup_targets(obs),[])
        lease.release(obs)
    def test_runtime_gpu_growth_exhaustion_stops_only_owned_work(self):
        lease=self.reserve();obs=copy.deepcopy(self.obs)
        obs['gpu_memory'][1]=44*G;obs['used']=45*G
        with self.assertRaisesRegex(a.Capacity,'stop owned worker only'):lease.health(obs)
        lease.release(self.obs)
    def test_explicit_growth_reservations_required(self):
        peers=copy.deepcopy(self.peers);peers[0]['gpu_budget_bytes']=None
        with self.assertRaisesRegex(a.Conflict,'growth budgets required'):
            a.Registry(self.root/'bad',protected_peers=peers,driver_headroom=G,host_free_floor=G,policy_sha256='f'*64)
    def test_live_registry_cannot_silently_change_protected_policy(self):
        lease=self.reserve()
        changed=a.Registry(self.root/'registry',locks=self.paths,gpu=a.GPU1,
                           roles={'one','two'},protected_peers=self.peers,
                           driver_headroom=G,host_free_floor=G,policy_sha256='e'*64)
        with self.assertRaisesRegex(a.Conflict,'reviewed policy differs'):
            changed.reserve('two',self.owner,{'gpu_bytes':G,'ram_bytes':G},self.obs)
        lease.release(self.obs)

class OwnedLifecycle(unittest.TestCase):
    def test_bootstrap_ignores_startup_hooks_until_registered_but_target_keeps_semantics(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);marker=root/'startup-marker'
            (root/'sitecustomize.py').write_text('import os\nfrom pathlib import Path\nPath(os.environ["GPU1_FIXTURE_MARKER"]).write_text("startup")\n')
            env=dict(os.environ,PYTHONPATH=str(root),GPU1_FIXTURE_MARKER=str(marker),CUDA_VISIBLE_DEVICES='')
            case=self
            class Lease:
                def fds(self):return []
                def attach(self,identity,observation):
                    # Give a buggy nonisolated interpreter time to run startup
                    # hooks; the corrected stdlib gate instead blocks on pipe.
                    end=time.monotonic()+.15
                    while not marker.exists() and time.monotonic()<end:time.sleep(.005)
                    case.assertFalse(marker.exists(),'startup hook executed before lease registration')
            worker,_=runner.gate_spawn([sys.executable,'-B','-c','pass'],env,Lease(),lambda:{})
            try:self.assertEqual(worker.wait(timeout=3),0)
            finally:
                if worker.poll() is None:worker.kill();worker.wait(timeout=3)
            self.assertEqual(marker.read_text(),'startup')
    def test_invalid_polling_interval_refused_before_policy_or_job(self):
        for value in [float('inf'),float('nan'),0,-1,3]:
            with self.assertRaisesRegex(ValueError,'health polling'):
                runner.run('/must-not-create',a.GPU1,'no-job',[],policy_path='/must-not-read',role='fixture',poll_seconds=value)
    def fixture_policy(self,root):
        root.mkdir();peers,obs=fixture();locks=[root/'admission.lock',root/'worker.lock']
        for p in locks:p.touch()
        cfg={'status':'PREPARED_FIXTURE_ONLY','gpu':a.GPU1,'registry_path':str(root/'registry'),
             'locks':list(map(str,locks)),'roles':{'fixture':{'gpu_bytes':G,'ram_bytes':G}},
             'driver_headroom_bytes':G,'host_free_floor_bytes':G,'protected_peers':peers,
             'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(a.__file__),Path(runner.__file__)]}}
        policy=root/'policy.json'
        def write():policy.write_text(json.dumps(cfg));return hashlib.sha256(policy.read_bytes()).hexdigest()
        return cfg,obs,policy,write
    def test_resolved_fixture_paths_cannot_escape_through_relative_or_symlink_alias(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)/'fixture';cfg,obs,policy,write=self.fixture_policy(root)
            original=cfg['locks'][0]
            for escape in [str(root/'../outside.lock'),str(root/'alias.lock')]:
                if escape.endswith('alias.lock'):(root/'alias.lock').symlink_to(root.parent/'outside.lock')
                cfg['locks'][0]=escape;pin=write()
                with self.assertRaisesRegex(a.Conflict,'resolve inside fixture root'):
                    runner.run(root/'ledger.json',a.GPU1,'no-job',[],policy_path=policy,role='fixture',observe=lambda:obs,test_mode=True,expected_policy_sha256=pin)
            self.assertFalse((root/'ledger.json').exists());self.assertFalse((root.parent/'outside.lock').exists())
    def test_policy_and_extra_source_tampering_refused_before_reservation(self):
        for target in ('policy','source'):
            with self.subTest(target=target),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                root=Path(directory)/'fixture';cfg,obs,policy,write=self.fixture_policy(root)
                dep=root/'dependency.py';dep.write_text('original');cfg['source_sha256'][str(dep)]=hashlib.sha256(dep.read_bytes()).hexdigest();pin=write()
                calls=[]
                def mutate_once():
                    if not calls:
                        calls.append(True)
                        (policy if target=='policy' else dep).write_text('changed')
                with self.assertRaisesRegex(a.Conflict,'policy changed|source changed'):
                    runner.run(root/'ledger.json',a.GPU1,'no-job',[],policy_path=policy,role='fixture',observe=lambda:obs,test_mode=True,expected_policy_sha256=pin,protect=mutate_once)
                self.assertFalse((root/'ledger.json').exists());self.assertFalse((root/'registry').exists())
    def test_imported_function_origin_must_match_pinned_admission_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)/'fixture';cfg,obs,policy,write=self.fixture_policy(root);pin=write()
            with patch.object(runner,'snapshot',lambda *args:obs):
                with self.assertRaisesRegex(a.Conflict,'source origin differs'):
                    runner.run(root/'ledger.json',a.GPU1,'no-job',[],policy_path=policy,role='fixture',observe=lambda:obs,test_mode=True,expected_policy_sha256=pin)
            self.assertFalse((root/'ledger.json').exists())
    def test_finite_cpu_job_preserves_peer_and_charges_failure_without_successor(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)/'fixture';root.mkdir()
            peer=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'])
            try:
                identity=a.protected_identity(peer.pid)
                paths=[root/'admission.lock',root/'worker.lock']
                for p in paths:p.touch()
                policy={'status':'PREPARED_FIXTURE_ONLY','gpu':a.GPU1,
                        'registry_path':str(root/'registry'),'locks':list(map(str,paths)),
                        'roles':{'fixture':{'gpu_bytes':G,'ram_bytes':G}},
                        'driver_headroom_bytes':G,'host_free_floor_bytes':G,
                        'protected_peers':[{'identity':identity,'gpu_kinds':['compute'],
                                            'gpu_budget_bytes':G,'ram_budget_bytes':G}],
                        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [Path(a.__file__),Path(runner.__file__)]}}
                policy_path=root/'policy.json';policy_path.write_text(json.dumps(policy))
                driver='import sys;import test_gpu0_candidate as t;raise SystemExit(t.run_fixture_driver(sys.argv[1]))'
                result=subprocess.run([sys.executable,'-B','-c',driver,str(root)],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,7,result.stderr);self.assertIsNone(peer.poll())
                ledger=json.loads((root/'ledger.json').read_text())
                self.assertEqual(len(ledger['jobs']),1);job=ledger['jobs'][0]
                self.assertEqual(job['status'],'failed');self.assertTrue(job['reservation_released'])
                self.assertGreater(job['charged_seconds'],0)
                self.assertEqual(json.loads((root/'registry/registry.json').read_text())['jobs'],{})
                duplicate=subprocess.run([sys.executable,'-B','-c',driver,str(root)],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(duplicate.returncode,0)
                self.assertIn('Prior attempt exists',duplicate.stderr)
                self.assertEqual(len(json.loads((root/'ledger.json').read_text())['jobs']),1)
            finally:
                if peer.poll() is None:peer.terminate()
                peer.wait(timeout=5)

class NVMLObservation(unittest.TestCase):
    def test_nvml_helper_uses_isolated_stdlib_startup_and_timeout(self):
        peers,obs=fixture();calls=[]
        def output(command,**kwargs):
            calls.append((command,kwargs))
            if command[0]=='nvidia-smi':return a.GPU1+', 49140, 1480, 32'
            return json.dumps({str(p):{'memory':n,'kinds':obs['gpu_kinds'][p]} for p,n in obs['gpu_memory'].items()})
        with patch.object(a.subprocess,'check_output',side_effect=output),patch.object(a,'proc_table',return_value=obs['procs']):
            result=a.snapshot()
        self.assertEqual(calls[1][0][1:4],['-I','-S','-B'])
        self.assertEqual(calls[1][1]['timeout'],2)
        self.assertEqual(result['gpu_kinds'],obs['gpu_kinds'])
    def test_process_disappearance_after_initial_identity_is_normal(self):
        with patch.object(a,'protected_identity',return_value=None):
            self.assertEqual(a.proc_table(),{})
    def library(self,compute,graphics):
        class Library:
            def nvmlInit_v2(self):return 0
            def nvmlShutdown(self):return 0
            def nvmlDeviceGetHandleByUUID(self,*args):return 0
            def nvmlDeviceGetUUID(self,handle,buffer,size):buffer.value=a.GPU1.encode();return 0
            def query(self,rows,count,target):
                count._obj.value=len(rows)
                if target is None:return 7 if rows else 0
                for i,(pid,memory) in enumerate(rows):
                    target[i].pid=pid;target[i].memory=memory
                    target[i].gpu_instance=target[i].compute_instance=0xffffffff
                return 0
            def nvmlDeviceGetComputeRunningProcesses_v2(self,handle,count,target):return self.query(compute,count,target)
            def nvmlDeviceGetGraphicsRunningProcesses_v2(self,handle,count,target):return self.query(graphics,count,target)
        return Library()
    def test_both_kinds_are_observed_and_shared_pid_memory_is_not_double_counted(self):
        lib=self.library([(1,100),(3,200)],[(2,300),(3,200)])
        with patch.object(a.ctypes,'CDLL',return_value=lib):result=a.nvml_allocations(a.GPU1)
        self.assertEqual(result,{1:{'memory':100,'kinds':['compute']},2:{'memory':300,'kinds':['graphics']},3:{'memory':200,'kinds':['compute','graphics']}})
    def test_unknown_memory_or_duplicate_pid_is_uncertain(self):
        for rows in [[(1,2**64-1)],[(1,100),(1,100)]]:
            with patch.object(a.ctypes,'CDLL',return_value=self.library(rows,[])):
                with self.assertRaisesRegex(a.Uncertain,'Ambiguous NVML'):a.nvml_allocations(a.GPU1)

if __name__=='__main__':unittest.main(verbosity=2)
