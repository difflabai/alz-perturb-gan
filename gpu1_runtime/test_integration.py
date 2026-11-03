"""Bounded isolated CPU fixtures; never canonical state, native models or leases."""
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import binding as b
import launch_core_gpu0 as owner
import project_runner_gpu0 as backend

G=2**30
guard_inventory=b.read(b.GUARD/'REVIEW_MANIFEST.json')['source_SHA256']
a=b.load_pinned(b.GUARD/'gpu0_admission.py',guard_inventory[str(b.GUARD/'gpu0_admission.py')],'gpu0_admission')

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def run_driver(root):
    layout=b.Layout.fixture(root);auth=b.read(layout.fixture_root/'authorization.json')
    peer=auth['protected_peers'][0]['identity']['pid']
    def observe():
        return {'gpu':b.GPU1,'total':48*G,'used':G//4,'gpu_memory':{peer:G//8},
                'gpu_kinds':{peer:['compute']},'procs':a.proc_table(),
                'available_ram':32*G,'total_ram':128*G,'complete':True}
    return owner.run(layout.fixture_root/'authorization.json',layout,observe=observe)

class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='task11-gpu0-integration-',dir='/tmp')
        self.root=Path(self.tmp.name)/'fixture';self.root.mkdir()
        self.layout=b.Layout.fixture(self.root);self.layout.project.mkdir();self.layout.science.mkdir()
        self.peer=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(30)'])
        self.value={'fold':0,'method':'pg','seed':0}
        s=self.layout.science;p=self.layout.project
        (s/'contracts.py').write_text('PG_UPDATES=20000\nSCVI_EPOCHS=400\n')
        (s/'native_model.py').write_text('SYNTHETIC_UNIMPORTED_NATIVE_SOURCE=True\n')
        cache=s/'caches/fold0';cache.mkdir(parents=True);(cache/'counts.bin').write_bytes(b'CPU_SYNTHETIC_COUNTS_NOT_REAL_DONORS')
        dump(cache/'manifest.json',{'files_SHA256':{'counts.bin':b.sha(cache/'counts.bin')}})
        traces=s/'traces';traces.mkdir();(traces/'trace.bin').write_bytes(b'CPU_FROZEN_TRACE')
        dump(traces/'manifest.json',{'files_SHA256':{'trace.bin':b.sha(traces/'trace.bin')}})
        dump(s/'CORE_PROTOCOL.json',{'pg_updates':20000,'scvi_epochs':400,'core_GPU_hours':180})
        dump(s/'NATIVE_SOURCE_PINS.json',{'files_SHA256':{str(s/'native_model.py'):b.sha(s/'native_model.py')}})
        science={'source_SHA256':{str(s/'contracts.py'):b.sha(s/'contracts.py')},
                 'native_source_inventory_SHA256':b.sha(s/'NATIVE_SOURCE_PINS.json'),
                 'cache_manifest_SHA256':{'0':b.sha(cache/'manifest.json')},
                 'protocol_SHA256':b.sha(s/'CORE_PROTOCOL.json'),
                 'shared_trace_manifest_SHA256':b.sha(traces/'manifest.json')}
        dump(s/'REVIEW_MANIFEST.json',science)
        # Exact unchanged Controller executes against this private synthetic
        # accounting implementation, never the canonical/default project.
        (p/'project_budget.py').write_text('''import json,hashlib
from pathlib import Path
def status(root):
 root=Path(root);state=json.loads((root/'project_gpu_budget.json').read_text());charged=0
 for entry in state['ledger_registry']:
  path=root/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['immutable_SHA256']
  ledger=json.loads(path.read_text());charged+=ledger['aggregate_charged_seconds']*entry['charged_device_count']
 reserved=sum(r['remaining_wall_seconds']*r['device_count'] for r in state['open_reservations'])
 return {'revision':state['revision'],'available_after_reservations_GPU_hours':(state['cap_GPU_seconds']-charged-reserved)/3600}
''')
        locks=[self.root/'gpu0-admission.lock',self.root/'gpu0-worker.lock']
        for path in locks:path.touch()
        peers=[{'identity':a.protected_identity(self.peer.pid),'gpu_kinds':['compute'],'gpu_budget_bytes':G,'ram_budget_bytes':G}]
        policy={'status':'PREPARED_FIXTURE_ONLY','gpu':b.GPU1,'registry_path':str(self.root/'registry'),
                'locks':list(map(str,locks)),'roles':{'alz_core_pg':{'gpu_bytes':G,'ram_bytes':G}},
                'driver_headroom_bytes':G,'host_free_floor_bytes':G,'protected_peers':peers,
                'source_sha256':{str(b.GUARD/name):b.sha(b.GUARD/name) for name in ['gpu0_admission.py','gpu0_runner.py']}}
        self.policy=self.root/'policy.json';dump(self.policy,policy)
        pins={}
        for path in list(s.rglob('*'))+[p/'project_budget.py',self.policy,b.CONTROLLER,b.runtime_path(self.layout,self.value),b.GUARD/'REVIEW_MANIFEST.json',*map(Path,guard_inventory),*b.HERE.glob('*.py')]:
            if path.is_file():pins[str(path)]=b.sha(path)
        self.auth={'status':'APPROVED_CPU_FIXTURE_ONLY','cpu_fixture':True,'cell':self.value,
                   'allocation_id':b.allocation_key(self.value),'attempt_id':b.attempt_key(self.value),'label':b.attempt_key(self.value),
                   'scheduler_allocation_id':'FIXTURE_GPU1_0001','source_review_passed':True,
                   'gpu_uuid':b.GPU1,'machine_id':socket.gethostname(),'expires_unix':time.time()+120,
                   'cap_seconds':15,'fixed_endpoint':20000,'core_GPU_hour_ceiling':180,
                   'project_root':str(p),'science_root':str(s),'cache_dir':str(cache),
                   'output_dir':str(b.output_path(self.layout,self.value)),'ledger_path':str(b.ledger_path(self.layout,self.value)),
                   'role':'alz_core_pg','policy_path':str(self.policy),'policy_SHA256':b.sha(self.policy),
                   'registry_path':policy['registry_path'],'protected_peers':peers,'files_SHA256':pins,
                   'fixture_science_review_SHA256':b.sha(s/'REVIEW_MANIFEST.json')}
        self.auth_path=self.root/'authorization.json';self.auth['command']=b.expected_command(self.layout,self.auth_path,self.value);dump(self.auth_path,self.auth)
        sourcepins={str(path):pins[str(path)] for path in [*b.HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']}
        self.state={'schema':2,'revision':0,'cap_GPU_seconds':1024*3600,
                    'grants':[{'grant_id':'CPU_FIXTURE_ONLY','GPU_seconds':1024*3600}],
                    'ledger_registry':[],'open_reservations':[],'allocation_history':[],
                    'future_accounting_policy':{'current_launch_admission':True},
                    'execution_controller':{'status':'REVIEWED_FOR_BOUNDED_EXECUTION','calibration_gpu1_backend':{'status':'REVIEWED_FOR_BOUNDED_EXECUTION','path':str(b.HERE/'project_runner_gpu0.py'),'source_SHA256':sourcepins}}}
        self.state['execution_controller']['calibration_gpu1_backend'].update(files_SHA256=pins,policy_SHA256=self.auth['policy_SHA256'],**{k:self.auth[k] for k in ['allocation_id','attempt_id','scheduler_allocation_id','expires_unix']})
        dump(p/'project_gpu_budget.json',self.state)
    def tearDown(self):
        if self.peer.poll() is None:self.peer.terminate()
        self.peer.wait(timeout=5);self.tmp.cleanup()
    def run_child(self):
        code='import sys;sys.path.insert(0,sys.argv[1]);import test_integration as t;raise SystemExit(t.run_driver(sys.argv[2]))'
        return subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(b.HERE),str(self.root)],capture_output=True,text=True,timeout=15)
    def test_held_owner_and_worker_refuse_before_reservation_or_model_import(self):
        self.auth['status']='HELD';dump(self.auth_path,self.auth);before=b.sha(self.layout.project/'project_gpu_budget.json')
        with self.assertRaisesRegex(b.Refusal,'remain held'):owner.run(self.auth_path,self.layout,observe=lambda:{})
        result=subprocess.run(self.auth['command'],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(result.returncode,0);self.assertIn('remain held',result.stderr)
        self.assertEqual(b.sha(self.layout.project/'project_gpu_budget.json'),before)
        self.assertFalse(Path(self.auth['output_dir']).exists());self.assertNotIn('torch',sys.modules)
    def test_complete_native_inventory_and_input_pins_required(self):
        for missing in [str(self.layout.science/'native_model.py'),str(self.layout.science/'caches/fold0/counts.bin')]:
            auth=copy.deepcopy(self.auth);del auth['files_SHA256'][missing];dump(self.auth_path,auth)
            with self.assertRaisesRegex(b.Refusal,'absent|complete'):b.verify(self.auth_path,self.layout)
        self.assertFalse(Path(self.auth['ledger_path']).exists())
    def test_wrong_device_expiry_endpoint_command_or_backend_binding_refused(self):
        for key,value in [('gpu_uuid','GPU-OTHER'),('expires_unix',0),('fixed_endpoint',1024),('source_review_passed',False),('output_dir',str(self.root/'other'))]:
            auth=copy.deepcopy(self.auth);auth[key]=value;dump(self.auth_path,auth)
            with self.assertRaises(b.Refusal):b.verify(self.auth_path,self.layout)
        state=copy.deepcopy(self.state);del state['execution_controller']['calibration_gpu1_backend']
        with self.assertRaisesRegex(b.Refusal,'backend source review'):backend.controller_gate(state,self.auth)
        changed=copy.deepcopy(self.auth)
        extra=self.root/'unreviewed-extra.py';extra.write_text('SYNTHETIC_NEW_DEPENDENCY=True\n')
        changed['files_SHA256'][str(extra)]=b.sha(extra)
        dump(self.auth_path,changed);b.verify(self.auth_path,self.layout)
        with self.assertRaisesRegex(b.Refusal,'full source/runtime/input/policy closure'):
            backend.controller_gate(self.state,changed)
    def test_source_or_cache_drift_refused_before_budget_or_model(self):
        for path in [self.layout.science/'native_model.py',self.layout.science/'caches/fold0/counts.bin']:
            previous=path.read_bytes();path.write_bytes(previous+b'TAMPER')
            with self.assertRaisesRegex(b.Refusal,'changed'):b.verify(self.auth_path,self.layout)
            path.write_bytes(previous)
        self.assertFalse(Path(self.auth['ledger_path']).exists())
    def test_missing_live_reservation_or_lease_refuses_worker(self):
        keys=['ALZ_PROJECT_RESERVATION_TOKEN','ALZ_PROJECT_BUDGET_ROOT','ALZ_BUDGET_LEDGER','COOPERATIVE_GPU_TOKEN']
        with patch.dict(os.environ,{k:'' for k in keys}):
            with self.assertRaisesRegex(b.Refusal,'no live canonical'):b.require_live(self.auth_path,self.layout)
        self.assertNotIn('torch',sys.modules)
    def test_worker_checks_private_canonical_lease_command_device_and_ancestor_bindings(self):
        current=a.identity(os.getpid());token='CPU_SYNTHETIC_CANONICAL_TOKEN';lease_token='CPU_SYNTHETIC_RESOURCE_TOKEN'
        reservation={'allocation_id':self.auth['allocation_id'],'token':token,'status':'reserved',
                     'gpu_uuids':[b.GPU1],'device_count':1,'declared_wall_seconds':15,'job_cap_seconds':15,
                     'remaining_wall_seconds':45,'owner':current,'label':self.auth['label'],
                     'machine_id':socket.gethostname(),'ledger_path':str(Path(self.auth['ledger_path']).relative_to(self.layout.project))}
        state=copy.deepcopy(self.state);state['open_reservations']=[reservation]
        job={'guard_identity':current,'status':'active','role':self.auth['role'],'lease_token':lease_token,
             'label':self.auth['label'],'command':self.auth['command'],'child_identity':current,'started_unix':time.time()}
        ledger={'gpu_uuid':b.GPU1,'cap_seconds':15,'jobs':[job]}
        resource={'owner':current,'worker':current,'status':'running','role':self.auth['role'],
                  'request':b.read(self.policy)['roles'][self.auth['role']]}
        registry={'schema':2,'gpu':b.GPU1,'policy_sha256':self.auth['policy_SHA256'],'jobs':{lease_token:resource}}
        env={'ALZ_PROJECT_RESERVATION_TOKEN':token,'ALZ_PROJECT_BUDGET_ROOT':str(self.layout.project),
             'ALZ_BUDGET_LEDGER':self.auth['ledger_path'],'COOPERATIVE_GPU_TOKEN':lease_token,'CUDA_VISIBLE_DEVICES':''}
        def write(s,l,r):
            dump(self.layout.project/'project_gpu_budget.json',s);dump(self.auth['ledger_path'],l)
            dump(Path(self.auth['registry_path'])/'registry.json',r)
        write(state,ledger,registry)
        with patch.dict(os.environ,env):
            auth,result=b.require_live(self.auth_path,self.layout)
            self.assertEqual(result['allocation_id'],self.auth['allocation_id'])
            for case in ['device','command','policy','lease','request','ancestor']:
                s,l,r=copy.deepcopy(state),copy.deepcopy(ledger),copy.deepcopy(registry)
                if case=='device':s['open_reservations'][0]['gpu_uuids']=['GPU-OTHER']
                if case=='command':l['jobs'][0]['command']=['UNREVIEWED_WORKER']
                if case=='policy':r['policy_sha256']='0'*64
                if case=='lease':r['jobs']={}
                if case=='request':r['jobs'][lease_token]['request']={'gpu_bytes':1,'ram_bytes':1}
                if case=='ancestor':
                    foreign={'pid':999999999,'birth':1,'boot':'SYNTHETIC_NOT_THIS_PROCESS'}
                    l['jobs'][0]['child_identity']=foreign;r['jobs'][lease_token]['worker']=foreign
                write(s,l,r)
                with self.subTest(case=case),self.assertRaises(b.Refusal):b.require_live(self.auth_path,self.layout)
        self.assertNotIn('torch',sys.modules)
    def test_old_gpu1_phase_lock_prevents_new_gpu0_race(self):
        before=b.sha(self.layout.project/'project_gpu_budget.json')
        with (self.layout.science/'core_phase.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result=self.run_child()
        self.assertNotEqual(result.returncode,0);self.assertIn('BlockingIOError',result.stderr)
        self.assertEqual(b.sha(self.layout.project/'project_gpu_budget.json'),before)
        self.assertFalse(Path(self.auth['ledger_path']).exists())
    def test_any_old_or_new_matrix_consumption_or_output_refuses_duplicate(self):
        for status in ['reserved','failed','uncertain','settled','cancelled_before_execution']:
            state=copy.deepcopy(self.state);state['allocation_history']=[{'allocation_id':self.auth['allocation_id'],'status':status}]
            with self.assertRaisesRegex(b.Refusal,'already consumed'):b.unseen(self.auth,self.layout,state)
        legacy=self.layout.science/'runs/f0_pg_s0';legacy.mkdir(parents=True)
        with self.assertRaisesRegex(b.Refusal,'existing GPU1/GPU1'):b.unseen(self.auth,self.layout,self.state)
    def test_core_180hour_budget_includes_closed_and_open_card_hours(self):
        state={'allocation_history':[{'allocation_id':self.auth['allocation_id'],'charged_GPU_seconds':180*3600-10,'status':'settled'}]}
        with self.assertRaisesRegex(b.Refusal,'180GPU-hour'):b.core_envelope(state,45)
        state={'open_reservations':[{'allocation_id':self.auth['allocation_id'],'remaining_wall_seconds':180*3600,'device_count':1,'status':'uncertain'}]}
        with self.assertRaisesRegex(b.Refusal,'uncertain'):b.core_envelope(state)
    def test_fixture_cannot_select_canonical_root(self):
        with self.assertRaises(b.Refusal):b.Layout.fixture(b.PROJECT)
        forged=b.Layout(b.PROJECT,b.SCIENCE,Path('/'))
        with self.assertRaisesRegex(b.Refusal,'fixture root'):b.verify('/must-not-read',forged)
        forged=b.Layout(b.PROJECT,b.SCIENCE,self.root)
        with self.assertRaisesRegex(b.Refusal,'Layout fields'):b.verify('/must-not-read',forged)
    def test_end_to_end_cpu_owner_worker_settle_then_other_card_same_key_refuses(self):
        result=self.run_child();self.assertEqual(result.returncode,0,result.stderr)
        out=Path(self.auth['output_dir']);proof=b.read(out/'CPU_FIXTURE_COMPLETE.json')
        self.assertFalse(proof['model_imported']);self.assertEqual(proof['CUDA_VISIBLE_DEVICES'],'')
        state=b.read(self.layout.project/'project_gpu_budget.json');self.assertEqual(state['open_reservations'],[])
        self.assertEqual(len(state['allocation_history']),1);entry=state['allocation_history'][0]
        self.assertEqual(entry['allocation_id'],self.auth['allocation_id']);self.assertEqual(entry['gpu_uuids'],[b.GPU1]);self.assertEqual(entry['device_count'],1);self.assertGreater(entry['charged_GPU_seconds'],0)
        self.assertEqual(b.read(Path(self.auth['registry_path'])/'registry.json')['jobs'],{})
        self.assertIsNone(self.peer.poll())
        c,_,_=b.modules(self.auth)
        with self.assertRaisesRegex(c.Conflict,'allocation ID already exists'):
            c.Controller(self.layout.project).reserve(allocation_id=self.auth['allocation_id'],label='otherGPU0',owner=a.identity(os.getpid()),machine_id=socket.gethostname(),gpu_uuids=['GPU-72c1b506-891d-b8bc-b353-e020585e1c47'],wall_seconds=15,job_cap_seconds=15,ledger_path='other-card-new-ledger.json')
        self.assertEqual(len(b.read(self.layout.project/'project_gpu_budget.json')['allocation_history']),1)
    def test_native_phase_is_card_local_without_touching_real_paths(self):
        layout=SimpleNamespace(project=self.layout.project,science=self.layout.science,fixture_mode=False)
        with (self.layout.science/'core_phase.lock').open('a+') as old_core:
            fcntl.flock(old_core,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(b,'verify',return_value=(self.auth,'CPU_FIXTURE_ONLY')):
                with patch.object(backend,'run',return_value=0) as call:
                    self.assertEqual(owner.run(self.auth_path,layout),0)
                    call.assert_called_once()
        self.assertTrue((self.layout.project/'calibration_gpu1_phase.lock').is_file())
        self.assertNotIn('torch',sys.modules)
    def test_failed_cpu_attempt_is_charged_and_consumes_same_scientific_key(self):
        self.auth['fixture_fail_exit']=True;dump(self.auth_path,self.auth)
        result=self.run_child();self.assertEqual(result.returncode,7,result.stderr)
        ledger=b.read(self.auth['ledger_path']);self.assertEqual(ledger['jobs'][0]['status'],'failed')
        state=b.read(self.layout.project/'project_gpu_budget.json');self.assertGreater(state['allocation_history'][0]['charged_GPU_seconds'],0)
        with self.assertRaisesRegex(b.Refusal,'already consumed'):b.unseen(self.auth,self.layout,state)
    def test_uncertain_resource_cleanup_preserves_canonical_reservation(self):
        c,admission,_=b.modules(self.auth)
        def fail(ledger,*args,**kwargs):
            dump(ledger,{'gpu_uuid':b.GPU1,'cap_seconds':15,'aggregate_charged_seconds':.5,'jobs':[{'guard_identity':a.identity(os.getpid()),'label':self.auth['label'],'status':'cleanup_uncertain','reservation_released':False,'charged_seconds':.5}]})
            raise RuntimeError('SYNTHETIC_UNCERTAIN_CLEANUP')
        with patch.object(b,'modules',return_value=(c,admission,SimpleNamespace(run=fail))):
            with self.assertRaisesRegex(c.Uncertain,'reservation retained'):backend.run(self.auth_path,self.layout,observe=lambda:{})
        state=b.read(self.layout.project/'project_gpu_budget.json')
        self.assertEqual(len(state['open_reservations']),1);self.assertEqual(state['open_reservations'][0]['status'],'uncertain')
        self.assertEqual(state['ledger_registry'],[]);self.assertEqual(state['cap_GPU_seconds'],1024*3600)
    def test_calibration_worker_refuses_partial_or_wrong_reference(self):
        import worker_gpu0
        auth=copy.deepcopy(self.auth)
        checkpoint={'schema':1,'method':'pg','fold':0,'seed':0,'status':'completed','completed_updates':20000,'fixed_endpoint':20000,'cache_manifest_SHA256':b.sha(Path(auth['cache_dir'])/'manifest.json')}
        worker_gpu0.validate_reference(checkpoint,auth)
        for key,value in [('completed_updates',19000),('status','recovery'),('seed',1),('fold',1),('cache_manifest_SHA256','wrong')]:
            changed=copy.deepcopy(checkpoint);changed[key]=value
            with self.assertRaises(b.Refusal):worker_gpu0.validate_reference(changed,auth)
        self.assertNotIn('torch',sys.modules)
    def test_native_runtime_import_roots_find_packages_without_importing_models(self):
        # find_spec of top-level names does not execute their package bodies.
        for method,names in [('pg',['particlegan','torch','numpy','scipy']),('scvi',['scvi','torch','numpy','scipy'])]:
            value={'fold':0,'method':method,'seed':0}
            pins={str(b.PG_SHARED_PTH):b.PG_SHARED_PTH_SHA}
            sites=b.native_site_roots(value,pins)
            code='import sys,json,importlib.util;sys.path.extend(json.loads(sys.argv[1]));names=json.loads(sys.argv[2]);specs={n:importlib.util.find_spec(n) for n in names};print(json.dumps({"origins":{n:(s.origin if s else None) for n,s in specs.items()},"model_imported":any(n in sys.modules for n in names)}))'
            result=subprocess.run([str(b.runtime_path(b.NATIVE_LAYOUT,value)),'-I','-S','-B','-c',code,json.dumps(list(map(str,sites))),json.dumps(names)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);data=json.loads(result.stdout)
            self.assertFalse(data['model_imported'])
            self.assertTrue(all(origin and any(Path(origin).is_relative_to(site) for site in sites) for origin in data['origins'].values()))

if __name__=='__main__':unittest.main(verbosity=2)
