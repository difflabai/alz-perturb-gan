"""Synthetic-only COMPUTE/RSS/host-peak sharing regressions."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gpu0_admission as a
import gpu0_runner as runner
import test_gpu0_candidate as fixtures

G = a.GiB


def rss_driver(root):
    root = Path(root)
    cfg = json.loads((root/'policy.json').read_text())
    peer = cfg['protected_peers'][0]['identity']['pid']
    def observe():
        procs = a.proc_table()
        procs[os.getpid()]['rss'] = 2*G
        return dict(gpu=a.GPU1,total=48*G,used=G//4,gpu_memory={peer:G//8},
                    gpu_kinds={peer:['compute']},procs=procs,available_ram=32*G,total_ram=128*G,complete=True)
    return runner.run(root/'ledger.json',a.GPU1,'sharing-rss',
                      [sys.executable,'-I','-S','-B','-c','import time;time.sleep(30)'],
                      cap_seconds=15,policy_path=root/'policy.json',role='fixture',observe=observe,
                      poll_seconds=.05,expected_policy_sha256=hashlib.sha256((root/'policy.json').read_bytes()).hexdigest(),test_mode=True)


class Sharing(unittest.TestCase):
    setUp = fixtures.Admission.setUp
    tearDown = fixtures.Admission.tearDown

    def shared_registry(self, *, peak=()):
        cfg = dict(external_compute_budget_bytes=15*G//4, moss_compute_pids=[1],
                   minimum_available_ram_bytes=8*G, enforce_own_rss=True, enforce_host_growth=True)
        return a.Registry(self.root/'registry', locks=self.paths, gpu=a.GPU1, roles={'one'},
                          protected_peers=self.peers, driver_headroom=G, host_free_floor=8*G,
                          policy_sha256='f'*64, sharing=cfg, host_peak_identities=list(peak))

    def test_external_cap_has_margin_and_counts_compute_not_graphics(self):
        registry = self.shared_registry()
        obs = copy.deepcopy(self.obs)
        obs['gpu_memory'][2] = 4*G
        value = registry.external_compute({'jobs':{}}, obs, {}, 3*G)
        self.assertFalse(value['external_compute_budget_exceeded'])
        obs['gpu_kinds'][2] = ['graphics','compute']
        value = registry.external_compute({'jobs':{}}, obs, {}, 3*G)
        self.assertTrue(value['external_compute_budget_exceeded'])
        self.assertEqual(value['external_compute_actual_bytes'], 4*G)
        with self.assertRaisesRegex(MemoryError, 'COMPUTE'):
            runner.enforce_health(value)

    def test_admission_refuses_aggregate_compute_forecast_before_resource_lease(self):
        registry = self.shared_registry()
        self.peers[1]['gpu_kinds'] = ['graphics','compute']
        self.obs['gpu_kinds'][2] = ['graphics','compute']
        self.peers[1]['gpu_budget_bytes'] = G
        self.obs['gpu_memory'][2] = G
        with self.assertRaisesRegex(a.Capacity, 'COMPUTE'):
            registry.reserve('one', self.owner, {'gpu_bytes':3*G,'ram_bytes':G}, self.obs)
        state = a.json.loads((registry.path/'registry.json').read_text()) if (registry.path/'registry.json').exists() else {'jobs':{}}
        self.assertFalse(state['jobs'])

    def test_runtime_external_aggregate_can_fail_while_owned_gpu_is_below_its_cap(self):
        registry = self.shared_registry()
        self.peers[1]['gpu_kinds'] = ['graphics','compute']
        self.peers[1]['gpu_budget_bytes'] = G//4
        obs = copy.deepcopy(self.obs)
        obs['gpu_kinds'][2] = ['graphics','compute']
        lease = registry.reserve('one', self.owner, {'gpu_bytes':3*G,'ram_bytes':G}, obs)
        worker = fixtures.proc(11, 10, 11)
        obs['procs'][11] = worker
        lease.attach({k:worker[k] for k in ('pid','birth','boot')}, obs)
        obs['gpu_memory'].update({2:G,11:29*G//10})
        obs['gpu_kinds'][11] = ['compute']
        health = lease.health(obs)
        self.assertFalse(health['own_gpu_budget_exceeded'])
        self.assertTrue(health['external_compute_budget_exceeded'])
        with self.assertRaisesRegex(MemoryError, 'COMPUTE'):
            runner.enforce_health(health)
        del obs['procs'][11]; del obs['gpu_memory'][11]; del obs['gpu_kinds'][11]
        lease.release(obs)

    def test_owned_rss_and_host_floor_are_stops_only_in_explicit_sharing_policy(self):
        registry = self.shared_registry()
        lease = registry.reserve('one', self.owner, {'gpu_bytes':G,'ram_bytes':G}, self.obs)
        obs = copy.deepcopy(self.obs)
        obs['procs'][10]['rss'] = 2*G
        health = lease.health(obs)
        self.assertTrue(health['own_ram_budget_exceeded'])
        with self.assertRaisesRegex(MemoryError, 'RSS'):
            runner.enforce_health(health)
        obs = copy.deepcopy(self.obs); obs['available_ram'] = 7*G
        health = lease.health(obs)
        self.assertTrue(health['shared_host_floor_breached'])
        with self.assertRaises(MemoryError):
            runner.enforce_health(health)
        lease.release(self.obs)

    def test_fresh_exact_host_peak_growth_is_reserved_and_identity_changes_refuse(self):
        saved = {k:self.obs['procs'][10][k] for k in a.IDENTITY_KEYS}
        registry = self.shared_registry(peak=[saved])
        obs = copy.deepcopy(self.obs)
        obs['procs'][10]['hwm'] = 7*G
        self.assertEqual(registry.host_peak_growth(obs), 7*G-G//4)
        obs['procs'][10]['argv_sha256'] = 'c'*64
        with self.assertRaisesRegex(a.Conflict, 'host-peak identity'):
            registry.host_peak_growth(obs)

    def test_host_growth_forecast_failure_stops_owned_even_above_8gib_free(self):
        registry = self.shared_registry()
        lease = registry.reserve('one', self.owner, {'gpu_bytes':G,'ram_bytes':G}, self.obs)
        obs = copy.deepcopy(self.obs); obs['available_ram'] = 9*G
        health = lease.health(obs)
        self.assertFalse(health['shared_host_floor_breached'])
        self.assertTrue(health['shared_host_growth_budget_exceeded'])
        with self.assertRaisesRegex(MemoryError, 'growth'):
            runner.enforce_health(health)
        lease.release(self.obs)

    def test_rss_runtime_breach_reaps_only_owned_cpu_worker_and_preserves_moss_peer(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)/'fixture';root.mkdir()
            peer=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(30)'])
            try:
                locks=[root/'admission.lock',root/'worker.lock']
                for path in locks:path.touch()
                cfg=dict(status='PREPARED_FIXTURE_ONLY',gpu=a.GPU1,registry_path=str(root/'registry'),
                         locks=list(map(str,locks)),roles={'fixture':dict(gpu_bytes=G,ram_bytes=G)},
                         driver_headroom_bytes=G,host_free_floor_bytes=8*G,
                         protected_peers=[dict(identity=a.protected_identity(peer.pid),gpu_kinds=['compute'],gpu_budget_bytes=G,ram_budget_bytes=G)],
                         source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.__file__,runner.__file__]},
                         sharing=dict(external_compute_budget_bytes=15*G//4,moss_compute_pids=[peer.pid],minimum_available_ram_bytes=8*G,enforce_own_rss=True,enforce_host_growth=True))
                (root/'policy.json').write_text(json.dumps(cfg))
                code='import sys;sys.path.insert(0,sys.argv[1]);import test_sharing as t;raise SystemExit(t.rss_driver(sys.argv[2]))'
                result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(Path(__file__).parent),str(root)],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIsNone(peer.poll())
                job=json.loads((root/'ledger.json').read_text())['jobs'][0]
                self.assertIn('RSS',job['error']);self.assertEqual(job['status'],'failed')
                self.assertTrue(job['reservation_released']);self.assertGreater(job['charged_seconds'],0)
                self.assertFalse(json.loads((root/'registry/registry.json').read_text())['jobs'])
            finally:
                if peer.poll() is None:peer.terminate()
                peer.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
