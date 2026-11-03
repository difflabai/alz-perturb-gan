"""Private CPU accounting interleavings and cross-card/source regression checks."""
import copy
import fcntl
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import binding as b
import launch_core_gpu0 as owner
import test_integration as fixtures

GPU0 = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'


class ReviewRegression(unittest.TestCase):
    setUp = fixtures.Integration.setUp
    tearDown = fixtures.Integration.tearDown

    def test_cross_card_unrecorded_output_and_ledger_refuse(self):
        other=b.attempt_key(self.value).replace('CAL_GPU1_V1_','CAL_GPU0_V1_',1)
        output=self.layout.science/'runs_calibration'/other
        output.mkdir(parents=True)
        with self.assertRaisesRegex(b.Refusal,'cross-card'):
            b.unseen(self.auth,self.layout,self.state)
        output.rmdir()
        ledger=self.layout.project/'calibration_gpu0_v1_ledgers'/(other+'.json')
        fixtures.dump(ledger,{})
        with self.assertRaisesRegex(b.Refusal,'cross-card'):
            b.unseen(self.auth,self.layout,self.state)

    def test_post_reserve_core_race_cancels_owned_unstarted_before_guard(self):
        control,admission,guard=b.modules(self.auth)
        self.state['allocation_history']=[dict(allocation_id='ALZ_CAL_V1_cpu_prior',status='settled',charged_GPU_seconds=180*3600-60)]
        fixtures.dump(self.layout.project/'project_gpu_budget.json',self.state)
        original=control.Controller.reserve
        competing=[];guard_calls=[]
        def interleave(instance,**kwargs):
            own=original(instance,**kwargs)
            competing.append(original(instance,allocation_id='ALZ_CAL_V1_cpu_competing_gpu0',label='fixture_competing',
                                      owner=admission.identity(os.getpid()),machine_id=self.auth['machine_id'],gpu_uuids=[GPU0],
                                      wall_seconds=15,job_cap_seconds=15,ledger_path='cpu_competing.json'))
            return own
        with patch.object(control.Controller,'reserve',interleave),patch.object(guard,'run',lambda *a,**kw:guard_calls.append(True)):
            with self.assertRaisesRegex(b.Refusal,'180GPU-hour'):
                owner.run(self.auth_path,self.layout,observe=lambda:{})
        state=b.read(self.layout.project/'project_gpu_budget.json')
        self.assertFalse(guard_calls);self.assertIsNone(self.peer.poll())
        self.assertEqual([r['allocation_id'] for r in state['open_reservations']],['ALZ_CAL_V1_cpu_competing_gpu0'])
        control.Controller(self.layout.project).cancel_unstarted(competing[0],resource_evidence=dict(no_bound_resource_lease=True,no_worker_started=True,basis='Private CPU race fixture; no guard or worker started'))
        self.assertFalse(b.read(self.layout.project/'project_gpu_budget.json')['open_reservations'])

    def test_two_card_locks_and_both_reservation_orderings_preserve_core_envelope(self):
        control,admission,_=b.modules(self.auth)
        for order in [(GPU0,b.GPU1),(b.GPU1,GPU0)]:
            with self.subTest(first=order[0]):
                state=copy.deepcopy(self.state)
                state['allocation_history']=[dict(allocation_id='ALZ_CAL_V1_cpu_prior',status='settled',charged_GPU_seconds=180*3600-60)]
                fixtures.dump(self.layout.project/'project_gpu_budget.json',state)
                controller=control.Controller(self.layout.project)
                # Both per-card owners legally pass prechecks on the same old
                # state while holding disjoint exclusive scientific locks.
                with (self.root/'card0.lock').open('a+') as c0,(self.root/'card1.lock').open('a+') as c1:
                    fcntl.flock(c0,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c1,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    b.core_envelope(state,45);b.core_envelope(state,45)
                    first=controller.reserve(allocation_id='ALZ_CAL_V1_cpu_first',label='first',owner=admission.identity(os.getpid()),
                                             machine_id=self.auth['machine_id'],gpu_uuids=[order[0]],wall_seconds=15,job_cap_seconds=15,ledger_path='first.json')
                    with controller.transaction() as current:b.core_envelope(current,0)
                    second=controller.reserve(allocation_id='ALZ_CAL_V1_cpu_second',label='second',owner=admission.identity(os.getpid()),
                                              machine_id=self.auth['machine_id'],gpu_uuids=[order[1]],wall_seconds=15,job_cap_seconds=15,ledger_path='second.json')
                    with self.assertRaisesRegex(b.Refusal,'180GPU-hour'):
                        with controller.transaction() as current:b.core_envelope(current,0)
                    evidence=dict(no_bound_resource_lease=True,no_worker_started=True,basis='CPU fixture; no resource guard or model imports')
                    controller.cancel_unstarted(second,resource_evidence=evidence)
                    b.core_envelope(b.read(self.layout.project/'project_gpu_budget.json'),0)
                    controller.cancel_unstarted(first,resource_evidence=evidence)

    def test_parent_checkpoint_drift_after_reserve_refuses_before_child(self):
        marker=self.root/'parent.pt';marker.write_bytes(b'CPU_CHECKPOINT_FIXTURE_ONLY')
        self.auth['checkpoint_path']=str(marker);self.auth['files_SHA256'][str(marker)]=b.sha(marker)
        fixtures.dump(self.auth_path,self.auth);fixtures.dump(self.layout.project/'project_gpu_budget.json',self.state)
        _,_,guard=b.modules(self.auth)
        original=guard.run
        def changed(*args,**kwargs):
            marker.write_bytes(b'CHANGED_CPU_FIXTURE')
            return original(*args,**kwargs)
        with patch.object(guard,'run',changed):
            with self.assertRaisesRegex(b.Refusal,'reviewed code changed'):
                owner.run(self.auth_path,self.layout,observe=lambda:{})
        self.assertFalse(Path(self.auth['ledger_path']).exists())
        self.assertFalse(b.read(self.layout.project/'project_gpu_budget.json')['open_reservations'])
        self.assertIsNone(self.peer.poll())


if __name__=='__main__':
    unittest.main()
