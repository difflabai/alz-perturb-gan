"""Distinct GPU0 backend around the unchanged canonical Controller semantics."""
import fcntl
import json
import os
from pathlib import Path
import time

import binding as b

def controller_gate(state,auth):
    entry=state.get('execution_controller',{})
    b.require(entry.get('status')=='REVIEWED_FOR_BOUNDED_EXECUTION' and state.get('future_accounting_policy',{}).get('current_launch_admission') is True,'canonical controller remains held')
    extra=(entry.get('calibration_backend',{}) if auth.get('cpu_fixture') and not auth.get('fixture_dual_backend_map') else entry.get('calibration_parallel_backends',{}).get(auth['allocation_id'],{}))
    required=[*b.HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']
    b.require(extra.get('status')=='REVIEWED_FOR_BOUNDED_EXECUTION' and extra.get('path')==str(Path(__file__).resolve()) and all(extra.get('source_SHA256',{}).get(str(p))==auth['files_SHA256'].get(str(p)) for p in required),'canonical GPU0 backend source review/deployment binding absent')
    b.require(extra.get('files_SHA256')==auth['files_SHA256'] and extra.get('policy_SHA256')==auth['policy_SHA256'],'canonical reviewed full source/runtime/input/policy closure differs')
    b.require(all(extra.get(key)==auth.get(key) for key in ['allocation_id','attempt_id','scheduler_allocation_id','expires_unix']),'canonical one-use scientific/device allocation scope differs')

def run(auth_path,layout=b.NATIVE_LAYOUT,*,observe=None):
    auth,auth_sha=b.verify(auth_path,layout)
    b.require((observe is not None)==layout.fixture_mode,'injected telemetry is fixture-only and mandatory in CPU fixtures')
    controller_module,admission,guard=b.modules(auth)
    controller=controller_module.Controller(layout.project)
    ledger=Path(auth['ledger_path']);policy=b.read(auth['policy_path'])
    code_paths=[*b.HERE.glob('*.py'),b.GUARD/'gpu0_admission.py',b.GUARD/'gpu0_runner.py']
    code_paths+=list(b.read(layout.science/'REVIEW_MANIFEST.json')['source_SHA256'])
    code_paths+=list(auth.get('research_source_SHA256',{}))
    def protect():
        b.require(b.sha(auth_path)==auth_sha,'one-use authorization changed during execution')
        if auth.get('parallel_profile') is True and not layout.fixture_mode:
            import sys
            sys.path.insert(0,str(b.HERE))
            import resource_profile as parallel_profile
            parallel_profile.validate(auth)
            parallel_profile.free_floor(auth['gpu_uuid'])
        for path in code_paths:b.require(b.sha(path)==auth['files_SHA256'][str(path)],'reviewed code changed during owned execution: '+str(path))
        if auth.get('dynamic_shared_foreign') is not True:
            for peer in auth['protected_peers']:b.require(admission.protected_identity(peer['identity']['pid'])==peer['identity'],'protected peer identity changed')
    protect()
    ledger.parent.mkdir(parents=True,exist_ok=True)
    # The outer owner holds the existing scientific core_phase.lock throughout.
    # This lock remains distinct from the unchanged Controller and guard locks.
    with ledger.with_suffix('.project-owner.lock').open('a+') as owner_lock:
        fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with controller.transaction() as state:
            controller_gate(state,auth);b.unseen(auth,layout,state)
            b.core_envelope(state,auth['cap_seconds']+30)
        reservation=controller.reserve(allocation_id=auth['allocation_id'],label=auth['label'],owner=admission.identity(os.getpid()),machine_id=auth['machine_id'],gpu_uuids=[b.device(auth)],wall_seconds=auth['cap_seconds'],job_cap_seconds=auth['cap_seconds'],ledger_path=str(ledger.relative_to(layout.project)))
        result=None;error=None
        old={k:os.environ.get(k) for k in ['ALZ_PROJECT_RESERVATION_TOKEN','ALZ_PROJECT_BUDGET_ROOT']}
        try:
            with controller.transaction() as state:
                controller_gate(state,auth);b.core_envelope(state,0)
            os.environ['ALZ_PROJECT_RESERVATION_TOKEN']=reservation['token']
            os.environ['ALZ_PROJECT_BUDGET_ROOT']=str(layout.project)
            result=guard.run(ledger,b.device(auth),auth['label'],auth['command'],cap_seconds=auth['cap_seconds'],policy_path=auth['policy_path'],role=auth['role'],protect=protect,absolute_deadline=auth['expires_unix'],observe=observe,test_mode=layout.fixture_mode,expected_policy_sha256=auth['policy_SHA256'])
            return result
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            for key,value in old.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            settlement=None;uncertain=None
            try:
                if ledger.exists():settlement=controller.settle(reservation)
                else:
                    registry_path=Path(policy['registry_path'])/'registry.json'
                    registry=b.read(registry_path) if registry_path.exists() else {'jobs':{}}
                    owner=reservation['owner']
                    own=[j for j in registry['jobs'].values() if j.get('owner')==owner and j.get('role')==auth['role']]
                    if own:raise controller_module.Uncertain('resource lease exists without a closed ledger')
                    controller.cancel_unstarted(reservation,resource_evidence={'no_bound_resource_lease':True,'no_worker_started':True,'basis':'versioned guard records ledger before releasing registered worker barrier; no matching resource lease exists'})
                    settlement={'status':'cancelled_before_execution','charged_GPU_seconds':0}
            except BaseException as exc:
                uncertain=repr(exc);controller.uncertain(reservation,uncertain)
            controller_module.atomic(ledger.with_name(auth['attempt_id']+'-settlement.json'),{'checked_unix':time.time(),'allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'gpu_uuid':b.device(auth),'guard_return_code':result,'guard_error':error,'settlement':settlement,'uncertain':uncertain,'original_job_cap_seconds':auth['cap_seconds'],'no_foreign_processes_signalled':True})
            if uncertain:raise controller_module.Uncertain('Project reservation retained: '+uncertain)
