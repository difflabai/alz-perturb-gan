"""Copy the qualified scientific frame into a disabled two-device namespace.

This never edits the running confirmation or the original serial benchmark.
The copied frame still needs its new concurrency implementation and tests.
"""
import hashlib,json
from pathlib import Path
R=Path('/ml2/alz');SRC=R/'benchmark097_runtime';DST=R/'benchmark097_dual_runtime'
def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)
def main():
    assert not DST.exists()
    DST.mkdir()
    for p in SRC.glob('*.py'):
        t=p.read_text().replace("ROOT/'analysis/run_benchmark097.py'","ROOT/'analysis/run_benchmark097_dual.py'").replace("s.ROOT/'analysis/run_benchmark097.py'","s.ROOT/'analysis/run_benchmark097_dual.py'")
        if p.name=='preset_scope.py':
            t=replace(t,"RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json'","RUNTIME_PROOF=STUDY/'dual_runtime_CPU_qualification.json'")
            t=replace(t,"assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and not auth['dynamic_shared_gpu0']","from dual_coordination import assigned_gpu\n assert auth['gpu_uuid']==assigned_gpu(v) and auth['dynamic_shared_gpu0']==(auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47')")
            t=replace(t,"return [PROTOCOL,NATIVE_PROOF", "return [HERE/'policy.prepared.json',HERE/'policy.gpu0.prepared.json',HERE/'policy.gpu1.prepared.json',ROOT/'analysis/benchmark097_dual_resource_protocol.json',PROTOCOL,NATIVE_PROOF")
        if p.name=='binding.py':
            t=replace(t,"parallel_profile.REGISTRY if auth.get('parallel_profile') is True", "parallel_profile.registry_path(device(auth)) if auth.get('parallel_profile') is True")
        if p.name=='project_runner_gpu0.py':
            t=replace(t,"extra=entry.get('calibration_backend',{})", "extra=(entry.get('calibration_backend',{}) if auth.get('cpu_fixture') else entry.get('calibration_parallel_backends',{}).get(auth['allocation_id'],{}))")
            t=replace(t,"parallel_profile.free_floor()", "parallel_profile.free_floor(auth['gpu_uuid'])")
        if p.name=='prepare.py':
            t=replace(t,"profile.DEPLOYED.parent.mkdir(parents=True,exist_ok=True)","deployed=profile.deployed_path(auth['gpu_uuid'])\n        deployed.parent.mkdir(parents=True,exist_ok=True)")
            t=t.replace('profile.DEPLOYED','deployed').replace('profile.deployed_bytes()','profile.deployed_bytes(auth[\'gpu_uuid\'])')
            t=replace(t,"forecast=profile.capacity(observation,admission);profile.free_floor()", "forecast=profile.capacity(observation,admission,auth['gpu_uuid']);profile.free_floor(auth['gpu_uuid'])\n    from dual_coordination import validate_active_peers\n    validate_active_peers(b.read(b.PROJECT/'project_gpu_budget.json'),auth)")
            t=replace(t,"reg=profile.REGISTRY/'registry.json'", "reg=Path(auth['registry_path'])/'registry.json'")
            t=replace(t,"b.require(not state['open_reservations'],'Canonical reservation still active')", "validate_active_peers(state,auth)")
            t=replace(t,"state['execution_controller']['calibration_backend']=metadata;backend.controller_gate(state,auth)", "metadata.update(gpu_uuid=auth['gpu_uuid'],cell=auth['cell'],benchmark_protocol_SHA256=auth['benchmark_protocol_SHA256'],resource_protocol_SHA256=b.sha(profile.RESOURCE_PROTOCOL),ledger_path=auth['ledger_path'])\n        state['execution_controller'].setdefault('calibration_parallel_backends',{})[auth['allocation_id']]=metadata;backend.controller_gate(state,auth)")
        if p.name=='launch_core_gpu0.py':
            start=t.index('    # Same lock already held')
            end=t.index('        b.verify(auth_path,layout)',start)
            t=t[:start]+"    from dual_coordination import execution_locks\n    with execution_locks(layout.science,auth):\n"+t[end:]
        if p.name=='qualify_prepared.py':
            t=t.replace("'prepared_CPU_qualification.json'","'dual_prepared_CPU_qualification.json'")
        (DST/p.name).write_text(t)
    (DST/'plan.json').write_text(json.dumps({'enabled':False},indent=2)+'\n')
    policies=[]
    for i,gpu in enumerate(('GPU-72c1b506-891d-b8bc-b353-e020585e1c47','GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69')):
        p=json.loads((SRC/'policy.prepared.json').read_text());p['gpu']=gpu
        p['registry_path']=str(R/f'artifacts/gpu{i}-benchmark097-dual-20261107/live_registry')
        p['locks']=[s.replace('gpu-1.lock',f'gpu-{i}.lock') for s in p['locks']]
        p['resource_basis']['foreign_control']='No foreign control. One bounded ALZ cell per assigned card, joint peer-ALZ host-growth reserve, unchanged physical locks, native methods, endpoints and scoring.'
        path=DST/f'policy.gpu{i}.prepared.json';path.write_text(json.dumps(p,indent=2)+'\n');policies.append(str(path))
    (DST/'policy.prepared.json').write_text(json.dumps(dict(status='PREPARED_TWO_DEVICE_BUNDLE_NOT_ACTIVE',policies=policies),indent=2)+'\n')
    cells=[]
    for seed in range(5):
        for fold in range(3):
            for method in ('pg','scvi','cfgen'):
                index=len(cells);lane=1-index%2
                cells.append(dict(experiment='external097',fold=fold,method=method,seed=seed,device_index=lane,gpu_uuid=('GPU-72c1b506-891d-b8bc-b353-e020585e1c47','GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69')[lane]))
    resource=R/'analysis/benchmark097_dual_resource_protocol.json'
    resource.write_text(json.dumps(dict(status='PREPARED_FIXED45_CELL_DEVICE_ASSIGNMENT',cells=cells,at_most_one_ALZ_cell_per_physical_card=True,peer_ALZ_host_reserve_bytes=6*2**30,scientific_protocol_unchanged=True,all45_fresh_fits_required=True,original57_and_prespecified53_preserved=True,CPU_handoff_performed=False,plan_enabled=False,GPU_allocation_created=False),indent=2)+'\n')
    print(json.dumps(dict(status='DISABLED_DUAL_FRAME_COPIED_IMPLEMENTATION_AND_TESTS_PENDING',files=len(list(DST.glob('*.py'))),original_files_unchanged=True)))
if __name__=='__main__':main()
