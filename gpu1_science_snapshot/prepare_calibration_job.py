"""Deploy one finite calibration sampling allocation after the reference settles."""
import copy
import datetime
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/ml2/alz')
RUNTIME = ROOT / 'runtime'
ORIGINAL = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration')
sys.path.insert(0, str(RUNTIME))
import binding as b


def main():
    original_deployment = b.read(ORIGINAL.parent.parent / 'alz-gpu0-one-use-deployment.json')
    original = b.read(original_deployment['authorization_path'])
    reference_output = Path(original['output_dir'])
    reference = b.read(reference_output / 'completed.json')
    b.require(reference['status'] == 'completed' and reference['endpoint'] == 20000,
              'fixed reference must complete before calibration sampling')
    checkpoint = reference_output / 'fit/final.pt'
    receipt = b.read(reference_output / 'fit/checkpoint_receipt.json')
    b.require(receipt['completed_updates'] == 20000 and receipt['status'] == 'completed'
              and b.sha(checkpoint) == receipt['SHA256'], 'reference checkpoint integrity differs')
    original_settlement = b.read(Path(original['ledger_path']).with_name(original['attempt_id'] + '-settlement.json'))
    b.require(original_settlement['settlement']['status'] == 'settled'
              and original_settlement['uncertain'] is None, 'reference cleanup/settlement uncertain')
    value = {'fold': 0, 'method': 'pg', 'seed': 0, 'experiment': 'reference_eval_000'}
    out = ROOT / 'artifacts/allocations' / b.attempt_key(value)
    out.mkdir(parents=True, exist_ok=False)
    auth_path = out / 'authorization.json'
    auth = copy.deepcopy(original)
    now = time.time()
    scheduler_id = 'ALZ_CAL_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    auth.update(status='APPROVED_FOR_BOUNDED_EXECUTION', source_review_passed=True,
                cell=value, allocation_id=b.allocation_key(value), attempt_id=b.attempt_key(value),
                label=b.attempt_key(value), scheduler_allocation_id=scheduler_id,
                operation='evaluate_reference', cap_seconds=900, expires_unix=now + 930,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT, value)),
                ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT, value)),
                checkpoint_path=str(checkpoint), checkpoint_SHA256=receipt['SHA256'],
                calibration_manifest_SHA256=b.sha(ROOT / 'data/calibration_fold0/manifest.json'),
                authority='Current user goal: iterate evaluation/debug/commits toward calibration energy score and distribution match; both GPUs authorized; preserve canonical1024GPUh cap.',
                source_review='Original unchanged scientific/runtime/input/guard closure plus local calibration source and source-derived15CPUownership/accounting tests. New native sampler not yet qualified; report outcome explicitly.')
    auth['command'] = b.expected_command(b.NATIVE_LAYOUT, auth_path, value)
    research = {str(p): b.sha(p) for p in [*ROOT.glob('*.py'), *RUNTIME.glob('*.py')]}
    auth['research_source_SHA256'] = research
    pins = auth['files_SHA256']
    pins.update(research)
    for p in (ROOT / 'data/calibration_fold0').iterdir():
        if p.is_file():
            pins[str(p)] = b.sha(p)
    for p in [checkpoint, ROOT / 'experiment_protocol.json', ROOT / 'OWNER_AUTHORIZATION_20261103.json']:
        pins[str(p)] = b.sha(p)
    b.source_closure(auth, b.NATIVE_LAYOUT)
    controller, admission, _ = b.modules(auth)
    observation = admission.snapshot()
    policy = b.read(auth['policy_path'])
    b.require(set(observation['gpu_memory']) == {p['identity']['pid'] for p in policy['protected_peers']},
              'unknown or missing GPU0 process; wait or reconcile before admission')
    for peer in policy['protected_peers']:
        pid = peer['identity']['pid']
        b.require(admission.protected_identity(pid) == peer['identity']
                  and set(observation['gpu_kinds'][pid]) == set(peer['gpu_kinds']),
                  'protected GPU0 process identity/membership changed')
    registry = b.read(Path(auth['registry_path']) / 'registry.json')
    b.require(not registry['jobs'], 'resource registry is not empty')
    backend = b.load_pinned(RUNTIME / 'project_runner_gpu0.py', research[str(RUNTIME / 'project_runner_gpu0.py')], 'project_runner_gpu0')
    metadata = {'status': 'REVIEWED_FOR_BOUNDED_EXECUTION',
                'path': str(RUNTIME / 'project_runner_gpu0.py'),
                'source_SHA256': {str(p): pins[str(p)] for p in [*RUNTIME.glob('*.py'), b.GUARD / 'gpu0_admission.py', b.GUARD / 'gpu0_runner.py']},
                'files_SHA256': pins, 'policy_SHA256': auth['policy_SHA256'],
                **{k: auth[k] for k in ['allocation_id', 'attempt_id', 'scheduler_allocation_id', 'expires_unix']}}
    control = controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'], 'another canonical job is reserved')
        b.unseen(auth, b.NATIVE_LAYOUT, state)
        b.core_envelope(state, 930)
        b.require(control._status(state)['available_after_reservations_GPU_hours'] >= 930 / 3600,
                  'insufficient canonical budget')
        state['future_accounting_policy']['current_launch_admission'] = True
        state['execution_controller']['status'] = 'REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend'] = metadata
        backend.controller_gate(state, auth)
    auth_path.write_text(json.dumps(auth, indent=2, allow_nan=False) + '\n')
    b.verify(auth_path)
    report = {'status': 'ONE_USE_CALIBRATION_METADATA_DEPLOYED_NO_JOB_STARTED',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'authorization_path': str(auth_path), 'authorization_SHA256': b.sha(auth_path),
              'allocation_id': auth['allocation_id'], 'attempt_id': auth['attempt_id'],
              'source_pin_count': len(pins), 'checkpoint_SHA256': receipt['SHA256'],
              'cap_seconds': 900, 'protected_peers_verified': len(policy['protected_peers']),
              'native_sampler_previously_qualified': False}
    (ROOT / 'calibration_deployment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
