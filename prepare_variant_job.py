"""Prospectively bind one committed training candidate to canonical accounting."""
import argparse
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/ml2/alz')
sys.path.insert(0, str(ROOT / 'runtime'))
import binding as b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    config = b.read(ROOT / 'candidate.json')
    b.require(set(config) == {'z_dim', 'stage_balance'} and config['z_dim'] in (2, 8)
              and type(config['stage_balance']) is bool, 'registered candidate config required')
    b.require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
              'commit the complete candidate before deploying an experiment')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    previous = b.read(ROOT / 'calibration_deployment.json')
    auth = copy.deepcopy(b.read(previous['authorization_path']))
    value = {'fold': 0, 'method': 'pg', 'seed': args.seed, 'experiment': args.experiment}
    b.cell({'cell': value, 'cpu_fixture': False})
    allocation_dir = ROOT / 'artifacts/allocations' / b.attempt_key(value)
    auth_path = allocation_dir / 'authorization.json'
    b.require(not allocation_dir.exists(), 'experiment allocation artifact already exists; inspect rather than retry')
    now = time.time()
    auth.update(cell=value, allocation_id=b.allocation_key(value), attempt_id=b.attempt_key(value),
                label=b.attempt_key(value), scheduler_allocation_id='ALZ_CAL_TRAIN_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                expires_unix=now + 21630, cap_seconds=21600, operation='train_variant',
                output_dir=str(b.output_path(b.NATIVE_LAYOUT, value)),
                ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT, value)),
                variant_config=config, candidate_commit=commit,
                source_review='Original science/runtime/input/guard preserved; exact optimizer/update AST and disabled-sampling original full-state parity; balanced8bags-per-stage test, complete continuation and sampling-config mismatch rejection; source-derived15CPU ownership/accounting tests. New native candidate quality unproven.')
    auth['command'] = b.expected_command(b.NATIVE_LAYOUT, auth_path, value)
    research_paths = [*ROOT.glob('*.py'), *(ROOT / 'runtime').glob('*.py'), *(ROOT / 'model').glob('*.py')]
    research = {str(p): b.sha(p) for p in research_paths}
    auth['research_source_SHA256'] = research
    pins = auth['files_SHA256']
    pins.update(research)
    pins[str(ROOT / 'candidate.json')] = b.sha(ROOT / 'candidate.json')
    for path in [ROOT / 'incumbent.json', ROOT / 'ITERATION_AUTHORIZATION_20261103.json']:
        pins[str(path)] = b.sha(path)
    b.source_closure(auth, b.NATIVE_LAYOUT)
    controller, admission, _ = b.modules(auth)
    observation = admission.snapshot()
    policy = b.read(auth['policy_path'])
    b.require(set(observation['gpu_memory']) == {p['identity']['pid'] for p in policy['protected_peers']},
              'unknown or missing GPU0 process; reconcile before admission')
    for peer in policy['protected_peers']:
        pid = peer['identity']['pid']
        b.require(admission.protected_identity(pid) == peer['identity']
                  and set(observation['gpu_kinds'][pid]) == set(peer['gpu_kinds']),
                  'protected GPU0 identity/membership changed')
    b.require(not b.read(Path(auth['registry_path']) / 'registry.json')['jobs'], 'existing resource lease')
    backend = b.load_pinned(ROOT / 'runtime/project_runner_gpu0.py', research[str(ROOT / 'runtime/project_runner_gpu0.py')], 'project_runner_gpu0')
    metadata = {'status': 'REVIEWED_FOR_BOUNDED_EXECUTION',
                'path': str(ROOT / 'runtime/project_runner_gpu0.py'),
                'source_SHA256': {str(p): pins[str(p)] for p in [*(ROOT / 'runtime').glob('*.py'), b.GUARD / 'gpu0_admission.py', b.GUARD / 'gpu0_runner.py']},
                'files_SHA256': pins, 'policy_SHA256': auth['policy_SHA256'],
                **{k: auth[k] for k in ['allocation_id', 'attempt_id', 'scheduler_allocation_id', 'expires_unix']}}
    control = controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'], 'existing canonical reservation')
        b.unseen(auth, b.NATIVE_LAYOUT, state)
        b.core_envelope(state, 21630)
        b.require(control._status(state)['available_after_reservations_GPU_hours'] >= 21630 / 3600,
                  'insufficient project hours')
        state['future_accounting_policy']['current_launch_admission'] = True
        state['execution_controller']['status'] = 'REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend'] = metadata
        backend.controller_gate(state, auth)
    allocation_dir.mkdir(parents=True, exist_ok=False)
    auth_path.write_text(json.dumps(auth, indent=2, allow_nan=False) + '\n')
    b.verify(auth_path)
    record = {'status': 'ONE_USE_COMMITTED_CANDIDATE_DEPLOYED_NO_JOB_STARTED',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'authorization_path': str(auth_path), 'authorization_SHA256': b.sha(auth_path),
              'allocation_id': auth['allocation_id'], 'attempt_id': auth['attempt_id'],
              'candidate_commit': commit, 'variant_config': config,
              'cap_seconds': 21600, 'endpoint': 20000, 'source_pin_count': len(pins),
              'output_dir': auth['output_dir'], 'ledger_path': auth['ledger_path']}
    (allocation_dir / 'deployment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
