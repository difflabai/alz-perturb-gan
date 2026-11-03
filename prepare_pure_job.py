"""Bind one finite, explicitly labeled PG-only retained-checkpoint evaluation."""
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
    parser.add_argument('--formulation', required=True)
    args = parser.parse_args()
    from pure_suite import load_entry
    base=load_entry(args.formulation)
    b.require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
              'commit the complete pure evaluation source before deployment')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    checkpoint = Path(base['checkpoint_path'])
    receipt = b.read(checkpoint.parent / 'checkpoint_receipt.json')
    b.require(receipt['status'] == 'completed' and receipt['completed_updates'] == base['completed_updates']
              and b.sha(checkpoint) == receipt['SHA256'] == base['checkpoint_SHA256'], 'complete base checkpoint required')
    previous = b.read(ROOT / 'calibration_deployment.json')
    auth = copy.deepcopy(b.read(previous['authorization_path']))
    value = {'fold': 0, 'method': 'pg', 'seed': base['training_seed'], 'experiment': args.experiment}
    b.cell({'cell': value, 'cpu_fixture': False})
    folder = ROOT / 'artifacts/allocations' / b.attempt_key(value)
    b.require(not folder.exists(), 'existing allocation artifact; inspect instead of duplicating')
    auth_path = folder / 'authorization.json'
    auth.update(cell=value, allocation_id=b.allocation_key(value), attempt_id=b.attempt_key(value), label=b.attempt_key(value),
                scheduler_allocation_id='ALZ_PURE_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                expires_unix=time.time() + 930, cap_seconds=900, operation='evaluate_pure_variant', fixed_endpoint=base['completed_updates'],
                output_dir=str(b.output_path(b.NATIVE_LAYOUT, value)), ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT, value)),
                variant_config=base['variant_config'], hybrid_config=None, candidate_commit=commit, pure_formulation=args.formulation, pure_manifest_SHA256=b.sha(ROOT/'pure_formulations.json'),
                checkpoint_path=str(checkpoint), checkpoint_SHA256=base['checkpoint_SHA256'],
                qualification_only=True,
                source_review='PG-only direct iter_pg, zero real-bag replacements; finite retained complete checkpoint/config bound by pure manifest. CPU binding regressions and control replay required. No candidate ranking against hybrid incumbent.')
    auth['command'] = b.expected_command(b.NATIVE_LAYOUT, auth_path, value)
    research = {str(p): b.sha(p) for p in [*ROOT.glob('*.py'), *(ROOT / 'runtime').glob('*.py'), *(ROOT / 'model').glob('*.py')]}
    auth['research_source_SHA256'] = research
    pins = auth['files_SHA256']
    pins.update(research)
    for path in [checkpoint, checkpoint.parent/'checkpoint_receipt.json', ROOT / 'pure_formulations.json', ROOT / 'incumbent.json', ROOT / 'ITERATION_AUTHORIZATION_20261103.json']:
        pins[str(path)] = b.sha(path)
    b.source_closure(auth, b.NATIVE_LAYOUT)
    controller, admission, _ = b.modules(auth)
    observation = admission.snapshot()
    policy = b.read(auth['policy_path'])
    b.require(set(observation['gpu_memory']) == {p['identity']['pid'] for p in policy['protected_peers']}, 'unknown or missing GPU0 peer')
    for peer in policy['protected_peers']:
        pid = peer['identity']['pid']
        b.require(admission.protected_identity(pid) == peer['identity']
                  and set(observation['gpu_kinds'][pid]) == set(peer['gpu_kinds']), 'protected identity/membership changed')
    b.require(not b.read(Path(auth['registry_path']) / 'registry.json')['jobs'], 'resource registry has active work')
    backend = b.load_pinned(ROOT / 'runtime/project_runner_gpu0.py', research[str(ROOT / 'runtime/project_runner_gpu0.py')], 'project_runner_gpu0')
    metadata = {'status': 'REVIEWED_FOR_BOUNDED_EXECUTION', 'path': str(ROOT / 'runtime/project_runner_gpu0.py'),
                'source_SHA256': {str(p): pins[str(p)] for p in [*(ROOT / 'runtime').glob('*.py'), b.GUARD / 'gpu0_admission.py', b.GUARD / 'gpu0_runner.py']},
                'files_SHA256': pins, 'policy_SHA256': auth['policy_SHA256'],
                **{k: auth[k] for k in ['allocation_id', 'attempt_id', 'scheduler_allocation_id', 'expires_unix']}}
    control = controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'], 'another canonical reservation is active')
        b.unseen(auth, b.NATIVE_LAYOUT, state); b.core_envelope(state, 930)
        b.require(control._status(state)['available_after_reservations_GPU_hours'] >= 930 / 3600, 'insufficient project hours')
        state['future_accounting_policy']['current_launch_admission'] = True
        state['execution_controller']['status'] = 'REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend'] = metadata
        backend.controller_gate(state, auth)
    folder.mkdir(parents=True, exist_ok=False)
    auth_path.write_text(json.dumps(auth, indent=2, allow_nan=False) + '\n')
    b.verify(auth_path)
    result = {'status': 'ONE_USE_PURE_SAMPLING_DEPLOYED_NO_JOB_STARTED', 'authorization_path': str(auth_path),
              'authorization_SHA256': b.sha(auth_path), 'allocation_id': auth['allocation_id'], 'attempt_id': auth['attempt_id'],
              'candidate_commit': commit, 'variant_config': base['variant_config'], 'hybrid_config': None, 'pure_formulation':args.formulation, 'endpoint':base['completed_updates'],
              'qualification_only': True, 'output_dir': auth['output_dir'], 'ledger_path': auth['ledger_path'],
              'cap_seconds': 900, 'source_pin_count': len(pins), 'base_model_updates': base['completed_updates'], 'new_model_updates': 0}
    (folder / 'deployment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
