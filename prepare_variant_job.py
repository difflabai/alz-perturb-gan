"""Prospectively bind one committed training candidate to canonical accounting."""
import argparse
import copy
import datetime
import json
import math
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
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--qualify-continuation', action='store_true')
    parser.add_argument('--continue-incumbent', action='store_true')
    args = parser.parse_args()
    endpoint=b.read(ROOT/'training_plan.json')['fixed_endpoint']
    b.require(endpoint in (20000,40000),'registered fixed endpoint required')
    b.require(not(args.qualify_continuation and args.continue_incumbent),'qualification is a separate finite operation')
    b.require(not args.continue_incumbent or endpoint==40000,'continuation requires prospectively fixed40k plan')
    b.require(endpoint==20000 or (args.hybrid and (args.continue_incumbent or args.seed in (1,2))),'40k scope is seed0 continuation or frozen seed1/2 confirmation')
    config = b.read(ROOT / 'candidate.json')
    b.require(set(config) == {'z_dim', 'stage_balance', 'cell_z_dim', 'theta_init_scale', 'fourier_feature_scale'} and config['z_dim'] in (2, 8)
              and type(config['stage_balance']) is bool and type(config['cell_z_dim']) is int
              and config['cell_z_dim'] in (8, 32) and type(config['theta_init_scale']) in (int, float)
              and config['theta_init_scale'] in (1., 10.) and type(config['fourier_feature_scale']) in (int, float)
              and config['fourier_feature_scale'] in (1., 5.), 'registered candidate config required')
    hybrid = b.read(ROOT / 'hybrid.json') if args.hybrid else None
    if hybrid is not None:
        fraction = hybrid.get('pg_bag_fraction')
        b.require(set(hybrid) == {'pg_bag_fraction'} and type(fraction) in (int, float)
                  and math.isfinite(fraction) and 0 <= fraction <= 1
                  and abs(200 * fraction - round(200 * fraction)) < 1e-9, 'exact whole-bag hybrid fraction required')
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
    cap = 900 if args.qualify_continuation else 21600
    operation='qualify_continuation' if args.qualify_continuation else ('continue_hybrid' if args.continue_incumbent else ('train_extended_hybrid' if endpoint==40000 else ('train_hybrid' if args.hybrid else 'train_variant')))
    auth.update(cell=value, allocation_id=b.allocation_key(value), attempt_id=b.attempt_key(value),
                label=b.attempt_key(value), scheduler_allocation_id='ALZ_CAL_TRAIN_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                expires_unix=now + cap + 30, cap_seconds=cap, operation=operation,fixed_endpoint=endpoint,initial_completed_updates=20000 if args.continue_incumbent else 0,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT, value)),
                ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT, value)),
                variant_config=config, hybrid_config=hybrid, candidate_commit=commit,qualification_only=False,
                source_review='Original science/runtime/input/guard preserved; exact optimizer/update AST and disabled-sampling original full-state parity; nineCPU parity/continuation tests including Fourier-block-only scaling and permutation invariance; source-derived15CPU ownership/accounting tests. New native candidate quality unproven.')
    if args.qualify_continuation or args.continue_incumbent:
        incumbent=b.read(ROOT/'incumbent.json')
        parent=Path(incumbent['checkpoint_path']);receipt=b.read(parent.parent/'checkpoint_receipt.json')
        b.require(args.seed==incumbent['training_seed']==0 and incumbent['completed_updates']==20000 and receipt['status']=='completed' and receipt['completed_updates']==20000,'complete retained seed0 parent required')
        normalized={key:incumbent['variant_config'].get(key,default) for key,default in [('z_dim',2),('stage_balance',False),('cell_z_dim',8),('theta_init_scale',1.),('fourier_feature_scale',1.)]}
        b.require(config==normalized and b.sha(parent)==receipt['SHA256']==incumbent['checkpoint_SHA256'],'qualification parent/config differs')
        auth.update(checkpoint_path=str(parent),checkpoint_SHA256=incumbent['checkpoint_SHA256'])
        if args.qualify_continuation:auth.update(qualification_only=True,qualification_steps=1,caller_execution_limit=40000)
    if endpoint==40000:
        proof_path=ROOT/'analysis/continuation_native_qualification.json';proof=b.read(proof_path)
        b.require(proof['status']=='VERIFIED_NATIVE_GPU0_QUALIFICATION_ONLY' and proof['clean_settlement_and_owned_release_verified'] and proof['proof']['complete_next_state_and_metrics_bitwise_equal'],'complete native continuation qualification missing')
        if args.continue_incumbent:b.require(proof['proof']['parent_checkpoint_SHA256']==auth['checkpoint_SHA256'],'parent differs from qualified complete checkpoint')
        auth['continuation_protocol_SHA256']=b.sha(ROOT/'continuation_protocol.json')
        auth['continuation_qualification_SHA256']=b.sha(proof_path)
    auth['command'] = b.expected_command(b.NATIVE_LAYOUT, auth_path, value)
    research_paths = [*ROOT.glob('*.py'), *(ROOT / 'runtime').glob('*.py'), *(ROOT / 'model').glob('*.py')]
    research = {str(p): b.sha(p) for p in research_paths}
    auth['research_source_SHA256'] = research
    pins = auth['files_SHA256']
    pins.update(research)
    if args.qualify_continuation or args.continue_incumbent:
        pins[str(parent)]=b.sha(parent)
        pins[str(parent.parent/'checkpoint_receipt.json')]=b.sha(parent.parent/'checkpoint_receipt.json')
    for p in [ROOT/'training_plan.json',ROOT/'continuation_protocol.json']:pins[str(p)]=b.sha(p)
    if endpoint==40000:pins[str(proof_path)]=b.sha(proof_path)
    pins[str(ROOT / 'candidate.json')] = b.sha(ROOT / 'candidate.json')
    if args.hybrid:
        pins[str(ROOT / 'hybrid.json')] = b.sha(ROOT / 'hybrid.json')
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
        b.core_envelope(state, cap + 30)
        b.require(control._status(state)['available_after_reservations_GPU_hours'] >= (cap + 30) / 3600,
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
              'candidate_commit': commit, 'variant_config': config, 'hybrid_config': hybrid,
              'cap_seconds': cap, 'endpoint': endpoint, 'initial_completed_updates': auth['initial_completed_updates'], 'qualification_only': args.qualify_continuation, 'source_pin_count': len(pins),
              'output_dir': auth['output_dir'], 'ledger_path': auth['ledger_path']}
    (allocation_dir / 'deployment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
