"""Prepare review evidence only; no leases, canonical writes or native launch."""
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import binding as b


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    budget = b.PROJECT / 'project_gpu_budget.json'
    before = b.sha(budget)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ROOT / 'artifacts/gpu1-held' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    previous = b.read(ROOT / 'calibration_deployment.json')
    auth = copy.deepcopy(b.read(previous['authorization_path']))
    policy = copy.deepcopy(b.read(auth['policy_path']))
    base = b.read(ROOT / 'hybrid_base.json')
    checkpoint = Path(base['checkpoint_path'])
    b.require(b.sha(checkpoint) == base['checkpoint_SHA256'], 'retained checkpoint changed')
    spec = importlib.util.spec_from_file_location('held_gpu1_observer', HERE / 'guard/gpu0_admission.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    snapshot = admission.snapshot()
    observation = {k: v for k, v in snapshot.items() if k != 'procs'}
    observation['identities'] = {pid: admission.protected_identity(pid) for pid in snapshot['gpu_memory']}
    observation['utc'] = stamp
    locks = []
    rows = Path('/proc/locks').read_text().splitlines()
    for name in admission.LOCKS:
        path = Path(name)
        inode = path.stat().st_ino if path.exists() else None
        locks.append({'path': name, 'inode': inode,
                      'observed_records': [row for row in rows if inode is not None and any(token.endswith(':' + str(inode)) for token in row.split())]})
    observation['physical_locks'] = locks
    write(folder / 'observation.json', observation)
    # This is a prospective allowlist for a free-card handoff. Live MOSS is
    # recorded above, never silently converted into an approved shared peer.
    peers = [p for p in policy['protected_peers'] if p['identity']['pid'] == 9387]
    b.require(len(peers) == 1 and admission.protected_identity(9387) == peers[0]['identity'], 'prospective desktop identity changed')
    foreign_compute = [pid for pid, kinds in snapshot['gpu_kinds'].items() if 'compute' in kinds]
    policy.update(schema='gpu1-coexistence-policy-v1', status='HELD_GPU1_NOT_DEPLOYED', gpu=b.GPU1,
                  registry_path=str(b.PROJECT / 'cooperative_gpu1_calibration_v1/live_registry'),
                  locks=list(admission.LOCKS), protected_peers=peers,
                  host_free_floor_bytes=38 * 1024**3,
                  prepared_note='Held prospective free-GPU1 policy. Requires explicit owner handoff, current source review, fresh peer/capacity verification and separate deployment. No allocation or registry created.')
    policy.pop('deployed_unix', None)
    policy.pop('snapshot_capacity_arithmetic', None)
    policy['source_sha256'] = {str(p): b.sha(p) for p in [*HERE.glob('*.py'), *(HERE / 'guard').glob('*.py')]}
    policy['resource_basis'] = {'status': 'CONSERVATIVE_PROPOSAL_NOT_GPU1_MEASUREMENT',
                              'own_GPU_bytes': 4 * 1024**3, 'own_host_RAM_bytes': 6 * 1024**3,
                              'host_floor_basis': '4GiB floor + 6GiB concurrent GPU0 ALZ + 7 other GPU0 peer forecasts at4GiB; shared gnome desktop separately reserved once onGPU1.',
                              'GPU0_observation': 'ALZ samples about1.08GiB GPU/1.8GiB host; GPU1 still unqualified.'}
    write(folder / 'policy.HELD.json', policy)
    value = dict(fold=0, method='pg', seed=base['training_seed'], experiment='gpu1_hybrid75_qualification_v1')
    research = {str(p): b.sha(p) for p in [*ROOT.glob('*.py'), *HERE.glob('*.py'), *(ROOT / 'model').glob('*.py')]}
    pins = auth['files_SHA256']
    pins.update(research)
    pins.update(policy['source_sha256'])
    pins[str(HERE / 'guard/REVIEW_MANIFEST.json')] = b.sha(HERE / 'guard/REVIEW_MANIFEST.json')
    pins[str(folder / 'policy.HELD.json')] = b.sha(folder / 'policy.HELD.json')
    for path in [checkpoint, ROOT / 'hybrid_base.json', ROOT / 'hybrid.json', ROOT / 'incumbent.json']:
        pins[str(path)] = b.sha(path)
    auth.update(status='HELD_GPU1_SOURCE_CAPACITY_AND_HANDOFF', schema='alz-calibration-gpu1-held-v1',
                source_review_passed=False, scheduler_allocation_id=None, expires_unix=0,
                cell=value, allocation_id=b.allocation_key(value), attempt_id=b.attempt_key(value), label=b.attempt_key(value),
                gpu_uuid=b.GPU1, cap_seconds=900, operation='evaluate_hybrid', qualification_only=True,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT, value)), ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT, value)),
                checkpoint_path=str(checkpoint), checkpoint_SHA256=base['checkpoint_SHA256'],
                variant_config=base['variant_config'], hybrid_config=b.read(ROOT / 'hybrid.json'),
                policy_path=str(folder / 'policy.HELD.json'), policy_SHA256=b.sha(folder / 'policy.HELD.json'),
                registry_path=policy['registry_path'], protected_peers=peers, research_source_SHA256=research,
                candidate_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_review='HELD. CPU preparation is not independent source review or native admission. Refresh all pins after active candidate finishes.')
    auth['command'] = b.expected_command(b.NATIVE_LAYOUT, folder / 'authorization.HELD.json', value)
    write(folder / 'authorization.HELD.json', auth)
    try:
        b.verify(folder / 'authorization.HELD.json')
    except b.Refusal as error:
        rejection = str(error)
    else:
        raise RuntimeError('Held packet unexpectedly admitted')
    b.require(b.sha(budget) == before, 'canonical budget changed during read-only packet preparation; reconcile concurrent change')
    result = {'status': 'HELD_REVIEW_PACKET_ONLY', 'utc': stamp, 'path': str(folder),
              'GPU1_native_qualified': False, 'job_started': False, 'reservation_started': False,
              'policy_deployed': False, 'canonical_backend_deployed': False,
              'canonical_budget_SHA256_before_and_after': before,
              'held_launch_rejected': rejection, 'current_foreign_compute_PIDs': foreign_compute,
              'next_task': 'Retained donor8/cell8 checkpoint, hybrid75, two fixed calibration replicates, no fitting;900s cap.',
              'required_before_launch': ['Explicit safe-boundary handoff from current GPU1 owner through Ember',
                                         'Review current exact source closure and fresh peer/capacity/physical lease observations',
                                         'Deploy exact GPU1 policy, empty registry and one-use controller backend; never reuse held authorization',
                                         'Create fresh unexpired finite authorization and launch through unchanged canonical accounting']}
    write(folder / 'REVIEW_PACKET.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
