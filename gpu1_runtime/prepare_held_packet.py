"""Freeze a literal-source, sharing-compatible review capsule; never deploy."""
import copy
import argparse
import datetime
import difflib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import binding as b

MOSS = Path('/ml2/particlesliders-working/moss_training')
GiB = 1024**3


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--free-card',action='store_true');args=parser.parse_args()
    budget = b.PROJECT / 'project_gpu_budget.json'
    before = b.sha(budget)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ROOT / 'artifacts/gpu1-sharing-capsules' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    auth = copy.deepcopy(b.read(b.read(ROOT / 'calibration_deployment.json')['authorization_path']))
    gpu0_policy = b.read(auth['policy_path'])
    base = b.read(HERE / 'qualification_base.json')
    sampler = b.read(HERE / 'qualification_sampler.json')
    checkpoint = Path(base['checkpoint_path'])
    b.require(b.sha(checkpoint) == base['checkpoint_SHA256'], 'retained checkpoint changed')
    for path, record in b.read(b.RESEARCH / 'SOURCE_ORIGINS.json')['files'].items():
        b.require(b.sha(path) == record['SHA256'], 'frozen science source changed')
    sharing = b.read(MOSS / 'gpu1-sharing.json')
    b.require(sharing['gpu_uuid'] == b.GPU1 and sharing['canonical_lock_mode'] == 'shared-read'
              and sharing['external_peer_gpu_MiB_cap'] == 4096 and sharing['owned_gpu_MiB_cap'] == 40960
              and sharing['owned_RSS_bytes_cap'] == 40 * GiB, 'published MOSS sharing envelope changed')
    spec = importlib.util.spec_from_file_location('held_gpu1_observer', HERE / 'guard/gpu0_admission.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    obs = admission.snapshot()
    desktop_samples=[obs['gpu_memory'].get(9387,0)]
    members=set(obs['gpu_memory'])
    for _ in range(2):
        time.sleep(.1)
        fresh=admission.snapshot()
        b.require(set(fresh['gpu_memory'])==members,'GPU1 peer turnover during desktop measurement; refresh held snapshot')
        desktop_samples.append(fresh['gpu_memory'].get(9387,0));obs=fresh
    compute = [pid for pid, kinds in obs['gpu_kinds'].items() if 'compute' in kinds]
    if args.free_card:
        b.require(not compute,'free-card profile requires zero current COMPUTE peers; never use stale or invented MOSS identity')
    else:
        b.require(len(compute)==1,'one exact stable MOSS compute peer required for sharing snapshot')
        moss_pid=compute[0];proc=Path('/proc')/str(moss_pid)
        argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        b.require(proc.joinpath('cwd').resolve()==MOSS and str(MOSS/'run_shared_pair.py') in argv,'GPU1 compute peer is not the coordinated MOSS sharing worker')
    gnome = [copy.deepcopy(p) for p in gpu0_policy['protected_peers'] if p['identity']['pid'] == 9387]
    b.require(len(gnome) == 1 and admission.protected_identity(9387) == gnome[0]['identity'], 'desktop identity changed')
    gnome[0]['gpu_kinds'] = list(obs['gpu_kinds'][9387])
    gnome[0]['gpu_budget_bytes'] = max(GiB//4,max(desktop_samples)+GiB//8)
    peers=gnome if args.free_card else [*gnome,{'identity':admission.protected_identity(moss_pid),'gpu_kinds':list(obs['gpu_kinds'][moss_pid]),'gpu_budget_bytes':40*GiB,'ram_budget_bytes':40*GiB}]
    growth = []
    for peer in gpu0_policy['protected_peers']:
        pid = peer['identity']['pid']
        if pid == 9387:
            continue
        b.require(admission.protected_identity(pid) == peer['identity'], 'GPU0 protected identity changed during host forecast')
        current = obs['procs'][pid]
        remaining = max(0, peer['ram_budget_bytes'] - current['rss'], current['hwm'] - current['rss'])
        growth.append({'identity': peer['identity'], 'rss': current['rss'], 'hwm': current['hwm'],
                       'reviewed_RAM_forecast': peer['ram_budget_bytes'], 'remaining_growth': remaining})
    # Reserve a future/concurrent GPU0 ALZ6GiB in full plus actual remaining
    # desktop growth, rather than double-counting already-resident peer RSS.
    host_floor = 8 * GiB + 6 * GiB + sum(q['remaining_growth'] for q in growth)
    policy = {'schema': 'gpu1-coexistence-policy-v1', 'status': 'HELD_GPU1_NOT_DEPLOYED', 'gpu': b.GPU1,
              'registry_path': str(b.PROJECT / 'cooperative_gpu1_calibration_v1/live_registry'),
              'locks': list(admission.LOCKS), 'alz_role_lock': '/mnt/nvme/sliders/locks/gpu-1-alz-training.lock',
              'driver_headroom_bytes': GiB, 'host_free_floor_bytes': host_floor,
              'roles': {'alz_core_pg': {'gpu_bytes': 3 * GiB, 'ram_bytes': 6 * GiB}}, 'protected_peers': peers,
              'sharing': dict(admission_mode='free_card_exact_peers' if args.free_card else 'with_moss',external_compute_budget_bytes=15*GiB//4,moss_compute_pids=compute,minimum_available_ram_bytes=8*GiB,enforce_own_rss=True,enforce_host_growth=True),
              'protected_host_peak_pids':[q['identity']['pid'] for q in growth],
              'protected_host_peak_identities':[q['identity'] for q in growth],
              'source_sha256': {str(p): b.sha(p) for p in [*HERE.glob('*.py'), *(HERE / 'guard').glob('*.py')]},
              'prepared_note': 'HELD exact-peer source proposal (free-card or active-MOSS profile). No deployment/registration/reservation/job. Exact current MOSS membership and all source/capacity gates must be refreshed after independent review.'}
    overhead = max(0, obs['used'] - sum(obs['gpu_memory'].values()))
    gpu_need = max(obs['used'], sum(obs['gpu_memory'].values())) + max(0, GiB - overhead) + 3 * GiB
    gpu_need += sum(max(0, p['gpu_budget_bytes'] - obs['gpu_memory'][p['identity']['pid']]) for p in peers)
    ram_need = 6 * GiB + host_floor + sum(max(0, p['ram_budget_bytes'] - obs['procs'][p['identity']['pid']]['rss'],
                                                      obs['procs'][p['identity']['pid']]['hwm'] - obs['procs'][p['identity']['pid']]['rss']) for p in peers)
    host_peak_extra=sum(max(0,q['hwm']-q['rss']) for q in growth)
    ram_need+=host_peak_extra
    basis = {'status': 'READ_ONLY_SHARING_FORECAST_NOT_ADMISSION', 'utc': stamp, 'admission_mode':'free_card_exact_peers' if args.free_card else 'with_moss', 'future_unknown_compute_peer':'Refuse/stop only owned ALZ on any new unallowlisted COMPUTE identity; no MOSS future-memory claim in absent-peer profile.',
             'MOSS_published_policy_path': str(MOSS / 'gpu1-sharing.json'), 'MOSS_published_policy_SHA256': b.sha(MOSS / 'gpu1-sharing.json'),
             'MOSS_shared_source_SHA256': b.sha(MOSS / 'shared_gpu.py'),
             'desktop_measurement':{'observed_GPU_bytes':desktop_samples,'observed_peak_bytes':max(desktop_samples),'forecast_GPU_bytes':gnome[0]['gpu_budget_bytes'],'forecast_rule':'max256MiB, measuredpeak+128MiB; reserve is a conservative forecast, not a measured peak.'},
             'authority_provenance':'Own currentuser explicitly permitsbothGPUs; sharing coordination follows authenticatedEmber9473. The Mikkel quote inMOSS JSON is publisher provenance, not a directmessage independentlyseenhere.',
             'compatibility': 'Free-card mode pins current graphics only and refuses/stops owned ALZ on any unexpected compute peer. Active-MOSS mode: MOSS shared_gpu.Peers dynamically pins new same-UID compute peers and caps external compute at4GiB; ALZ3GiB plus desktop.25GiB forecasts fit even if graphics are conservatively included. Both physical leasesREAD; distinct exclusiveALZrole. Never signal/adopt MOSS.',
             'GPU_required_bytes': gpu_need, 'GPU_total_bytes': obs['total'], 'GPU_fits_snapshot': gpu_need <= obs['total'],
             'host_required_available_bytes': ram_need, 'host_peak_extra_bytes':host_peak_extra,
             'observed_protected_RAM':[{**q['identity'],'rss':obs['procs'][q['identity']['pid']]['rss'],'hwm':obs['procs'][q['identity']['pid']]['hwm']} for q in peers], 'host_available_bytes': obs['available_ram'], 'host_fits_snapshot': ram_need <= obs['available_ram'],
             'enforcement':'Native sharingguard stops ownedALZ on externalCOMPUTE actual/projected>3.75GiB, ownRSSbudgetbreach, MemAvailable<8GiB or hostgrowthforecastfailure. Graphics-only desktop excluded fromMOSScompute cap, separatelyforecast; fresh exact hostpeak identities reserveadditionalHWM growth.',
             'host_floor_basis': {'minimum_host_floor_bytes': 8 * GiB, 'future_or_concurrent_GPU0_ALZ_RAM_bytes': 6 * GiB,
                                 'GPU0_other_protected_remaining_growth': growth},
             'native_GPU1_qualified': False, 'foreign_worker_turnover': 'Any exact peer membership/identity change refuses/stops owned ALZ; refresh finite attempt after review, never relax peer gates.'}
    write(folder / 'resource_basis.json', basis)
    policy['resource_basis'] = basis
    write(folder / 'policy.HELD.json', policy)
    observation = {k: v for k, v in obs.items() if k != 'procs'}
    observation['identities'] = {pid: admission.protected_identity(pid) for pid in obs['gpu_memory']}
    records = Path('/proc/locks').read_text().splitlines()
    observation['physical_locks'] = [{'path': name, 'records': [line for line in records if any(token.endswith(':' + str(Path(name).stat().st_ino)) for token in line.split())]}
                                    for name in admission.LOCKS]
    write(folder / 'observation.json', observation)
    value = dict(fold=0, method='pg', seed=0, experiment='gpu1_free_h75_qualification_v1' if args.free_card else 'gpu1_shared_h75_qualification_v1')
    research = {str(p): b.sha(p) for p in [*b.RESEARCH.glob('*.py'), *(b.RESEARCH / 'model').glob('*.py'), *HERE.glob('*.py'), MOSS / 'shared_gpu.py', MOSS / 'gpu1-sharing.json']}
    pins = {path: pin for path, pin in auth['files_SHA256'].items()
            if not Path(path).is_relative_to(ROOT) or Path(path).is_relative_to(ROOT / 'data')
            or Path(path).name in ('OWNER_AUTHORIZATION_20261103.json', 'ITERATION_AUTHORIZATION_20261103.json')}
    pins.update(research)
    pins.update(policy['source_sha256'])
    for p in [HERE / 'guard/REVIEW_MANIFEST.json', HERE / 'qualification_base.json', HERE / 'qualification_sampler.json',
              b.RESEARCH / 'SOURCE_ORIGINS.json', checkpoint, checkpoint.parent / 'checkpoint_receipt.json',
              folder / 'policy.HELD.json', folder / 'resource_basis.json']:
        pins[str(p)] = b.sha(p)
    auth.update(status='HELD_GPU1_SHARING_REVIEW_REQUIRED', schema='alz-calibration-gpu1-held-v2',
                source_review_passed=False, scheduler_allocation_id=None, expires_unix=0, files_SHA256=pins,
                cell=value, allocation_id=b.allocation_key(value), attempt_id=b.attempt_key(value), label=b.attempt_key(value),
                gpu_uuid=b.GPU1, cap_seconds=900,torch_allocator_cap_bytes=2*GiB, operation='evaluate_hybrid', qualification_only=True,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT, value)), ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT, value)),
                checkpoint_path=str(checkpoint), checkpoint_SHA256=base['checkpoint_SHA256'], variant_config=base['variant_config'], hybrid_config=sampler,
                policy_path=str(folder / 'policy.HELD.json'), policy_SHA256=b.sha(folder / 'policy.HELD.json'),
                resource_proposal_path=str(folder / 'resource_basis.json'), resource_proposal_SHA256=b.sha(folder / 'resource_basis.json'),
                registry_path=policy['registry_path'], protected_peers=peers, research_source_SHA256=research,
                candidate_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_review='Independent exact-source review pending. Frozen literal science copy; only retained hybrid75 zero-update qualification. This held authorization is never reusable for launch.')
    auth['command'] = b.expected_command(b.NATIVE_LAYOUT, folder / 'authorization.HELD.json', value)
    write(folder / 'authorization.HELD.json', auth)
    b.source_closure(auth, b.NATIVE_LAYOUT)
    try:
        b.verify(folder / 'authorization.HELD.json')
    except b.Refusal as error:
        rejection = str(error)
    else:
        raise RuntimeError('Held packet unexpectedly admitted')
    literal = {}
    selected = {Path(path) for path in pins if Path(path).suffix in ('.py', '.json', '.cfg', '.pth')}
    for p in [*folder.glob('*.json'), ROOT / 'gpu1_runtime/CPU_PREPARATION_RECEIPT.json', ROOT / 'analysis/fourier_cpu_qualification.json']:
        selected.add(p)
    parent = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence')
    reviewed = parent / 'integration/REVIEW_MANIFEST.json'
    b.require(b.sha(reviewed) == '36a0a6975e1fd01806bc6cc72ae5c639bcbfb723023fde628b373e555b93ce87', 'parent reviewed integration identity differs')
    selected.update([reviewed, *(parent / 'integration').glob('*.py'), *(parent / 'v2').glob('*.py'), *(ROOT / 'runtime').glob('*.py')])
    for path in sorted(selected):
        target = folder / 'literal' / str(path).lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        b.require(b.sha(target) == b.sha(path), 'literal source copy changed')
        literal[str(path)] = {'path': str(target.relative_to(folder)), 'SHA256': b.sha(target)}
    for target in HERE.glob('*.py'):
        baseline = ROOT / 'runtime' / target.name
        if baseline.exists():
            diff = ''.join(difflib.unified_diff(baseline.read_text().splitlines(keepends=True), target.read_text().splitlines(keepends=True), fromfile=str(baseline), tofile=str(target)))
            path = folder / 'diffs' / (target.name + '.diff')
            path.parent.mkdir(exist_ok=True)
            path.write_text(diff)
    result = {'status': 'FROZEN_LITERAL_SHARING_CAPSULE_REVIEW_PENDING', 'utc': stamp, 'path': str(folder),
              'parent_reviewed_integration_SHA256': b.sha(reviewed), 'source_inventory': literal,
              'GPU1_native_qualified': False, 'job_started': False, 'reservation_started': False,
              'policy_deployed': False, 'canonical_backend_deployed': False,
              'canonical_budget_SHA256_before_and_after': before, 'held_launch_rejected': rejection,
              'resource_basis': basis, 'scope': 'Only retained G8/cell8/defaultprecision/defaultFourier checkpoint444c22a,hybrid75,twofixedreplicates,zero updates,900s cap. LiveGPU0 source evolution cannot alter the frozen science copy.',
              'next_requirements': ['Independent exact literal-source/policy review', 'Fresh exact MOSS worker membership, compatible READ physical leases and resource admission', 'Separate finite native deployment/accounting step under existing user authority; never deploy or launch this held packet']}
    write(folder / 'REVIEW_PACKET.json', result)
    manifest = {str(path.relative_to(folder)): b.sha(path) for path in folder.rglob('*') if path.is_file()}
    write(folder / 'CAPSULE_MANIFEST.json', manifest)
    digest = b.sha(folder / 'CAPSULE_MANIFEST.json')
    archive = folder.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(folder, arcname=folder.name)
    b.require(b.sha(budget) == before, 'canonical budget changed during read-only capsule preparation; reconcile concurrent change')
    print(json.dumps({'status': result['status'], 'path': str(folder), 'manifest_SHA256': digest,
                      'archive': str(archive), 'archive_SHA256': b.sha(archive), 'literal_file_count': len(literal),
                      'GPU_fits_snapshot': basis['GPU_fits_snapshot'], 'host_fits_snapshot': basis['host_fits_snapshot'],
                      'held_launch_rejected': rejection, 'job_or_reservation_started': False}, indent=2))


if __name__ == '__main__':
    main()
