#!/usr/bin/env python3
"""Execute the frozen native EMPF host under the current application selection."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import datetime
import fcntl
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / 'analysis/default_empf_20261117'
SOURCE_MANIFEST = SELECTION / 'runtime_source_manifest.json'
CACHE_ANCHORS = ROOT / 'analysis/empirical_challenge_20261114/cache_and_fit_anchors.json'
# Derives the native EMPF host with checkout-relative import paths (vendored model sources);
# analysis/empf_comparison_20261117/host.py itself stays unchanged with its concluded study.
HOST = SELECTION / 'vendored_host.py'
# Host activation derives or confirms the host files on every start; serialize it across processes.
HOST_LOCK = ROOT / 'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native-vendored.lock'
# Machine-wide CPU slot locks, shared by every checkout of this repository on the host.
CPU_LOCK_DIR = Path(os.environ.get('MICROGLIA_CPU_LOCK_DIR') or f'/tmp/microglia-cpu-slots-{os.getuid()}')
THREADS = 3
_CPU_SLOT = None  # open slot-lock handle; held for the life of the process


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for name, digest in manifest['sources'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.exists() or sha(path) != digest:
            raise ValueError(f'pinned EMPF runtime source missing or changed: {name}')
    old = manifest['archived_previous_entrypoint']
    if sha(ROOT / old['archive']) != old['sha256']:
        raise ValueError('previous public entrypoint archive changed')
    return manifest


def check_unchanged_sources(started, digest):
    if sha(SOURCE_MANIFEST) != digest or check_sources() != started or sha(SOURCE_MANIFEST) != digest:
        raise ValueError('runtime source manifest changed during execution')


def check_cache(cache, fold, held=False):
    anchor = json.loads(CACHE_ANCHORS.read_text())['caches'].get(str(fold))
    if anchor is None:
        raise ValueError(f'no frozen development cache anchor for fold {fold}')
    if sha(cache/'manifest.json') != anchor['manifest_sha256']:
        raise ValueError('development cache manifest differs from the frozen fold')
    names = anchor['payload_sha256'] if held else ('space.npz', 'train/counts.npz', 'train/metadata.npz')
    payloads = {name: anchor['payload_sha256'][name] for name in names}
    for name, digest in payloads.items():
        if sha(cache/name) != digest:
            raise ValueError(f'development cache payload changed: {name}')
    return {'manifest_sha256': anchor['manifest_sha256'], 'payload_sha256': payloads}


def check_checkpoint(q, saved):
    expected = {key: saved[key] for key in ('arm', 'flags', 'panel', 'fold', 'seed',
                'cache_manifest_SHA256', 'recipe', 'particlegan_version', 'native_recipe')}
    expected.update(status='completed', completed_updates=20000, fixed_endpoint=20000,
                    released_hyperparameters=saved['hyperparameters'],
                    penalty_options={'input_domain': 'continuous'})
    for key, value in expected.items():
        if json.loads(json.dumps(q.get(key))) != value:
            raise ValueError(f'final checkpoint header differs from the selected fit: {key}')
    try:
        policy = q['policy']
        record = policy['optimizers'][1]['regularizer']['record']
        packet = record['empf']
        valid = (json.loads(json.dumps(policy['recipe'])) == saved['recipe'] and
                 policy['completed_steps'] == record['calls'] == record['observed_steps'] == 20000 and
                 packet['schema'] == 2 and packet['initialized'] is True and
                 packet['proto_initialized'] is False and packet['proto_sum'] is None and
                 packet['proto_count'] is None)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('final checkpoint lacks the completed native EMPF state') from exc
    if not valid:
        raise ValueError('final checkpoint does not verify the native recipe and every-update regularization')


def application():
    spec = importlib.util.spec_from_file_location('microglia_application', ROOT/'scripts/microglia.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cpu_busy(cpus, seconds=.2):
    """Per-CPU busy fraction over a short /proc/stat sample; {} when unavailable."""
    def sample():
        out = {}
        for line in Path('/proc/stat').read_text().splitlines():
            name, *values = line.split()
            if name.startswith('cpu') and name[3:].isdigit():
                values = [int(v) for v in values]
                out[int(name[3:])] = (sum(values), values[3] + (values[4] if len(values) > 4 else 0))
        return out
    try:
        a = sample(); time.sleep(seconds); b = sample()
    except (OSError, ValueError, IndexError):
        return {}
    return {c: 1 - (b[c][1]-a[c][1]) / max(b[c][0]-a[c][0], 1) for c in cpus if c in a and c in b}


def choose_cpus(allowed, threads=THREADS, env=None, lock_dir=None, load=cpu_busy):
    """Pick the CPUs for this process on a shared host.

    An inherited mask of at most `threads` CPUs is respected. MICROGLIA_CPU_SLOT=k
    selects the k-th block of `threads` allowed CPUs. Otherwise the process claims
    the least-loaded free block under a machine-wide flock; the claim is released
    when the process exits. If every block is claimed, or locking is unavailable,
    the least-loaded CPUs are used without a claim.
    """
    global _CPU_SLOT
    env = os.environ if env is None else env
    lock_dir = CPU_LOCK_DIR if lock_dir is None else Path(lock_dir)
    allowed = sorted(allowed)
    if len(allowed) <= threads:
        return allowed, {'source': 'inherited_mask'}
    blocks = [allowed[i:i+threads] for i in range(0, len(allowed)-threads+1, threads)]
    slot = env.get('MICROGLIA_CPU_SLOT')
    if slot not in (None, ''):
        k = int(slot)
        if not 0 <= k < len(blocks):
            raise ValueError(f'MICROGLIA_CPU_SLOT must be in 0..{len(blocks)-1} '
                             f'({len(blocks)} blocks of {threads} allowed CPUs), got {k}')
        return blocks[k], {'source': 'MICROGLIA_CPU_SLOT', 'slot': k}
    try:
        lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(lock_dir/'select.lock', 'a+') as select:
            fcntl.flock(select, fcntl.LOCK_EX)
            busy = load(allowed)
            order = sorted(range(len(blocks)),
                           key=lambda i: (round(sum(busy.get(c, 0.) for c in blocks[i]), 1), i))
            for i in order:
                handle = open(lock_dir/('cpus-' + '-'.join(map(str, blocks[i])) + '.lock'), 'a+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                if _CPU_SLOT is not None:
                    _CPU_SLOT.close()
                _CPU_SLOT = handle
                return blocks[i], {'source': 'slot_lock', 'lock_dir': str(lock_dir)}
    except OSError:
        pass
    busy = load(allowed)
    return sorted(sorted(allowed, key=lambda c: (busy.get(c, 0.), c))[:threads]), {'source': 'least_loaded'}


def pin_cpus():
    cpus, info = choose_cpus(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus)
    info['cpus'] = sorted(os.sched_getaffinity(0))
    return info


def activate_host(host=None, lock=None):
    """Run the host's activate under an exclusive lock so concurrent starts never see half-written files."""
    host = HOST if host is None else Path(host)
    lock = HOST_LOCK if lock is None else Path(lock)
    if str(host.parent) not in sys.path:
        sys.path.insert(0, str(host.parent))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, 'a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return importlib.import_module(host.stem).activate('native')


def activate(device, memory):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('CUDA_VISIBLE_DEVICES must explicitly select a shared GPU')
    cpus = pin_cpus()
    print(f'microglia runtime: CPU affinity {cpus["cpus"]} ({cpus["source"]})', flush=True)
    T,G,provenance = activate_host()
    importlib.import_module('mode_cellshape')
    T.torch.set_num_threads(THREADS)
    T.torch.use_deterministic_algorithms(True)
    T.torch.backends.cuda.matmul.allow_tf32 = False
    T.torch.backends.cudnn.allow_tf32 = False
    T.torch.backends.cudnn.benchmark = False
    T.torch.backends.cudnn.deterministic = True
    if device == 'cuda':T.torch.cuda.set_per_process_memory_fraction(memory)
    else:assert not T.torch.cuda.is_initialized()
    return T,G,provenance,cpus


def check_weights(fit, receipt):
    expected={'final.pt':receipt['final_SHA256'],
              **{f'snap_{step}.pt':digest for step,digest in receipt['snapshots_SHA256'].items()}}
    if set(expected) != {'final.pt', *(f'snap_{step}.pt' for step in (5000,10000,15000,16000,17000,18000,19000))}:
        raise ValueError('fit does not retain the full declared snapshot set')
    for name,digest in expected.items():
        if sha(fit/name) != digest:raise ValueError(f'checkpoint hash mismatch: {name}')
    return expected


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def receipt_path(out):
    return out.with_name(out.stem+'.receipt.json')


def pending_path(out):
    return out.with_name('.'+out.stem+'.pending.npz')


def pending_sidecars(pending):
    """The pending NPZ plus the files the generator writes beside it.

    The generator dumps its own receipt next to its `out` and stages np.savez through a
    hidden `.partial.npz`; both would otherwise survive next to the final output.
    """
    receipt=receipt_path(pending)
    return (pending,receipt,receipt.with_name('.'+receipt.name+'.tmp'),
            pending.with_name('.'+pending.stem+'.partial.npz'))


def resolve_profile(M,cfg,profile,mode):
    """Resolve the read-out profile; with no --profile, infer it from --mode (the default wins ties)."""
    inferred=profile is None and mode is not None
    if profile is None and mode is None:profile='default'
    if inferred:
        names=['default',*cfg.get('experimental_profiles',{}),*cfg.get('reference_profiles',{})]
        matches=[n for n in names if M.resolve_profile(cfg,n)['generation']['mode']==mode]
        if not matches:raise ValueError('readout is outside the declared application profiles')
        profile=matches[0]
    try:resolved=M.resolve_profile(cfg,profile)
    except KeyError:raise ValueError(f'unknown application profile: {profile}') from None
    if mode is None:mode=resolved['generation']['mode']
    if mode!=resolved['generation']['mode']:
        raise ValueError(f'--mode {mode} differs from profile {profile} ({resolved["generation"]["mode"]})')
    return profile,resolved,mode,inferred


def audited_digest(fit):
    path=fit/'no_skip_audit.json'
    return json.loads(path.read_text()).get('application_source_manifest_sha256') if path.exists() else None


def check_training_provenance(fit,digest):
    """Refuse to resume, or re-audit, a fit under a source manifest other than the one it started or was audited under."""
    audited=audited_digest(fit)
    if audited is not None and audited!=digest:
        raise ValueError(f'{fit/"no_skip_audit.json"} records source manifest {audited}; '
                         f'refusing to relabel it with {digest}')
    start=fit/'train_start.json'
    started=json.loads(start.read_text()) if start.exists() else None
    if started is not None and started.get('application_source_manifest_sha256')!=digest:
        raise ValueError(f'fit started under source manifest {started.get("application_source_manifest_sha256")}; '
                         f'refusing to resume or audit it under {digest}')
    fresh=not any((fit/name).exists() for name in ('config.json','recovery.pt','final.pt'))
    return started,fresh


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','generate'))
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--gpu-mem-fraction',type=float,default=.04)
    p.add_argument('--arm',default='pbulk');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--updates',type=int,default=20000);p.add_argument('--out-root',type=Path)
    p.add_argument('--fit',type=Path);p.add_argument('--out',type=Path);p.add_argument('--mode')
    p.add_argument('--profile',help='application profile; inferred from --mode when omitted')
    p.add_argument('--force',action='store_true',help='replace an existing --out and its receipt after a successful run')
    args=p.parse_args(argv)
    if args.threads!=3 or not 0<args.gpu_mem_fraction<=.04 or args.arm!='pbulk' or args.updates!=20000 or args.seed<0:
        p.error('the default requires pbulk/20000,three threads and at most.04 GPU fraction')
    manifest_digest=sha(SOURCE_MANIFEST);manifest=check_sources()
    check_unchanged_sources(manifest,manifest_digest)
    M=application();cfg=M.load_config()
    cache=args.cache.resolve()
    fold=json.loads((cache/'manifest.json').read_text())['inner_fold']
    cache_anchor=check_cache(cache,fold,held=args.command=='generate')
    if args.command=='generate':
        if args.fit is None or args.out is None:p.error('generate requires --fit and --out')
        fit=args.fit.resolve();out=args.out.resolve()
        if fit==out.parent or fit in out.parents:p.error('generation output must be outside the fit')
        if out.suffix!='.npz':p.error('--out must end in .npz')
        profile,resolved,mode,inferred=resolve_profile(M,cfg,args.profile,args.mode)
        existing=[str(x) for x in (out,receipt_path(out)) if x.exists()]
        if existing and not args.force:
            raise FileExistsError(f'generation output exists: {", ".join(existing)} (use --force to replace)')
        M.check_fit(fit,cfg,fold,args.seed)
        saved=json.loads((fit/'config.json').read_text())
        if sha(cache/'manifest.json')!=saved['cache_manifest_SHA256']:
            raise ValueError('generation cache differs from trained fit')
        weights=check_weights(fit,json.loads((fit/'fit_receipt.json').read_text()))
        if fit==(ROOT/cfg['runtime']['selected_fit']).resolve():
            selected=json.loads((SELECTION/'selected_model.json').read_text())
            if weights!=selected['weights_sha256']:
                raise ValueError('selected model weights differ from the completed native EMPF pilot')
            for name,digest in selected['metadata_sha256'].items():
                if sha(fit/name)!=digest:raise ValueError(f'selected model metadata changed: {name}')
        print(f'microglia runtime: generating from fit {fit} (profile {profile}, mode {mode}) -> {out}',flush=True)
    elif args.out_root is None:p.error('train requires --out-root')
    else:
        fit=(args.out_root/args.arm/cfg['training']['panel']/f'fold{fold}'/f'seed{args.seed}').resolve()
        started,fresh=check_training_provenance(fit,manifest_digest)
        print(f'microglia runtime: training fit {fit} ({"fresh" if fresh else "resume or audit"})',flush=True)
    T,G,provenance,cpus=activate(args.device,args.gpu_mem_fraction)
    expected=json.loads(M.RECIPE.read_text())['native']['recipe']
    if json.loads(json.dumps(T.recipe().to_dict()))!=expected:
        raise ValueError('live trainer differs from selected native EMPF recipe')
    if args.command=='train':
        if fresh and started is None:
            fit.mkdir(parents=True,exist_ok=True)
            started=dict(schema=1,application_source_manifest_sha256=manifest_digest,started_utc=utc_now(),
                         fold=fold,seed=args.seed,cache_anchor=cache_anchor,cpu_affinity=cpus,provenance=provenance)
            write(fit/'train_start.json',started)
        native_args=T.parser().parse_args(['--cache',str(cache),'--arm','pbulk','--seed',str(args.seed),
            '--updates','20000','--out-root',str(args.out_root),'--device',args.device,'--threads','3',
            '--gpu-mem-fraction',str(args.gpu_mem_fraction)])
        original=G.generate_fit;G.generate_fit=lambda *a,**kw:(None,{'status':'deferred_to_explicit_application_readout'})
        try:trained=T.run(native_args)
        finally:G.generate_fit=original
        if Path(trained).resolve()!=fit:
            raise ValueError(f'trainer wrote {trained}, not the expected fit {fit}')
        q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
        saved=json.loads((fit/'config.json').read_text())
        check_checkpoint(q,saved)
        record=q['policy']['optimizers'][1]['regularizer']['record']
        if not record['calls']==record['observed_steps']==q['completed_updates']==20000:
            raise ValueError('native final checkpoint does not verify regularization every update')
        check_cache(cache,fold)
        check_unchanged_sources(manifest,manifest_digest)
        check_training_provenance(fit,manifest_digest)
        audit=dict(status='passed_actual_full_EMPF_no_skips',penalty_calls=record['calls'],
            observed_steps=record['observed_steps'],completed_updates=q['completed_updates'],lazy_k=1,
            final_checkpoint_sha256=sha(fit/'final.pt'),application_source_manifest_sha256=manifest_digest,
            training_start_source_manifest_sha256=None if started is None else started['application_source_manifest_sha256'],
            cache_anchor=cache_anchor)
        if started is None:
            audit['training_start_note']='fit predates train_start.json; the sources it started under are not recorded'
        write(fit/'no_skip_audit.json',audit)
        M.check_fit(fit,cfg,fold,args.seed)
        check_weights(fit,json.loads((fit/'fit_receipt.json').read_text()))
        print(f'completed native EMPF training: {fit}',flush=True)
    else:
        q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
        check_checkpoint(q,saved)
        del q
        # Same five named streams as the fixed EMPF pilot, repeat0.
        def streams(fold,index):
            children=T.np.random.SeedSequence([20261114,6,fold,index,0]).spawn(5)
            return {key:int(child.generate_state(1,dtype=T.np.uint64)[0])%(2**63-16)
                    for key,child in zip(G.GEN_STREAMS,children)}
        G.generation_streams=streams
        pending=pending_path(out);pending.unlink(missing_ok=True)
        try:
            _,receipt=G.generate_fit(fit,cache,mode=mode,out=pending,device=args.device)
            check_cache(cache,fold,held=True)
            check_unchanged_sources(manifest,manifest_digest)
            if sha(pending)!=receipt['generated_SHA256']:
                raise ValueError('generated output changed before it was receipted')
            receipt.update(generation_streams=[20261114,6,fold,'development donor index',0],
                cache_anchor=cache_anchor,output=str(out),cpu_affinity=cpus,
                application_default=dict(name=resolved['name'],status=resolved['status'],profile=profile,
                    profile_inferred_from_mode=inferred,mode=mode,
                    calibration=resolved['generation']['calibration'],evidence=resolved['evidence'],
                    population_qualified=False,
                    config_sha256=sha(M.CONFIG),source_manifest_sha256=manifest_digest,
                    native_recipe='empf',particlegan_version='0.10.0',provenance=provenance))
            if not args.force and (out.exists() or receipt_path(out).exists()):
                raise FileExistsError(f'generation output appeared during the run: {out}')
            # The receipt is written first; the NPZ only appears once it is receipted. Both
            # steps are atomic replacements, so with --force the previous output and receipt
            # stay in place until each is overwritten.
            write(receipt_path(out),receipt)
            pending.replace(out)
        finally:
            for leftover in pending_sidecars(pending):
                leftover.unlink(missing_ok=True)
    check_unchanged_sources(manifest,manifest_digest)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,RuntimeError,KeyError) as exc:raise SystemExit(f'microglia runtime: {type(exc).__name__}: {exc}')
