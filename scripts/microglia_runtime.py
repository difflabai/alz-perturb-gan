#!/usr/bin/env python3
"""Execute the frozen native EMPF host under the current application selection."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / 'analysis/default_empf_20261117'
SOURCE_MANIFEST = SELECTION / 'runtime_source_manifest.json'
CACHE_ANCHORS = ROOT / 'analysis/empirical_challenge_20261114/cache_and_fit_anchors.json'


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


def activate(device, memory):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('CUDA_VISIBLE_DEVICES must explicitly select a shared GPU')
    study = ROOT/'analysis/empf_comparison_20261117'
    sys.path.insert(0,str(study))
    import host
    T,G,provenance = host.activate('native')
    importlib.import_module('mode_cellshape')
    T.torch.set_num_threads(3)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3])
    T.torch.use_deterministic_algorithms(True)
    T.torch.backends.cuda.matmul.allow_tf32 = False
    T.torch.backends.cudnn.allow_tf32 = False
    T.torch.backends.cudnn.benchmark = False
    T.torch.backends.cudnn.deterministic = True
    if device == 'cuda':T.torch.cuda.set_per_process_memory_fraction(memory)
    else:assert not T.torch.cuda.is_initialized()
    return T,G,provenance


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
    elif args.out_root is None:p.error('train requires --out-root')
    T,G,provenance=activate(args.device,args.gpu_mem_fraction)
    expected=json.loads(M.RECIPE.read_text())['native']['recipe']
    if json.loads(json.dumps(T.recipe().to_dict()))!=expected:
        raise ValueError('live trainer differs from selected native EMPF recipe')
    if args.command=='train':
        native_args=T.parser().parse_args(['--cache',str(cache),'--arm','pbulk','--seed',str(args.seed),
            '--updates','20000','--out-root',str(args.out_root),'--device',args.device,'--threads','3',
            '--gpu-mem-fraction',str(args.gpu_mem_fraction)])
        original=G.generate_fit;G.generate_fit=lambda *a,**kw:(None,{'status':'deferred_to_explicit_application_readout'})
        try:fit=T.run(native_args)
        finally:G.generate_fit=original
        q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
        saved=json.loads((fit/'config.json').read_text())
        check_checkpoint(q,saved)
        record=q['policy']['optimizers'][1]['regularizer']['record']
        if not record['calls']==record['observed_steps']==q['completed_updates']==20000:
            raise ValueError('native final checkpoint does not verify regularization every update')
        check_cache(cache,fold)
        check_unchanged_sources(manifest,manifest_digest)
        audit=dict(status='passed_actual_full_EMPF_no_skips',penalty_calls=record['calls'],
            observed_steps=record['observed_steps'],completed_updates=q['completed_updates'],lazy_k=1,
            final_checkpoint_sha256=sha(fit/'final.pt'),application_source_manifest_sha256=manifest_digest,
            cache_anchor=cache_anchor)
        write(fit/'no_skip_audit.json',audit)
        M.check_fit(fit,cfg,fold,args.seed)
        check_weights(fit,json.loads((fit/'fit_receipt.json').read_text()))
        print(f'completed native EMPF training: {fit}',flush=True)
    else:
        q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
        check_checkpoint(q,saved)
        del q
        allowed={cfg['generation']['mode'],*(v['mode'] for v in cfg['reference_profiles'].values())}
        if args.mode not in allowed:raise ValueError('readout is outside the declared application profiles')
        # Same five named streams as the fixed EMPF pilot, repeat0.
        def streams(fold,index):
            children=T.np.random.SeedSequence([20261114,6,fold,index,0]).spawn(5)
            return {key:int(child.generate_state(1,dtype=T.np.uint64)[0])%(2**63-16)
                    for key,child in zip(G.GEN_STREAMS,children)}
        G.generation_streams=streams
        _,receipt=G.generate_fit(fit,cache,mode=args.mode,out=out,device=args.device)
        check_cache(cache,fold,held=True)
        check_unchanged_sources(manifest,manifest_digest)
        receipt.update(generation_streams=[20261114,6,fold,'development donor index',0],
            cache_anchor=cache_anchor,
            application_default=dict(name=cfg['name'],status=cfg['status'],population_qualified=False,
                config_sha256=sha(M.CONFIG),source_manifest_sha256=manifest_digest,
                native_recipe='empf',particlegan_version='0.10.0',provenance=provenance))
        write(out.with_name(out.stem+'.receipt.json'),receipt)
    check_unchanged_sources(manifest,manifest_digest)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError) as exc:raise SystemExit(f'microglia runtime: {exc}')
