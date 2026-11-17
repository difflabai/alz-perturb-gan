"""Fresh-version comparison and real ALZ checkpoint replay on shared GPU1."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time
import host


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def setup(version, lane='compat'):
    if version == '0.10.0':
        T, G, provenance = host.activate(lane)
    else:
        sys.path.insert(0, str(host.OLD))
        T, G = importlib.import_module('trainer'), importlib.import_module('generate')
        provenance = {'release': T.VERSION, 'trainer_sha256': host.sha(T.__file__)}
    torch = T.torch
    torch.set_num_threads(3)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(.04)
    assert T.VERSION == version
    return T, G, provenance


def scope(fold=0):
    cache = host.ROOT / f'runs/improve_20261109/caches/hvg2000/fold{fold}'
    permitted = {str((cache / n).resolve()) for n in
                 ('manifest.json', 'space.npz', 'train/counts.npz', 'train/metadata.npz')}
    opened = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(host.ROOT / 'runs/improve_20261109/caches'):
            if str(p) not in permitted:
                raise PermissionError(f'TRAIN verification attempted non-TRAIN payload: {p}')
            opened.add(str(p))
    sys.addaudithook(audit)
    return cache, opened, permitted


def snapshot(T, ctx, b, g, r, u, streams):
    return T.snapshot(ctx, b, g, r, u, 0, streams, 'incomplete',
        endpoint=20000, stream_root=T.STREAM_ROOT, wall_seconds=0.)


def semantic_observer(T, b, lane):
    if lane != 'rhbi':
        return {}
    rec,G,D,P,og,od,pen,loss,reg,policy = b
    assert rec.rhbi_simplex_weight == 0 and rec.prior_reg == .02
    assert loss.record is od.record and reg.thermal_source is od.record
    assert og._rhbi_record is od.record and D._rhbi_record is od.record
    stats = {'conditional_transport_calls':0, 'nonzero_values':0,
             'differentiable_values':0, 'shared_record':True,
             'sample_feature_width':2150, 'condition_width':7,
             'simplex_weight':0, 'prior_coefficient':.02}
    original = loss.rhbi.conditional_fiber_transport_loss
    def observe(x, y, c, *args, **kwargs):
        assert x.shape == y.shape == (32,2150) and c.shape == (32,7)
        assert kwargs['record'] is od.record and kwargs['mode']=='diag_imq_cmmd'
        value = original(x,y,c,*args,**kwargs)
        assert T.torch.isfinite(value)
        stats['conditional_transport_calls'] += 1
        stats['nonzero_values'] += int(float(value.detach()) != 0)
        stats['differentiable_values'] += int(value.requires_grad)
        return value
    loss.rhbi.conditional_fiber_transport_loss = observe
    def forbidden(*args,**kwargs):
        raise AssertionError('declared no-simplex variant invoked simplex')
    loss.rhbi.simplicial_manifold_loss = forbidden
    return stats


def verify(version, lane='compat'):
    started = time.monotonic()
    T, G, provenance = setup(version, lane)
    torch = T.torch
    cache, opened, permitted = scope()
    from common import cache_anchor
    anchor=cache_anchor()
    ctx = T.Context(cache, 'cuda', T.parse_arm('pbulk'))
    b, g, r, u, streams = T.build(ctx, 0)
    semantics = semantic_observer(T,b,lane)
    initial = snapshot(T, ctx, b, g, r, u, streams)
    metrics = []
    for _ in range(5):
        metrics.append(T.advance(ctx, b, g, r, u, execution_limit=20000))
    first = snapshot(T, ctx, b, g, r, u, streams)
    rejected=[]
    for field,value in [('particlegan_version','incompatible-release'),('native_recipe','incompatible-recipe')]:
        try:
            T.restore(ctx,first|{field:value},0)
        except ValueError:
            rejected.append(field)
        else:
            raise AssertionError('accepted incompatible checkpoint '+field)
    root = host.RUNS / 'verification'
    root.mkdir(parents=True, exist_ok=True)
    suffix = version if lane == 'compat' else version + '_' + lane
    output = root / f'fresh_{suffix}.pt'
    torch.save({'initial': initial, 'after5': first, 'metrics': metrics}, output)
    proof = {'version': version, 'provenance': provenance, 'verification_code_sha256':host.sha(__file__),
             'rejected_checkpoint_mismatches':rejected,
             'fresh_five_step_sha256': host.sha(output)}
    if version == '0.10.0':
        for _ in range(195):
            T.advance(ctx, b, g, r, u, execution_limit=20000)
        middle = snapshot(T, ctx, b, g, r, u, streams)
        replay_path = root / ('replay200_' + lane + '.pt'); torch.save(middle, replay_path)
        fresh = [T.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(100)]
        endpoint = snapshot(T, ctx, b, g, r, u, streams)
        expected_path=root/('expected300_'+lane+'.pt')
        torch.save({'snapshot':endpoint,'metrics':fresh},expected_path)
        del b, g, r, u, middle
        torch.cuda.empty_cache()
        q = torch.load(replay_path, map_location='cpu', weights_only=False)
        ctx2 = T.Context(cache, 'cuda', T.parse_arm('pbulk'))
        b, g, r, u, streams = T.restore(ctx2, q, 0)
        del q
        replay = [T.advance(ctx2, b, g, r, u, execution_limit=20000) for _ in range(100)]
        assert T._same(fresh, replay), 'restored metrics differ'
        assert T._same(endpoint, snapshot(T, ctx2, b, g, r, u, streams)), 'full restored snapshot differs'
        proof.update(restart_at=200, replay_updates=100, full_checkpoint_and_metrics_exact=True,
                     replay_checkpoint_sha256=host.sha(replay_path),
                     expected_checkpoint_sha256=host.sha(expected_path))
    assert opened == permitted, (opened, permitted)
    assert cache_anchor()==anchor
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
    assert max(allocated, reserved) <= total * .04
    if semantics:
        assert semantics['conditional_transport_calls']==300
        assert semantics['nonzero_values']>0 and semantics['differentiable_values']==300
        proof['rhbi_semantics']=semantics
        assert b[5].record.rhbi_cond_initialized and not b[7].rhbi._discrete_prior_step
    proof.update(status='passed', seconds=time.monotonic()-started,
        TRAIN_files_opened=sorted(opened), cache_anchor=anchor, affinity=sorted(os.sched_getaffinity(0)),
        gpu={'selector':'1','memory_fraction':.04,'max_allocated_bytes':allocated,
             'max_reserved_bytes':reserved,'total_bytes':total})
    write(root / f'verification_{suffix}.json', proof)
    print(json.dumps(proof, allow_nan=False), flush=True)


def replay_process(lane):
    started=time.monotonic()
    T,G,provenance=setup('0.10.0',lane)
    cache,opened,permitted=scope()
    from common import cache_anchor
    anchor=cache_anchor()
    root=host.RUNS/'verification'
    q=T.torch.load(root/('replay200_'+lane+'.pt'),map_location='cpu',weights_only=False)
    expected=T.torch.load(root/('expected300_'+lane+'.pt'),map_location='cpu',weights_only=False)
    ctx=T.Context(cache,'cuda',T.parse_arm('pbulk'))
    b,g,r,u,streams=T.restore(ctx,q,0)
    semantics=semantic_observer(T,b,lane)
    metrics=[T.advance(ctx,b,g,r,u,execution_limit=20000) for _ in range(100)]
    assert T._same(metrics,expected['metrics']), 'fresh-process metrics differ'
    assert T._same(snapshot(T,ctx,b,g,r,u,streams),expected['snapshot']), 'fresh-process full snapshot differs'
    assert opened==permitted
    assert cache_anchor()==anchor
    if lane=='rhbi':
        assert semantics['conditional_transport_calls']==100
        assert semantics['nonzero_values']>0 and semantics['differentiable_values']==100
        assert b[5].record.rhbi_cond_initialized and not b[7].rhbi._discrete_prior_step
    total=T.torch.cuda.get_device_properties(0).total_memory
    allocated=T.torch.cuda.max_memory_allocated();reserved=T.torch.cuda.max_memory_reserved()
    assert max(allocated,reserved)<=total*.04 and len(os.sched_getaffinity(0))<=3
    proof={'status':'passed_fresh_process_exact_200_to_300','lane':lane,
           'verification_code_sha256':host.sha(__file__),
           'seconds':time.monotonic()-started,'full_checkpoint_and_metrics_exact':True,
           'rhbi_semantics':semantics,'provenance':provenance,
           'cache_anchor':anchor,'TRAIN_files_opened':sorted(opened),
           'gpu':{'selector':'1','memory_fraction':.04,'max_allocated_bytes':allocated,
                  'max_reserved_bytes':reserved,'total_bytes':total},
           'replay_checkpoint_sha256':host.sha(root/('replay200_'+lane+'.pt')),
           'expected_checkpoint_sha256':host.sha(root/('expected300_'+lane+'.pt'))}
    write(root/('fresh_process_'+lane+'.json'),proof)
    print(json.dumps(proof),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', choices=('0.9.7','0.10.0'), required=True)
    p.add_argument('--lane', choices=('compat','rhbi'), default='compat')
    p.add_argument('--replay',action='store_true')
    a = p.parse_args()
    if a.version == '0.9.7' and a.lane != 'compat':
        p.error('RHBI requires v0.10.0')
    if a.replay:
        replay_process(a.lane)
    else:
        verify(a.version, a.lane)
