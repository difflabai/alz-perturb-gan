"""One isolated v0.10 native fit or matched CPU generation/scoring worker."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import common as C
import host


def train(lane):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from verify import setup, scope
    sources = C.preflight(); C.check_disk()
    T, G, provenance = setup('0.10.0', lane)
    cache, opened, permitted = scope()
    anchor=C.cache_anchor()
    args = T.parser().parse_args(['--cache',str(cache),'--arm','pbulk','--seed','0',
        '--updates','20000','--out-root',str(host.RUNS/'fits'/lane),
        '--device','cuda','--threads','3','--gpu-mem-fraction','.04'])
    original = G.generate_fit
    G.generate_fit = lambda *a,**kw: (None, {'status':'deferred_to_CPU_worker'})
    try:
        fit = T.run(args)
    finally:
        G.generate_fit = original
    assert fit == C.fit_path(lane) and opened == permitted
    assert C.cache_anchor()==anchor
    assert C.check_sources() == sources
    C.check_disk()
    receipt = json.loads((fit/'fit_receipt.json').read_text())
    assert receipt['completed_updates'] == receipt['endpoint'] == 20000 and not receipt['snapshots_missing']
    total = T.torch.cuda.get_device_properties(0).total_memory
    allocated=T.torch.cuda.max_memory_allocated(); reserved=T.torch.cuda.max_memory_reserved()
    assert max(allocated,reserved) <= total*.04
    C.write(fit/'release_test_receipt.json', {
        'status':'completed_native_TRAIN_only','lane':lane,'particlegan':'0.10.0',
        'provenance':provenance,'source_manifest_sha256':host.sha(C.PIN),
        'TRAIN_files_opened':sorted(opened),'default_changed':False,
        'cache_manifest_sha256':host.sha(cache/'manifest.json'),
        'cache_anchor':anchor,
        'fit_receipt_sha256':host.sha(fit/'fit_receipt.json'),
        'metadata_sha256':{n:host.sha(fit/n) for n in
            ('config.json','fit_receipt.json','checkpoint_receipt.json','learning_curve.jsonl')},
        'weights_sha256':{p.name:host.sha(p) for p in fit.glob('*.pt')},
        'affinity':sorted(os.sched_getaffinity(0)),
        'gpu':{'selector':'1','memory_fraction':.04,'max_allocated_bytes':allocated,
               'max_reserved_bytes':reserved,'total_bytes':total}})


def generate(lane):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sources = C.preflight(); C.check_disk()
    T,G,provenance = host.activate(lane)
    anchor=C.cache_anchor(held=True)
    T.torch.set_num_threads(3)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
    assert not T.torch.cuda.is_initialized()
    fit=C.fit_path(lane); cache=host.ROOT/'runs/improve_20261109/caches/hvg2000/fold0'
    assert C.valid_fit(lane)
    train=json.loads((fit/'release_test_receipt.json').read_text())
    assert train['source_manifest_sha256']==host.sha(C.PIN) and train['provenance']==provenance
    for name,digest in train['weights_sha256'].items():
        assert host.sha(fit/name)==digest
    output=host.RUNS/f'generated/{lane}.npz'; output.parent.mkdir(parents=True,exist_ok=True)
    score=host.RUNS/f'results/{lane}.json'
    # Exact existing covariance-control common-random-number namespace, repeat0.
    def streams(fold,index):
        children=T.np.random.SeedSequence([20261114,6,fold,index,0]).spawn(5)
        return {key:int(child.generate_state(1,dtype=T.np.uint64)[0])%(2**63-16)
                for key,child in zip(G.GEN_STREAMS,children)}
    G.generation_streams=streams
    _,receipt=G.generate_fit(fit,cache,mode=C.MODE,out=output,device='cpu')
    assert receipt['M']==32 and receipt['K']==256 and receipt['targets']==37 and receipt['heldout_targets']==12
    assert receipt['states_used']==[f'snap_{u}' for u in range(15000,20000,1000)]+['final']
    E=importlib.import_module('architecture_score')
    E.score(output,cache,score)
    row=json.loads(score.read_text())
    fingerprints={}
    with T.np.load(output,allow_pickle=False) as arrays:
        for name in arrays.files:
            a=T.np.ascontiguousarray(arrays[name])
            fingerprints[name]={'shape':list(a.shape),'dtype':str(a.dtype),
                                'sha256':__import__('hashlib').sha256(a.tobytes()).hexdigest()}
    receipt.update(lane=lane,release='0.10.0',source_manifest_sha256=host.sha(C.PIN),
        generation_streams=[20261114,6,0,'development donor index',0],
        generated_sha256=host.sha(output),array_fingerprints=fingerprints,
        fit_release_receipt_sha256=host.sha(fit/'release_test_receipt.json'))
    C.write(output.with_suffix('.receipt.json'),receipt)
    C.write(score,row|{'release_test':{'lane':lane,'release':'0.10.0',
        'source_manifest_sha256':host.sha(C.PIN),
        'generation_receipt_sha256':host.sha(output.with_suffix('.receipt.json')),
        'pruned_generated_sha256':host.sha(output),'default_changed':False}})
    assert C.check_sources()==sources
    assert C.cache_anchor(held=True)==anchor
    output.unlink()
    C.check_disk()
    assert C.valid_score(lane)
    print(json.dumps({'lane':lane,'historical':row['historical'],
        'heldout':row['intervals']['heldout']['percentile']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('train','generate'),required=True)
    p.add_argument('--lane',choices=('compat','rhbi'),required=True)
    a=p.parse_args(); (train if a.phase=='train' else generate)(a.lane)
