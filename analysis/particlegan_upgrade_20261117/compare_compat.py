"""Compare fresh old/new same-device behavior; enumerate additive metadata."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')
import json
import numpy as np
import torch
import host


def compare(a, b, path='', additions=None):
    if additions is None:
        additions = []
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a,b), path
    elif isinstance(a, dict):
        assert set(a) <= set(b), (path, 'removed fields', set(a)-set(b))
        for k in b.keys()-a.keys():
            additions.append(path+'/'+str(k))
        for k in a:
            compare(a[k], b[k], path+'/'+str(k), additions)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i,(x,y) in enumerate(zip(a,b)):
            compare(x,y,path+'/'+str(i),additions)
    else:
        assert a == b, (path,a,b)
    return additions


def main():
    root = host.RUNS / 'verification'
    files = [root/f'fresh_{v}.pt' for v in ('0.9.7','0.10.0')]
    a,b = [torch.load(p,map_location='cpu',weights_only=False) for p in files]
    additions = []
    for name in ('initial','after5'):
        left,right = a[name],b[name]
        # Distribution version necessarily differs; every other old field must
        # preserve its value. Additive fields remain explicitly listed below.
        assert (left.pop('particlegan_version'),right.pop('particlegan_version')) == ('0.9.7','0.10.0')
        additions.extend(compare(left,right,name))
    compare(a['metrics'],b['metrics'],'five_metrics',additions)
    # This exact tagged pair had no additive fields. Future releases need a
    # separately reviewed schema comparison, not a wildcard allowance.
    assert not additions, additions
    proof = {'status':'passed_exact_initial_and_five_updates_on_shared_GPU1',
             'comparison_code_sha256':host.sha(__file__),
             'old_version':'0.9.7','new_version':'0.10.0','explicit_recipe':'bispf_optimized',
             'compared':'all old checkpoint fields, including tensors/optimizers/policy/RNG/usage/shadows and all five metrics',
             'additive_metadata_fields':additions,
             'fresh_checkpoints_sha256':{p.name:host.sha(p) for p in files},
             'default_changed':False,'scope':'software compatibility, no quality/default qualification'}
    (root/'compatibility.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__ == '__main__':
    main()
