"""Score a new fit or frozen read-out with historical and interval metrics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'3')
import argparse
import json
from pathlib import Path
import numpy as np
import evaluate as E
from interval_metrics import summarize


def observed_summaries(real, k):
    return np.stack([np.stack([E.MX.pc_bag_summary(real.pcs[i][E.MX.det_subsample(
        len(real.pcs[i]),k,E.MX.stable_seed('pc-summary-rep',donor,repeat))])
        for i,donor in enumerate(real.donors)]) for repeat in range(E.DEFAULT_PC_REPEATS)])


def score(path, cache, out):
    path,cache,out = map(Path,(path,cache,out))
    fingerprint = {'generated_sha256':E.sha256(path),'cache_sha256':E.sha256(cache/'manifest.json'),
                   'score_code_sha256':E.sha256(Path(__file__)),
                   'interval_code_sha256':E.sha256(Path(__file__).with_name('interval_metrics.py'))}
    if out.exists():
        existing=json.loads(out.read_text())
        if existing.get('fingerprint') == fingerprint:
            return existing
        raise ValueError(f'stale score {out}; inspect before replacing')
    real = E.load_real(cache)
    generated = E.align_generated(E.load_generated(path),real)
    result,_ = E.score_generated(generated,real,{'all':(np.arange(len(real.genes)),False)})
    Y = E.MX.pc_bag_summary(generated['pcs'].astype(np.float32))
    actual=observed_summaries(real,generated['pcs'].shape[2])
    intervals = {split:{method:summarize(Y,actual,mask,method) for method in ('percentile','rank')}
                 for split,mask in [('train',~real.is_heldout),('heldout',real.is_heldout)]}
    value={'fingerprint':fingerprint,'source':str(path.resolve()),'fold':real.fold,
           'M':int(Y.shape[1]),'historical':E.headline(result),'intervals':intervals}
    E.write_json(out,value)
    return value


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    value=score(a.generated,a.cache,a.out)
    print(json.dumps({'M':value['M'],'historical':value['historical'],
                      'intervals':value['intervals']['heldout']},indent=2))
