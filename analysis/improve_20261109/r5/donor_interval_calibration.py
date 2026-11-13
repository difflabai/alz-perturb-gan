"""Development donor-split CQR: calibration donors excluded from GAN fitting.

The cohort was previously used in model selection, so this is not a final
coverage guarantee. Each donor supplies one calibration bag per coordinate;
cells, coordinates, and repeated subsamples never inflate n_calibration.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(k,'3')
import json
from pathlib import Path
import numpy as np
import evaluate as E
from architecture_score import observed_summaries
from interval_metrics import bounds, interval_score

OUT=E.RUNS/'results_r5/architecture_pass/donor_calibration'


def fit_correction(low,high,observed,calibration,alpha=.2):
    n=int(np.sum(calibration))
    rank=int(np.ceil((n+1)*(1-alpha)))
    if not n or rank>n:return np.full(low.shape[1],np.inf),rank
    score=np.maximum(0,np.maximum(low[calibration]-observed[calibration],observed[calibration]-high[calibration]))
    return np.sort(score,axis=0)[rank-1],rank


def group_scores(low,high,observed,mask):
    out={}
    for name,sl in [('mean',slice(0,50)),('sd',slice(50,100)),('combined',slice(0,100))]:
        lo,hi=low[mask,sl],high[mask,sl]
        y=observed[:,mask,sl]
        out[name]={'coverage':float(((y>=lo)&(y<=hi)).mean()),
                   'width':float((hi-lo).mean()),'interval_score':float(interval_score(lo,hi,y).mean())}
    return out


def run(fold,seed):
    real=E.load_real(E.RUNS/f'caches/hvg2000/fold{fold}')
    source=E.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}/generated_biascal@stagehalf.npz'
    gen=E.align_generated(E.load_generated(source),real)
    samples=E.MX.pc_bag_summary(gen['pcs'].astype(np.float32))
    actual=observed_summaries(real,64)
    held=np.flatnonzero(real.is_heldout)
    # Donor assignment depends on donor identity and a fixed seed, never values.
    order=sorted(held,key=lambda i:E.MX.stable_seed('interval-calibration-split',20261113,str(real.donors[i])))
    halves=np.array_split(order,2)
    records=[]
    for method in ('percentile','rank'):
        low,high=bounds(samples,method)
        assembled_low=low.copy();assembled_high=high.copy()
        for swap in (0,1):
            calibration=np.isin(np.arange(len(real.donors)),halves[swap])
            test=np.isin(np.arange(len(real.donors)),halves[1-swap])
            assert not (calibration&~real.is_heldout).any() and not (test&calibration).any()
            correction,rank=fit_correction(low,high,actual[0],calibration)
            # A test-data perturbation must not change a fitted correction.
            altered=actual[0].copy();altered[test]=1e6
            assert np.array_equal(correction,fit_correction(low,high,altered,calibration)[0])
            assembled_low[test]=low[test]-correction
            assembled_high[test]=high[test]+correction
            records.append({'method':method,'swap':swap,'n_calibration_donors':int(calibration.sum()),
                 'calibration_donors':real.donors[calibration].tolist(),'test_donors':real.donors[test].tolist(),
                 'rank':rank,'mean_correction':float(correction.mean()),
                 'raw':group_scores(low,high,actual,test),
                 'calibrated':group_scores(low-correction,high+correction,actual,test)})
        records.append({'method':method,'all_heldout_crossfitted':True,
            'raw':group_scores(low,high,actual,real.is_heldout),
            'calibrated':group_scores(assembled_low,assembled_high,actual,real.is_heldout)})
    value={'fold':fold,'seed':seed,'records':records,'source_sha256':E.sha256(source),
           'scope':'donor-split development calibration; no independent model-selection/coverage guarantee'}
    E.write_json(OUT/f'fold{fold}/seed{seed}.json',value)
    return value


def main():
    all_records=[]
    for f in range(9):
        for s in (0,1):
            path=OUT/f'fold{f}/seed{s}.json'
            value=json.loads(path.read_text()) if path.exists() else run(f,s)
            all_records.extend(r for r in value['records'] if r.get('all_heldout_crossfitted'))
            print(f'completed calibration f{f}s{s}',flush=True)
    summary={method:{recipe:{group:{stat:float(np.mean([r[recipe][group][stat] for r in all_records if r['method']==method]))
                 for stat in ('coverage','width','interval_score')} for group in ('mean','sd','combined')}
                 for recipe in ('raw','calibrated')} for method in ('percentile','rank')}
    E.write_json(OUT/'summary.json',{'n_fits':18,'summary':summary,
         'scope':'37 reused development donors; calibration and test donors disjoint within each fit'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
