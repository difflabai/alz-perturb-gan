"""Report both strict and distribution-focused guards for the fixed read-out."""
import json
from pathlib import Path
import numpy as np
from pzero_compare import paired

ROOT=Path('/ml2/alz/runs/improve_20261109/results_r5/architecture_pass/shape_confirmation')
METRICS=('pb_rmse','meanlog_rmse_gene','nonzero_rmse_gene','logvar_rmse_gene',
         'pc_energy_rep','cell_energy_rep','pc_spread_rep','pc_coverage80_rep')
TARGETS=('meanlog_rmse_gene','nonzero_rmse_gene','sd_interval_score')


def read(label):
    return {(int(p.parent.name[4:]),int(p.stem[4:])):json.loads(p.read_text())
            for p in (ROOT/label).glob('fold*/seed*.json')}


def metric(row,name):
    if name=='sd_interval_score':return row['intervals']['heldout']['percentile']['sd']['interval_score']
    value=row['historical'][name]
    if name=='pc_spread_rep':return abs(np.log(value))
    if name=='pc_coverage80_rep':return abs(value-.8)
    return value


def means(rows):
    return {'historical':{m:float(np.mean([r['historical'][m] for r in rows])) for m in METRICS},
            'intervals':{g:{m:float(np.mean([r['intervals']['heldout']['percentile'][g][m] for r in rows]))
                for m in ('coverage','width','interval_score')} for g in ('mean','sd','combined')}}


def main():
    baseline=read('default');candidate=read('count_shape')
    keys=sorted(set(baseline)&set(candidate))
    expected={(f,s) for f in range(3,9) for s in (0,1)}
    if set(keys)-expected:raise ValueError('unexpected fold or seed')
    if not keys:
        print('No complete shape-confirmation pairs yet')
        return
    for k in keys:
        for r in (baseline[k],candidate[k]):
            if r.get('generation_device')!='cpu' or r['M']!=32:
                raise ValueError('confirmation needs matched CPU,M32 outputs')
    comparison={}
    for name in (*METRICS,'sd_interval_score'):
        differences={f:[metric(candidate[(f,s)],name)-metric(baseline[(f,s)],name)
                        for ff,s in keys if ff==f] for f in sorted({f for f,s in keys})}
        comparison[name]=paired(differences)
    complete=set(keys)==expected
    baseline_mean=float(np.mean([baseline[k]['historical']['pb_rmse'] for k in keys]))
    profile=comparison['pb_rmse']
    seed_se=profile['se_seed_noise']
    per_fold_relative={f:float(np.mean([candidate[(int(f),s)]['historical']['pb_rmse']-
        baseline[(int(f),s)]['historical']['pb_rmse'] for ff,s in keys if ff==int(f)])/
        np.mean([baseline[(int(f),s)]['historical']['pb_rmse'] for ff,s in keys if ff==int(f)])) for f in profile['per_fold']}
    noninferior=bool(complete and profile['mean_diff']+2*seed_se <= .005*baseline_mean
                    and all(v<=.005 for v in per_fold_relative.values()))
    targets={}
    for name in TARGETS:
        block=comparison[name]
        repeat_means={str(repeat):float(np.mean([d for f,d in block['per_fold'].items() if int(f)//3==repeat]))
                      for repeat in (1,2) if any(int(f)//3==repeat for f in block['per_fold'])}
        targets[name]={'passes':bool(complete and block['mean_diff'] < -2*block['se_seed_noise']
                           and block['n_folds_lower']>=5 and all(v<0 for v in repeat_means.values())),
                       'repeat_means':repeat_means}
    guards={name:bool(complete and comparison[name]['mean_diff']<=2*comparison[name]['se_seed_noise'])
            for name in ('logvar_rmse_gene','pc_energy_rep','cell_energy_rep','pc_spread_rep','pc_coverage80_rep')}
    strict=bool(complete and profile['mean_diff']<=2*seed_se)
    value={'n_pairs':len(keys),'complete':complete,'means':{'default':means([baseline[k] for k in keys]),
        'count_shape':means([candidate[k] for k in keys])},'comparison':comparison,
        'profile_relative_cost_by_fold':per_fold_relative,'profile_noninferiority_margin':.005,
        'profile_cost_upper_2seedSE':None if seed_se is None else profile['mean_diff']+2*seed_se,
        'passes_05percent_profile_guard':noninferior,'passes_strict_profile_guard_on_extra_folds':strict,
        'targets':targets,'guards':guards,'passes_distribution_focused_rule':bool(noninferior and all(
            t['passes'] for t in targets.values()) and all(guards.values())),
        'original_strict_screen':'failed; not reclassified by this confirmation',
        'scope':'12 prospective CPU read-out pairs on six extra splits of the same37 development donors'}
    (ROOT/'comparison.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
