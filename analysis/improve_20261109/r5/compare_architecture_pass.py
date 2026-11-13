"""Paired comparison of predeclared architecture/interval experiments."""
import json
from pathlib import Path
import numpy as np
from pzero_compare import paired

ROOT=Path('/ml2/alz/runs/improve_20261109/results_r5/architecture_pass')
METRICS=('pb_rmse','meanlog_rmse_gene','nonzero_rmse_gene','logvar_rmse_gene','pc_energy_rep','cell_energy_rep')


def load(label):
    return {(int(p.parent.name[4:]),int(p.stem[4:])):json.loads(p.read_text())
            for p in (ROOT/label).glob('fold*/seed*.json')}


def compare(data,baseline,method='percentile',base_method='percentile'):
    keys=sorted(set(data)&set(baseline))
    metrics={}
    def value(row,metric,construction):
        if metric in METRICS:return row['historical'][metric]
        group,stat=metric.split('/')
        val=row['intervals']['heldout'][construction][group][stat]
        return abs(val-.8) if stat=='coverage' else val
    for metric in [*METRICS,*[g+'/'+s for g in ('mean','sd','combined') for s in ('coverage','width','interval_score')]]:
        differences={f:[value(data[(f,s)],metric,method)-value(baseline[(f,s)],metric,base_method)
                         for ff,s in keys if ff==f] for f in sorted({k[0] for k in keys})}
        metrics[metric]=paired(differences)
    complete=set(keys)=={(f,s) for f in range(3) for s in (0,1)}
    targets=metrics.get('sd/interval_score')
    def guard(p):return p and p['se_seed_noise'] is not None and p['mean_diff']<=2*p['se_seed_noise']
    passes=bool(complete and targets['n_folds_lower']==3 and targets['mean_diff']<-2*targets['se_seed_noise']
                and metrics['sd/coverage']['mean_diff']<0 and all(guard(metrics[m]) for m in METRICS))
    return {'n_pairs':len(keys),'complete_screen':complete,'comparison':metrics,
            'passes_initial_architecture_rule':passes,
            'decision':'requires_extra_fold_confirmation' if passes else 'incomplete' if not complete else 'retain_current_default'}


def means(data,method):
    if not data:return {}
    return {'n_fits':len(data),'historical':{m:float(np.mean([r['historical'][m] for r in data.values()])) for m in METRICS},
            'intervals':{g:{s:float(np.mean([r['intervals']['heldout'][method][g][s] for r in data.values()]))
                            for s in ('coverage','width','interval_score')} for g in ('mean','sd','combined')}}


def main():
    baseline=load('baseline32')
    out={'scope':'overlapping development splits, no independent final test','arms':{},'intervals':{}}
    for label in ('d.randproj,pbulk','g.split,pbulk','g.hier,pbulk'):
        data=load(label)
        out['arms'][label]={'means':means(data,'percentile'),**compare(data,baseline)}
    for label in ('baseline32','baseline128'):
        data=load(label)
        for method in ('percentile','rank'):
            out['intervals'][label+'/'+method]={'means':means(data,method),**compare(data,baseline,method)}
    path=ROOT/'comparison.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
