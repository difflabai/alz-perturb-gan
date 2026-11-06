"""Plot measured500step objectives and paired fresh-seed comparisons only."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/hpo500'
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/hpo500/plot-cache'))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(Path(p).read_bytes())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='measured500step-results');args=parser.parse_args()
    rows=[read(p) for p in sorted((STUDY/'results').glob('*.json'))]
    selection=[r for r in rows if r['phase']=='search' and r['finite']]
    assert selection and all(r['endpoint']==500 and r['calibration_only'] for r in rows)
    control=next(r for r in selection if r['candidate_id']=='c000')['metrics']
    valid=[r for r in selection if .7<=r['metrics']['spread_ratio']<=1.3 and r['metrics']['cell_energy_distance']<=control['cell_energy_distance']]
    best=min(valid,key=lambda r:(r['metrics']['energy_score'],r['metrics']['cell_energy_distance'],r['candidate_id']))
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),gridspec_kw={'width_ratios':[1.45,1,1]})
    ax=axes[0]
    for feasible,color,label in [(False,'#b3b8c2','Outside matching/spread constraints'),(True,'#2878ad','Inside matching/spread constraints')]:
        subset=[r for r in selection if (r in valid)==feasible]
        if subset:ax.scatter([r['metrics']['cell_energy_distance'] for r in subset],[r['metrics']['energy_score'] for r in subset],c=color,s=43,label=label)
    ax.axvline(control['cell_energy_distance'],c='#555',ls='--',lw=.9)
    ax.axhline(control['energy_score'],c='#555',ls='--',lw=.9)
    for cid,marker,color,label in [('c000','s','#252525','Current hyperparameter control'),(best['candidate_id'],'*','#cd5e13','Best feasible selection-seed result')]:
        r=next(r for r in selection if r['candidate_id']==cid);m=r['metrics']
        ax.scatter(m['cell_energy_distance'],m['energy_score'],marker=marker,c=color,s=130 if marker=='*' else 62,zorder=5,label=label)
        ax.annotate(cid,(m['cell_energy_distance'],m['energy_score']),xytext=(6,5),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Cell distribution distance (lower is better)',ylabel='Donor energy score (lower is better)',title=f'Selection seed: {len(selection)}/25 finite results')
    ax.legend(fontsize=7,loc='best',framealpha=.85)
    confirmation=[r for r in rows if r['phase']=='confirm']
    paired={(r['candidate_id'],r['seed']):r for r in confirmation}
    ids=sorted({r['candidate_id'] for r in confirmation if r['candidate_id']!='c000'})
    for ax,metric,title in zip(axes[1:],['energy_score','cell_energy_distance'],['Fresh-seed energy differences','Fresh-seed cell-distance differences']):
        found=False
        for index,cid in enumerate(ids):
            seeds=[];delta=[]
            for seed in (1,2,3,4):
                a=paired.get((cid,seed));b=paired.get(('c000',seed))
                if a and b and a['finite'] and b['finite']:
                    seeds.append(seed);delta.append(a['metrics'][metric]-b['metrics'][metric])
            if seeds:
                ax.plot(np.asarray(seeds)+(index-.5)*.05,delta,'o-',label=cid,ms=4,lw=1);found=True
        ax.axhline(0,c='#555',ls='--',lw=.9);ax.set(xlim=(.65,4.35),xticks=[1,2,3,4],xlabel='Fresh training seed',ylabel='Candidate minus matched control',title=title)
        if found:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Confirmation pending\nNo fresh-seed results yet',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='#555')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16)
    fig.suptitle('ParticleGAN 0.9.6 · non-EMA · fixed formulation · exactly 500 fresh updates',fontsize=13)
    complete=(STUDY/'complete_report.json').exists()
    caption=('Complete bounded search; ' if complete else 'Provisional partial search; ')+f'{len([r for r in rows if r["phase"]=="search"])}/25 selection fits audited. Four adaptive calibration donors; two generation replicates. '
    caption+='Fresh seeds test initialization sensitivity on these same donors. This is not independent cohort validation.'
    fig.text(.5,.015,caption,ha='center',fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.07,1,.92))
    out=ROOT/'artifacts/hpo500/figures';out.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf'):fig.savefig(out/(args.name+'.'+extension),dpi=160,bbox_inches='tight')
    manifest=dict(status='MEASURED500STEP_FIGURE_RENDERED',selection_fits=len(selection),fresh_confirmation_fits=len(confirmation),complete=complete,source_results=[r['attempt'] for r in rows],best_selection_candidate=best['candidate_id'])
    (out/(args.name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))
if __name__=='__main__':main()
