"""Plot actual kappa endpoints and every observed paired fresh-seed difference."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');NEW=ROOT/'analysis/hpo500_boundary';OLD=ROOT/'analysis/hpo500'
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(P/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(P/'.venv312/lib/python3.12/site-packages')]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/hpo500/plot-cache'))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(Path(p).read_bytes())
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',default='fixed500step-kappa-boundary-results');args=p.parse_args()
 anchor=read(NEW/'anchor.json');rows=[read(p) for p in sorted((NEW/'results').glob('*.json'))];selection=[r for r in rows if r['phase']=='search']
 assert selection and all(r['endpoint']==500 and r['calibration_only'] for r in rows)
 current=read(OLD/'results/CAL_GPU0_V1_h5_c000_search_f0_pg_s0.json')['metrics'];complete=read(NEW/'complete_report.json') if (NEW/'complete_report.json').exists() else None
 points=[dict(candidate_id=anchor['anchor_candidate'],hyperparameters=anchor['anchor_hyperparameters'],metrics=anchor['anchor_metrics'],finite=True)]+selection
 finite=sorted([r for r in points if r['finite']],key=lambda r:r['hyperparameters']['reg_kappa'])
 fig,axes=plt.subplots(2,2,figsize=(11.5,8))
 for ax,metric,title,ylabel in [(axes[0,0],'energy_score','Selection seed: donor score','Energy score (lower is better)'),(axes[0,1],'cell_energy_distance','Selection seed: cell distribution','Cell distribution distance (lower is better)')]:
  ax.plot([r['hyperparameters']['reg_kappa'] for r in finite],[r['metrics'][metric] for r in finite],'o-',c='#2878ad',ms=5,lw=1)
  ax.scatter(anchor['anchor_hyperparameters']['reg_kappa'],anchor['anchor_metrics'][metric],marker='*',c='#cd5e13',s=140,zorder=4,label='Anchor from original 25 settings')
  ax.axhline(current[metric],c='#555',ls='--',lw=.9,label='Starting hyperparameter control')
  for r in selection:
   if not r['finite']:ax.scatter(r['hyperparameters']['reg_kappa'],.03,transform=ax.get_xaxis_transform(),marker='x',c='#a33',label='Nonfinite trial (no numerical score)')
  ax.set(xscale='log',xlabel='Native reg_kappa; other five parameters fixed',ylabel=ylabel,title=title);ax.legend(fontsize=8)
 records=[read(p) for p in sorted((OLD/'results').glob('*confirm*.json'))]+[r for r in rows if r['phase']=='confirm'];paired={(r['candidate_id'],r['seed']):r for r in records};ids=[cid for cid in read(OLD/'finalists.json')['candidates'] if cid!='c000']+sorted({r['candidate_id'] for r in rows if r['phase']=='confirm'})
 for ax,metric,title in [(axes[1,0],'energy_score','Fresh-seed energy differences'),(axes[1,1],'cell_energy_distance','Fresh-seed distribution differences')]:
  for i,cid in enumerate(ids):
   seeds=[];delta=[]
   for seed in (1,2,3,4):
    a=paired.get((cid,seed));b=paired.get(('c000',seed))
    if a and b and a['finite'] and b['finite']:seeds.append(seed);delta.append(a['metrics'][metric]-b['metrics'][metric])
   if seeds:ax.plot(np.array(seeds)+(i-(len(ids)-1)/2)*.04,delta,'o-',label=cid,ms=4,lw=1.5 if complete and complete['selected_candidate']==cid else .9)
  ax.axhline(0,c='#555',ls='--',lw=.9);ax.set(xlim=(.7,4.3),xticks=[1,2,3,4],xlabel='Fresh training seed',ylabel='Candidate minus matched starting control',title=title);ax.legend(fontsize=8)
 for ax in axes.ravel():ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.17)
 fig.suptitle('ParticleGAN 0.9.6 · non-EMA · unchanged formulation · 500 updates from scratch',fontsize=13)
 caption=('Complete study. ' if complete else 'Provisional results. ')+f'{len(selection)} lower-kappa selection trials; {sum(not r["finite"] for r in selection)} nonfinite trials retained. '
 caption+='Four adaptive calibration donors and two generation replicates. Fresh seeds test initialization on these same donors; this is not independent cohort validation.'
 fig.text(.5,.01,caption,ha='center',fontsize=8,wrap=True);fig.tight_layout(rect=(0,.06,1,.95))
 out=ROOT/'artifacts/hpo500/figures';out.mkdir(exist_ok=True)
 for extension in ('png','pdf'):fig.savefig(out/(args.name+'.'+extension),dpi=160,bbox_inches='tight')
 print(json.dumps(dict(status='MEASURED_FIXED500STEP_BOUNDARY_FIGURE_RENDERED',name=args.name,selection_trials=len(selection),new_confirmation_fits=sum(r['phase']=='confirm' for r in rows),complete=bool(complete))))
if __name__=='__main__':main()
