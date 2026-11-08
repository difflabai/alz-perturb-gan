"""Frozen45 outcome figures; real values require both completed scientific reports."""
import argparse,datetime,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
METHODS=('pg','scvi','cfgen','hierarchical_resampling')
LABELS=('ParticleGAN','scVI','CFGen','Resampling')
COLORS=('#1676b7','#31936a','#9561ac','#768391')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
 x=float(x)
 if not math.isfinite(x):raise ValueError('Finite figure values required')
 return x
def validate_summary(summary):
 assert set(summary)==set(METHODS)
 for m in METHODS:
  row=summary[m]
  for key in ('cell_energy_distance','pooled_donor_energy_score','mean_spread_ratio','mean_within_condition_spread_ratio','mean_coverage80'):finite(row[key])
  points=row['fold_seed_rows'];assert {(v['fold'],v['seed']) for v in points}=={(f,s) for f in range(3) for s in range(5)} and len(points)==15
  assert np.isclose(np.mean([finite(v['cell_energy_distance']) for v in points]),row['cell_energy_distance'],rtol=0,atol=1e-10)
 return summary

def load_real():
 # Metadata/hash barrier precedes every quality-report read and plotting output.
 sys.path.insert(0,str(ROOT/'scalarcal097_runtime'))
 import preset_scope
 pins=preset_scope.prior_study_ready()
 primary=read(ROOT/'analysis/benchmark097/comparison.json');sensitivity=read(ROOT/'analysis/benchmark097_calibration_excluded53_report.json')
 assert primary['complete_cells']==45 and sensitivity['complete_fresh_generative_models']==45 and sensitivity['original57_primary_preserved']
 assert not primary['SOTA_claim'] and not sensitivity['SOTA_claim']
 cohorts=[('Original 57 donors',validate_summary(primary['methods']),primary['PG_pairwise_primary_contrasts']),('Prespecified 53 donors',validate_summary(sensitivity['models']),sensitivity['contrasts'])]
 return cohorts,pins

def synthetic():
 cohorts=[]
 for cohort,shift in [('SYNTHETIC preview: 57',0.),('SYNTHETIC preview: 53',.015)]:
  summary={};contrasts={}
  for i,m in enumerate(METHODS):
   values=[.12+i*.02+shift+(f-1)*.006+(s-2)*.002 for f in range(3) for s in range(5)]
   summary[m]=dict(cell_energy_distance=float(np.mean(values)),pooled_donor_energy_score=1.7+i*.08,mean_spread_ratio=[.45,.9,1.15,1.35][i],mean_within_condition_spread_ratio=[.4,.85,1.05,1.2][i],mean_coverage80=[.4,.7,.8,.85][i],fold_seed_rows=[dict(fold=f,seed=s,cell_energy_distance=values[f*5+s]) for f in range(3) for s in range(5)])
   if i:
    delta=-.02*i;contrasts[m]=dict(PG_minus_other_primary_distance=delta,conditional_real_donor_cluster_interval=dict(mean_delta=delta,lower=delta-.015,upper=delta+.025,interval_percent=98.33333333333333))
  cohorts.append((cohort,validate_summary(summary),contrasts))
 return cohorts,{}

def dashboard(cohorts,output,preview):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,4,figsize=(16,9.2));fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.16,wspace=.36,hspace=.5)
 x=np.arange(4)
 for row,(cohort,summary,_) in enumerate(cohorts):
  ax=axes[row,0]
  for i,m in enumerate(METHODS):
   points=sorted(summary[m]['fold_seed_rows'],key=lambda p:(p['fold'],p['seed']))
   ys=[p['cell_energy_distance'] for p in points];jitter=np.linspace(-.16,.16,len(points))
   ax.scatter(i+jitter,ys,s=15,c=COLORS[i],alpha=.5,edgecolors='none')
   mean=summary[m]['cell_energy_distance'];ax.scatter(i,mean,s=45,marker='D',c=COLORS[i],edgecolors='black',linewidths=.7,zorder=4);ax.annotate(f'{mean:.4f}',(i,max(ys)),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
  ax.set_title(cohort+'\nCell energy distance ↓',loc='left',fontsize=11,pad=12);ax.margins(y=.27)
  for col,key,title in [(1,'pooled_donor_energy_score','Donor energy score ↓'),(3,'mean_coverage80','80% interval coverage')]:
   ax=axes[row,col];values=[summary[m][key] for m in METHODS];ax.bar(x,values,color=COLORS,width=.64,alpha=.85)
   for i,y in enumerate(values):ax.text(i,y,f'{y:.3f}',ha='center',va='bottom',fontsize=9)
   if col==3:ax.axhline(.8,color='#4b5661',ls='--',lw=1);ax.set_ylim(0,1.04)
   else:ax.set_ylim(0,max(values)*1.22)
   ax.set_title(title,loc='left',fontsize=11,pad=12)
  ax=axes[row,2];ax.axhspan(.7,1.3,color='#b8c8b8',alpha=.22);ax.axhline(1,color='#4b5661',lw=1,ls='--')
  for i,m in enumerate(METHODS):
   total=summary[m]['mean_spread_ratio'];within=summary[m]['mean_within_condition_spread_ratio']
   ax.scatter(i-.09,total,s=46,c=COLORS[i]);ax.scatter(i+.09,within,s=46,facecolors='white',edgecolors=COLORS[i],linewidths=1.7)
  ax.set_ylim(bottom=0);ax.set_title('Spread ratios\nfilled: total; open: within condition',loc='left',fontsize=11,pad=12)
  for ax in axes[row]:ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_xlim(-.45,3.45);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 title='SYNTHETIC LAYOUT PREVIEW — no ALZ results' if preview else 'Released ParticleGAN: frozen 45-fit external comparison'
 fig.suptitle(title,x=.065,y=.965,ha='left',fontsize=17)
 fig.text(.065,.9,'Three folds × five fresh training seeds per model; two generation replicates per point. Lower distance and energy are better.',fontsize=11,color='#394959')
 fig.text(.065,.075,'Distance dots: 15 fold/seed means. Diamonds: equal-fold means. Donor energy pools donors; coverage and spread average folds.',fontsize=10,color='#394959')
 fig.text(.065,.035,'Historical cohort, overlapping fitting folds and unequal tuning budgets. Shaded spread band: 0.7–1.3 diagnostic. No population SOTA claim.',fontsize=10,color='#394959')
 for ext in ('.png','.svg'):fig.savefig(output.with_suffix(ext),dpi=150,facecolor='white')
 plt.close(fig)

def paired_intervals(cohorts,output,preview):
 fig,axes=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(left=.13,right=.97,top=.72,bottom=.25,wspace=.5)
 for ax,(cohort,summary,comparisons) in zip(axes,cohorts):
  ax.axvline(0,color='#4b5661',lw=1)
  for i,m in enumerate(METHODS[1:]):
   c=comparisons[m];delta=finite(summary['pg']['cell_energy_distance']-summary[m]['cell_energy_distance']);interval=c.get('conditional_real_donor_cluster_interval',c.get('conditional_donor_cluster_interval'))
   if interval is None:ax.scatter(delta,i,c=COLORS[i+1],marker='x');ax.annotate('interval unavailable',(delta,i),xytext=(5,8),textcoords='offset points',fontsize=8)
   else:
    assert np.isclose(interval['mean_delta'],delta,rtol=0,atol=1e-10)
    low,high=map(finite,(interval['lower'],interval['upper']));assert low<=high
    ax.plot([low,high],[i,i],c=COLORS[i+1],lw=2);ax.scatter(delta,i,c=COLORS[i+1],s=44,zorder=3)
  ax.set_yticks(range(3),LABELS[1:]);ax.set_ylim(2.6,-.6);ax.set_title(cohort,loc='left',fontsize=12);ax.set_xlabel('PG − comparator distance (negative favors PG)');ax.grid(axis='x',alpha=.15)
 fig.suptitle('SYNTHETIC INTERVAL PREVIEW' if preview else 'Paired distribution-distance contrasts',x=.13,y=.95,ha='left',fontsize=16)
 fig.text(.13,.83,'98⅓% donor-cluster intervals; three Bonferroni-adjusted contrasts.',fontsize=11)
 fig.text(.13,.075,'Conditional on fixed fitted models and generation reservoirs; donor clusters are resampled within fold and stage.',fontsize=9)
 for ext in ('.png','.svg'):fig.savefig(output.with_suffix(ext),dpi=150,facecolor='white')
 plt.close(fig)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--synthetic-preview',action='store_true');args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 cohorts,pins=synthetic() if args.synthetic_preview else load_real()
 folder=ROOT/'artifacts/benchmark097-figure-preview' if args.synthetic_preview else ROOT/'analysis/benchmark097/figures'
 assert not folder.exists();folder.mkdir(parents=True)
 dashboard(cohorts,folder/'quality',args.synthetic_preview);paired_intervals(cohorts,folder/'paired_distance',args.synthetic_preview)
 q=dict(status='SYNTHETIC_LAYOUT_ONLY' if args.synthetic_preview else 'COMPLETE_FROZEN45_BOTH_COHORTS_FIGURES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),synthetic=args.synthetic_preview,real_target_values_read=not args.synthetic_preview,new_training=False,SOTA_claim=False,full_goal_complete=False,input_SHA256=pins,source_SHA256={str(Path(__file__).resolve()):sha(__file__)},figure_SHA256={str(p):sha(p) for p in folder.iterdir() if p.suffix in ('.png','.svg')})
 (folder/'provenance.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(folder=str(folder),status=q['status'])))
if __name__=='__main__':main()
