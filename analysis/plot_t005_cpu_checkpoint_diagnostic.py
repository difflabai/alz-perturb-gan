"""Visualize a completed read-only CPU calibration checkpoint diagnostic."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');sys.path[:0]=list(map(str,[P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 path=ROOT/'analysis/t005_checkpoint_CPU_calibration_result_v3_20261108.json';q=json.loads(path.read_text());assert q['status']=='COMPLETE_READ_ONLY_T005_SEED0_CPU_CALIBRATION_CHECKPOINT_DIAGNOSTIC' and q['planned_saved_updates']==[5000,10000,20000]
 assert q['new_optimizer_updates']==0 and q['only_four_calibration_donors'] and not q['external_evaluation_target_values_read'] and not q['CUDA_initialized'] and q['generator_prior_state_unchanged']
 assert sha(q['protocol_path'])==q['protocol_SHA256']
 for source,pin in q['output_SHA256'].items():assert sha(source)==pin
 rows=q['rows'];assert [r['completed_updates'] for r in rows]==[5000,10000,20000]
 fields=['cell_energy_distance','energy_score','spread_ratio'];titles=['Cell energy distance ↓','Donor energy score ↓','Predictive spread ratio'];colors=['#1676b7','#bf6f26'];x=np.array([5000,10000,20000])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(13.5,5.6));fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.29,wspace=.31)
 for ax,field,title in zip(axes,fields,titles):
  v=np.array([[r['replicates'][rep][field] for r in rows] for rep in (0,1)]);assert np.isfinite(v).all()
  mean=v.mean(0);np.testing.assert_allclose(mean,[r['mean'][field] for r in rows],rtol=0,atol=1e-12)
  baseline=q['matched_CPU_resampling'][field];ax.axhline(baseline,color='#768391',lw=1.6,ls='--')
  for rep in (0,1):ax.plot(x,v[rep],marker='o',c=colors[rep],lw=1.2,ms=5,alpha=.75)
  ax.plot(x,mean,marker='D',c='#202b36',lw=1.6,ms=5)
  values=np.r_[v.ravel(),baseline];span=max(np.ptp(values),.03)
  for step,y in zip(x,mean):ax.annotate(f'{y:.5f}',(step,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
  ax.set_ylim(values.min()-.22*span,values.max()+.35*span);ax.set_xticks(x,['5,000','10,000','20,000']);ax.set_xlim(3500,21500);ax.set_title(title,fontsize=12,pad=12);ax.set_xlabel('Saved training updates');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 fig.suptitle('Selected ParticleGAN setting: saved-checkpoint CPU diagnostic',x=.07,y=.965,ha='left',fontsize=16)
 fig.text(.07,.855,'One existing training initialization (seed 0); matching fitting-only traces and prescribed generation seeds at all three checkpoints.',fontsize=10,color='#3c4b59')
 handles=[Line2D([],[],marker='o',c=c,lw=1,label=f'Generation replicate {r}') for r,c in enumerate(colors)]+[Line2D([],[],marker='D',c='#202b36',lw=1.6,label='Two-replicate mean'),Line2D([],[],c='#768391',lw=1.6,ls='--',label='Matched resampling mean')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.15),ncol=4,frameon=False,fontsize=9)
 fig.text(.07,.085,'Four adaptively reused calibration donors. Sampling replicates are not biological replicates or confidence intervals.',fontsize=10,color='#3c4b59')
 fig.text(.07,.035,'CPU float32 sampling; compare checkpoints within this diagnostic. No new training, endpoint reselection or SOTA conclusion.',fontsize=10,color='#3c4b59')
 stem=ROOT/'analysis/t005_saved_checkpoint_CPU_calibration_20261108';outputs=[stem.with_suffix('.png'),stem.with_suffix('.svg')];assert not any(p.exists() for p in outputs)
 for p in outputs:fig.savefig(p,dpi=150,facecolor='white')
 plt.close(fig);dest=ROOT/'analysis/t005_saved_checkpoint_CPU_calibration_plot_provenance_20261108.json';assert not dest.exists();dest.write_text(json.dumps(dict(status='COMPLETED_CPU_CHECKPOINT_DIAGNOSTIC_PLOT',data_role='reused_calibration_diagnostic',input_SHA256={str(path):sha(path)},source_SHA256={str(Path(__file__).resolve()):sha(__file__)},figure_SHA256={str(p):sha(p) for p in outputs},new_training=False,SOTA_claim=False,full_goal_complete=False),indent=2)+'\n');print(json.dumps(dict(png=str(outputs[0]),provenance=str(dest))))
if __name__=='__main__':main()
