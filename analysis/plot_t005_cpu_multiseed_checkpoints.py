"""Retrospective five-initialization checkpoint plot; calibration only."""
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
 path=ROOT/'analysis/t005_checkpoint_CPU_multiseed_result_20261108.json';q=json.loads(path.read_text());assert q['status']=='COMPLETE_FIVE_EXISTING_INITIALIZATIONS_READ_ONLY_CPU_CHECKPOINT_CALIBRATION_DIAGNOSTICS' and q['training_seeds']==[0,1,2,3,4] and q['saved_updates']==[5000,10000,20000]
 assert q['new_optimizer_updates']==q['new_training_fits']==0 and q['calibration_donors']==4 and not q['external_evaluation_target_values_read'] and not q['CUDA_initialized'];assert sha(q['protocol_path'])==q['protocol_SHA256']
 for source,pin in q['per_seed_output_SHA256'].items():assert sha(source)==pin
 records=q['all_seed_records'];assert set(records)==set(map(str,range(5)));x=np.array(q['saved_updates']);colors=['#1676b7','#bf6f26','#31936a','#9561ac','#c34c63'];fields=['cell_energy_distance','energy_score','spread_ratio'];titles=['Cell energy distance ↓','Donor energy score ↓','Predictive spread ratio']
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(14,6.2));fig.subplots_adjust(left=.07,right=.985,top=.77,bottom=.32,wspace=.32)
 for ax,field,title in zip(axes,fields,titles):
  values=np.array([[r['mean'][field] for r in records[str(seed)]['rows']] for seed in range(5)]);assert np.isfinite(values).all()
  mean=values.mean(0);sd=values.std(0,ddof=1);np.testing.assert_allclose(mean,[q['summary'][str(step)][field]['mean'] for step in x],rtol=0,atol=1e-12);np.testing.assert_allclose(sd,[q['summary'][str(step)][field]['initialization_and_sampling_SD'] for step in x],rtol=0,atol=1e-12)
  baseline=q['mean_matched_resampling'][field];ax.axhline(baseline,c='#768391',ls='--',lw=1.6)
  for seed,color in enumerate(colors):ax.plot(x,values[seed],c=color,marker='o',lw=1,ms=4,alpha=.65)
  ax.plot(x,mean,c='#202b36',marker='D',lw=2.1,ms=6,zorder=5)
  span=max(np.ptp(np.r_[values.ravel(),baseline]),.03)
  for step,y in zip(x,mean):ax.annotate(f'{y:.5f}',(step,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9,fontweight='bold')
  ax.set_ylim(min(values.min(),baseline)-.15*span,max(values.max(),baseline)+.18*span);ax.set_xlim(3500,21500);ax.set_xticks(x,['5,000','10,000','20,000']);ax.set_xlabel('Saved training updates');ax.set_title(title,fontsize=12,pad=12);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('Selected ParticleGAN setting: CPU checkpoint diagnostics across five existing seeds',x=.07,y=.965,ha='left',fontsize=16)
 fig.text(.07,.86,'Each point averages two matching-trace generation replicates. All planned 5k, 10k and 20k checkpoints are retained.',fontsize=11,color='#3c4b59')
 handles=[Line2D([],[],c=c,marker='o',lw=1,label=f'Seed {s} (existing '+('exploration)' if s<2 else 'confirmation)')) for s,c in enumerate(colors)]+[Line2D([],[],c='#202b36',marker='D',lw=2,label='Five-seed mean'),Line2D([],[],c='#768391',ls='--',lw=1.6,label='Mean matched resampling')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.14),ncol=4,frameon=False,fontsize=9)
 fig.text(.07,.085,'Retrospective analysis of a calibration-selected setting on four reused donors. Seeds and sampling draws are not biological replicates.',fontsize=10,color='#3c4b59')
 fig.text(.07,.035,'CPU float32 sampling; no new training or checkpoint selection. The frozen45 and separate six fresh LR fits remain unchanged.',fontsize=10,color='#3c4b59')
 stem=ROOT/'analysis/t005_five_seed_checkpoint_CPU_calibration_20261108';outputs=[stem.with_suffix('.png'),stem.with_suffix('.svg')];assert not any(p.exists() for p in outputs)
 for p in outputs:fig.savefig(p,dpi=150,facecolor='white')
 plt.close(fig);out=ROOT/'analysis/t005_five_seed_checkpoint_CPU_calibration_plot_provenance_20261108.json';assert not out.exists();out.write_text(json.dumps(dict(status='COMPLETE_FIVE_EXISTING_SEED_CPU_CHECKPOINT_CALIBRATION_PLOT',input_SHA256={str(path):sha(path)},source_SHA256={str(Path(__file__).resolve()):sha(__file__)},figure_SHA256={str(p):sha(p) for p in outputs},retrospective_calibration_only=True,new_training=False,SOTA_claim=False,full_goal_complete=False),indent=2)+'\n');print(json.dumps(dict(png=str(outputs[0]),provenance=str(out))))
if __name__=='__main__':main()
