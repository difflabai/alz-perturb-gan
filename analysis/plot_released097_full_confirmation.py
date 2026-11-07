"""Render the complete frozen three-setting calibration confirmation."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 reportpath=ROOT/'analysis/confirmfaith097/report.json';auditpath=ROOT/'analysis/confirmation097_pipeline_independent_audit_20261107.json';protocolpath=ROOT/'analysis/benchmark097/protocol.json'
 report=read(reportpath);audit=read(auditpath);protocol=read(protocolpath)
 assert audit['status']=='INDEPENDENT_RELEASED097_MATCHED_CONFIRMATION_AUDIT_PASS' and audit['required_attempts']==9 and audit['report_SHA256']==sha(reportpath)
 assert report['summary']==audit['summary'] and audit['summary']['ranking_seeds']==[2,3,4]
 assert protocol['PG_candidate_id']==audit['summary']['ranking'][0]=='t005'
 candidates={c['candidate_id']:c for c in audit['summary']['candidates']};order=['f000','t005','t004'];seeds=[2,3,4]
 for c in candidates.values():
  assert c['all_seeds_finite'] and [x['seed'] for x in c['per_seed']]==seeds
  assert all(x['baseline_metrics']==candidates['f000']['per_seed'][i]['baseline_metrics'] for i,x in enumerate(c['per_seed']))
 fields=['cell_energy_distance','energy_score','spread_ratio'];labels=['Nominal\nreference','Particle LR\n0.06\nSELECTED','Critic LR\n0.036','Resampling']
 colors=['#b65f00','#34804c','#7a5ab5'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(13.8,5.4));fig.subplots_adjust(left=.065,right=.985,top=.78,bottom=.28,wspace=.31)
 summaries={}
 for ax,field,title in zip(axes,fields,['Pooled cell distance ↓','Donor energy score ↓','Predictive spread ratio']):
  values=np.array([[candidates[c]['per_seed'][i]['metrics'][field] for c in order]+[candidates['f000']['per_seed'][i]['baseline_metrics'][field]] for i in range(3)])
  mean,sd=values.mean(0),values.std(0,ddof=1);x=np.arange(4);span=max(np.ptp(values),.05)
  ax.axvspan(.62,1.38,color='#1971c2',alpha=.08,zorder=0)
  for i,(seed,color) in enumerate(zip(seeds,colors)):
   ax.plot(x,values[i],color=color,alpha=.28,lw=1,zorder=1)
   ax.scatter(x,values[i],s=29,color=color,edgecolors='white',linewidths=.6,zorder=3)
  ax.errorbar(x,mean,yerr=sd,fmt='D',color='#202b36',ms=5,capsize=4,lw=1.6,zorder=4)
  for i,y in enumerate(mean):
   ax.text(i,max(y+sd[i],values[:,i].max())+.07*span,f'{y:.5f}',ha='center',va='bottom',fontsize=9,fontweight='bold' if i==1 else 'normal',color='#13558a' if i==1 else '#202b36')
  ax.set_ylim(min((mean-sd).min(),values.min())-.13*span,max((mean+sd).max(),values.max())+.24*span)
  ax.set_xticks(x,labels,fontsize=9);ax.set_xlim(-.45,3.45);ax.set_title(title,fontsize=12,pad=14);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
  summaries[field]={label:dict(mean=float(mean[i]),sample_SD=float(sd[i]),per_initialization={str(seed):float(values[j,i]) for j,seed in enumerate(seeds)}) for i,label in enumerate(order+['resampling'])}
 fig.suptitle('Released ParticleGAN: complete fresh 20,000-update confirmation',fontsize=16,x=.065,ha='left',y=.965)
 fig.text(.065,.875,'Particle LR 0.06 gives the lowest PG distance; the nominal reference gives the lowest PG energy score.',fontsize=11,color='#354555')
 handles=[Line2D([],[],color=c,marker='o',lw=1,label=f'Seed {s}') for s,c in zip(seeds,colors)]+[Line2D([],[],color='#202b36',marker='D',lw=1,label='Mean ± initialization SD')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.135),ncol=4,frameon=False,fontsize=10)
 fig.text(.065,.078,'Three matched fresh training initializations; each point averages two generation replicates. Lines link matched seeds.',fontsize=9,color='#455465')
 fig.text(.065,.034,'Four adaptive calibration donors. Error bars describe training variability. The frozen 45-fit external comparison is separate.',fontsize=9,color='#455465')
 stem=ROOT/'analysis/released097_full_confirmation_selection_20261107';png=stem.with_suffix('.png');svg=stem.with_suffix('.svg')
 assert not png.exists() and not svg.exists();fig.savefig(png,dpi=150,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
 out=ROOT/'analysis/released097_full_confirmation_plot_provenance_20261107.json';assert not out.exists()
 q=dict(status='COMPLETE_FRESH_SEED234_CALIBRATION_CONFIRMATION_PLOT_RENDERED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),data_role='adaptive_calibration_selection',training_seeds=seeds,training_updates_per_fit=20000,new_models_from_scratch=True,generation_replicates_per_point=2,adaptive_calibration_donors=4,selected_candidate=protocol['PG_candidate_id'],selection_criterion='Lowest mean calibration pooled cell energy distance over all three fresh confirmation seeds2..4; energy tracked separately.',summary=summaries,error_bars='Sample SD across three matched training initializations.',external_target_arrays_read=False,new_training=False,full45_external_comparison_complete=False,full_goal_complete=False,SOTA_claim=False,input_SHA256={str(p):sha(p) for p in (reportpath,auditpath,protocolpath)},source_SHA256={str(Path(__file__).resolve()):sha(__file__)},figure_SHA256={str(p):sha(p) for p in (png,svg)})
 out.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(png=str(png),svg=str(svg),provenance=str(out))))
if __name__=='__main__':main()
