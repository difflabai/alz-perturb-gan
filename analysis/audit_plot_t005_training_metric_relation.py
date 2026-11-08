"""Independent tied-rank arithmetic and exact logged-window audit; static figure."""
import datetime,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ranks(values):
 order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);start=0
 while start<len(order):
  end=start+1
  while end<len(order) and values[order[end]]==values[order[start]]:end+=1
  for i in order[start:end]:out[i]=(start+1+end)/2
  start=end
 return out
def correlation(a,b):
 a,b=ranks(a),ranks(b);ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b)
 aa=[x-ma for x in a];bb=[y-mb for y in b];denom=math.sqrt(math.fsum(x*x for x in aa)*math.fsum(y*y for y in bb))
 return math.fsum(x*y for x,y in zip(aa,bb))/denom if denom else None
def close(a,b):assert abs(a-b)<1e-12,(a,b)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 close(correlation([1,2,3],[1,2,3]),1.);close(correlation([1,2,3],[3,2,1]),-1.)
 assert ranks([2,1,2])==[2.5,1.,2.5] and correlation([1,1,1],[1,2,3]) is None
 path=ROOT/'analysis/t005_training_loss_calibration_relation_CPU_20261108.json';q=read(path)
 for p,pin in q['input_SHA256'].items():assert sha(p)==pin,p
 source=ROOT/'analysis/diagnose_t005_training_metric_relation.py';assert sha(source)==q['source_SHA256']
 original=read(ROOT/'analysis/t005_checkpoint_CPU_multiseed_result_20261108.json')
 curves={}
 for seed in range(5):
  attempt=('CAL_GPU0_V1_lf097_t005_f0_pg_s' if seed<2 else 'CAL_GPU0_V1_cf097_t005_f0_pg_s')+str(seed)
  curves[seed]=[json.loads(line) for line in (ROOT/'runs'/attempt/'fit/learning_curve.jsonl').read_text().splitlines()]
 for row in q['records']:
  seed,step=row['training_seed'],row['saved_update'];selected=[v for v in curves[seed] if step-1000<v['step']<=step]
  assert len(selected)==8
  snapshot=next(v for v in original['all_seed_records'][str(seed)]['rows'] if v['completed_updates']==step)
  assert row['cell_energy_distance']==snapshot['mean']['cell_energy_distance'] and row['energy_score']==snapshot['mean']['energy_score']
  for key in q['predeclared_descriptive_proxies']:
   values=[v['penalty_diagnostics']['ema_residual'] if key=='penalty_ema_residual' else v[key] for v in selected]
   close(row['window_proxy_means'][key],math.fsum(values)/8)
   assert row['last_reported_proxy_values'][key]==values[-1]
 for key in q['predeclared_descriptive_proxies']:
  x=[v['window_proxy_means'][key] for v in q['records']];y=[v['cell_energy_distance'] for v in q['records']]
  close(correlation(x,y),q['correlations'][key]['pooled15_spearman_with_calibration_ED'])
  close(correlation([v['last_reported_proxy_values'][key] for v in q['records']],y),q['correlations'][key]['last_iteration15_spearman_with_ED'])
  dx=[];dy=[]
  for row in q['records']:
   group=[v for v in q['records'] if v['training_seed']==row['training_seed']]
   dx.append(row['window_proxy_means'][key]-math.fsum(v['window_proxy_means'][key] for v in group)/3)
   dy.append(row['cell_energy_distance']-math.fsum(v['cell_energy_distance'] for v in group)/3)
  close(correlation(dx,dy),q['correlations'][key]['within_seed_demeaned_spearman_with_ED'])
 paired=[]
 for seed in range(5):
  a=next(v for v in q['records'] if v['training_seed']==seed and v['saved_update']==10000)
  b=next(v for v in q['records'] if v['training_seed']==seed and v['saved_update']==20000)
  delta=b['cell_energy_distance']-a['cell_energy_distance'];loss=b['window_proxy_means']['G_total']-a['window_proxy_means']['G_total']
  assert delta*loss<0
  paired.append(dict(seed=seed,ED_delta_20k_minus10k=delta,G_total_delta_20k_minus10k=loss))
 fig,axes=plt.subplots(2,2,figsize=(11.8,7.4));colors=plt.get_cmap('tab10').colors;markers={5000:'o',10000:'s',20000:'^'}
 labels={'G_total':'Generator total loss','D_total':'Critic total loss','critic_gap':'Critic gap','penalty_ema_residual':'Penalty EMA residual'}
 for ax,key in zip(axes.flat,q['predeclared_descriptive_proxies']):
  for seed in range(5):
   group=sorted([v for v in q['records'] if v['training_seed']==seed],key=lambda v:v['saved_update'])
   ax.plot([v['window_proxy_means'][key] for v in group],[v['cell_energy_distance'] for v in group],color=colors[seed],alpha=.4,lw=1)
   for row in group:ax.scatter(row['window_proxy_means'][key],row['cell_energy_distance'],marker=markers[row['saved_update']],color=colors[seed],s=58,edgecolors='white',linewidths=.5,zorder=3)
  ax.set_title(labels[key]+'; rank correlation '+format(q['correlations'][key]['pooled15_spearman_with_calibration_ED'],'.2f'),fontsize=11)
  ax.set_xlabel('Mean of 8 logged iterations in preceding 1,000 updates',fontsize=9);ax.set_ylabel('Calibration cell energy distance',fontsize=9)
  ax.grid(alpha=.2);ax.tick_params(labelsize=9)
 fig.suptitle('Training losses do not consistently track distribution quality',fontsize=14,y=.98)
 fig.text(.5,.935,'15 existing t005 checkpoints / 5 initialization seeds / 4 reused calibration donors; lower distance is better; exploratory only',ha='center',fontsize=9)
 handles=[Line2D([],[],color=colors[s],marker='o',linestyle='none',label='Seed '+str(s)) for s in range(5)]+[Line2D([],[],color='gray',marker=mark,linestyle='none',label=str(step//1000)+'k') for step,mark in markers.items()]
 fig.legend(handles=handles,loc='lower center',ncol=8,frameon=False,bbox_to_anchor=(.5,.025),fontsize=9)
 fig.text(.5,.008,'Lines join saved checkpoints within each seed. No metric, checkpoint or setting is selected by this diagnostic.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.08,1,.925))
 outputs=[]
 for suffix in ('png','svg'):
  dest=ROOT/('analysis/t005_training_loss_calibration_relation_20261108.'+suffix);assert not dest.exists();fig.savefig(dest,dpi=160);outputs.append(dest)
 plt.close(fig)
 audit=dict(status='PASSED_INDEPENDENT_LOG_WINDOW_CALIBRATION_JOIN_AND_TIED_RANK_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),result_path=str(path),result_SHA256=sha(path),all15_actual_logged_windows_recomputed=True,all12_rank_correlations_independently_recomputed=True,tie_and_constant_and_monotone_rank_math_checked=True,paired_20k_minus10k=paired,generator_loss_and_ED_changes_opposite_all_five_seeds=True,all_input_and_checkpoint_hashes_verified=True,plot_files_SHA256={str(p):sha(p) for p in outputs},plot_visual_inspection_pending=True,source_SHA256=sha(__file__),no_training_or_LR_or_checkpoint_selection=True,no_new_ALZ_target_or_GPU_access=True,full_goal_complete=False,SOTA_claim=False)
 dest=ROOT/'analysis/t005_training_loss_calibration_relation_root_audit_20261108.json';assert not dest.exists();dest.write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(dict(status=audit['status'],opposite_generator_loss_ED_directions=5,plot_paths=list(audit['plot_files_SHA256']))))
if __name__=='__main__':main()
