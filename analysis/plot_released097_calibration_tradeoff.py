"""Descriptive matched-seed calibration plot; no model selection or new scores."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def main():
 source=ROOT/'analysis/released097_matched_seed0_calibration_comparison_20261107.json';q=json.loads(source.read_text());assert q['status']=='MATCHED_SEED0_CALIBRATION_EXTERNAL_REFERENCE_COMPARISON_NOT_FRESH45_BENCHMARK'
 rows=q['rows_sorted_by_cell_distance'];assert len(rows)==5 and all(r['training_seed']==0 and r['generation_replicates']==2 for r in rows)
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(8.4,6.3));fig.subplots_adjust(left=.12,right=.84,bottom=.30,top=.86)
 ref=next(r for r in rows if r['method']=='Hierarchical resampling');ax.axvline(ref['cell_energy_distance'],color='#aab2bd',ls=':',lw=1);ax.axhline(ref['energy_score'],color='#aab2bd',ls=':',lw=1)
 norm=Normalize(0,1.2);cmap=plt.get_cmap('viridis')
 labels={'scVI':('scVI',5,10),'CFGen':('CFGen',5,-20),'Hierarchical resampling':('Resampling',5,12),'f000':('PG nominal',-80,-16),'f001':('PG lower rates / κ',-75,13)}
 for r in rows:
  key=r['configuration'] if r['method']=='ParticleGAN' else r['method'];label,dx,dy=labels[key]
  ax.scatter(r['cell_energy_distance'],r['energy_score'],s=110,marker='o' if r['method']=='ParticleGAN' else 's',color=cmap(norm(r['spread_ratio'])),edgecolor='#223044',linewidth=.9,zorder=3)
  ax.annotate(label,(r['cell_energy_distance'],r['energy_score']),xytext=(dx,dy),textcoords='offset points',fontsize=9,color='#223044')
 ax.set(xlim=(.219,.290),ylim=(1.84,2.43),xlabel='Pooled cell energy distance  ↓',ylabel='Donor energy score  ↓');ax.grid(alpha=.16);ax.set_axisbelow(True)
 bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.045,pad=.05);bar.set_label('Donor-summary spread ratio\n1 = reference total spread',fontsize=9)
 fig.text(.12,.94,'Released ParticleGAN: calibration tradeoff',fontsize=14,weight='bold',color='#17263b');fig.text(.12,.89,'Lower is better on both axes; dotted lines show matched resampling.',fontsize=9,color='#455468')
 fig.text(.12,.065,'Four adaptive calibration donors · initialization seed 0 · two generation replicates\nPG: fresh 20,000-update runs. scVI / CFGen: previously measured fixed calibration models.\nDescriptive comparison; no biological confidence interval or external SOTA conclusion.',fontsize=8,color='#455468',linespacing=1.6)
 outputs=[]
 for extension in ('png','svg'):
  path=ROOT/'analysis'/('released097_matched_seed0_calibration_tradeoff.'+extension);assert not path.exists();fig.savefig(path,dpi=220,facecolor='white');outputs.append(path)
 plt.close(fig)
 proof=dict(status='PLOT_RENDERED_FROM_MATCHED_CALIBRATION_RECORDS',input_path=str(source),input_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),source_path=str(Path(__file__).resolve()),source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},new_model_fits=0,new_scores_computed=False,SOTA_claim=False)
 (ROOT/'analysis/released097_matched_seed0_calibration_plot_provenance.json').write_text(json.dumps(proof,indent=2)+'\n');print(outputs[0])
if __name__=='__main__':main()
