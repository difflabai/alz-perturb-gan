"""Static stage-wise calibration comparison from independently recomputed saved scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/ml2/alz');q=json.loads((ROOT/'analysis/pure_ensemble_stage_diagnostics.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.8,5.2))
colors={'Resampling':'#555555','PG8':'#277DA8','Learned':'#B8781B','Pure ensemble':'#16866B'}
for ax,key,title in zip(axes,['donor_energy_score','cell_energy_distance'],['Donor energy score','Conditional cell energy distance']):
 for label,color in colors.items():
  values=[stage['methods'][label][key] for stage in q['stages']]
  ax.plot(range(4),values,'o--' if label=='Resampling' else 'o-',label=label,color=color,lw=2 if label=='Pure ensemble' else 1.2,ms=6 if label=='Pure ensemble' else 4)
 ax.set(title=title,xlabel='ADNC stage code',ylabel='Score (lower is better)',xticks=range(4));ax.grid(alpha=.15)
ax=axes[0];ax.legend(fontsize=9,loc='upper right')
fig.suptitle('Pure ensemble: seed-0 calibration stage breakdown',fontsize=14)
fig.text(.03,.055,'Stage 2 is slightly worse than resampling on both scores; stages 0, 1 and 3 improve.',fontsize=10)
fig.text(.03,.017,'Four adaptive donors (one per stage); two sampling replicates. Conditional cell ED differs from pooled cell ED.',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.1,1,.94))
fig.savefig(ROOT/'analysis/pure_ensemble_stage_comparison.png',dpi=170);fig.savefig(ROOT/'analysis/pure_ensemble_stage_comparison.pdf');plt.close(fig)
