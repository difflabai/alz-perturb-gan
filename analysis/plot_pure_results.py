"""Static scientific comparison of fixed seed0 calibration measurements."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/ml2/alz');q=json.loads((ROOT/'analysis/pure_formulations_leaderboard.json').read_text());b=q['baseline'];h=json.loads((ROOT/'incumbent.json').read_text())
fig,ax=plt.subplots(figsize=(9.8,6.4))
ax.axvline(b['energy_score'],color='#666666',lw=1,ls='--',alpha=.7);ax.axhline(b['cell_energy_distance'],color='#666666',lw=1,ls='--',alpha=.7)
labels={'latent8':'Latent 8 (20k)','theta10':'Theta scale 10','fourier5':'Fourier scale 5','cell32':'Cell latent 32','continue40k':'Latent 8 continued to 40k','reference2':'Reference latent 2','balanced8':'Stage balanced latent 8'}
for row in q['rows']:
 name=row['formulation'];x,y=row['energy_score'],row['cell_energy_distance'];ax.scatter(x,y,s=75,c='#127C8A',zorder=4)
 delta={'theta10':(10,-15),'fourier5':(10,3),'cell32':(10,-3),'latent8':(-3,12),'continue40k':(10,-2),'reference2':(-20,-18),'balanced8':(-20,10)}[name]
 ax.annotate(labels[name],(x,y),xytext=delta,textcoords='offset points',ha='right' if name in ['reference2','balanced8'] else 'left',fontsize=10)
ax.scatter([],[],s=75,c='#127C8A',label='Pure ParticleGAN (no resampling replacements)')
ax.scatter(b['energy_score'],b['cell_energy_distance'],marker='*',c='#111111',s=180,zorder=5,label='Matched resampling baseline')
ax.scatter(h['energy_score'],h['cell_energy_distance'],marker='s',facecolor='none',edgecolor='#A25225',s=85,lw=1.7,zorder=4,label='Hybrid 75% PG / 25% training resampling')
ax.set(xlabel='Donor energy score (lower is better)',ylabel='Cell energy distance (lower is better)',title='Pure ParticleGAN: donor and cell distribution matching\nCalibration only · four donors · fold 0 · training seed 0 · two fixed sampling replicates')
ax.set_xlim(1.94,2.3);ax.set_ylim(.233,.35);ax.grid(alpha=.15);ax.legend(loc='lower right',framealpha=.95,fontsize=9)
fig.text(.015,.005,'40k uses extra training compute. No independent validation or external SOTA claim.',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.045,1,1))
fig.savefig(ROOT/'analysis/pure_calibration_comparison.png',dpi=170);fig.savefig(ROOT/'analysis/pure_calibration_comparison.pdf');plt.close(fig)
print('Saved analysis/pure_calibration_comparison.png and .pdf')
