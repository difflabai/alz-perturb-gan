"""Static scientific comparison of fixed seed0 calibration measurements."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/ml2/alz');q=json.loads((ROOT/'analysis/pure_formulations_leaderboard.json').read_text());b=q['baseline'];h=json.loads((ROOT/'incumbent.json').read_text())
fig,ax=plt.subplots(figsize=(9.8,6.4))
ax.axvline(b['energy_score'],color='#666666',lw=1,ls='--',alpha=.7);ax.axhline(b['cell_energy_distance'],color='#666666',lw=1,ls='--',alpha=.7)
labels={'latent8':'Latent 8 (20k)','theta10':'Theta scale 10','fourier5':'Fourier scale 5','cell32':'Cell latent 32','continue40k':'Latent 8 continued to 40k','reference2':'Reference latent 2','balanced8':'Stage balanced latent 8','original_bispf':'Native original BISPF','semantic_observation':'Frozen subtype observations','bcap':'Native BCAP','learned_observation':'Learned cell observations','network_rate_half':'Half network step sizes'}
for row in q['rows']:
 name=row['formulation'];x,y=row['energy_score'],row['cell_energy_distance'];ax.scatter(x,y,s=75,c='#127C8A',zorder=4)
 delta={'theta10':(-8,-15),'fourier5':(10,3),'cell32':(10,-3),'latent8':(-3,12),'continue40k':(10,-6),'reference2':(-20,-18),'balanced8':(-20,10),'original_bispf':(-20,10),'semantic_observation':(45,14),'bcap':(12,12),'learned_observation':(-12,-20),'network_rate_half':(12,12)}[name]
 ax.annotate(labels[name],(x,y),xytext=delta,textcoords='offset points',ha='right' if name in ['reference2','balanced8','original_bispf','learned_observation'] else 'left',fontsize=10)
ax.scatter([],[],s=75,c='#127C8A',label='Pure ParticleGAN (no resampling replacements)')
ax.scatter(b['energy_score'],b['cell_energy_distance'],marker='*',c='#111111',s=180,zorder=5,label='Matched resampling baseline')
ax.scatter(h['energy_score'],h['cell_energy_distance'],marker='s',facecolor='none',edgecolor='#A25225',s=85,lw=1.7,zorder=4,label='Hybrid 75% PG / 25% training resampling')
ax.set(xlabel='Donor energy score (lower is better)',ylabel='Cell energy distance (lower is better)',title='Pure ParticleGAN: donor and cell distribution matching\nCalibration only · four donors · fold 0 · training seed 0 · two fixed sampling replicates')
ax.set_xlim(min(1.94,min(row['energy_score'] for row in q['rows'])-.035),max(2.3,max(row['energy_score'] for row in q['rows'])+.035));ax.set_ylim(min(.233,min(row['cell_energy_distance'] for row in q['rows'])-.018),max(.35,max(row['cell_energy_distance'] for row in q['rows'])+.018));ax.grid(alpha=.15);ax.legend(loc='upper left',framealpha=.95,fontsize=9)
fig.text(.015,.005,'40k uses extra training compute. No independent validation or external SOTA claim.',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.045,1,1))
fig.savefig(ROOT/'analysis/pure_calibration_comparison.png',dpi=170);fig.savefig(ROOT/'analysis/pure_calibration_comparison.pdf');plt.close(fig)
print('Saved analysis/pure_calibration_comparison.png and .pdf')
