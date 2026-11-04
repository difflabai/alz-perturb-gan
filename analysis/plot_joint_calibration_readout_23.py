"""Saved-score readout with a full-range panel; no fitting or selection."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT=Path('/ml2/alz')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plot(leaderboard,prefix):
    confirmation=ROOT/'analysis/pure_ensemble_confirmation_v6_all_seeds.json'
    d=json.loads(leaderboard.read_text());c=json.loads(confirmation.read_text())
    assert d['completed_seed0_comparisons']==len(d['rows'])==23
    assert not c['stable_joint_target_passed'] and c['per_seed_joint_improvement']==[True,False,False]
    points=[{'name':r.get('formulation',r.get('pure_formulation',r.get('method'))),
             'ES':r['energy_score'],'cell_ED':r['cell_energy_distance']} for r in d['rows']]
    assert all(np.isfinite([p['ES'],p['cell_ED']]).all() and p['ES']>0 and p['cell_ED']>0 for p in points)
    by={p['name']:p for p in points};assert len(by)==23
    native_names={'latent8','original_bispf','bcap','bcap_adam','ka2','k3p','halloween'}
    assert set(d['completed_native_preset_names'])=={'bispf_optimized','bispf','bcap','bcap_adam','ka2','k3p','halloween'}
    native=[by[n] for n in sorted(native_names)]
    pareto=[p for p in points if not any(q['ES']<=p['ES'] and q['cell_ED']<=p['cell_ED'] and
            (q['ES']<p['ES'] or q['cell_ED']<p['cell_ED']) for q in points)]
    selected=by['pure_ensemble75'];inc=d['selected_pure_incumbent']
    assert selected['ES']==inc['energy_score'] and selected['cell_ED']==inc['cell_energy_distance']
    assert sha(ROOT/'pure_incumbent.json')==d['selected_pure_incumbent_SHA256']
    records=c['records'];assert [r['seed'] for r in records]==[0,1,2]
    relative=[{'seed':r['seed'],'ES_percent':100*(1-r['energy_score']/r['matched_baseline_energy_score']),
               'cell_ED_percent':100*(1-r['cell_energy_distance']/r['matched_baseline_cell_energy_distance']),
               'joint_pass':passed} for r,passed in zip(records,c['per_seed_joint_improvement'])]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,(full,zoom,confirm)=plt.subplots(1,3,figsize=(15.8,6.3),gridspec_kw={'width_ratios':[.9,1.22,1.]})
    fig.subplots_adjust(left=.052,right=.985,top=.80,bottom=.25,wspace=.37)
    fig.suptitle('No stable joint win in donor energy and cell distribution match',x=.052,y=.96,ha='left',fontsize=16,weight='bold')
    fig.text(.052,.91,'23 completed seed-0 pure-count comparisons, including seven native trainer presets. Lower is better for both metrics.',color='#444444',fontsize=10)
    limits=(1.88,2.345,.218,.89)
    for ax in (full,zoom):
        ax.scatter([p['ES'] for p in points],[p['cell_ED'] for p in points],s=34,color='#b0b8c2',zorder=2,label='Other pure-count variants')
        ax.scatter([p['ES'] for p in native],[p['cell_ED'] for p in native],s=42,color='#176ba0',marker='s',zorder=3,label='Native trainer presets')
        ax.scatter([p['ES'] for p in pareto],[p['cell_ED'] for p in pareto],s=110,facecolors='none',edgecolors='#333333',linewidths=.8,zorder=4)
        ax.scatter(selected['ES'],selected['cell_ED'],s=140,marker='*',color='#d77a16',edgecolors='#5f390f',linewidths=.6,zorder=5,label='Selected pure ensemble')
        ax.scatter(records[0]['matched_baseline_energy_score'],records[0]['matched_baseline_cell_energy_distance'],s=65,marker='X',color='#925c9e',zorder=4,label='Matched resampling, seed 0')
        ax.set_yscale('log');ax.minorticks_off();ax.grid(alpha=.14)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Donor energy score')
        ax.set_ylabel('Pooled cell energy distance (log scale)')
    full.set(title='All completed scores',xlim=(1.80,4.48),ylim=(.20,7.4))
    full.set_yticks([.25,.5,1.,2.,5.])
    full.add_patch(Rectangle((limits[0],limits[2]),limits[1]-limits[0],limits[3]-limits[2],fill=False,edgecolor='#777777',linestyle='--',linewidth=.8,zorder=1))
    full.text(2.52,.47,'Dashed box:\nexpanded at right',fontsize=8,color='#555555')
    halloween=by['halloween']
    full.annotate('Halloween\nsevere collapse',(halloween['ES'],halloween['cell_ED']),xytext=(-96,-72),textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':'#777777','linewidth':.6})
    zoom.set(title='Expanded view: 22 scores',xlim=limits[:2],ylim=limits[2:])
    zoom.set_yticks([.23,.3,.4,.6,.8])
    assert sum(limits[0]<=p['ES']<=limits[1] and limits[2]<=p['cell_ED']<=limits[3] for p in points)==22
    labels={'network_rate_half':('Half network rate',(8,30)),
            'native_MoG_prior_family':('MoG prior',(35,8)),
            'pure_ensemble75':('Selected ensemble',(22,-8)),
            'stage_conditioned_particle_prior_task_extension':('Stage prior',(24,20)),
            'bcap_adam':('BCAP / Adam',(-86,-8))}
    for name,(label,offset) in labels.items():
        p=by[name];zoom.annotate(label,(p['ES'],p['cell_ED']),xytext=offset,textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':'#777777','linewidth':.6})
    full.legend(loc='upper left',fontsize=7.5,frameon=False)
    confirm.set(title='Selected ensemble: fixed-seed confirmation',xlim=(0,10),ylim=(-26,11),
                xlabel='Donor ES improvement vs resampling (%)',ylabel='Cell ED improvement vs resampling (%)')
    confirm.axhspan(0,11,color='#e7f2e9',zorder=0);confirm.axhline(0,color='#666666',linewidth=.8)
    confirm.text(.55,8.5,'Both metrics improve',fontsize=9,color='#2b6a3d')
    for p in relative:
        color='#268044' if p['joint_pass'] else '#bd4938'
        confirm.scatter(p['ES_percent'],p['cell_ED_percent'],s=75,color=color,zorder=2)
        confirm.annotate(f"Seed {p['seed']}",(p['ES_percent'],p['cell_ED_percent']),xytext=(6,6),textcoords='offset points',fontsize=9,color=color)
    confirm.grid(alpha=.14)
    fig.text(.052,.15,f"Three-seed mean donor ES improves {100*c['mean_ES_relative_improvement']:.2f}%, but seeds 1 and 2 worsen cell match. Joint confirmation: FAILED.",fontsize=10,weight='bold')
    fig.text(.052,.104,'Same four adaptive calibration donors; seeds are training repeats. The 20 evaluation donors remain excluded.',fontsize=9,color='#444444')
    fig.text(.052,.064,'Routed E22 has no completed score in this readout. Pareto rings describe saved seed-0 scores; they do not replace the fixed selection rule.',fontsize=9,color='#444444')
    for ext in ('.png','.pdf'):
        assert not prefix.with_suffix(ext).exists();fig.savefig(prefix.with_suffix(ext),dpi=180,facecolor='white')
    plt.close(fig)
    out={'status':'SAVED_23_COMPLETED_SCORE_READOUT_NO_NEW_SELECTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'completed_seed0_comparisons':23,'completed_native_trainer_presets':7,'all_points':points,'nondominated_seed0_points':pareto,
         'fixed_seed_confirmation_relative_metrics':relative,'stable_joint_target_passed':False,
         'source_SHA256':{str(p):sha(p) for p in (Path(__file__).resolve(),leaderboard,confirmation,ROOT/'pure_incumbent.json')},
         'output_SHA256':{str(prefix.with_suffix(ext)):sha(prefix.with_suffix(ext)) for ext in ('.png','.pdf')},
         'incumbent_changed':False,'new_generation_or_training':False,
         'scope':'All completed23savedseed0comparisons,including negative stageprior/Halloween. Original fixedseed0/1/2confirmation retained; no replacement/reranking. Four adaptivecalibdonors,20evaluationexcluded. Native trainer presets are distinct from additional task adaptations and prior families. The selectedensemble has40kparentupdates vs20kfor mostsinglemodels. Descriptive readout, no new score, target access, ranking threshold or generalization claim.'}
    assert not prefix.with_suffix('.json').exists();prefix.with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'PNG':str(prefix.with_suffix('.png')),'PDF':str(prefix.with_suffix('.pdf')),'points':23,'nondominated':pareto},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--leaderboard',type=Path,required=True);parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();plot(args.leaderboard,args.output_prefix)
