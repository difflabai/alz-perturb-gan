"""Full audited calibration search; run-to-run SD is not biological uncertainty."""
import hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    audit_path=ROOT/'analysis/local097_pipeline_independent_audit_20261107.json'
    protocol_path=ROOT/'analysis/confirmfaith097/protocol.json'
    audit=json.loads(audit_path.read_text());protocol=json.loads(protocol_path.read_text())
    assert audit['status']=='INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS'
    assert audit['required_attempts']==16 and audit['required_new_attempts']==12
    candidates=audit['summary']['candidates'];assert len(candidates)==8
    selected={c['candidate_id'] for c in protocol['selected_configurations']}
    assert selected=={'t004','t005','f000'}
    metrics=['cell_energy_distance','energy_score'];rows=[]
    labels={'f000':'Nominal ★','f001':'G / D / P rates and κ ×0.5',
        't000':'Prior weight ×0.3','t001':'Prior weight ×3','t002':'κ ×2',
        't003':'Generator LR ×2','t004':'Critic LR ×2 ★','t005':'Particle LR ×2 ★'}
    for c in candidates:
        assert [r['seed'] for r in c['per_seed']]==[0,1] and c['all_seeds_finite']
        rows.append(dict(id=c['candidate_id'],label=labels[c['candidate_id']],
            mean={m:c['mean'][m] for m in metrics},SD={m:c['SD'][m] for m in metrics}))
    controls=[r['baseline_metrics'] for r in candidates[0]['per_seed']]
    for c in candidates:assert [r['baseline_metrics'] for r in c['per_seed']]==controls
    baseline=dict(id='resampling',label='Matched resampling',
        mean={m:statistics.mean(r[m] for r in controls) for m in metrics},
        SD={m:statistics.stdev(r[m] for r in controls) for m in metrics})
    rows=sorted([*rows,baseline],key=lambda r:(r['mean']['cell_energy_distance'],r['mean']['energy_score']))
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.3,6.7),sharey=True)
    fig.subplots_adjust(left=.285,right=.97,bottom=.25,top=.80,wspace=.14)
    colors={'chosen':'#2364a1','other':'#9aa3ad','control':'#263346'}
    for ax,metric,title,limits in zip(axes,metrics,['Pooled cell energy distance ↓','Donor energy score ↓'],[(.20,.49),(1.72,2.44)]):
        ax.axvline(baseline['mean'][metric],color='#8b96a3',ls=':',lw=1)
        for y,r in enumerate(rows):
            if y%2==0:ax.axhspan(y-.47,y+.47,color='#f4f6f8',zorder=0)
            chosen=r['id'] in selected;control=r['id']=='resampling'
            color=colors['control' if control else 'chosen' if chosen else 'other']
            ax.errorbar(r['mean'][metric],y,xerr=r['SD'][metric],fmt='s' if control else 'D' if chosen else 'o',
                color=color,ecolor=color,markersize=5.5,capsize=3,lw=1.1,zorder=3)
            ax.text(.99,y,f"{r['mean'][metric]:.4f}",transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color='#263346')
        ax.set(xlim=limits,xlabel=title,ylim=(len(rows)-.35,-.65))
        ax.set_yticks(range(len(rows)));ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    axes[0].set_yticklabels([r['label'] for r in rows],fontsize=9)
    axes[1].tick_params(axis='y',left=False,labelleft=False)
    fig.text(.04,.94,'Released ParticleGAN: full 20,000-update calibration search',fontsize=14,weight='bold',color='#18283c')
    fig.text(.04,.89,'All eight settings and matched resampling; sorted by mean cell distance. Lower is better.',fontsize=10,color='#46566a')
    fig.legend(handles=[Line2D([0],[0],marker='D',ls='',color=colors['chosen'],label='★ Frozen for fresh-seed confirmation'),
        Line2D([0],[0],marker='o',ls='',color=colors['other'],label='Other PG settings'),
        Line2D([0],[0],marker='s',ls='',color=colors['control'],label='Resampling')],
        loc='upper center',bbox_to_anchor=(.63,.85),ncol=3,fontsize=8,frameon=False)
    fig.text(.04,.055,'Means ± sample SD across two complete runs (seeds 0/1); two generation replicates per run.\n'
        'Same four adaptive calibration donors. Error bars show run variability, not a biological confidence interval.\n'
        'Nominal G / D / P rates = 0.012 / 0.018 / 0.03; κ = 1; critic penalty weight = 1; prior weight = 0.01.\n'
        'Three settings now require fresh seeds 2/3/4; frozen external comparison and any SOTA conclusion remain pending.',
        fontsize=8,color='#46566a',linespacing=1.6)
    outputs=[]
    for ext in ('png','svg'):
        p=ROOT/'analysis'/('released097_full_calibration_search_20261107.'+ext);assert not p.exists()
        fig.savefig(p,dpi=210,facecolor='white');outputs.append(p)
    plt.close(fig)
    proof=dict(status='FULL_SEARCH_CALIBRATION_PLOT_RENDERED_PENDING_VISUAL_QA',
        source_SHA256={str(Path(__file__).resolve()):sha(__file__)},
        input_SHA256={str(p):sha(p) for p in (audit_path,protocol_path)},
        output_SHA256={str(p):sha(p) for p in outputs},plotted_rows=rows,
        original_scores_changed=False,new_scores_computed=False,new_model_fits=0,
        external_benchmark_values_plotted=False,SOTA_claim=False,full_goal_complete=False)
    p=ROOT/'analysis/released097_full_calibration_plot_provenance_20261107.json';assert not p.exists()
    p.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(outputs[0])
if __name__=='__main__':main()
