"""Saved native learning curves and independently audited calibration endpoints."""
import argparse,datetime,hashlib,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/ml2/alz')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plot(attempt,prefix):
    assert re.fullmatch(r'CAL_GPU0_V1_released_rate_[a-z0-9_]+_f0_pg_s0',attempt)
    authpath=ROOT/'artifacts/allocations'/attempt/'authorization.json'
    auth=read(authpath);out=Path(auth['output_dir']);hyper=auth['released_hyperparameters']
    protocol=read(ROOT/'analysis/released_tuning/protocol.json')
    assert auth['native_recipe']=='bispf_optimized' and auth['fixed_endpoint']==20000
    assert [s for s in protocol['stage1']['configs'] if s['id']==auth['tuning_candidate_id']]==[{'id':auth['tuning_candidate_id'],'parameters':hyper}]
    logfile=out/'fit/learning_curve.jsonl';raw=logfile.read_bytes()
    complete=b''.join(line for line in raw.splitlines(keepends=True) if line.endswith(b'\n'))
    records=[json.loads(line) for line in complete.splitlines()]
    assert records and records[0]['step']==1
    steps=np.array([r['step'] for r in records]);assert (np.diff(steps)>0).all() and steps[-1]<=20000
    for row in records:
        assert all(np.isfinite(row[k]) for k in ('D_total','G_total','D_penalty','G_adversarial','D_adversarial'))
        assert row['particle_usage']['total_draws']==32*row['step']
        assert row['particle_usage']['total_unsampled_row_moves']==0
    incumbent=read(ROOT/'pure_incumbent.json');baseline=read(ROOT/'analysis/baseline_calibration_seed0.json')
    assert sha(ROOT/'pure_incumbent.json')==auth['frozen_pure_incumbent_SHA256']
    curvepath=out/'convergence_curve.json';resultpath=ROOT/'analysis'/(attempt+'-result.json')
    entries=[]
    if resultpath.exists():
        result=read(resultpath)
        assert result['predeclared_all_three_checkpoint_saved_scores_and_counts_verified'] is True
        assert result['owned_release_and_clean_settlement_verified'] is True
        assert result['candidate']['convergence_curve_SHA256']==sha(curvepath)
        entries=read(curvepath)['entries']
        assert [e['completed_updates'] for e in entries]==[5000,10000,20000]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(13,11.2))
    fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.11,hspace=.52,wspace=.25)
    rates=', '.join(f'{label} {hyper[key]:g}' for label,key in [('G','generator_lr'),('D','critic_lr'),('particles','particle_lr')])
    fig.suptitle('Released ParticleGAN: learning-rate convergence readout',x=.075,y=.965,ha='left',fontsize=17,weight='bold')
    fig.text(.075,.925,f"Native bispf_optimized | fold 0, seed 0 | {rates}",fontsize=10)
    status='audited complete calibration curve' if entries else 'calibration audit pending'
    fig.text(.075,.895,f"Latest saved update {steps[-1]:,} / 20,000 | {status}",fontsize=10,color='#444444')
    for ax in axes.flat:
        ax.set_xlim(0,20000);ax.set_xticks([0,5000,10000,15000,20000]);ax.set_xlabel('Completed optimizer pairs')
        ax.ticklabel_format(axis='x',style='plain');ax.grid(alpha=.17)
    for ax,key,title in [(axes[0,0],'energy_score','Donor energy score (lower is better)'),
                         (axes[0,1],'cell_energy_distance','Pooled cell energy distance (lower is better)')]:
        ax.axhline(incumbent[key],color='#b66d16',linewidth=1.3,label='Frozen seed-0 incumbent')
        ax.axhline(baseline[key],color='#777777',linestyle='--',linewidth=1,label='Matched resampling, seed 0')
        if entries:
            ax.plot([e['completed_updates'] for e in entries],[e['metrics'][key] for e in entries],marker='o',
                linewidth=1.7,color='#176ba0',label='Audited checkpoint scores')
        else:
            ax.text(.5,.48,'Calibration endpoints pending',transform=ax.transAxes,ha='center',color='#555555')
        values=[incumbent[key],baseline[key],*[e['metrics'][key] for e in entries]]
        margin=.12*max(max(values)-min(values),max(abs(v) for v in values)*.05)
        ax.set_ylim(min(values)-margin,max(values)+margin)
        ax.set_title(title,loc='left',fontsize=11);ax.legend(fontsize=8,loc='best')
    ax=axes[1,0]
    for key,label,color in [('G_adversarial','Generator','#176ba0'),('D_adversarial','Critic','#b65d37')]:
        ax.plot(steps,[r[key] for r in records],label=label,color=color,linewidth=1.1,marker='.',markersize=3)
    ax.set_title('Native adversarial losses',loc='left',fontsize=11);ax.set_ylabel('Saved loss');ax.legend(fontsize=8)
    ax=axes[1,1]
    zero_gradient_records={}
    for key,label,color in [('G_gradient_norm','Generator','#176ba0'),('D_gradient_norm','Critic','#b65d37'),('prior_gradient_norm','Particles','#5c8b41')]:
        sampled=[r for r in records if r[key] is not None]
        assert all(np.isfinite(r[key]) and r[key]>=0 for r in sampled)
        zero_gradient_records[key]=sum(r[key]==0 for r in sampled)
        ax.plot([r['step'] for r in sampled],[r[key] for r in sampled],label=label,color=color,linewidth=1.1,marker='.',markersize=3)
    if any(zero_gradient_records.values()):
        ax.set_yscale('symlog',linthresh=1e-6);ax.set_ylabel('L2 norm (symmetric log scale)')
    else:ax.set_yscale('log');ax.set_ylabel('L2 norm (log scale)')
    ax.set_title('Observed gradient norms',loc='left',fontsize=11);ax.legend(fontsize=8)
    ax=axes[2,0]
    ax.plot(steps,[r['particle_usage']['particle_covariance_trace'] for r in records],color='#5c8b41',linewidth=1.3)
    ax.set_title('Latent particle covariance trace',loc='left',fontsize=11);ax.set_ylabel('Sum of latent coordinate variances')
    ax=axes[2,1]
    for getter,label,color in [(lambda r:r['G_prior_LRs'][0],'Generator','#176ba0'),
                              (lambda r:r['D_LRs'][0],'Critic','#b65d37'),
                              (lambda r:r['G_prior_LRs'][1],'Particles','#5c8b41')]:
        values=[getter(r) for r in records];assert all(np.isfinite(v) and v>0 for v in values)
        ax.plot(steps,values,label=label,color=color,linewidth=1.4)
    ax.set_title('Actual native optimizer learning rates',loc='left',fontsize=11);ax.set_ylabel('Learning rate');ax.legend(fontsize=8)
    fig.text(.075,.058,'Fixed 20,000-update endpoint selects the model. The 5k and 10k checkpoints are convergence diagnostics.',fontsize=9,color='#444444')
    fig.text(.075,.034,'Four adaptive calibration donors; two generation replicates per checkpoint. Evaluation donors are separate.',fontsize=9,color='#444444')
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    snapshot=prefix.with_suffix('.learning_curve.jsonl');snapshot.write_bytes(complete)
    png=prefix.with_suffix('.png');pdf=prefix.with_suffix('.pdf')
    fig.savefig(png,dpi=160);fig.savefig(pdf);plt.close(fig)
    metadata={'status':'SAVED_NATIVE_CONVERGENCE_READOUT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'attempt':attempt,'candidate_id':auth['tuning_candidate_id'],'nominal_hyperparameters':hyper,
        'latest_saved_update':int(steps[-1]),'saved_records':len(records),'audited_calibration_endpoints':len(entries),
        'zero_gradient_norm_records':zero_gradient_records,
        'training_complete':int(steps[-1])==20000,'calibration_audited':bool(entries),
        'truncated_trailing_line_ignored':complete!=raw,'plot_source_path':str(Path(__file__).resolve()),'plot_source_SHA256':sha(__file__),
        'authorization_path':str(authpath),'authorization_SHA256':sha(authpath),
        'original_learning_curve_path':str(logfile),'learning_curve_snapshot_path':str(snapshot),'learning_curve_snapshot_SHA256':sha(snapshot),
        'calibration_curve_path':str(curvepath) if entries else None,'calibration_curve_SHA256':sha(curvepath) if entries else None,
        'audited_result_path':str(resultpath) if entries else None,'audited_result_SHA256':sha(resultpath) if entries else None,
        'PNG_path':str(png),'PNG_SHA256':sha(png),'PDF_path':str(pdf),'PDF_SHA256':sha(pdf),
        'incumbent_changed':False,'ATH_claimed':False,
        'scope':'Read-only saved measurements. Raw loss/gradient/latent covariance curves diagnose optimization; they do not establish count distribution quality. Calibration shown only after all three checkpoint scores pass independent saved-array/count/settlement audit. No early stopping or best-checkpoint selection.'}
    prefix.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k in ('status','latest_saved_update','saved_records','audited_calibration_endpoints','training_complete','calibration_audited','PNG_path')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();plot(args.attempt,args.output_prefix)
