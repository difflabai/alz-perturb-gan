"""Audit one-field latest-release quality changes against the completed reference."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/ml2/alz')
sys.path.insert(0,str(ROOT/'analysis'))
from aggregate_validation import scientific_sources
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(attempt,prefix):
    reference_attempt='CAL_GPU0_V1_released_v095_bridge_f0_pg_s0'
    paths=[ROOT/'analysis'/(a+'-result.json') for a in (reference_attempt,attempt)]
    results=[read(p) for p in paths];reference,candidate=[r['candidate'] for r in results]
    for result in results:
        for key in ('owned_release_and_clean_settlement_verified','official095_origin_version_and_continuous_domain_verified',
                    'independent_saved_scores_checkpoint_and_count_exemplars_verified',
                    'predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                    'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified'):
            assert result[key] is True,key
    protocol_path=ROOT/'analysis/released_penalty_v095/protocol.json';protocol=read(protocol_path)
    settings=[s for s in protocol['stage1']['configs'] if s['id']==candidate['tuning_candidate_id']]
    assert len(settings)==1;setting=settings[0]
    assert candidate['preset_protocol_SHA256']==sha(protocol_path)
    assert candidate['released_hyperparameters']==setting['parameters']
    assert reference['released_hyperparameters']==protocol['baseline_parameters']
    differences={k for k in reference['released_hyperparameters']
                 if reference['released_hyperparameters'][k]!=candidate['released_hyperparameters'][k]}
    assert differences=={setting['changed_field']}
    for key in ('training_seed','generation_replicates','donors','sampling_contract','variant_config',
                'calibration_manifest_SHA256','trace_manifest_SHA256','native_recipe','particlegan_version','penalty_options'):
        assert reference[key]==candidate[key],key
    assert candidate['particlegan_version']=='0.9.5' and candidate['penalty_options']=={'input_domain':'continuous'}
    assert candidate['training_seed']==0 and candidate['generation_replicates']==2
    for c in (reference,candidate):
        assert c['status']=='completed' and c['completed_updates']==c['fixed_endpoint']==20000
        assert c['integer_count_contract_passed'] and c['real_bag_replacements']==0
        assert sha(c['checkpoint_path'])==c['checkpoint_SHA256']
    auth_paths=[ROOT/'artifacts/allocations'/a/'authorization.json' for a in (reference_attempt,attempt)]
    auths=[read(p) for p in auth_paths]
    sources=[scientific_sources(a) for a in auths];assert sources[0]==sources[1]
    assert all(sha(p)==h for p,h in sources[0].items())
    curves=[];curve_paths=[]
    for c in (reference,candidate):
        path=Path(c['convergence_curve_path']);assert sha(path)==c['convergence_curve_SHA256']
        curve=read(path);assert curve['training_finished_before_calibration'] is True
        assert [e['completed_updates'] for e in curve['entries']]==[5000,10000,20000]
        curves.append(curve['entries']);curve_paths.append(path)
    donor_deltas=[{'donor':donor,'ES_difference_candidate_minus_reference':c-r}
        for donor,r,c in zip(reference['donors'],reference['per_donor_energy_score'],candidate['per_donor_energy_score'])]
    assert len(donor_deltas)==4
    report={'status':'AUDITED_OFFICIAL095_ONE_FIELD_PENALTY_QUALITY_COMPARISON',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference_attempt':reference_attempt,'attempt':attempt,
        'changed_field':setting['changed_field'],'factor':setting['factor'],
        'reference_parameters':reference['released_hyperparameters'],'candidate_parameters':candidate['released_hyperparameters'],
        'reference20k':{k:reference[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
        'candidate20k':{k:candidate[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
        'difference_candidate_minus_reference':{k:candidate[k]-reference[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
        'per_donor_ES_differences':donor_deltas,
        'per_stage_cell_ED_differences':[c-r for r,c in zip(reference['per_stage_cell_energy_distance'],candidate['per_stage_cell_energy_distance'])],
        'pooled_cell_ED_is_not_the_mean_of_per_stage_ED':True,
        'candidate_dominates_reference_ES_and_cell_ED':candidate['energy_score']<reference['energy_score'] and candidate['cell_energy_distance']<=reference['cell_energy_distance'],
        'joint_frozen_incumbent_decision':results[1]['decision'],
        'all_scientific_source_hashes_equal_and_current':True,'scientific_sources_count':len(sources[0]),
        'sources_SHA256':{str(p):sha(p) for p in [Path(__file__),protocol_path,*paths,*auth_paths,*curve_paths]},
        'only_final20k_checkpoint_eligible':True,'incumbent_changed':False,'stable_joint_ATH_claimed':False,
        'scope':'Actual completed audited0.9.5one-field comparison, same four adaptive calibration donors and two generation replicates. Descriptive donor/stage differences, no biological replication or inference claim;5k/10kdiagnostics do not select the endpoint. Evaluation excluded.'}
    prefix=Path(prefix)
    for suffix in ('.json','.png','.pdf'):assert not prefix.with_suffix(suffix).exists()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    incumbent=read(ROOT/'pure_incumbent.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14.5,5.5));fig.subplots_adjust(left=.065,right=.98,top=.73,bottom=.24,wspace=.3)
    fig.suptitle('ParticleGAN 0.9.5: one-field regularization control',x=.065,y=.96,ha='left',fontsize=16,weight='bold')
    field=setting['changed_field']
    fig.text(.065,.875,f"{field}: {reference['released_hyperparameters'][field]:g} → {candidate['released_hyperparameters'][field]:g}; G 0.006 / D 0.009 / particles 0.015 fixed",fontsize=11)
    fig.text(.065,.815,'Same model sources, seed 0 and sampling traces; only final 20k endpoints select candidates.',fontsize=10)
    for ax,key,title in zip(axes[:2],('energy_score','cell_energy_distance'),('Donor energy score (lower is better)','Pooled cell energy distance (lower is better)')):
        ax.axhline(incumbent[key],color='#a56817',label='Frozen incumbent')
        for entries,label,color in zip(curves,('Reference','Penalty control'),('#777777','#176ba0')):
            ax.plot([e['completed_updates']/1000 for e in entries],[e['metrics'][key] for e in entries],marker='o',label=label,color=color)
        ax.set_title(title,loc='left',fontsize=10);ax.set_xlim(0,21);ax.set_xticks([5,10,20]);ax.set_xlabel('Completed updates (thousands)');ax.grid(alpha=.17)
    ax=axes[2];delta=[x['ES_difference_candidate_minus_reference'] for x in donor_deltas]
    ax.bar(range(4),delta,color=['#176ba0' if d<=0 else '#b95c3e' for d in delta]);ax.axhline(0,color='#777777')
    ax.set_xticks(range(4),reference['donors'],rotation=20,ha='right',fontsize=8)
    ax.set_title('Per-donor ES: control − reference',loc='left',fontsize=10);ax.set_ylabel('Negative values improve');ax.grid(axis='y',alpha=.17)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.06,.105),ncol=3,frameon=False,fontsize=9)
    fig.text(.065,.05,'Four adaptive calibration donors. Curves are saved-checkpoint diagnostics; generation replicates add no biological donors.',fontsize=9)
    fig.savefig(prefix.with_suffix('.png'),dpi=150);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    report['PNG_SHA256']=sha(prefix.with_suffix('.png'));report['PDF_SHA256']=sha(prefix.with_suffix('.pdf'))
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','changed_field','factor','difference_candidate_minus_reference','joint_frozen_incumbent_decision')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    compare(args.attempt,args.output_prefix)
