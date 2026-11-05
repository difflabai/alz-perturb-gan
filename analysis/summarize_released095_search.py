"""Read audited latest-release outcomes without pooling historical executions."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/ml2/alz')
sys.path.insert(0,str(ROOT))
from selection import decide
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    paths=[ROOT/'analysis/released_tuning_v095/protocol.json',ROOT/'analysis/released_penalty_v095/protocol.json']
    protocols=[read(p) for p in paths]
    settings={s['id']:(s['parameters'],p,sha(p)) for p,protocol in zip(paths,protocols) for s in protocol['stage1']['configs']}
    assert len(settings)==27+12
    incumbent_path=ROOT/'pure_incumbent.json';incumbent=read(incumbent_path)
    assert sha(incumbent_path)=='869c9a55c3f3bb5628a9e0ab44357f83297a5b74b8cb7d67faf420794de487fa'
    baseline=read(ROOT/'analysis/baseline_calibration_seed0.json')
    rows=[];completed=set()
    for path in sorted((ROOT/'analysis').glob('CAL_GPU0_V1_*-result.json')):
        result=read(path);candidate=result.get('candidate',{})
        if candidate.get('particlegan_version')!='0.9.5':continue
        identity=candidate['tuning_candidate_id'];assert identity in settings and identity not in completed
        hyper,protocol_path,protocol_sha=settings[identity]
        for key in ('owned_release_and_clean_settlement_verified',
                    'official095_origin_version_and_continuous_domain_verified',
                    'independent_saved_scores_checkpoint_and_count_exemplars_verified',
                    'predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                    'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified'):
            assert result[key] is True,(path,key)
        assert candidate['released_hyperparameters']==hyper and candidate['preset_protocol_SHA256']==protocol_sha
        assert candidate['status']=='completed' and candidate['completed_updates']==candidate['fixed_endpoint']==20000
        assert candidate['penalty_options']=={'input_domain':'continuous'}
        assert candidate['training_seed']==0 and candidate['generation_replicates']==2
        assert candidate['integer_count_contract_passed'] and candidate['real_bag_replacements']==0
        assert candidate['calibration_manifest_SHA256']==baseline['calibration_manifest_SHA256']
        assert candidate['trace_manifest_SHA256']==baseline['trace_manifest_SHA256']
        assert sha(candidate['checkpoint_path'])==candidate['checkpoint_SHA256']
        assert result['decision']==decide(candidate,incumbent)
        curve=read(candidate['convergence_curve_path'])
        assert sha(candidate['convergence_curve_path'])==candidate['convergence_curve_SHA256']
        assert [e['completed_updates'] for e in curve['entries']]==[5000,10000,20000]
        rows.append(dict(candidate_id=identity,study='rate_grid' if identity.startswith('v095_lr_') else 'targeted_penalty_followup',
            energy_score=candidate['energy_score'],cell_energy_distance=candidate['cell_energy_distance'],spread_ratio=candidate['spread_ratio'],
            released_hyperparameters=hyper,decision=result['decision'],charged_GPU_seconds=result['charged_GPU_seconds'],
            result_path=str(path),result_SHA256=sha(path),protocol_path=str(protocol_path),
            ES_improvement_vs_resampling_percent=100*(1-candidate['energy_score']/baseline['energy_score']),
            cell_ED_difference_vs_resampling=candidate['cell_energy_distance']-baseline['cell_energy_distance']))
        completed.add(identity)
    allocations=[]
    for path in sorted((ROOT/'artifacts/allocations').glob('*/authorization.json')):
        auth=read(path)
        if auth.get('particlegan_version')!='0.9.5':continue
        ledger=Path(auth['ledger_path']);jobs=read(ledger).get('jobs',[]) if ledger.exists() else []
        allocations.append({'attempt':auth['attempt_id'],'candidate_id':auth['tuning_candidate_id'],
            'ledger_status':jobs[0].get('status') if jobs else 'not_started',
            'reservation_released':jobs[0].get('reservation_released') if jobs else None,
            'independently_audited_result_present':auth['tuning_candidate_id'] in completed,
            'process_liveness_checked':False})
    rate_count=sum(r['study']=='rate_grid' for r in rows)
    result={'status':'LATEST_RELEASE095_AUDITED_SEARCH_READOUT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'native_version':'0.9.5','completed_latest_release_seed0_endpoints':len(rows),
        'completed_latest_rate_grid_configurations':rate_count,'declared_latest_rate_grid_configurations':27,
        'completed_targeted_penalty_configurations':len(rows)-rate_count,'declared_targeted_penalty_configurations':12,
        'historical0_9_4_rate_configurations':8,'historical0_9_4_endpoints_counted_as0_9_5':False,
        'rows':sorted(rows,key=lambda r:r['energy_score']),'allocations':allocations,
        'original_full_rate_grid_complete':rate_count==27,'original_full_grid_stage2_anchor_selected':False,
        'targeted_penalty_priority_declared_before_bridge_scores':True,
        'fixed_incumbent':{k:incumbent[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
        'incumbent_changed':False,'stable_joint_ATH_claimed':False,'evaluation_used_for_tuning':False,
        'sources_SHA256':{str(p):sha(p) for p in [Path(__file__),incumbent_path,*paths]},
        'scope':'Only independently audited0.9.5 executions populate rows. Historical0.9.4 results remain separate. The targeted12penalty controls are a prospective priority amendment, not completion of the original27point grid or its stage2 anchor. Allocation status is ledger evidence, not current process liveness. No automatic selection/promotion.'}
    destination=ROOT/'analysis/released095_search_readout.json'
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources_SHA256','scope')},indent=2))


if __name__=='__main__':build()
