"""Add only audited final seed0 rate-grid models; preserve all prior negatives."""
import argparse,datetime,hashlib,json
from pathlib import Path
ROOT=Path('/ml2/alz')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(destination):
    source=ROOT/'analysis/pure_formulations_leaderboard_after_E22.json';old=read(source)
    protocolpath=ROOT/'analysis/released_tuning/protocol.json';protocol=read(protocolpath)
    baseline=read(ROOT/'analysis/baseline_calibration_seed0.json')
    assert len(old['rows'])==old['completed_seed0_comparisons']==24
    settings={s['id']:s['parameters'] for s in protocol['stage1']['configs']}
    new={}
    for path in (ROOT/'analysis').glob('CAL_GPU0_V1_*-result.json'):
        result=read(path);candidate=result.get('candidate',{});identity=candidate.get('tuning_candidate_id')
        if identity not in settings or candidate.get('training_seed')!=0:continue
        assert identity not in new,'Duplicate completed setting needs explicit scientific lineage reconciliation'
        for key in ('owned_release_and_clean_settlement_verified','predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                    'independent_saved_scores_checkpoint_and_count_exemplars_verified','native_public_CUDA_reload_and_optimizer_state_byte_exact_verified'):
            assert result[key] is True
        assert candidate['status']=='completed' and candidate['completed_updates']==candidate['fixed_endpoint']==20000
        assert candidate['released_hyperparameters']==settings[identity]
        assert candidate['native_recipe']=='bispf_optimized' and candidate['pure_formulation']=='released_hyperparameters'
        assert candidate['generation_replicates']==2 and candidate['integer_count_contract_passed']
        assert candidate['calibration_manifest_SHA256']==baseline['calibration_manifest_SHA256']
        assert candidate['trace_manifest_SHA256']==baseline['trace_manifest_SHA256']
        assert candidate['real_bag_replacements']==0 and candidate['preset_protocol_SHA256']==sha(protocolpath)
        assert sha(candidate['checkpoint_path'])==candidate['checkpoint_SHA256']
        if candidate.get('resumed_same_model'):
            assert result['native10842_parent_all_scientific_streams_and_diagnostics_lineage_verified'] is True
        charged=result.get('charged_GPU_seconds_total_scientific_trial',result['charged_GPU_seconds'])
        row=dict(candidate,formulation=identity,complete_clean_settlement_verified=True,
            charged_GPU_seconds=charged,accepted=result['decision']['accept'],decision_evidence=str(path),decision_evidence_SHA256=sha(path))
        row['ES_improvement_vs_resampling_percent']=100*(1-row['energy_score']/baseline['energy_score'])
        row['cell_ED_difference_vs_resampling']=row['cell_energy_distance']-baseline['cell_energy_distance']
        row['joint_baseline_target_met']=row['ES_improvement_vs_resampling_percent']>=5 and row['cell_ED_difference_vs_resampling']<=0 and .7<=row['spread_ratio']<=1.3
        new[identity]=row
    assert new,'No independently audited complete rate configuration yet'
    old.update(status='PURE_COUNT_COMPARISONS_WITH_AUDITED_RELEASED_RATE_SETTINGS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prior_leaderboard_path=str(source),prior_leaderboard_SHA256=sha(source),
        rows=sorted([*old['rows'],*new.values()],key=lambda r:r['energy_score']),completed_seed0_comparisons=24+len(new),
        completed_released_rate_configurations=len(new),declared_released_rate_configurations=27,
        additional_training_updates_since_prior_leaderboard=20000*len(new),additional_charged_GPU_seconds=sum(r['charged_GPU_seconds'] for r in new.values()),
        released_rate_protocol_path=str(protocolpath),released_rate_protocol_SHA256=sha(protocolpath),
        incumbent_changed=False,stable_joint_ATH_claimed=False,
        scope='All previous24 pure-count seed0 comparisons plus complete independently audited released bispf_optimized rate settings. Each scientific model counts once across operational checkpoint continuations; lineage charges summed. Intermediate5k/10k diagnostic checkpoints do not count as completed comparisons or select models. Eight native preset names unchanged; all negative results retained. Four adaptive calibration donors,20evaluation donors excluded; stable3seed joint ATH remains unproven.')
    assert sha(ROOT/'pure_incumbent.json')==old['selected_pure_incumbent_SHA256']
    Path(destination).write_text(json.dumps(old,indent=2,allow_nan=False)+'\n')
    return old,new
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    leaderboard,rows=build(args.output)
    print(json.dumps({'complete_seed0_comparisons':leaderboard['completed_seed0_comparisons'],
        'completed_rate_settings':len(rows),'settings':[{k:r[k] for k in ('formulation','energy_score','cell_energy_distance','spread_ratio','accepted','charged_GPU_seconds')} for r in rows.values()]},indent=2))
