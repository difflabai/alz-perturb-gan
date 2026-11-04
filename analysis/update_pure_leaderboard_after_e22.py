"""Add one completed, audited E22 model, including both charged attempts."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path('/ml2/alz');ATTEMPT='CAL_GPU0_V1_routed_e22_resume1_f0_pg_s0'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    source=ROOT/'analysis/pure_formulations_leaderboard_after_Halloween.json'
    result_path=ROOT/'analysis'/(ATTEMPT+'-result.json')
    result=read(result_path);candidate=result['candidate']
    assert result['status']=='VERIFIED_NATIVE_ROUTED_E22_COUNT_ADAPTER_RESULT'
    for key in ('independent_saved_scores_checkpoint_and_count_exemplars_verified','full_original_device_native_public_policy_reload_verified',
            'owned_release_and_clean_settlement_verified','native_7013_parent_and_remaining12987_update_lineage_verified'):
        assert result[key] is True
    assert candidate['status']=='completed' and candidate['completed_updates']==candidate['fixed_endpoint']==20000
    assert candidate['initial_completed_updates']==7013 and candidate['new_model_updates']==12987
    assert candidate['training_seed']==0 and candidate['generation_replicates']==2
    assert candidate['integer_count_contract_passed'] and candidate['real_bag_replacements']==0
    assert sha(candidate['checkpoint_path'])==candidate['checkpoint_SHA256']
    row=dict(candidate,formulation='native_routed_E22_count_adapter',complete_clean_settlement_verified=True,
        charged_GPU_seconds=result['charged_GPU_seconds_total_scientific_trial'],accepted=result['decision']['accept'],
        decision_evidence=str(result_path),decision_evidence_SHA256=sha(result_path))
    old=read(source);assert len(old['rows'])==23
    baseline=read(ROOT/'baseline_calibration.json')
    row['ES_improvement_vs_resampling_percent']=100*(1-row['energy_score']/baseline['energy_score'])
    row['cell_ED_difference_vs_resampling']=row['cell_energy_distance']-baseline['cell_energy_distance']
    row['joint_baseline_target_met']=row['ES_improvement_vs_resampling_percent']>=5 and row['cell_ED_difference_vs_resampling']<=0 and .7<=row['spread_ratio']<=1.3
    old.update(status='PURE_COUNT_COMPARISONS_UPDATED_WITH_AUDITED_CONDITIONAL_ROUTED_E22',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prior_leaderboard_path=str(source),prior_leaderboard_SHA256=sha(source),
        rows=sorted([*old['rows'],row],key=lambda r:r['energy_score']),completed_seed0_comparisons=24,
        completed_native_preset_names=sorted(set(old['completed_native_preset_names'])|{'e22_routed'}),
        additional_training_updates_since_prior_leaderboard=20000,additional_charged_GPU_seconds=row['charged_GPU_seconds'],
        routed_E22_result_path=str(result_path),routed_E22_result_SHA256=sha(result_path),
        routed_E22_checkpoint_audit=result['checkpoint_audit'],
        routed_E22_parent_7013_updates_GPU_seconds=result['parent_charged_GPU_seconds'],
        routed_E22_remaining_12987_updates_and_generation_GPU_seconds=result['charged_GPU_seconds'],
        routed_E22_count_task_completed=True,independent_row_E22_or_Atlas_count_task_completed=False,
        current_priority='Released ParticleGAN fixed-family public learning-rate and regularization search; VAE prototype aside.',
        incumbent_changed=False,
        scope='24 completed pure-count seed0 comparisons,8native preset names including the conditional e22_routed adaptation. This is one E22 scientific model across a7013-update GPU0 host-pressure stop and exact GPU1 continuation to20000, not two completed models. Full native CUDA policy and count/score/cleanup/charge audit; clean coupled serving. Original independent-row E22/Atlas conditional-contract refusals remain. AE/VAE are CPU feasibility only, no ALZ quality credit; DDGAN untrained. Same4adaptive calibration donors,20evaluation donors excluded. All3seed stable joint ATH remains unproven.')
    assert sha(ROOT/'pure_incumbent.json')==old['selected_pure_incumbent_SHA256']
    dest=ROOT/'analysis/pure_formulations_leaderboard_after_E22.json';assert not dest.exists()
    dest.write_text(json.dumps(old,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:row[k] for k in ('energy_score','cell_energy_distance','spread_ratio','accepted','charged_GPU_seconds','ES_improvement_vs_resampling_percent','cell_ED_difference_vs_resampling')},indent=2))
if __name__=='__main__':main()
