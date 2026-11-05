"""Freeze a released PG comparator before new evaluation; no target values read."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
def read(path):
    return json.loads(Path(path).read_bytes())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    destination = ROOT/'analysis/external_benchmark_frozen_candidate_20261105.json'
    assert not destination.exists(), 'A frozen comparison must not be overwritten'
    pg_path = ROOT/'analysis/CAL_GPU0_V1_released_v095_regkappa050_f0_pg_s0-result.json'
    scvi_path = ROOT/'analysis/CAL_GPU0_V1_matched_scvi_eval1_f0_scvi_s0-result.json'
    baseline_path = ROOT/'analysis/baseline_calibration_seed0.json'
    version_path = ROOT/'artifacts/native-releases/latest_at_external_benchmark_freeze_20261105.json'
    protocol_path = SCIENCE/'CORE_PROTOCOL.json'
    pg_audit, scvi_audit = read(pg_path), read(scvi_path)
    pg, scvi, baseline = pg_audit['candidate'], scvi_audit['candidate'], read(baseline_path)
    assert pg_audit['owned_release_and_clean_settlement_verified']
    assert pg_audit['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    assert scvi_audit['owned_release_and_clean_settlement_verified']
    assert scvi_audit['independent_saved_scores_and_count_exemplars_verified']
    assert pg['status'] == scvi['status'] == 'completed'
    assert pg['completed_updates'] == 20000 and scvi['completed_epochs'] == 400
    assert pg['particlegan_version'] == read(version_path)['version'] == '0.9.5'
    assert pg['released_hyperparameters'] == dict(generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_coeff=1.,reg_kappa=.5,prior_reg=.01)
    assert pg['pure_only'] and pg['real_bag_replacements'] == scvi['real_bag_replacements'] == 0
    for key in ('donors','training_seed','generation_replicates','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
        assert pg[key] == scvi[key] == baseline[key]
    assert pg['energy_score'] <= .95 * baseline['energy_score']
    assert pg['cell_energy_distance'] <= scvi['cell_energy_distance']
    assert .7 <= pg['spread_ratio'] <= 1.3
    checkpoint = Path(pg['checkpoint_path'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT/checkpoint
    assert sha(checkpoint) == pg['checkpoint_SHA256']
    original = read(protocol_path)
    rows = []
    for fold in range(3):
        for method in ('pg','scvi'):
            for seed in range(5):
                existing = fold == seed == 0
                rows.append(dict(fold=fold,method=method,training_seed=seed,
                                 fit_status='EXISTING_MATCHED_FINAL_MODEL_AUDITED' if existing else 'NOT_FIT_FOR_THIS_FROZEN_COMPARISON',
                                 evaluation_status='NOT_RUN_FOR_THIS_FROZEN_COMPARISON',
                                 fit_reuse_requires_native_identity_and_generation_qualification=True))
    evidence = [Path(__file__),pg_path,scvi_path,baseline_path,version_path,protocol_path,
                ROOT/'analysis/released_tuning_v095/protocol.json',
                ROOT/'analysis/released_tuning_v095/native_release.json',
                ROOT/'analysis/matched_scvi/protocol.json',
                ROOT/'analysis/matched_external_scvi_first_result_and_tradeoffs.json']
    result = dict(status='FROZEN_EXTERNAL_BENCHMARK_CANDIDATE_NOT_INCUMBENT_PROMOTION',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  user_request='Test how well we do against external methods, not only our own calibration incumbent.',
                  selection=dict(candidate='released_v095_regkappa050',
                                 data='Four adaptive fold0 calibration donors only; no evaluation score used for this selection.',
                                 rule='Among completed released0.9.5 fixed20k pure trials, require calibration ES at least5percent below matched resampling, valid0.7..1.3 spread and cellED no greater than matched fixed400epoch scVI; choose lowest ES among feasible endpoints.',
                                 rule_timing='This selection rule is adaptive after calibration results, frozen prospectively before the new evaluation. It was not preregistered before calibration.',
                                 fixed_endpoint_only=True,intermediate_checkpoint_selection=False,
                                 original_internal_incumbent_gate_passed=False,
                                 incumbent_changed=False,
                                 purpose='Measure the best current feasible released pure PG configuration in the external benchmark. This does not label it an internal ATH or waive its failed incumbent ES gate.'),
                  PG=dict(version='0.9.5',wheel_SHA256=read(version_path)['wheels'][0]['digests']['sha256'],
                          hyperparameters=pg['released_hyperparameters'],variant_config=pg['variant_config'],
                          fixed_updates=20000,native_recipe=pg['native_recipe'],penalty_options=pg['penalty_options'],
                          fold0_seed0_checkpoint=str(checkpoint),fold0_seed0_checkpoint_SHA256=pg['checkpoint_SHA256'],
                          training_initialization='Fresh per fold/seed; unchanged registered training stream derivation; optimization donors only. Reuse already completed exact matched fold0seed0 model after qualification.'),
                  scVI=dict(version='1.3.3',fixed_epochs=400,formulation=original['scVI_formulation'],
                            fold0_seed0_model='runs/CAL_GPU0_V1_matched_scvi_calibration_f0_scvi_s0/fit/model/model.pt',
                            fold0_seed0_model_SHA256='e413fe3a7290bfb16eef49ee80b6f15eeb893cc580daebb5931d54253233fa67',
                            reload='Already qualified device=auto helper for one visible CUDA card; no refit or data-driven endpoint adjustment.'),
                  comparison=dict(folds=3,training_seeds=[0,1,2,3,4],matrix_cells=30,
                                  additional_comparators='Finish live matched CFGen and prepare scDesign3 in parallel where CPU-only; optional readiness does not block the core PG/scVI/resampling matrix.',
                                  common_trace_manifest_SHA256=pg['trace_manifest_SHA256'],
                                  original_evaluation_generation=original['generation'],
                                  original_statistics=original['statistics'],decision_rules=original['decision_rules'],
                                  no_calibration_equal_stage_reservoir_substitution=True,
                                  no_favorable_model_or_endpoint_replacement_after_evaluation=True,
                                  retain_failures_and_all_seeds=True,
                                  partial_results_are_descriptive_only=True,
                                  final_decision_requires_complete_matrix=True),
                  execution=dict(current_matched_CFGen_fit_unchanged=True,
                                 next_scientific_priority_after_current_phase='Qualify and execute this frozen external core comparison before further PG-only calibration tuning.',
                                 launch_enabled=False,
                                 required_before_launch='Exact source/runtime qualification for final evaluation, native saved-state restore, counts, common resampling, source/resource review and canonical admission. No artificial new owner approval.',
                                 project_GPU_hour_budget=1024,core_GPU_hour_envelope=180,finite_phase_cap_seconds=21600,
                                 sharing_allowed=True,foreign_jobs_unchanged=True),
                  matrix=rows,
                  limitations=['All57cohortdonors historically exposed; cross-fitting is not independent fresh-cohort confirmation.',
                               'PG received many adaptive calibration trials; scVI and CFGen currently fixed recipes, not equal tuning effort.',
                               'No single-cell/celltype replication can replace real donor replication.',
                               'No completed SOTA result or full-matrix claim at freeze time.'],
                  evaluation_target_values_read=False,SOTA_claim=False,goal_complete=False,
                  evidence_SHA256={str(p):sha(p) for p in evidence})
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],path=str(destination),SHA256=sha(destination),matrix_cells=len(rows),
                          calibration_ES=pg['energy_score'],calibration_cellED=pg['cell_energy_distance'])))

if __name__ == '__main__':
    main()
