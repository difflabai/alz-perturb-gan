"""Compare completed native methods on the identical adaptive calibration set."""
import datetime, hashlib, importlib.util, json
from pathlib import Path

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
def read(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=ROOT/'analysis/matched_PG_scVI_CFGen_calibration_20261105.json'
    assert not out.exists(), 'Do not overwrite measured evidence'
    paths={
        'PG095_half_kappa':ROOT/'analysis/CAL_GPU0_V1_released_v095_regkappa050_f0_pg_s0-result.json',
        'scVI133':ROOT/'analysis/CAL_GPU0_V1_matched_scvi_eval1_f0_scvi_s0-result.json',
        'CFGen':ROOT/'analysis/CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0-result.json',
    }
    audits={name:read(path) for name,path in paths.items()}
    candidates={name:q['candidate'] for name,q in audits.items()}
    baseline_path=ROOT/'analysis/baseline_calibration_seed0.json'
    baseline=read(baseline_path)
    freeze_path=ROOT/'analysis/external_benchmark_frozen_candidate_20261105.json'
    frozen=read(freeze_path)
    assert candidates['PG095_half_kappa']['checkpoint_SHA256']==frozen['PG']['fold0_seed0_checkpoint_SHA256']
    for name,q in audits.items():
        c=q['candidate']
        assert q['owned_release_and_clean_settlement_verified'] and c['status']=='completed'
        assert c['real_bag_replacements']==0
        for key in ('donors','training_seed','generation_replicates','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
            assert c[key]==baseline[key],(name,key)
        assert len(c['donors'])==4 and c['training_seed']==0 and c['generation_replicates']==2
    assert audits['PG095_half_kappa']['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    assert audits['scVI133']['independent_saved_scores_and_count_exemplars_verified']
    cfgen=audits['CFGen']
    assert cfgen['native_CPU_checkpoint_reload_matches_fresh_GPU_state']
    assert cfgen['fixed_AE300_FM1500_endpoints_and_encoder_freeze_verified']
    assert cfgen['independent_saved_scores_and_count_exemplars_verified']
    assert cfgen['candidate']['AE_epochs']==300 and cfgen['candidate']['FM_epochs']==1500
    assert cfgen['candidate']['total_optimizer_updates']==102600 and cfgen['candidate']['evaluation_used'] is False
    assert sha(cfgen['candidate']['checkpoint_path'])==cfgen['candidate']['checkpoint_SHA256']
    for path,pin in cfgen['array_files_SHA256'].items(): assert sha(path)==pin
    attempt='CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0'
    auth=read(ROOT/'artifacts/allocations'/attempt/'authorization.json')
    ledger=read(auth['ledger_path']); job=ledger['jobs'][0]
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    for key in ('child_identity','guard_identity'):
        identity=job[key]
        p=Path('/proc')/str(identity['pid'])/'stat'
        assert not p.exists(), 'Expected completed owned process absent'
    state=read(PROJECT/'project_gpu_budget.json')
    assert not any(q['allocation_id']==auth['allocation_id'] for q in state['open_reservations'])
    settlement=[q for q in state['allocation_history'] if q['allocation_id']==auth['allocation_id']]
    assert len(settlement)==1 and settlement[0]['status']=='settled'
    assert settlement[0]['charged_GPU_seconds']==cfgen['charged_GPU_seconds']
    registry=read(Path(auth['registry_path'])/'registry.json')
    assert job['lease_token'] not in registry['jobs']
    budget_source=PROJECT/'project_budget.py'
    assert sha(budget_source)=='85ea495a54dfc524fb493a34002305b534964cf1992fc659d07d2c38951e4872'
    spec=importlib.util.spec_from_file_location('verified_readonly_project_budget',budget_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    status=module.status(PROJECT)
    budget={k:status[k] for k in ('cap_GPU_hours','charged_GPU_hours','reserved_GPU_hours','remaining_GPU_hours')}
    keys=('energy_score','cell_energy_distance','spread_ratio','coverage80','per_donor_energy_score','per_stage_cell_energy_distance')
    methods={'resampling':{k:baseline[k] for k in keys}}
    methods.update({name:{k:c[k] for k in keys} for name,c in candidates.items()})
    pg=candidates['PG095_half_kappa']
    contrasts={name:{key:100*(1-pg[key]/other[key]) for key in ('energy_score','cell_energy_distance')}
               for name,other in methods.items() if name!='PG095_half_kappa'}
    report=dict(status='THREE_NATIVE_MODELS_COMPLETED_MATCHED_CALIBRATION_COMPARISON',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),methods=methods,
                PG_half_kappa_percent_lower=contrasts,
                scope='Same4adaptivecalibrationdonors,fold0seed0,2generationreplicates. Unequal adaptivePG versus fixedscVI/CFGen tuning. No final evaluation, seed-stability or SOTA claim.',
                full_core_matrix='Frozen3fold5seed PG/scVI/resampling comparison remains outstanding; prepare its actual evaluation runtime next before further PG-only tuning.',
                CFGen_actual_optimizer_updates=102600,CFGen_actual_GPU_seconds=cfgen['charged_GPU_seconds'],
                CFGen_completed_and_clean_settled=True,CFGen_owned_processes_absent=True,
                CFGen_spread_outside_band=True,scVI_spread_outside_band=True,
                all_negative_comparator_results_retained=True,incumbent_changed=False,
                PG_internal_incumbent_gate_passed=False,goal_complete=False,SOTA_claim=False,
                canonical_budget=budget,
                activation_to_disable_after_commit='3e97e64',
                evidence_SHA256={str(p):sha(p) for p in [Path(__file__),*paths.values(),baseline_path,freeze_path,budget_source]})
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','methods','PG_half_kappa_percent_lower','canonical_budget')},indent=2))

if __name__=='__main__':main()
