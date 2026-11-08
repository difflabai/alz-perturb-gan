"""Read-only V7 forecast including both queued six-fit calibration studies.

This is a conditional cost forecast, not a training authorization. It never
parses a quality-result JSON or imports a model, Torch or a GPU library.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

ROOT = Path('/ml2/alz')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    dispatcher = ROOT/'analysis/continue_benchmark097_after_capacity_v7.py'
    spec = importlib.util.spec_from_file_location('current_capacity_dispatcher', dispatcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = time.time()
    owner = read(module.STATE/'identity.json')
    assert module.identity(owner['pid']) == {k: owner[k] for k in ('pid', 'birth', 'boot')}
    assert owner['source_SHA256'] == sha(dispatcher)
    assert not (module.STATE/'failure.json').exists()
    progress = read(module.STATE/'progress.json')
    assert not progress['errors'] and not progress['unsettled']
    rows = module.successful_costs()
    costs, sources = module.cost_table(rows)
    active = progress['active']
    initial = [0., 0.]
    allocation_remaining = [0., 0.]
    active_rows = []
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for handle in active:
        attempt = handle['attempt']
        auth = read(ROOT/'artifacts/allocations'/attempt/'authorization.json')
        job = read(auth['ledger_path'])['jobs'][0]
        lease = read(Path(auth['registry_path'])/'registry.json')['jobs'][job['lease_token']]
        assert job['status'] == 'active' and lease['status'] == 'running'
        deployment = read(ROOT/'artifacts/allocations'/attempt/'deployment.json')
        assert sha(deployment['authorization_path']) == deployment['authorization_SHA256']
        assert lease['owner'] == job['guard_identity'] == handle['identity']
        assert lease['worker'] == job['child_identity']
        for identity in (lease['owner'], lease['worker']):
            assert identity['boot'] == boot
            stat = Path('/proc', str(identity['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
            assert stat[0] != 'Z' and int(stat[19]) == identity['birth']
        assert -5 < time.time()-lease['heartbeat_unix'] < 30
        elapsed = max(0., now-job['started_unix'])
        estimated = max(0., costs[auth['cell']['method']][module.GPUS.index(auth['gpu_uuid'])]-elapsed)
        initial[module.GPUS.index(auth['gpu_uuid'])] = estimated
        remaining_cap = max(0., job['started_unix'] + job['allowed_seconds'] - now)
        allocation_remaining[module.GPUS.index(auth['gpu_uuid'])] = remaining_cap
        active_rows.append(dict(attempt=attempt, cell=auth['cell'], gpu_uuid=auth['gpu_uuid'], guard=lease['owner'], worker=lease['worker'], elapsed_seconds=elapsed, conditional_remaining_seconds=estimated, remaining_allocation_seconds=remaining_cap, estimate_method='Latest three settled same-method same-card median minus current charged elapsed; not an optimizer-throughput prediction. A zero estimate would not mean completion.'))
    resolved = {path.stem for path in (ROOT/'analysis/benchmark097/results').glob('*.json')}
    active_keys = {h['attempt'] for h in active}
    assert not resolved & active_keys
    cells = module.old.matrix()
    pending = [c for c in cells if module.b.attempt_key(c) not in resolved | active_keys]
    counts = {m: sum(c['method'] == m for c in pending) for m in ('pg', 'scvi', 'cfgen')}
    assert len(cells) == 45 and len(pending) == progress['pending_original_cells']
    assert len(rows) == progress['complete_original_cells']
    assert len(resolved) == len(rows)+3
    from released097_remaining_runtime_balance import balance_counts as unrestricted
    from itertools import product
    def balance(counts, costs, *, initial=(0.,0.)):
        # Match the live V7/original and planned recovery dispatch rule.
        # PG/scVI can use either card; future CFGen uses GPU1 only.
        methods=sorted(counts);candidates=[]
        assert set(methods)==set(costs)
        for m in methods:
            assert type(counts[m]) is int and counts[m]>=0
            assert len(costs[m])==2 and all(math.isfinite(v) and v>0 for v in costs[m])
        for n0 in product(*(range(counts[m]+1) for m in methods)):
            if any(m=='cfgen' and n for m,n in zip(methods,n0)):continue
            n1=tuple(counts[m]-n for m,n in zip(methods,n0))
            loads=[initial[i]+sum(n*costs[m][i] for m,n in zip(methods,ns)) for i,ns in enumerate((n0,n1))]
            candidates.append((max(loads),sum(loads),n0,n1,loads))
        _,_,n0,n1,loads=min(candidates)
        return dict(counts_per_lane={m:[n0[i],n1[i]] for i,m in enumerate(methods)},predicted_lane_seconds=loads,predicted_makespan_seconds=max(loads),CFGen_GPU0_excluded=True,conditional_on_cost_assumptions=True,not_a_native_admission=True)
    # Analytic scheduling boundary and unchanged PG-only allocation checks.
    assert balance({'cfgen':2},{'cfgen':[1.,3.]})['predicted_makespan_seconds']==6.
    assert balance({'pg':6},{'pg':[4.,1.]})['predicted_makespan_seconds']==unrestricted({'pg':6},{'pg':[4.,1.]})['predicted_makespan_seconds']
    originals = balance(counts, costs, initial=initial)
    full_active_allocation = balance(counts, costs, initial=allocation_remaining)
    replacements = balance({'pg': 1, 'scvi': 0, 'cfgen': 2}, costs)
    base = originals['predicted_makespan_seconds']+replacements['predicted_makespan_seconds']
    expires = read(module.scope.WINDOW)['expires_unix']
    active_PG_training_projection=[]
    for row in active_rows:
        if row['cell']['method']!='pg':continue
        auth=read(ROOT/'artifacts/allocations'/row['attempt']/'authorization.json')
        progress=read(Path(auth['output_dir'])/'fit/progress.json')
        steps=progress['step'];assert 0<steps<=20000
        projected=row['elapsed_seconds']*20000/steps+120.
        active_PG_training_projection.append(dict(attempt=row['attempt'],gpu_uuid=row['gpu_uuid'],completed_updates=steps,elapsed_seconds=row['elapsed_seconds'],projected_full20k_plus120s_seconds=projected,training_pace_projection_not_a_settled_model_cost=True,future_shared_service_not_guaranteed=True))
    theta = balance({'pg': 6}, {'pg': costs['pg']})
    theta_window_allowance = 1.25*theta['predicted_makespan_seconds'] + 1200
    scenarios = []
    for multiplier in (1.15, 2., 5.):
        scalar = balance({'pg': 6}, {'pg': [v*multiplier for v in costs['pg']]})
        total = base+scalar['predicted_makespan_seconds']
        all_studies = total + theta_window_allowance
        conservative_allocation_total = (full_active_allocation['predicted_makespan_seconds']
            + replacements['predicted_makespan_seconds'] + scalar['predicted_makespan_seconds']
            + theta_window_allowance)
        scenarios.append(dict(scalar_cost_multiplier=multiplier, scalar_load_forecast=scalar,
            originals_plus_replacements_hours=base/3600,
            scalar_hours=scalar['predicted_makespan_seconds']/3600,
            originals_replacements_scalar_hours=total/3600,
            originals_replacements_scalar_window_margin_hours=(expires-now-total)/3600,
            theta_raw_forecast_hours=theta['predicted_makespan_seconds']/3600,
            theta_window_forecast_with_1p25_safety_and_1200s_cleanup_hours=theta_window_allowance/3600,
            all_current_studies_hours=all_studies/3600,
            all_current_studies_window_margin_hours=(expires-now-all_studies)/3600,
            all_studies_if_active_runs_use_full_remaining_allocations_hours=conservative_allocation_total/3600,
            full_active_allocation_scenario_window_margin_hours=(expires-now-conservative_allocation_total)/3600))
    pace_scenarios=[]
    if active_PG_training_projection:
        projected_costs={m:list(values) for m,values in costs.items()}
        projected_initial=list(initial)
        for row in active_PG_training_projection:
            i=module.GPUS.index(row['gpu_uuid'])
            projected_costs['pg'][i]=row['projected_full20k_plus120s_seconds']
            projected_initial[i]=max(0.,projected_costs['pg'][i]-row['elapsed_seconds'])
        projected_originals=balance(counts,projected_costs,initial=projected_initial)
        projected_replacements=balance({'pg':1,'scvi':0,'cfgen':2},projected_costs)
        projected_theta=balance({'pg':6},{'pg':projected_costs['pg']})
        for multiplier in (1.15,2.,5.):
            scalar=balance({'pg':6},{'pg':[v*multiplier for v in projected_costs['pg']]})
            total=(projected_originals['predicted_makespan_seconds']+projected_replacements['predicted_makespan_seconds']
                +scalar['predicted_makespan_seconds']+1.25*projected_theta['predicted_makespan_seconds']+1200)
            pace_scenarios.append(dict(scalar_cost_multiplier=multiplier,all_current_studies_hours=total/3600,
                all_current_studies_window_margin_hours=(expires-now-total)/3600,
                projected_method_card_costs_seconds=projected_costs,
                extrapolates_one_active_PG_training_pace_to_future_same_card_PG_fits=True,
                not_measured_future_fit_cost_or_admission_guarantee=True))
    controller_path = module.b.PROJECT/'budget_executor_v2_20261103_r2/budget_controller.py'
    assert sha(controller_path) == 'c8520f426e6b7a94741dd96804b96d9d22a224bff973b7e85d765e5b83c09848'
    budget_spec = importlib.util.spec_from_file_location('read_only_budget_status', controller_path)
    budget_module = importlib.util.module_from_spec(budget_spec)
    budget_spec.loader.exec_module(budget_module)
    status = budget_module.Controller(module.b.PROJECT)._status(read(module.b.PROJECT/'project_gpu_budget.json'))
    result = dict(status='CURRENT_V7_CONSTRAINED_COST_ONLY_FORECAST_ORIGINALS_REPLACEMENTS_SCALAR6_THETA6',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        successful_logical_models=len(rows), failed_original_attempts_retained=3,
        active_CPU_owner_identity_verified={k:owner[k] for k in ('pid','birth','boot')},
        active_models_identity_verified=active_rows, pending_untouched_original_counts=counts,
        costs_seconds_by_method_card=costs, cost_source_attempts=sources, untouched_originals=originals,
        untouched_originals_if_active_allocations_fully_consumed=full_active_allocation,
        three_fresh_replacements=replacements, six_theta_fixed_rate_fits=theta, overhead_scenarios=scenarios,
        canonical_budget_snapshot={k:status[k] for k in ('revision','cap_GPU_hours',
            'charged_GPU_hours','reserved_GPU_hours','available_after_reservations_GPU_hours')},
        active_elapsed_hours_not_yet_settled=sum(r['elapsed_seconds'] for r in active_rows)/3600,
        canonical_settled_charge_does_not_include_active_elapsed_as_final=True,
        authorized_window_remaining_hours=(expires-now)/3600,
        authorized_GPU_window_end_UTC=datetime.datetime.fromtimestamp(expires, datetime.timezone.utc).isoformat(),
        scalar_measurement_overhead_measured=False,
        active_PG_training_pace_projections=active_PG_training_projection,
        active_PG_same_card_future_pace_what_if_scenarios=pace_scenarios,
        future_model_costs_still_use_only_settled_complete_models=True,
        current_method_device_exclusion=dict(method='cfgen',GPU_index=0,basis='verified original postfit native cap failure'),
        no_GPU_admission_or_active_source_changes=True, quality_or_evaluation_target_values_read=False,
        caveats='Conditional two-card forecasts from settled costs, with future CFGen restricted toGPU1 and allthree fresh replacements included. Active PG pace projections are supplemental, not future settled costs. Current native handles are reported explicitly; future card eligibility is not guaranteed. Excludes future capacity waits, failures/replacement amendments and most CPU report time. Scalar online-scoring overhead remains unmeasured; multipliers are scenarios, not estimates. Theta has its registered1.25 cost safety factor plus1200s cleanup. All four phases remain sequential. A full remaining-allocation scenario is a duration assumption, not a guarantee those models succeed by their caps. Neither forecast changes the owner window or any live authorization.',
        source_SHA256={str(Path(__file__).resolve()):sha(__file__), str(dispatcher):sha(dispatcher),
            str(ROOT/'analysis/released097_remaining_runtime_balance.py'):sha(ROOT/'analysis/released097_remaining_runtime_balance.py'),
            str(controller_path):sha(controller_path)}, full_goal_complete=False, SOTA_claim=False)
    assert all(math.isfinite(s['all_current_studies_hours']) and s['all_current_studies_hours'] > 0 for s in scenarios)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['UTC', 'successful_logical_models', 'pending_untouched_original_counts', 'canonical_budget_snapshot', 'authorized_window_remaining_hours', 'overhead_scenarios']}))


if __name__ == '__main__':
    main()
