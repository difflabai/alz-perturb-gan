"""Read-only V5 forecast including both queued six-fit calibration studies.

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
    dispatcher = ROOT/'analysis/continue_benchmark097_after_capacity_v5.py'
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
    assert len(resolved) == len(rows)+2
    from released097_remaining_runtime_balance import balance_counts as balance
    originals = balance(counts, costs, initial=initial)
    full_active_allocation = balance(counts, costs, initial=allocation_remaining)
    replacements = balance({'pg': 1, 'scvi': 0, 'cfgen': 1}, costs)
    base = originals['predicted_makespan_seconds']+replacements['predicted_makespan_seconds']
    expires = read(module.scope.WINDOW)['expires_unix']
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
    controller_path = module.b.PROJECT/'budget_executor_v2_20261103_r2/budget_controller.py'
    assert sha(controller_path) == 'c8520f426e6b7a94741dd96804b96d9d22a224bff973b7e85d765e5b83c09848'
    budget_spec = importlib.util.spec_from_file_location('read_only_budget_status', controller_path)
    budget_module = importlib.util.module_from_spec(budget_spec)
    budget_spec.loader.exec_module(budget_module)
    status = budget_module.Controller(module.b.PROJECT)._status(read(module.b.PROJECT/'project_gpu_budget.json'))
    result = dict(status='CURRENT_V5_COST_ONLY_FORECAST_ORIGINALS_REPLACEMENTS_SCALAR6_THETA6',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_goal_turn_classification='PROGRESS: prepared an independent theta endpoint audit,11 tests and40 existing calibration-array recomputations passed; active runtime sources unchanged.',
        successful_logical_models=len(rows), failed_original_attempts_retained=2,
        active_CPU_owner_identity_verified={k:owner[k] for k in ('pid','birth','boot')},
        active_models_identity_verified=active_rows, pending_untouched_original_counts=counts,
        costs_seconds_by_method_card=costs, cost_source_attempts=sources, untouched_originals=originals,
        untouched_originals_if_active_allocations_fully_consumed=full_active_allocation,
        two_fresh_replacements=replacements, six_theta_fixed_rate_fits=theta, overhead_scenarios=scenarios,
        canonical_budget_snapshot={k:status[k] for k in ('revision','cap_GPU_hours',
            'charged_GPU_hours','reserved_GPU_hours','available_after_reservations_GPU_hours')},
        active_elapsed_hours_not_yet_settled=sum(r['elapsed_seconds'] for r in active_rows)/3600,
        canonical_settled_charge_does_not_include_active_elapsed_as_final=True,
        authorized_window_remaining_hours=(expires-now)/3600,
        authorized_GPU_window_end_UTC=datetime.datetime.fromtimestamp(expires, datetime.timezone.utc).isoformat(),
        scalar_measurement_overhead_measured=False,
        no_GPU_admission_or_active_source_changes=True, quality_or_evaluation_target_values_read=False,
        caveats='Conditional two-card forecasts from settled costs. Both cards are currently running real native jobs, but this does not guarantee future eligibility. Excludes future capacity waits, failures/replacement amendments and most CPU report time. Scalar online-scoring overhead remains unmeasured; multipliers are scenarios, not estimates. Theta has its registered1.25 cost safety factor plus1200s cleanup. All four phases remain sequential. A full remaining-allocation scenario is a duration assumption, not a guarantee those models succeed by their caps. Neither forecast changes the owner window or any live authorization.',
        source_SHA256={str(Path(__file__).resolve()):sha(__file__), str(dispatcher):sha(dispatcher),
            str(ROOT/'analysis/released097_remaining_runtime_balance.py'):sha(ROOT/'analysis/released097_remaining_runtime_balance.py'),
            str(controller_path):sha(controller_path)}, full_goal_complete=False, SOTA_claim=False)
    assert all(math.isfinite(s['all_current_studies_hours']) and s['all_current_studies_hours'] > 0 for s in scenarios)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['UTC', 'successful_logical_models', 'pending_untouched_original_counts', 'canonical_budget_snapshot', 'authorized_window_remaining_hours', 'overhead_scenarios']}))


if __name__ == '__main__':
    main()
