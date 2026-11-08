"""Current frozen-study runtime forecast using settled costs, never quality."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

ROOT = Path('/ml2/alz')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location('current_capacity_dispatcher', ROOT/'analysis/continue_benchmark097_after_capacity_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = time.time()
    progress = read(module.STATE/'progress.json')
    assert not progress['errors'] and not progress['unsettled']
    rows = module.successful_costs()
    costs, sources = module.cost_table(rows)
    active = progress['active']
    initial = [0., 0.]
    active_rows = []
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for handle in active:
        attempt = handle['attempt']
        auth = read(ROOT/'artifacts/allocations'/attempt/'authorization.json')
        job = read(auth['ledger_path'])['jobs'][0]
        lease = read(Path(auth['registry_path'])/'registry.json')['jobs'][job['lease_token']]
        assert job['status'] == 'active' and lease['status'] == 'running'
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
        active_rows.append(dict(attempt=attempt, cell=auth['cell'], gpu_uuid=auth['gpu_uuid'], guard=lease['owner'], worker=lease['worker'], elapsed_seconds=elapsed, conditional_remaining_seconds=estimated, estimate_method='Latest three settled same-method same-card median minus current charged elapsed; not an optimizer-throughput prediction.'))
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
    replacements = balance({'pg': 1, 'scvi': 0, 'cfgen': 1}, costs)
    base = originals['predicted_makespan_seconds']+replacements['predicted_makespan_seconds']
    expires = read(module.scope.WINDOW)['expires_unix']
    scenarios = []
    for multiplier in (1.15, 2., 5.):
        scalar = balance({'pg': 6}, {'pg': [v*multiplier for v in costs['pg']]})
        total = base+scalar['predicted_makespan_seconds']
        scenarios.append(dict(scalar_cost_multiplier=multiplier, scalar_load_forecast=scalar, originals_plus_replacements_hours=base/3600, scalar_hours=scalar['predicted_makespan_seconds']/3600, combined_hours=total/3600, conditional_remaining_window_margin_hours=(expires-now-total)/3600))
    result = dict(status='CURRENT_TWO_CARD_COST_ONLY_REMAINING_ORIGINALS_REPLACEMENTS_SCALAR_FORECAST', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_goal_turn_classification='PROGRESS: completed and independently audited ten current count-variance decompositions, checked live native identities and committed evidence.', successful_logical_models=len(rows), failed_original_attempts_retained=2, active_models_identity_verified=active_rows, pending_untouched_original_counts=counts, costs_seconds_by_method_card=costs, cost_source_attempts=sources, untouched_originals=originals, two_fresh_replacements=replacements, overhead_scenarios=scenarios, authorized_window_remaining_hours=(expires-now)/3600, authorized_GPU_window_end_UTC=datetime.datetime.fromtimestamp(expires, datetime.timezone.utc).isoformat(), scalar_measurement_overhead_measured=False, no_GPU_admission_or_source_changes=True, quality_or_evaluation_target_values_read=False, caveats='Conditional allocation lower forecast from recent settled costs, assuming both cards remain available. Excludes future capacity waits/preemptions, CPU reports/audits, cell-dependent cost variation and unknown scalar measurement overhead. A nonnegative margin is not a completion guarantee. Original and recovery phases are sequential under the actual current pipeline.', source_SHA256={str(Path(__file__).resolve()):sha(__file__), str(ROOT/'analysis/continue_benchmark097_after_capacity_v2.py'):sha(ROOT/'analysis/continue_benchmark097_after_capacity_v2.py'), str(ROOT/'analysis/released097_remaining_runtime_balance.py'):sha(ROOT/'analysis/released097_remaining_runtime_balance.py')}, full_goal_complete=False, SOTA_claim=False)
    assert all(math.isfinite(s['combined_hours']) and s['combined_hours'] > 0 for s in scenarios)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['UTC', 'successful_logical_models', 'pending_untouched_original_counts', 'authorized_window_remaining_hours', 'overhead_scenarios']}))


if __name__ == '__main__':
    main()
