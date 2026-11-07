"""Independent read-only audit of the first new coordinate endpoint, not selection."""
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0] = list(map(str, [ROOT/'analysis/faithful097/model', ROOT, ROOT/'model',
    PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',
    PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import numpy as np
import torch
from calibration import average, score


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live(identity):
    path = Path('/proc', str(identity['pid']), 'stat')
    if not path.exists():
        return False
    fields = path.read_text().rsplit(')', 1)[1].split()
    return (fields[0] != 'Z' and int(fields[19]) == identity['birth'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot'])


def main():
    destination = ROOT/'analysis/localfaith097_first20k_readout_20261107.json'
    assert not destination.exists()
    attempt = 'CAL_GPU0_V1_lf097_t000_f0_pg_s0'
    result_path = ROOT/'analysis/localfaith097/results'/(attempt+'.json')
    result = read(result_path)
    auth_path = ROOT/'artifacts/allocations'/attempt/'authorization.json'
    auth = read(auth_path)
    assert result['finite'] and result['clean_settlement'] and result['from_scratch']
    assert result['physical_start_update'] == result['initial_completed_updates'] == 0
    assert result['new_phase_model_updates'] == result['endpoint'] == 20000
    assert not result['resumed_same_trial'] and result['owned_processes_and_lease_absent']
    assert auth['source_model_path'] is auth['source_model_SHA256'] is None
    assert not auth['is_continuation'] and not auth['reuse_completed_model']
    assert auth['initial_completed_updates'] == 0 and auth['operation'] == 'localfaith097_cell'
    assert read(auth_path.parent/'deployment.json')['authorization_SHA256'] == sha(auth_path)
    assert result['hyperparameters'] == auth['released_hyperparameters']
    assert result['hyperparameters']['prior_reg'] == 0.003
    assert result['seed'] == auth['cell']['seed'] == 0
    pins = {str(result_path): sha(result_path), str(auth_path): sha(auth_path)}
    for name in ('released_update.py', 'released_adapter.py', 'release_env.py'):
        path = ROOT/'analysis/faithful097/model'/name
        assert sha(path) == auth['files_SHA256'][str(path)]
        pins[str(path)] = sha(path)
    for path, digest in result['files_SHA256'].items():
        assert sha(path) == digest
    out = Path(auth['output_dir'])
    q = torch.load(out/'fit/final.pt', map_location='cpu', weights_only=False)
    assert q['status'] == 'completed' and q['particlegan_version'] == '0.9.7'
    assert q['completed_updates'] == q['fixed_endpoint'] == q['policy']['completed_steps'] == 20000
    assert q['released_hyperparameters'] == result['hyperparameters']
    assert q['recipe']['ema_decay'] == 0
    assert all(torch.isfinite(v).all() for state in q['policy']['models'].values() for v in state.values())
    for prefix, field, replicate_field in [('r', 'metrics', 'replicate_metrics'),
                                           ('resampling_r', 'baseline_metrics', 'baseline_replicate_metrics')]:
        records = []
        for rep in (0, 1):
            folder = out/(prefix+str(rep))
            metric = read(folder/'metrics.json')
            assert metric == result[replicate_field][rep]
            with np.load(folder/'arrays.npz', allow_pickle=False) as arrays:
                measured = score(arrays['stage_synthetic_summaries'], arrays['generated_cells_by_stage'])
            for key, value in measured.items():
                if isinstance(value, (int, float, list)) and key != 'donors':
                    np.testing.assert_allclose(value, metric[key], rtol=0, atol=1e-10)
            records.append(metric)
        assert average(records) == result[field]
    job = read(auth['ledger_path'])['jobs'][0]
    assert job['reservation_released'] and not live(job['child_identity']) and not live(job['guard_identity'])
    settlement_path = Path(auth['ledger_path']).with_name(attempt+'-settlement.json')
    settlement = read(settlement_path)
    assert settlement['uncertain'] is None and settlement['settlement']['status'] == 'settled'
    assert math.isclose(result['charged_GPU_seconds'], settlement['settlement']['charged_GPU_seconds'], rel_tol=0, abs_tol=1e-8)
    pins[str(settlement_path)] = sha(settlement_path)
    baseline_path = ROOT/'analysis/faithful097/results/CAL_GPU0_V1_faith097_f000_f0_pg_s0.json'
    baseline = read(baseline_path)
    baseline_audit_path = ROOT/'analysis/faithful097_final_independent_audit_20261107.json'
    assert sha(baseline_path) == read(baseline_audit_path)['result_SHA256'][baseline['attempt']]
    assert baseline['baseline_metrics'] == result['baseline_metrics']
    assert {k:v for k,v in baseline['hyperparameters'].items() if k != 'prior_reg'} == {
        k:v for k,v in result['hyperparameters'].items() if k != 'prior_reg'}
    pins[str(baseline_path)] = sha(baseline_path)
    pins[str(baseline_audit_path)] = sha(baseline_audit_path)
    keys = ('cell_energy_distance', 'energy_score', 'spread_ratio', 'coverage80')
    assert not torch.cuda.is_initialized()
    output = dict(status='FIRST_FRESH_COORDINATE20K_SAVED_SCORE_AND_SETTLEMENT_AUDIT_PASS',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), attempt=attempt,
                  initial_completed_updates=0, endpoint=20000, seed=0,
                  hyperparameters=result['hyperparameters'], metrics=result['metrics'],
                  matched_nominal_seed0_metrics=baseline['metrics'], matched_resampling=result['baseline_metrics'],
                  delta_vs_nominal={k:result['metrics'][k]-baseline['metrics'][k] for k in keys},
                  actual_GPU_seconds=result['charged_GPU_seconds'], saved_artifacts_hashed=len(result['files_SHA256']),
                  calibration_replicates_independently_rescored=4, owned_trial_closed=True,
                  scope='One exploratory seed only; paired seed1 remains required. No model selection or search change.',
                  full12fit_audit_complete=False, external45fit_comparison_complete=False,
                  new_model_fits=0, new_optimizer_updates=0, CUDA_initialized=False,
                  evaluation_target_values_read=False, SOTA_claim=False, full_goal_complete=False,
                  source_SHA256={str(Path(__file__).resolve()):sha(__file__), str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')},
                  input_SHA256=pins, saved_artifact_SHA256=result['files_SHA256'])
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ('status','delta_vs_nominal','actual_GPU_seconds')}))


if __name__ == '__main__':
    main()
