"""Read-only training-state diagnostics; never initialize or advance a model."""
import datetime
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path.extend([str(PROJECT / 'current_base_20261029/runtime/lib/python3.12/site-packages'),
                 str(PROJECT / '.venv312/lib/python3.12/site-packages')])
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize(fit):
    """Require a closed endpoint; immutable snapshots and logs are diagnostics."""
    fit = Path(fit)
    final = fit / 'final.pt'
    q = torch.load(final, map_location='cpu', weights_only=False)
    assert q['status'] == 'completed'
    assert q['completed_updates'] == q['policy']['completed_steps'] == 20000
    rows = [json.loads(x) for x in (fit / 'learning_curve.jsonl').read_text().splitlines()]
    steps = [r['step'] for r in rows]
    assert steps == sorted(set(steps)) and steps[-1] == 20000
    fields = ['D_adversarial', 'D_penalty', 'G_adversarial', 'critic_gap',
              'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm']
    windows = []
    for lo, hi in ((0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)):
        selected = [r for r in rows if lo < r['step'] <= hi]
        def median(field):
            values = [r[field] for r in selected if r[field] is not None]
            return statistics.median(values) if values else None
        windows.append(dict(after_update=lo, through_update=hi, logged_pairs=len(selected),
                            median={key: median(key) for key in fields}))
    snapshots = []
    for name, endpoint in [('checkpoint05000.pt', 5000), ('checkpoint10000.pt', 10000), ('final.pt', 20000)]:
        path = fit / name
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['completed_updates'] == state['policy']['completed_steps'] == endpoint
        assert state['usage']['steps'] == endpoint
        assert int(state['usage']['draws'].sum()) == endpoint * 32
        assert int(state['usage']['unsampled_moved_steps'].sum()) == 0
        table = state['policy']['table'].detach()
        assert isinstance(table, torch.Tensor) and table.shape == (512, 8)
        assert bool(torch.isfinite(table).all())
        snapshots.append(dict(path=str(path), SHA256=sha(path), completed_updates=endpoint,
                              total_particle_draws=int(state['usage']['draws'].sum()),
                              unsampled_particle_moves=0,
                              particle_covariance_trace=float(table.var(0, unbiased=False).sum()),
                              particle_mean_squared_norm=float(table.square().sum(-1).mean())))
    rates = {'G_prior_LRs': {tuple(r['G_prior_LRs']) for r in rows},
             'D_LRs': {tuple(r['D_LRs']) for r in rows}}
    assert all(len(v) == 1 for v in rates.values())
    return dict(fit=str(fit), native_version=q['particlegan_version'],
                hyperparameters=q['released_hyperparameters'],
                fixed_observed_rates={k: list(next(iter(v))) for k, v in rates.items()},
                windows=windows, snapshots=snapshots,
                learning_curve_SHA256=sha(fit / 'learning_curve.jsonl'))

def main():
    names = ['CAL_GPU0_V1_h20_c014_f0_pg_s0'] + [f'CAL_GPU0_V1_dist097_d00{i}_f0_pg_s0' for i in range(6)]
    completed = [summarize(ROOT / 'runs' / name / 'fit') for name in names
                 if (ROOT / 'runs' / name / 'fit/final.pt').exists()]
    assert not torch.cuda.is_initialized()
    result = dict(status='READ_ONLY_COMPLETED_FRESH20K_TRAINING_STATE_DIAGNOSTICS',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  completed_runs=completed, source_SHA256=sha(__file__),
                  no_optimizer_or_model_constructed=True, CUDA_initialized=False,
                  checkpoint_initialization_used=False, generated_counts=False,
                  heldout_values_read=False,
                  interpretation='Native clock/particle coverage, rates and finite training diagnostics only. Losses, gradient norms and critic gap are not distribution-distance estimates or evidence of monotonic generative improvement. Compare separately scored fixed endpoints; snapshot inspection does not select an alternate endpoint or warm-start training.')
    path = ROOT / 'analysis/distribution097_completed_training_diagnostics_20261106.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(path=str(path), completed_runs=len(completed), CUDA_initialized=False)))

if __name__ == '__main__':
    main()
