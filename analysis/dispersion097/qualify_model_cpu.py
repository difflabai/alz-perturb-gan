"""Synthetic fullwidth native parity for one isolated theta-initialization trial."""
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
HERE = Path(__file__).resolve().parent
sys.path[:0] = list(map(str, [HERE/'model', ROOT/'analysis/duration096', ROOT/'refinement097_runtime', ROOT/'model', ROOT, SCIENCE, PROJECT, PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages', PROJECT/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import torch
import released_adapter as app
import released_update as up
from native_fixture import context
from state_fingerprint import fingerprint

HYPER = dict(generator_lr=.012, critic_lr=.018, particle_lr=.06, reg_coeff=1., reg_kappa=1., prior_reg=.01)
BASE = dict(z_dim=8, stage_balance=False, cell_z_dim=8, theta_init_scale=1., fourier_feature_scale=1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    assert Path(app.__file__).resolve() == HERE/'model/released_adapter.py'
    assert Path(up.__file__).resolve() == HERE/'model/released_update.py'
    lineage_path = HERE/'model_source_lineage.json'
    lineage = json.loads(lineage_path.read_bytes())
    for name, row in lineage['files'].items():
        assert sha(row['parent_path']) == row['parent_SHA256']
        assert sha(HERE/'model'/name) == row['new_SHA256']
    assert lineage['files']['released_adapter.py']['byte_identical']
    assert lineage['files']['release_env.py']['byte_identical']
    parent_parity = ROOT/'analysis/faithful097/native_trainer_parity.py'
    original_proof = json.loads((ROOT/'analysis/faithful097/native_trainer_parity_CPU.json').read_bytes())
    assert original_proof['source_SHA256'][str(parent_parity)] == sha(parent_parity)
    spec = importlib.util.spec_from_file_location('original_actual_GANTrainer_parity', parent_parity)
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    assert parity.app is app and parity.up is up
    rows = []
    for scale in (1., 5.):
        up.configure(dict(BASE, theta_init_scale=scale))
        for seed in (0, 1):
            row = parity.run(HYPER, seed, pairs=3)
            row['theta_init_scale'] = scale
            rows.append(row)
    for invalid in [dict(BASE, z_dim=2), dict(BASE, stage_balance=True), dict(BASE, theta_init_scale=10.), dict(BASE, fourier_feature_scale=5.)]:
        try:
            up.configure(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Unregistered experimental coordinate accepted')
    with tempfile.TemporaryDirectory() as folder:
        cache = Path(folder)
        (cache/'manifest.json').write_text('{}')
        ctx = context(cache, fullwidth=True)
        initial = []
        for scale in (1., 5.):
            up.configure(dict(BASE, theta_init_scale=scale))
            b, g, rng, usage, streams = app.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            assert b[-1].completed_steps == 0 and not b[4].state and not b[5].state
            assert b[0].ema_decay == 0 and b[0].num_particles == 512
            initial.append(copy.deepcopy([m.state_dict() for m in b[1:4]]))
        for name in initial[0][0]:
            if name != 'log_theta':
                assert torch.equal(initial[0][0][name], initial[1][0][name]), name
        assert fingerprint(initial[0][1:]) == fingerprint(initial[1][1:])
        theta1 = torch.nn.functional.softplus(initial[0][0]['log_theta'])+.05
        theta5 = torch.nn.functional.softplus(initial[1][0]['log_theta'])+.05
        torch.testing.assert_close(theta5, 5*theta1, rtol=1e-6, atol=1e-7)
        # Exercise strict checkpoint binding after an actual fresh native pair.
        up.advance(ctx, b, g, rng, usage, execution_limit=20000)
        saved = app.snapshot(ctx, b, g, rng, usage, 0, streams, 'synthetic')
        restored, rg, rr, ru, rs = app.restore(ctx, saved)
        again = app.snapshot(ctx, restored, rg, rr, ru, 0, rs, 'synthetic')
        assert fingerprint(saved) == fingerprint(again)
        assert saved['variant_config']['theta_init_scale'] == 5.
        assert saved['completed_updates'] == saved['policy']['completed_steps'] == 1
        rec, G, D, P, *rest = restored
        z = torch.zeros(2, 8)
        c = torch.zeros(2, 7)
        lib = torch.full((2, 4), 10000.)
        with torch.no_grad():
            counts = G.sample_counts(z, c, 4, lib, torch.Generator().manual_seed(999))
        assert counts.shape == (2, 4, 2000) and torch.isfinite(counts).all()
        assert (counts >= 0).all() and torch.equal(counts, counts.floor())
    assert not torch.cuda.is_initialized()
    files = [Path(__file__).resolve(), lineage_path, parent_parity, *sorted((HERE/'model').glob('*.py')), ROOT/'analysis/duration096/native_fixture.py', Path(sys.modules['pg_model'].__file__), Path(sys.modules['particlegan.training'].__file__)]
    q = dict(status='ISOLATED_THETA_INIT1_VS5_SYNTHETIC_FULLWIDTH_NATIVE_GANTRAINER_PARITY_AND_RESTORE_PASS', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), native_version='0.9.7', rows=rows, actual_cases=4, actual_optimizer_pairs_per_implementation=12, additional_actual_fresh_checkpoint_pair=1, fullwidth_genes=2000, prior_rows=512, all_prior_rows_regularized=True, explicit_bound_critic_thermal=True, nonEMA=True, same_seed_initial_models_differ_only_in_generator_log_theta=True, initial_theta5_equals5_times_theta1=True, strict_native_snapshot_restore_exact=True, native_integer_count_sampling_checked=True, invalid_coordinate_changes_rejected=True, real_ALZ_data_read=False, new_ALZ_fits=0, GPU_allocation_created=False, CUDA_initialized=False, source_models_and_native_package_modified=False, model_only_CPU_qualification=True, GPU_admission_runtime_qualified=False, live45_and_six_scalar_sources_unchanged=True, source_SHA256={str(p):sha(p) for p in files}, full_goal_complete=False, SOTA_claim=False)
    path = HERE/'model_CPU_qualification.json'
    assert not path.exists()
    path.write_text(json.dumps(q, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in q.items() if k not in ('source_SHA256', 'rows')}))


if __name__ == '__main__':
    main()
