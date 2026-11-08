"""Actual native GANTrainer prefix parity for the six proposed fresh cells."""
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0] = list(map(str, [HERE/'model', ROOT/'analysis/duration096', ROOT/'refinement097_runtime', ROOT/'model', ROOT, SCIENCE, PROJECT, PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages', PROJECT/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
import fresh_seed_adapter as app
import released_adapter as reference
import released_update as up


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    def timeout(signum, frame):
        raise TimeoutError('Finite synthetic CPU fresh-namespace qualification ended')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    proof_path = HERE/'model_CPU_qualification.json'
    original = json.loads(proof_path.read_bytes())
    for path, pin in original['source_SHA256'].items():
        assert sha(path) == pin, path
    lineage_path = HERE/'fresh_seed_adapter_lineage.json'
    lineage = json.loads(lineage_path.read_bytes())
    assert sha(lineage['parent_path']) == lineage['parent_SHA256']
    assert sha(lineage['path']) == lineage['SHA256']
    text = Path(lineage['parent_path']).read_text()
    edit = lineage['only_source_edit']
    assert text.count(edit['before']) == 1
    assert text.replace(edit['before'], edit['after']) == Path(lineage['path']).read_text()
    parent_parity = ROOT/'analysis/faithful097/native_trainer_parity.py'
    spec = importlib.util.spec_from_file_location('actual_public_trainer_fresh_prefix', parent_parity)
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    # Select the isolated production caller for the existing exact native fixture.
    # No native package or live application module is patched.
    parity.app = app
    assert parity.up is up
    assert Path(app.__file__).resolve() == HERE/'model/fresh_seed_adapter.py'
    old = [reference.training_streams(0, seed) for seed in range(5)]
    streams = {seed:app.training_streams(0, seed) for seed in (8,9,10)}
    assert all(value not in old for value in streams.values())
    assert len({json.dumps(value, sort_keys=True) for value in streams.values()}) == 3
    for seed in (0,1,2,3,4,5,6,7,11):
        try:
            app.training_streams(0, seed)
        except ValueError:
            pass
        else:
            raise AssertionError('Seed outside isolated study namespace accepted')
    hyper = dict(generator_lr=.012, critic_lr=.018, particle_lr=.06, reg_coeff=1., reg_kappa=1., prior_reg=.01)
    config = dict(z_dim=8, stage_balance=False, cell_z_dim=8, theta_init_scale=1., fourier_feature_scale=1.)
    paths = [Path(__file__).resolve(), lineage_path, proof_path, parent_parity, *sorted((HERE/'model').glob('*.py'))]
    pins = {str(p):sha(p) for p in paths}
    rows = []
    for scale in (1.,5.):
        up.configure(dict(config, theta_init_scale=scale))
        for seed in (8,9,10):
            row = parity.run(hyper, seed, pairs=3)
            row['theta_init_scale'] = scale
            rows.append(row)
            print(json.dumps(dict(completed_native_parity_case=[scale,seed],pairs_per_implementation=3)), flush=True)
    for path,pin in pins.items():
        assert sha(path) == pin, path
    assert not torch.cuda.is_initialized()
    q = dict(status='ALL_SIX_FRESH_THETA1_THETA5_SEED8TO10_FULLWIDTH_NATIVE_TRAINER_PREFIX_CASES_PASS', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows, actual_cases=6, actual_optimizer_pairs_per_implementation=18, fullwidth_genes=2000, native_version='0.9.7', native_GANTrainer_used=True, same_released_formulation=True, all_models_and_optimizers_started_at_zero=True, prior_rows=512, new_initialization_streams_unseen_in_original_seed0to4=True, fresh_training_seed_namespaces_qualified=[8,9,10], old_seed_namespaces_rejected=True, real_ALZ_data_read=False, real_new_ALZ_fits=0, CUDA_initialized=False, new_GPU_allocations=0, GPU_admission_runtime_qualified=False, live45_and_six_LR_sources_unchanged=True, source_SHA256=pins, full_goal_complete=False, SOTA_claim=False)
    path = HERE/'fresh_seed_model_CPU_qualification.json'
    assert not path.exists()
    path.write_text(json.dumps(q,indent=2)+'\n')
    signal.alarm(0)
    print(json.dumps({k:v for k,v in q.items() if k not in ('rows','source_SHA256')}), flush=True)


if __name__ == '__main__':
    main()
