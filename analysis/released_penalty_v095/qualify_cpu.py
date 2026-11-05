"""Exercise each new one-field penalty control on a full-width synthetic model."""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path('/ml2/alz')
HERE = Path(__file__).resolve().parent
MODEL = ROOT / 'analysis/released_tuning_v095'
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path[:0] = list(map(str, [MODEL, ROOT/'model', ROOT, ROOT/'routed_resume_runtime', SCIENCE,
    PROJECT, PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',
    PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate, VERSION, PENALTY_OPTIONS
native = activate()
import numpy as np
import torch
import released_update as update
import released_adapter as adapter
from test_variant import fixture
from state_fingerprint import fingerprint

read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def context(cache, fullwidth=False):
    ctx = fixture(cache)
    ctx.__class__ = adapter.Context
    if fullwidth:
        ctx.X = torch.randint(1, 8, (24, 2000), generator=torch.Generator().manual_seed(774)).float()
        ctx.mean, ctx.sd = torch.zeros(2000), torch.ones(2000)
        ctx.W = torch.randn(2000, 50, generator=torch.Generator().manual_seed(775))/2000
        ctx.trainpcs, ctx.mu = ctx.project(ctx.X), ctx.X.mean(0)
        ctx.theta = (ctx.mu.square()/(ctx.X.var(0, unbiased=False)-ctx.mu).clamp_min(1e-4)).clamp(.1,10000)
    return ctx


def main():
    protocol_path = HERE/'protocol.json'
    protocol = read(protocol_path)
    base = protocol['baseline_parameters']
    assert VERSION == '0.9.5' and PENALTY_OPTIONS == {'input_domain':'continuous'}
    common_proof = MODEL/'CPU_qualification.json'
    assert read(common_proof)['tests_passed'] == 8
    for path, pin in read(common_proof)['source_SHA256'].items(): assert sha(path) == pin
    def fixture_ast(path):
        return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='context'),include_attributes=False)
    assert fixture_ast(__file__) == fixture_ast(MODEL/'qualify_cpu.py')
    settings = protocol['stage1']['configs']
    assert len(settings) == 12 and len({s['id'] for s in settings}) == 12
    rows = []
    with tempfile.TemporaryDirectory(prefix='alz095-penalty-cpu-') as temp:
        cache = Path(temp); (cache/'manifest.json').write_text('{}')
        ctx = context(cache, fullwidth=True)
        for setting in settings:
            hyper = setting['parameters']; field = setting['changed_field']
            assert {key for key in hyper if hyper[key] != base[key]} == {field}
            assert hyper[field] == base[field] * setting['factor']
            update.configure_hyperparameters(hyper)
            bundle, generators, rng, usage, streams = adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
            assert bundle[6].regularizer.input_domain == 'continuous'
            assert bundle[6].regularizer.coeff == hyper['reg_coeff']
            assert bundle[6].regularizer.kappa == hyper['reg_kappa']
            actual_rates = [bundle[4].param_groups[0]['lr'],bundle[5].param_groups[0]['lr'],bundle[4].param_groups[1]['lr']]
            np.testing.assert_allclose(actual_rates,[.006,.009,.015],rtol=1e-14,atol=0)
            first = adapter.advance(ctx,bundle,generators,rng,usage,execution_limit=20000)
            for key in ('G_gradient_norm','D_gradient_norm','prior_gradient_norm'):
                assert np.isfinite(first[key]) and first[key] > 0, (setting['id'],key)
            checkpoint = adapter.snapshot(ctx,bundle,generators,rng,usage,0,streams,'synthetic')
            assert checkpoint['released_hyperparameters'] == hyper
            assert checkpoint['particlegan_version'] == VERSION and checkpoint['penalty_options'] == PENALTY_OPTIONS
            # Native optimizer/control/RNG plus external data/count streams must
            # replay the next actual pair for each declared scalar setting.
            expected_metrics = adapter.advance(ctx,bundle,generators,rng,usage,execution_limit=20000)
            expected = adapter.snapshot(ctx,bundle,generators,rng,usage,0,streams,'synthetic')
            rb,rg,rr,ru,rs = adapter.restore(ctx,copy.deepcopy(checkpoint))
            actual_metrics = adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
            actual = adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')
            assert fingerprint(actual_metrics) == fingerprint(expected_metrics)
            assert fingerprint(actual) == fingerprint(expected)
            assert ru.steps == 2 and int(ru.draws.sum()) == 64
            assert int(ru.unsampled_moved_steps.sum()) == 0
            rows.append(dict(candidate_id=setting['id'],parameters=hyper,actual_optimizer_LRs=actual_rates,
                finite_connected_fullwidth_gradients=True,complete_next_pair_state_byte_exact=True))
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__),protocol_path,common_proof,MODEL/'released_update.py',MODEL/'released_adapter.py',MODEL/'release_env.py',MODEL/'qualify_cpu.py']
    output = dict(status='ALL12_OFFICIAL095_ONE_FIELD_PENALTY_CONTROLS_FULLWIDTH_CPU_REPLAY_PASS',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_version=VERSION,
        actual_import_origin=str(Path(native.__file__).resolve()),penalty_options=PENALTY_OPTIONS,
        configurations_checked=len(rows),actual_optimizer_pairs_executed=36,checks=rows,
        source_SHA256={str(p):sha(p) for p in sources},CUDA_initialized=False,GPU_job_created=False,
        convergence_or_ALZ_quality_known=False,
        scope='Twelve declared one-field settings on the actual2000gene ALZ architecture with synthetic data. Each executes one pair, saves all state, executes the next pair, restores and replays that next pair exactly. CPU qualification is not calibration evidence or an original full-rate-grid stage2 anchor.')
    destination = HERE/'CPU_qualification.json'; assert not destination.exists()
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('checks','source_SHA256')},indent=2))


if __name__ == '__main__':
    main()
