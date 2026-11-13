"""CPU checks for the optional critic projector and hierarchical decoders."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import json
import torch
import numpy as np
import trainer as T
import arms

torch.set_num_threads(3)
CACHE = T.RUNS / 'caches/hvg2000/fold0'


def run():
    checks = {}
    fast = ',t.bag@16,t.batch@16'
    for name in ('d.randproj,pbulk', 'g.split,pbulk', 'g.hier,pbulk'):
        ctx = T.Context(CACHE, 'cpu', arms.parse_arm(name + fast))
        W = torch.tensor(ctx.space['W'] / ctx.space['pc_sd'])
        assert torch.equal(ctx.W, W), 'evaluation projector changed'
        bundle, gens, rng, usage, streams = T.build(ctx, 0)
        if name.startswith('d.'):
            idx = torch.arange(256).reshape(16,16)
            cf = T.standardize_counts(ctx.X[idx], ctx.mean, ctx.sd)
            assert torch.allclose(ctx.features.real(idx), ctx.features.fake(cf, ctx.X[idx], ctx.X[idx]), atol=1e-5)
            assert not torch.equal(ctx.features.W, ctx.W)
            assert ctx.features.n_features == 2150
        for _ in range(2):
            T.advance(ctx, bundle, gens, rng, usage, execution_limit=3)
        q = copy.deepcopy(T.snapshot(ctx, bundle, gens, rng, usage, 0, streams, 'incomplete',
                                    endpoint=3, stream_root=T.STREAM_ROOT, wall_seconds=0.))
        T.advance(ctx, bundle, gens, rng, usage, execution_limit=3)
        expected = [copy.deepcopy(m.state_dict()) for m in bundle[1:4]]
        restored_ctx = T.Context(CACHE, 'cpu', arms.parse_arm(name + fast))
        restored, gs, nr, us, _ = T.restore(restored_ctx, q, 0)
        T.advance(restored_ctx, restored, gs, nr, us, execution_limit=3)
        for model, exp in zip(restored[1:4], expected):
            assert all(torch.equal(v, exp[k]) for k,v in model.state_dict().items()), 'resume differs'
        G = restored[1]
        z = torch.zeros(2, 8)
        c = restored_ctx.conditions[:2]
        library = torch.full((2,16), 100.)
        mu, theta = G.parameters_nb(z,c,16,library,torch.Generator().manual_seed(2))
        assert bool(torch.isfinite(mu).all()) and bool((mu > 0).all())
        assert torch.allclose(mu.sum(-1),library,atol=2e-5)
        assert bool((theta > 0).all())
        # Bias calibration must still change output gene log odds exactly
        # once. Centered state/cell residuals must not cancel or duplicate it.
        import mode_biascal as MB
        delta = torch.linspace(-.1,.1,ctx.n_genes)
        with torch.no_grad():
            original_bias = MB.head(G).clone()
            MB.head(G).add_(delta)
            shifted, shifted_theta = G.parameters_nb(z,c,16,library,torch.Generator().manual_seed(2))
            expected_rates = mu.detach()*delta.exp()
            expected_rates *= library[...,None]/expected_rates.sum(-1,keepdim=True)
            assert torch.allclose(shifted,expected_rates,rtol=2e-5,atol=2e-5), 'bias calibration is not a direct logit shift'
            assert torch.equal(theta,shifted_theta), 'bias calibration changed dispersion'
            MB.head(G).copy_(original_bias)
        checks[name] = {'finite_updates':3, 'resume_bitwise':True, 'evaluation_PCA_unchanged':True,
                        'output_bias_calibration_compatible':True}
        del ctx, bundle, restored_ctx, restored
    # Registration is inert for the default path and its released feature values.
    ctx = T.Context(CACHE, 'cpu', arms.parse_arm('pbulk' + fast))
    idx = torch.arange(256).reshape(16,16)
    from released_update import bag_features
    assert torch.equal(ctx.features.real(idx)[:,:150],bag_features(ctx.trainpcs[idx]))
    from arm_d_randproj import fixed_directions
    R = fixed_directions(2000)
    assert np.max(np.abs(R.T @ R - np.eye(50))) < 1e-12
    checks['default_features_unchanged'] = True
    out = T.RUNS / 'results_r5/architecture_pass/tests.json'
    T.dump(out, checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    run()
