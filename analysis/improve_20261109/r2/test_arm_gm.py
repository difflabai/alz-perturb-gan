"""Tests for F6: genemean@s (core), gmlite@k (r2/arm_gm.py) and r2/gm_diagnose.py.  CPU, 3 threads.

    python3 -B r2/test_arm_gm.py flags     # parsing / refusals (seconds)
    python3 -B r2/test_arm_gm.py parity    # genemean (s=1) vs ROUND-1 genemean, initial tensors + 3 updates bitwise;
                                            # genemean@s: scale = s*s_gm exactly, same initial tensors, released streams
    python3 -B r2/test_arm_gm.py gmlite    # groups, scale, block = linear map of the genemean features, build + 3 updates,
                                            # step-0 block gradient, released streams/initial tensors = base, checkpoint round trip
    python3 -B r2/test_arm_gm.py cli       # tiny r2 CLI fits (20 updates): genemean@0.5 seeds 0,1 and gmlite@64 seed 0,
                                            # snapens mode, then every gm_diagnose subcommand on the scratch fits
    python3 -B r2/test_arm_gm.py cleanup   # delete the scratch tree (keeps results/*.json)
    python3 -B r2/test_arm_gm.py all

Real fold-0 hvg2000 cache (read-only).  Scratch:
/ml2/alz/runs/improve_20261109/smoke_r2/gm_test/.  Round-1 code runs only in
a subprocess whose sys.path is the round-1 folder (as r2/test_core.py).
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

torch.set_num_threads(3)
import trainer  # noqa: E402
import arms  # noqa: E402
import arm_gm  # noqa: E402  (registers gmlite)

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
R1 = HERE.parent
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r2/gm_test'
RES = SCR / 'results'
TINY = 20
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
METRIC_KEYS = ['D_adversarial', 'D_penalty', 'D_total', 'G_adversarial', 'prior_regularization_unweighted', 'G_total', 'critic_gap',
               'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm', 'penalty_stats']


def log(*a):
    print('[test_arm_gm]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', obj)


def sh(cmd, check=True, **kw):
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p


def states(*mods):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in mods]


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def refused(text):
    try:
        arms.parse_arm(text)
        return False
    except ValueError:
        return True


# ============================================================================ flags
def flags():
    res = {'accepted': {}, 'refused': {}}
    for ok, want in [('genemean@0.5', ('genemean@0.5',)), ('genemean@0.25', ('genemean@0.25',)), ('gmlite@64', ('gmlite@64',)),
                     ('gmlite@64,covars', ('covars', 'gmlite@64')), ('genemean', ('genemean',))]:
        res['accepted'][ok] = list(arms.parse_arm(ok))
        assert arms.parse_arm(ok) == want, (ok, arms.parse_arm(ok))
    for bad in ['genemean@1', 'genemean@0.50', 'genemean@0', 'genemean@-0.5', 'genemean@.5', 'gmlite', 'gmlite@64.0', 'gmlite@064',
                'gmlite@1', 'gmlite@64,genemean', 'gmlite@64,genemean@0.5', 'gmlite@64,donorpc', 'genemean@0.5,donorpc']:
        res['refused'][bad] = refused(bad)
    assert all(res['refused'].values()), res['refused']
    res['PASS'] = True
    save('flags', res)
    log('flags PASS')
    return res


# ============================================================================ parity
def parity(n=3):
    """genemean (s = 1) in r2 == round-1 genemean (subprocess), bitwise; genemean@s consistency."""
    SCR.mkdir(parents=True, exist_ok=True)
    out = SCR / 'parity_r1_genemean.pt'
    code = f"""
        import sys; sys.dont_write_bytecode = True; sys.path.insert(0, {str(R1)!r})
        import torch; torch.set_num_threads(3)
        import trainer, arms
        ctx = trainer.Context({str(CACHE0)!r}, 'cpu', arms.parse_arm('genemean'))
        b, g, r, u, s = trainer.build(ctx, 0)
        init = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range({n})]
        fin = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
        torch.save(dict(init=init, metrics=mets, final=fin, numpy=r.bit_generator.state, scale=ctx.features.scale,
                        arm_state=ctx.arm_state(), torch_rng=torch.get_rng_state()), {str(out)!r})
    """
    sh([PY, '-B', '-c', textwrap.dedent(code)], cwd=str(R1))
    R = torch.load(out, weights_only=False)
    out.unlink()
    res = {}
    run = {}
    for arm in ('genemean', 'genemean@0.5', 'genemean@0.25'):
        ctx = trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))
        b, g, r, u, _ = trainer.build(ctx, 0)
        init = states(b[1], b[2], b[3])
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
        run[arm] = dict(init=init, metrics=mets, final=states(b[1], b[2], b[3]), numpy=r.bit_generator.state, scale=ctx.features.scale,
                        info=ctx.features.info, arm_state=ctx.arm_state(), torch_rng=torch.get_rng_state())
        del ctx, b
    a = run['genemean']
    res['genemean_vs_round1'] = {
        'initial_tensors_bitwise': [sd_equal(x, y) for x, y in zip(a['init'], R['init'])],
        'metrics_bitwise': [all(m1[k] == m2[k] for k in METRIC_KEYS) for m1, m2 in zip(a['metrics'], R['metrics'])],
        'final_tensors_bitwise': [sd_equal(x, y) for x, y in zip(a['final'], R['final'])],
        'numpy_rng_equal': a['numpy'] == R['numpy'], 'torch_global_rng_equal': bool(torch.equal(a['torch_rng'], R['torch_rng'])),
        'scale_equal': a['scale'] == R['scale'], 'arm_state_equal': trainer._same(a['arm_state'], R['arm_state'])}
    v = res['genemean_vs_round1']
    assert all(v['initial_tensors_bitwise']) and all(v['metrics_bitwise']) and all(v['final_tensors_bitwise']), v
    assert v['numpy_rng_equal'] and v['torch_global_rng_equal'] and v['scale_equal'] and v['arm_state_equal'], v
    for arm, s in (('genemean@0.5', 0.5), ('genemean@0.25', 0.25)):
        q = run[arm]
        row = {'scale_ratio': q['scale'] / a['scale'], 'scale_exact': q['scale'] == a['scale'] * s,
               'info_s_gm_unchanged': q['info']['genemean_scale'] == a['scale'] and q['info']['genemean_multiplier'] == s,
               'initial_tensors_equal_genemean': [sd_equal(x, y) for x, y in zip(q['init'], a['init'])],
               'donor_sampler_equal_after_updates': q['numpy'] == a['numpy'],
               'torch_global_rng_equal': bool(torch.equal(q['torch_rng'], a['torch_rng'])),
               'first_update_differs': not all(q['metrics'][0][k] == a['metrics'][0][k] for k in ('D_adversarial', 'critic_gap')),
               'final_critic_differs': not sd_equal(q['final'][1], a['final'][1]),
               'arm_state_scale': q['arm_state']['features']['genemean_scale']}
        assert row['scale_exact'] and row['info_s_gm_unchanged'] and all(row['initial_tensors_equal_genemean']), row
        assert row['donor_sampler_equal_after_updates'] and row['first_update_differs'] and row['final_critic_differs'], row
        res[arm] = row
    res['PASS'] = True
    save('parity', res)
    log('parity PASS', json.dumps(res, default=str)[:800])
    return res


# ============================================================================ gmlite
def gmlite():
    res = {}
    k = 64
    ctx = trainer.Context(CACHE0, 'cpu', arms.parse_arm(f'gmlite@{k}'))
    a = ctx.plugins[0]
    G = ctx.n_genes
    groups = a.groups.numpy()
    sizes = np.bincount(groups, minlength=k)
    mean = ctx.mean.double().numpy()
    gmax = np.array([mean[groups == j].max() for j in range(k)])
    gmin = np.array([mean[groups == j].min() for j in range(k)])
    res['group_sizes'] = [int(sizes.min()), int(sizes.max()), int(sizes.sum())]
    res['groups_expression_ordered'] = bool((gmax[:-1] <= gmin[1:]).all())
    res['A_column_sums_one'] = bool(torch.allclose(a.A.sum(0), torch.ones(k), atol=1e-6))
    assert res['group_sizes'] == [G // k, -(-G // k), G] and res['groups_expression_ordered'] and res['A_column_sums_one'], res
    assert ctx.features.n_features == 150 + k and [(b.name, lo, hi) for _, b, lo, hi in ctx.features.blocks] == [('groupmean', 150, 150 + k)]
    # scale: same reference bags as genemean -> recompute independently
    from released_update import bag_features
    rng = np.random.default_rng(20261109)
    num = den = 0.0
    with torch.no_grad():
        for _ in range(8):
            ds = rng.integers(len(ctx.by), size=32)
            idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]))
            num += float(bag_features(ctx.trainpcs[idx]).double().square().sum())
            gm = ctx.Z[idx].mean(1).double()
            den += float((gm @ a.A.double()).square().sum())
    res['scale'] = a.scale
    res['scale_rel_diff_independent'] = abs(a.scale / np.sqrt(num / den) - 1)
    assert res['scale_rel_diff_independent'] < 1e-5, res
    # block = scale * genemean features @ A (real and fake paths), gradient flows through fake
    ctxg = trainer.Context(CACHE0, 'cpu', ('genemean',))
    idx = torch.as_tensor(np.stack([np.random.default_rng(3).choice(r, 128) for r in ctx.by[:4]]))
    xr = ctx.features.real(idx, np.arange(4))
    gmr = ctxg.features.real(idx, np.arange(4))[:, 150:] / ctxg.features.scale
    res['real_block_equals_genemean_map'] = float((xr[:, 150:] - (gmr @ a.A) * a.scale).abs().max())
    cf = ctx.Z[idx].clone().requires_grad_(True)
    xf = ctx.features.fake(cf, cf, cf.detach())
    xf[:, 150:].sum().backward()
    res['fake_block_gradient_nonzero'] = bool(cf.grad.abs().sum() > 0)
    res['fake_equals_real_on_real_cells'] = float((xf.detach() - xr).abs().max())
    assert res['real_block_equals_genemean_map'] < 1e-4 and res['fake_block_gradient_nonzero'] and res['fake_equals_real_on_real_cells'] < 1e-4, res
    del ctxg
    # build + 3 updates (step-0 block gradient asserted by the trainer); released streams and released initial tensors
    ctxb = trainer.Context(CACHE0, 'cpu', ())
    bb, gb, rb, ub, _ = trainer.build(ctxb, 0)
    b1, g1, r1, u1, s1 = trainer.build(ctx, 0)
    res['generator_prior_init_equal_base'] = sd_equal(states(bb[1])[0], states(b1[1])[0]) and sd_equal(states(bb[3])[0], states(b1[3])[0])
    res['critic'] = [type(b1[2]).__name__, b1[2].n_features]
    for _ in range(3):
        trainer.advance(ctxb, bb, gb, rb, ub, execution_limit=20000)
        trainer.advance(ctx, b1, g1, r1, u1, execution_limit=20000)
    res['donor_sampler_equal_base_after_3'] = rb.bit_generator.state == r1.bit_generator.state
    res['no_arm_streams'] = ctx.arm_rngs == {}
    assert res['generator_prior_init_equal_base'] and res['critic'] == ['WideVectorCritic', 150 + k], res
    assert res['donor_sampler_equal_base_after_3'] and res['no_arm_streams'], res
    # checkpoint round trip, exact arm state, tamper refusal
    q = trainer.snapshot(ctx, b1, g1, r1, u1, 0, s1, 'incomplete', endpoint=20000, stream_root=trainer.STREAM_ROOT, wall_seconds=0.)
    for _ in range(2):
        trainer.advance(ctx, b1, g1, r1, u1, execution_limit=20000)
    ctx2 = trainer.Context(CACHE0, 'cpu', arms.parse_arm(f'gmlite@{k}'))
    b2, g2, r2, u2, _ = trainer.restore(ctx2, q, 0)
    for _ in range(2):
        trainer.advance(ctx2, b2, g2, r2, u2, execution_limit=20000)
    res['restore_then_2_updates_bitwise'] = all(sd_equal(x, y) for x, y in zip(states(b1[1], b1[2], b1[3]), states(b2[1], b2[2], b2[3])))
    st = q['arm_state']['r2'][f'gmlite@{k}']
    res['arm_state_keys'] = sorted(st['arm']) + sorted(st['blocks'])
    st['arm']['groups'][0] += 1
    try:
        trainer.restore(trainer.Context(CACHE0, 'cpu', arms.parse_arm(f'gmlite@{k}')), q, 0)
        res['tampered_arm_state_refused'] = False
    except ValueError:
        res['tampered_arm_state_refused'] = True
    assert res['restore_then_2_updates_bitwise'] and res['tampered_arm_state_refused'], res
    res['PASS'] = True
    save('gmlite', res)
    log('gmlite PASS', json.dumps(res, default=str)[:800])
    return res


# ============================================================================ cli
def cli():
    fits = SCR / 'fits'
    jobs = [('genemean@0.5', 0), ('genemean@0.5', 1), ('gmlite@64', 0)]
    res = {'fits': {}}
    for arm, seed in jobs:
        fd = fits / arm / 'hvg2000' / 'fold0' / f'seed{seed}'
        if not (fd / 'generated.npz').exists():
            t = time.monotonic()
            sh([PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', str(seed), '--out-root', fits,
                '--updates', str(TINY), '--device', 'cpu', '--threads', '3'])
            log('trained', arm, seed, '%.0fs' % (time.monotonic() - t))
        cfg = json.loads((fd / 'config.json').read_text())
        snaps = sorted(p.name for p in fd.glob('snap_*.pt'))
        res['fits'][f'{arm}|{seed}'] = {'critic_features': cfg['critic_features'], 'snapshots': snaps, 'r2_arms': cfg['r2_arms'],
                                        'arm_info': {k: v for k, v in cfg['arm_info'].items() if k != 'r2_blocks'}}
    assert res['fits']['gmlite@64|0']['critic_features'] == 214 and res['fits']['genemean@0.5|0']['critic_features'] == 2150
    assert res['fits']['genemean@0.5|0']['arm_info']['genemean_multiplier'] == 0.5
    assert res['fits']['genemean@0.5|0']['snapshots'] == ['snap_16.pt', 'snap_17.pt', 'snap_18.pt', 'snap_19.pt']
    fd = fits / 'genemean@0.5' / 'hvg2000' / 'fold0' / 'seed0'
    if not (fd / 'generated_snapens.npz').exists():
        sh([PY, '-B', HERE / 'generate.py', '--fit', fd, '--cache', CACHE0, '--mode', 'snapens', '--threads', '3'])
    with np.load(fd / 'generated_snapens.npz') as z:
        res['snapens_state_counts'] = np.bincount(z['state_index'][0].astype(int)).tolist()
        res['snapens_states'] = z['state_names'].tolist()
    assert res['snapens_state_counts'] == [7, 7, 6, 6, 6]
    # gm_diagnose on the scratch fits (no capture dir: r2 fits use their snapshots)
    out = SCR / 'diag'
    common = ['--roots', fits, '--panels', 'hvg2000', '--capture-dir', SCR / 'nonexistent']
    sh([PY, '-B', HERE / 'gm_diagnose.py', 'all', *common, '--out-dir', out])
    cur = json.loads((out / 'gm_curves.json').read_text())
    cri = json.loads((out / 'gm_critic.json').read_text())
    see = json.loads((out / 'gm_seeds.json').read_text())
    dri = json.loads((out / 'gm_drift.json').read_text())
    res['curves_fits'] = sorted(cur['per_fit'])
    res['critic_states'] = sorted(cri['per_state'])
    c0 = cri['per_state']['genemean@0.5~r2|hvg2000|0|0|final']
    res['critic_gm05'] = {k: c0[k] for k in ('gap', 'gap_ig_share', 'gap_ig_completeness', 'preact_var_share_real', 'push_vs_error_bags',
                                              'push_vs_error_generated')}
    cl = cri['per_state']['gmlite@64~r2|hvg2000|0|0|final']
    res['critic_gmlite_blocks'] = cl['blocks']
    res['seeds_groups'] = sorted(see['per_group'])
    res['seeds_gm05'] = see['per_group']['genemean@0.5~r2|hvg2000|0']
    res['drift_fits'] = sorted(dri['per_fit'])
    d0 = dri['per_fit']['genemean@0.5~r2|hvg2000|0|0']
    res['drift_gm05'] = {k: d0[k] for k in ('states', 'drift_rms_to_final', 'mc_rms', 'rmse_state_average', 'seed_pair_rms_final')}
    assert len(res['curves_fits']) == 3 and len(res['critic_states']) == 3
    assert 'genemean@0.5~r2#snapens|hvg2000|0' in res['seeds_groups'] and res['seeds_gm05']['n_seeds'] == 2
    assert d0['states'] == ['snap_16', 'snap_17', 'snap_18', 'snap_19', 'final'] and d0['seed_pair_rms_final'] is not None
    assert abs(c0['gap_ig_completeness'] - 1) < 0.05 and set(cl['blocks']) == {'released150', 'gmlite@64/groupmean'}
    res['PASS'] = True
    save('cli', res)
    log('cli PASS', json.dumps(res, default=str)[:1500])
    return res


def cleanup():
    if SCR.exists():
        for p in SCR.iterdir():
            if p.name != 'results':
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    log('cleanup done')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    steps = {'flags': flags, 'parity': parity, 'gmlite': gmlite, 'cli': cli, 'cleanup': cleanup}
    for name in (['flags', 'parity', 'gmlite', 'cli'] if which == 'all' else [which]):
        t = time.monotonic()
        steps[name]()
        log(name, 'done in %.0fs' % (time.monotonic() - t))
