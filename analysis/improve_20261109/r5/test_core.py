"""Tests for the round-3 core (r3/trainer.py, r3/arms.py, r3/generate.py, r3/run_screen.sh, r3/arm_example.py).

CPU only, 3 threads per process, real fold-0 hvg2000 cache (read-only).  Run::

    python3 -B r3/test_core.py parity    # r3 == round 1 (base, genemean 5 updates; offset/donorpc/covars 2) and == r2 (pbulk 3,
                                         #   genemean@0.5 2); t.ema leaves training bitwise unchanged; native EMA == shadow EMA
    python3 -B r3/test_core.py recipes   # recipe resolution, every t. flag's fields, refusals, native incompatibilities
    python3 -B r3/test_core.py anneal    # LR read from the optimizer groups every update == documented cosine; t.steps
                                         #   horizon; annealstart / netfloor / nethorizon; training past the horizon
    python3 -B r3/test_core.py capacity  # t.particles / t.zdim / t.batch / t.bag / t.glr|dlr|plr build and train
    python3 -B r3/test_core.py hooks     # g.example / d.example (generator kwargs, class chain, critic module, per-cell
                                         #   block, layout, update-0 checks, checkpoint round trip, refusals)
    python3 -B r3/test_core.py cli       # tiny CLI fits: r1 vs r3 trainer+generation bitwise with snapshots + shadow EMA on;
                                         #   annealed t.ema fit via run_screen.sh; interrupted mid-anneal + resume bitwise;
                                         #   generation modes; refuse-to-overwrite; out-root guards
    python3 -B r3/test_core.py oldfits   # round-1 and round-2 fits through r3 generation (== r2 code on CPU; --out guards)
    python3 -B r3/test_core.py drift     # real-data read-out: per-update step size vs late-snapshot drift of r2 fits
    python3 -B r5/test_core.py guards    # round-5 repair F1: generate._check_out refuses fits_r3 without --out
    python3 -B r3/test_core.py cleanup   # delete the scratch fits (results/*.json are kept)
    python3 -B r3/test_core.py all

Round-5 edits (r5 trainer task): scratch moved to smoke_r5/core_test; ``parity``
also compares with the FROZEN r3 trainer (pbulk 3 updates, g.dispz,pbulk,t.kappa@0.25
2 updates) in a subprocess.
Scratch: /ml2/alz/runs/improve_20261109/smoke_r5/core_test/ (results/*.json are
small and kept).  Round-1 and round-2 code is executed only in subprocesses
(sys.path[0] = that round's folder) so modules named trainer/arms/generate of
different rounds never meet in one interpreter; nothing is written into
round-1/2 folders.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import json
import math
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
import trainer  # noqa: E402  (r3)
import arms  # noqa: E402
import generate  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
R1 = HERE.parent
R2 = HERE.parent / 'r2'
R3 = HERE.parent / 'r3'
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r5/core_test'
RES = SCR / 'results'
R1_GPU_FIT = RUNS / 'fits/base/hvg2000/fold0/seed0'
R2_GPU_FIT = RUNS / 'fits_r2/pbulk/hvg2000/fold0/seed0'
TINY = 20
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
FAST = 't.bag@16,t.batch@16'   # small batches for schedule tests (the LR law does not depend on them)


def log(*a):
    print('[test_core_r3]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', json.loads(json.dumps(obj, default=str)))


def sh(cmd, check=True, **kw):
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


def other_round_python(folder, code):
    """Run code with ANOTHER round's modules (cwd and sys.path[0] = that folder)."""
    pre = ("import sys,os; sys.dont_write_bytecode=True; sys.path.insert(0, %r)\n"
           "import torch; torch.set_num_threads(3)\n") % str(folder)
    return sh([PY, '-B', '-c', pre + textwrap.dedent(code)], cwd=str(folder))


def states(*mods):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in mods]


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def models_equal(a, b):
    return set(a) == set(b) and all(sd_equal(a[k], b[k]) for k in a)


def ctx_of(arm):
    return trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))


def run_updates(ctx, n, seed=0, limit=None):
    b, g, r, u, s = trainer.build(ctx, seed)
    mets = [trainer.advance(ctx, b, g, r, u, execution_limit=limit or ctx.topt.steps) for _ in range(n)]
    return b, g, r, u, s, mets


# ============================================================================ parity
R1_ARMS = {'base': 5, 'genemean': 5, 'offset': 2, 'donorpc': 2, 'covars': 2}
R2_ARMS = {'pbulk': 3, 'genemean@0.5': 2}
R3_ARMS = {'pbulk': 3, 'g.dispz,pbulk,t.kappa@0.25': 2}
METRIC_KEYS = ['D_adversarial', 'D_penalty', 'D_total', 'G_adversarial', 'prior_regularization_unweighted', 'G_total', 'critic_gap',
               'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm', 'G_prior_LRs', 'D_LRs', 'penalty_stats']
REF_CODE = """
    import trainer, arms
    res = {{}}
    for arm, n in {arms_!r}.items():
        ctx = trainer.Context({cache!r}, 'cpu', arms.parse_arm(arm))
        b, g, r, u, s = trainer.build(ctx, 0)
        init = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
        fin = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
        res[arm] = dict(init=init, metrics=mets, final=fin, numpy=r.bit_generator.state,
                        gens={{k: v.get_state() for k, v in g.items()}}, torch_rng=torch.get_rng_state(),
                        conditions=ctx.conditions.clone(), arm_state=ctx.arm_state(), n_features=ctx.features.n_features,
                        critic=type(b[2]).__name__, generator=type(b[1]).__name__, streams=s,
                        optimizers=[b[4].state_dict(), b[5].state_dict()])
    torch.save(res, {out!r})
"""


def _compare(arm, n, R):
    ctx = ctx_of(arm)
    b, g, r, u, s = trainer.build(ctx, 0)
    init = states(b[1], b[2], b[3])
    mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
    fin = states(b[1], b[2], b[3])
    row = {'updates': n,
           'initial_tensors_bitwise': [sd_equal(x, y) for x, y in zip(init, R['init'])],
           'metrics_bitwise': [all(m1[k] == m2[k] for k in METRIC_KEYS) for m1, m2 in zip(mets, R['metrics'])],
           'final_tensors_bitwise': [sd_equal(x, y) for x, y in zip(fin, R['final'])],
           'numpy_rng_equal': r.bit_generator.state == R['numpy'],
           'torch_generators_equal': all(torch.equal(g[k].get_state(), R['gens'][k]) for k in g),
           'torch_global_rng_equal': torch.equal(torch.get_rng_state(), R['torch_rng']),
           'conditions_equal': torch.equal(ctx.conditions, R['conditions']),
           'arm_state_equal': trainer._same(ctx.arm_state(), R['arm_state']),
           'streams_equal': s == R['streams'],
           'optimizer_state_equal': trainer._same([b[4].state_dict(), b[5].state_dict()], R['optimizers']),
           'classes': [type(b[1]).__name__, type(b[2]).__name__, R['generator'], R['critic']],
           'n_features': [ctx.features.n_features, R['n_features']]}
    row['PASS'] = bool(all(row['initial_tensors_bitwise']) and all(row['metrics_bitwise']) and all(row['final_tensors_bitwise'])
                       and row['numpy_rng_equal'] and row['torch_generators_equal'] and row['torch_global_rng_equal']
                       and row['conditions_equal'] and row['arm_state_equal'] and row['streams_equal'] and row['optimizer_state_equal']
                       and row['classes'][:2] == row['classes'][2:] and row['n_features'][0] == row['n_features'][1])
    return row, (b, g, r, u, s, mets, fin)


def parity():
    """r3 (shadow EMA on) == round-1 and == round-2 trainers bitwise; native EMA changes nothing and equals the shadow EMA."""
    SCR.mkdir(parents=True, exist_ok=True)
    res = {}
    for label, folder, arm_set in (('round1', R1, R1_ARMS), ('round2', R2, R2_ARMS), ('round3', R3, R3_ARMS)):
        out = SCR / f'parity_{label}.pt'
        other_round_python(folder, REF_CODE.format(arms_=arm_set, cache=str(CACHE0), out=str(out)))
        ref = torch.load(out, weights_only=False)
        out.unlink()
        for arm, n in arm_set.items():
            row, keep = _compare(arm, n, ref[arm])
            res[f'{label}:{arm}'] = row
            log('parity', label, arm, row['PASS'])
            if label == 'round1' and arm == 'base':
                base_keep = keep
    # t.ema@0.999: training trajectory bitwise == base; native EMA (policy averages) == shadow EMA at 0.999, bitwise
    b0, g0, r0, u0, s0, m0, fin0 = base_keep
    ctx = ctx_of('t.ema@0.999')
    b, g, r, u, s, mets = run_updates(ctx, 5, limit=20000)
    pol = b[-1].state_dict()
    sh_ = trainer.shadow_of(b).state()['0.999']
    shadow0 = trainer.shadow_of(b0).state()
    row = {'raw_final_bitwise_base': [sd_equal(x, y) for x, y in zip(states(b[1], b[2], b[3]), fin0)],
           'metrics_bitwise_base': [all(x[k] == y[k] for k in METRIC_KEYS) for x, y in zip(mets, m0)],
           'native_ema_equals_shadow_0.999': sd_equal(pol['averages']['generator'], sh_['generator']) and sd_equal(pol['averages']['prior'], sh_['prior']),
           'shadow_same_with_and_without_native_ema': sd_equal(shadow0['0.999']['generator'], sh_['generator']),
           'native_ema_differs_from_raw': not sd_equal(pol['averages']['generator'], pol['models']['generator']),
           'recipe_ema_decay': b[0].ema_decay,
           'shadow_decays': list(trainer.shadow_of(b).decays),
           'shadow_differs_from_live': not sd_equal(shadow0['0.9995']['generator'], states(b0[1])[0])}
    row['PASS'] = bool(all(row['raw_final_bitwise_base']) and all(row['metrics_bitwise_base']) and row['native_ema_equals_shadow_0.999']
                       and row['shadow_same_with_and_without_native_ema'] and row['native_ema_differs_from_raw'] and row['shadow_differs_from_live'])
    res['t.ema_training_invariance'] = row
    log('parity t.ema', row['PASS'])
    res['PASS'] = all(v['PASS'] for v in res.values())
    save('parity', res)
    assert res['PASS'], json.dumps(res, indent=1, default=str)
    return res


# ============================================================================ recipes
def recipes():
    from particlegan import get_recipe
    res = {}
    rel = trainer.resolved_recipe(trainer.T005)
    res['default_general_equals_released'] = trainer.general_recipe(arms.TrainOptions()).to_dict() == rel.to_dict()
    res['default_recipe_is_released_object_path'] = trainer.recipe(arms.TrainOptions()).to_dict() == rel.to_dict()
    assert res['default_general_equals_released'] and res['default_recipe_is_released_object_path']
    res['released'] = {k: getattr(rel, k) for k in ('total_steps', 'batch_size', 'num_particles', 'z_dim', 'lr', 'd_lr_mult', 'prior_lr_mult',
                                                   'lr_schedule', 'lr_anneal_start', 'lr_floor', 'network_lr_floor', 'network_lr_horizon_cap',
                                                   'ema_decay', 'optimizer_family', 'optimizer_momentum', 'optimizer_smoothing',
                                                   'constraint_geometry_mode', 'beta2_end', 'reg_coeff_end', 'serve_average')}
    cases = {
        't.steps@40000': {'total_steps': 40000},
        't.anneal@0.1': {'lr_floor': 0.1, 'network_lr_floor': 0.1, 'lr_anneal_start': 0.6, 'lr_schedule': 'cosine'},
        't.anneal@0,t.annealstart@0.5': {'lr_floor': 0.0, 'network_lr_floor': 0.0, 'lr_anneal_start': 0.5},
        't.anneal@0.2,t.netfloor@0.05,t.nethorizon@15000': {'lr_floor': 0.2, 'network_lr_floor': 0.05, 'network_lr_horizon_cap': 15000},
        't.ema@0.999': {'ema_decay': 0.999},
        't.glr@2': {'lr': 0.024, 'd_lr_mult': 0.75, 'prior_lr_mult': 2.5},
        't.dlr@0.5': {'lr': 0.012, 'd_lr_mult': 0.75},
        't.plr@2': {'prior_lr_mult': 10.0},
        't.particles@2048': {'num_particles': 2048},
        't.zdim@16': {'z_dim': 16},
        't.batch@64': {'batch_size': 64},
        't.bag@256': {},
    }
    rows = {}
    for arm, want in cases.items():
        flags = arms.parse_arm(arm)
        topt = arms.train_options(flags)
        rec = trainer.recipe(topt)
        trainer.check_recipe(rec, topt)
        got = {k: getattr(rec, k) for k in want}
        ctx_free = trainer.departures(type('C', (), {'flags': flags, 'plugins': []})(), rec)
        rows[arm] = {'fields': got, 'ok': all(math.isclose(got[k], v, rel_tol=1e-15) if isinstance(v, float) else got[k] == v for k, v in want.items()),
                     'effective_rates': [rec.lr, rec.lr * rec.d_lr_mult, rec.lr * rec.prior_lr_mult], 'departures': ctx_free}
        assert rows[arm]['ok'], (arm, got, want)
        assert set(ctx_free) == set(f for f in flags if f.startswith('t.')), ctx_free
    res['cases'] = rows
    assert math.isclose(rows['t.dlr@0.5']['effective_rates'][1], 0.009, rel_tol=1e-15)
    assert math.isclose(rows['t.plr@2']['effective_rates'][2], 0.12, rel_tol=1e-15)
    assert 'prior_regulariser_rows' in rows['t.particles@2048']['departures']['t.particles@2048']
    refused = {}
    for bad in ['t.steps@20000', 't.anneal@1', 't.annealstart@0.5', 't.netfloor@0.1', 't.nethorizon@100', 't.foo@1', 't.glr@1',
                't.glr@0', 't.particles@64', 't.particles@8193', 't.zdim@8', 't.batch@8', 't.bag@8', 't.ema@1', 't.ema@0',
                't.anneal@0.10', 't.steps@040', 't.steps', 'T.steps@40', 't.steps@-5']:
        try:
            arms.parse_arm(bad)
            refused[bad] = 'ACCEPTED'
        except ValueError as e:
            refused[bad] = str(e)
    res['refused'] = refused
    assert 'ACCEPTED' not in refused.values(), refused
    # plug-ins cannot register t. names; arm strings sort with t. flags
    try:
        @arms.register
        class _Bad(arms.Arm):
            name = 't.bad'
        res['t_registration_refused'] = False
    except ValueError:
        res['t_registration_refused'] = True
    res['sorted'] = arms.parse_arm('t.steps@40000,t.anneal@0.1,genemean@0.5')
    assert res['t_registration_refused'] and res['sorted'] == ('genemean@0.5', 't.anneal@0.1', 't.steps@40000')
    # native options that cannot be combined with bispf_optimized: exact native error text
    base = dict(trainer='bispf_optimized', total_steps=20000, batch_size=32, num_particles=512, z_dim=8, lr=.012, d_lr_mult=1.5,
                prior_lr_mult=5., reg_coeff=1., reg_kappa=1., prior_reg=.01)
    incompat = {}
    for label, extra in {'beta2_end': {'beta2_end': 0.9}, 'reg_coeff_end': {'reg_coeff_end': 0.1},
                         'optimizer_momentum': {'optimizer_momentum': 0.9}, 'serve_average': {'serve_average': 4.0},
                         'continuous_policy_dv12': {'continuous_policy': 'dv12', 'total_steps': None}}.items():
        try:
            get_recipe('gan', **{**base, **extra})
            incompat[label] = 'ACCEPTED by Recipe validation'
        except (ValueError, TypeError) as e:
            incompat[label] = str(e)
    res['native_incompatible'] = incompat
    assert all(v != 'ACCEPTED by Recipe validation' for k, v in incompat.items() if k != 'serve_average'), incompat
    res['snapshot_updates'] = {U: trainer.snapshot_updates(U) for U in (20, 40, 300, 20000, 40000)}
    assert res['snapshot_updates'][20000] == [5000, 10000, 15000, 16000, 17000, 18000, 19000]
    assert res['snapshot_updates'][20] == [5, 10, 15, 16, 17, 18, 19]
    assert res['snapshot_updates'][40000][-10:] == list(range(30000, 40000, 1000))
    res['PASS'] = True
    save('recipes', res)
    log('recipes', json.dumps(res['native_incompatible'])[:800])
    return res


# ============================================================================ anneal
def lr_scale(k, T, S, F):
    """Independent re-statement of the documented native cosine (arms.py docstring, t.anneal)."""
    if F == 1:
        return 1.0
    f = min(1.0, max(0.0, (k - S * T) / ((1 - S) * T)))
    return F + (1 - F) * 0.5 * (1 + math.cos(math.pi * f))


def _lr_rows(arm, n, limit=None):
    ctx = ctx_of(arm)
    b, g, r, u, s = trainer.build(ctx, 0)
    og, od = b[4], b[5]
    rows = []
    for k in range(n):
        trainer.advance(ctx, b, g, r, u, execution_limit=limit or ctx.topt.steps)
        rows.append((k, og.param_groups[0]['lr'], og.param_groups[1]['lr'], od.param_groups[0]['lr']))   # read from the groups
    return ctx, b, rows


def _check_rows(rows, T, S, F_net, F_prior, T_net=None):
    T_net = T if T_net is None else min(T, T_net)
    bad = []
    for k, g, p, d in rows:
        want = (0.012 * lr_scale(k, T_net, S, F_net), 0.06 * lr_scale(k, T, S, F_prior), 0.018 * lr_scale(k, T_net, S, F_net))
        if not all(math.isclose(x, y, rel_tol=1e-12, abs_tol=0) for x, y in zip((g, p, d), want)):
            bad.append((k, (g, p, d), want))
    return bad


def anneal():
    res = {}
    # t.steps@16 vs t.steps@32 with the same floor: the horizon sets the curve
    for arm, T, S, Fn, Fp, Tn, n in [(f'{FAST},t.anneal@0.1,t.steps@16', 16, 0.6, 0.1, 0.1, None, 16),
                                     (f'{FAST},t.anneal@0.1,t.steps@32', 32, 0.6, 0.1, 0.1, None, 24),
                                     (f'{FAST},t.anneal@0,t.annealstart@0.25,t.steps@12', 12, 0.25, 0.0, 0.0, None, 12),
                                     (f'{FAST},t.anneal@0.5,t.netfloor@0.05,t.nethorizon@8,t.steps@16', 16, 0.6, 0.05, 0.5, 8, 16)]:
        ctx, b, rows = _lr_rows(arm, n)
        bad = _check_rows(rows, T, S, Fn, Fp, Tn)
        res[arm] = {'horizon': b[0].total_steps, 'lr_rows': rows, 'mismatches': bad,
                    'changes': len({r[1] for r in rows}) > 1}
        log('anneal', arm, 'mismatches', len(bad))
        assert not bad and b[0].total_steps == T and res[arm]['changes'], (arm, bad[:3])
    a16 = res[f'{FAST},t.anneal@0.1,t.steps@16']['lr_rows']
    a32 = res[f'{FAST},t.anneal@0.1,t.steps@32']['lr_rows']
    res['horizon_changes_curve'] = [a16[k][1] != a32[k][1] for k in (10, 12, 15)]
    assert all(res['horizon_changes_curve']) and a16[9][1] == a32[9][1] == 0.012
    # constant (no t.anneal): every update at the base rates, exactly as round 1
    ctx, b, rows = _lr_rows(f'{FAST},t.steps@8', 8)
    res['constant'] = rows
    assert all(r[1:] == (0.012, 0.06, 0.018000000000000002) for r in rows)
    # past the declared horizon (execution_limit > total_steps): well defined, rates hold at the floor
    ctx, b, rows = _lr_rows(f'{FAST},t.anneal@0.1,t.steps@8', 11, limit=11)
    res['past_horizon'] = rows
    assert not _check_rows(rows, 8, 0.6, 0.1, 0.1) and all(math.isclose(r[1], 0.0012, rel_tol=1e-12) for r in rows[8:])
    res['PASS'] = True
    save('anneal', res)
    return res


# ============================================================================ capacity options
def capacity():
    res = {}
    for arm, check in [('t.particles@256,t.zdim@4', lambda b, ctx: tuple(b[3].z.shape) == (256, 4) and b[1].donor[0].in_features == 4 + 7),
                       ('t.particles@2048', lambda b, ctx: tuple(b[3].z.shape) == (2048, 8)),
                       ('t.zdim@16', lambda b, ctx: tuple(b[3].z.shape) == (512, 16) and b[1].donor[0].in_features == 16 + 7),
                       ('t.batch@48,t.bag@64', lambda b, ctx: ctx.batch_size == 48 and ctx.bag == 64),
                       ('t.batch@16,t.bag@256', lambda b, ctx: ctx.batch_size == 16 and ctx.bag == 256),
                       ('t.dlr@0.5,t.glr@2,t.plr@0.5,' + FAST, lambda b, ctx: [g['lr'] for g in b[4].param_groups] == [0.024, 0.024 * 1.25]
                        and math.isclose(b[5].param_groups[0]['lr'], 0.009, rel_tol=1e-15))]:
        ctx = ctx_of(arm)
        t = time.monotonic()
        b, g, r, u, s, mets = run_updates(ctx, 3)
        ok = bool(check(b, ctx))
        G = b[1]
        z, _ = b[3].sample(2, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            rate = G.rates(z, ctx.conditions[:2], 8, ctx.fitlibs[:16].reshape(2, 8), torch.Generator().manual_seed(1))
        res[arm] = {'ok': ok, 'finite_losses': all(math.isfinite(m['G_total']) and math.isfinite(m['D_total']) for m in mets),
                    'rate_finite': bool(torch.isfinite(rate).all()), 'seconds': time.monotonic() - t,
                    'usage_histogram_bins': len(u.unique_batch_histogram), 'prior_shape': list(b[3].z.shape)}
        log('capacity', arm, res[arm])
        assert ok and res[arm]['finite_losses'] and res[arm]['rate_finite'], arm
    # released initial tensors are untouched by t.batch / t.bag / t.glr (only shapes/rates change)
    ctx_b, ctx_c = ctx_of('base'), ctx_of('t.batch@48,t.bag@64')
    bb, *_ = trainer.build(ctx_b, 0)
    bc, *_ = trainer.build(ctx_c, 0)
    res['batch_bag_same_initial_tensors'] = all(sd_equal(x, y) for x, y in zip(states(bb[1], bb[2], bb[3]), states(bc[1], bc[2], bc[3])))
    assert res['batch_bag_same_initial_tensors']
    res['PASS'] = True
    save('capacity', res)
    return res


# ============================================================================ architecture hooks
def hooks():
    import arm_example  # noqa: F401
    res = {}
    # g.example: kwargs + class chain; composition with the round-2 example arm
    ctx = ctx_of('g.example@64')
    b, g, r, u, s, mets = run_updates(ctx, 2)
    G = b[1]
    res['g_kwargs'] = trainer.generator_kwargs(ctx, b[0])
    res['g_class_chain'] = [k.__name__ for k in type(G).__mro__[:3]]
    res['g_shapes'] = {n: list(p.shape) for n, p in G.named_parameters()}
    assert res['g_kwargs'] == {'zd': 8, 'zc': 8, 'h': 64, 'depth': 2} and G.donor[0].out_features == 64 and len(G.extra) == 1
    assert any(p is G.extra[0][1].weight for p in b[4].param_groups[0]['params'])
    ctxc = ctx_of('example@0.5,g.example@32')
    bc, *_ = trainer.build(ctxc, 0)
    res['composed_chain'] = [k.__name__ for k in type(bc[1]).__mro__[:4]]
    assert res['composed_chain'][:2] == ['DeeperGenerator', 'ExampleGenerator'] and hasattr(bc[1], 'example_bias')
    # d.example: per-cell block, layout, plug-in critic, update-0 dD/dx check, combination with genemean and t.bag
    for arm, width in [('d.example', 150 + 128 * 8), ('d.example,genemean,t.bag@32', 150 + 2000 + 32 * 8)]:
        ctx = ctx_of(arm)
        lay = ctx.features.layout()
        b, g, r, u, s, mets = run_updates(ctx, 2)
        res[arm] = {'layout': lay.as_dict(), 'critic': type(b[2]).__name__, 'n_features': ctx.features.n_features,
                    'critic_params_in_optimizer': sum(p.numel() for p in b[2].parameters()) == sum(p.numel() for g_ in b[5].param_groups for p in g_['params']),
                    'losses_finite': all(math.isfinite(m['D_total']) for m in mets)}
        assert ctx.features.n_features == width and lay.block('cellpc8').cell_dim == 8 and lay.blocks[0].name == 'released'
        assert res[arm]['critic'] == 'CellSetCritic' and res[arm]['critic_params_in_optimizer'] and res[arm]['losses_finite']
    assert [x.name for x in ctx.features.layout().blocks] == ['released', 'genemean', 'cellpc8']
    # the update-0 check catches a plug-in critic that ignores a block
    from torch import nn

    class _Blind(arm_example.CellSetCritic):
        def forward(self, x, c):
            return self.phi(torch.cat([x[:, self.vector_cols], c], 1)).sum(1) + 0 * self.rho[0].weight.sum() + 0 * sum(p.sum() for p in self.cell.parameters())

    @arms.register
    class _BlindArm(arms.Arm):
        name = 'd.testblind'

        def feature_blocks(self, ctx):
            return [arm_example.CellPCBlock(ctx.bag)]

        def critic_module(self, ctx, layout):
            return _Blind(layout)
    try:
        run_updates(ctx_of('d.testblind'), 1)
        res['blind_critic_refused'] = False
    except AssertionError as e:
        res['blind_critic_refused'] = 'cellpc8' in str(e)
    try:
        arms.parse_arm('d.example,d.testblind')
        res['two_critic_modules_refused'] = False
    except ValueError:
        res['two_critic_modules_refused'] = True
    assert res['blind_critic_refused'] and res['two_critic_modules_refused']
    # bad per-cell width refused
    class _BadBlock(arm_example.CellPCBlock):
        name = 'badcell'

        def __init__(self, cells):
            super().__init__(cells)
            self.width = cells * 8 + 1

    @arms.register
    class _BadArm(arms.Arm):
        name = 'd.testbadwidth'

        def feature_blocks(self, ctx):
            return [_BadBlock(ctx.bag)]
    try:
        ctx_of('d.testbadwidth')
        res['bad_cell_width_refused'] = False
    except ValueError:
        res['bad_cell_width_refused'] = True
    assert res['bad_cell_width_refused']
    # checkpoint round trip (shadow EMA, plug-in critic, g. generator), bitwise
    for arm in ('d.example,g.example@32', 't.anneal@0.1,t.ema@0.9,t.steps@6,' + FAST):
        ctx1 = ctx_of(arm)
        b1, g1, r1, u1, s1 = trainer.build(ctx1, 0)
        for _ in range(2):
            trainer.advance(ctx1, b1, g1, r1, u1, execution_limit=ctx1.topt.steps)
        q = trainer.snapshot(ctx1, b1, g1, r1, u1, 0, s1, 'incomplete', endpoint=ctx1.topt.steps, stream_root=trainer.STREAM_ROOT, wall_seconds=0.)
        for _ in range(2):
            trainer.advance(ctx1, b1, g1, r1, u1, execution_limit=ctx1.topt.steps)
        ctx2 = ctx_of(arm)
        b2, g2, r2, u2, _ = trainer.restore(ctx2, q, 0)
        for _ in range(2):
            trainer.advance(ctx2, b2, g2, r2, u2, execution_limit=ctx2.topt.steps)
        res[f'restore:{arm}'] = {
            'models_bitwise': all(sd_equal(x, y) for x, y in zip(states(b1[1], b1[2], b1[3]), states(b2[1], b2[2], b2[3]))),
            'shadow_bitwise': trainer._same(trainer.shadow_of(b1).state(), trainer.shadow_of(b2).state()),
            'policy_averages_bitwise': trainer._same(b1[-1].state_dict()['averages'], b2[-1].state_dict()['averages']),
            'lrs_equal': [g_['lr'] for g_ in b1[4].param_groups] == [g_['lr'] for g_ in b2[4].param_groups]}
        assert all(res[f'restore:{arm}'].values()), res[f'restore:{arm}']
    # config departures for g./d. flags
    ns = argparse.Namespace(seed=0, cache=str(CACHE0), updates=20, device='cpu', checkpoint_every=2000)
    cfg = trainer.make_config(ctx_of('d.example,g.example@32,t.ema@0.99'), ns, trainer.STREAM_ROOT)
    res['config_departures'] = cfg['departures']
    res['config_critic_module'] = cfg['critic_module']
    res['config_generator_kwargs'] = cfg['generator_kwargs']
    assert set(cfg['departures']) == {'d.example', 'g.example@32', 't.ema@0.99'} and cfg['critic_module'] == 'd.example'
    assert cfg['default_generation_state'] == 'native_ema' and cfg['generator_kwargs']['h'] == 32
    res['PASS'] = True
    save('hooks', res)
    log('hooks', 'PASS')
    return res


# ============================================================================ CLI fits
def r3_train(arm, root, *extra, check=True, updates=None):
    cmd = [PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', '0', '--out-root', root, '--device', 'cpu', '--threads', '3']
    if updates is not None:
        cmd += ['--updates', str(updates)]
    return sh(cmd + list(extra), check=check)


def fitpath(root, arm):
    return Path(root) / arm / 'hvg2000' / 'fold0' / 'seed0'


def final_q(fit):
    return torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)


def curve(fit):
    rows = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines()]
    for r in rows:
        r.pop('wall_seconds')
    return rows


def npz(p):
    with np.load(p, allow_pickle=False) as q:
        return {k: q[k] for k in q.files}


ANNEAL_ARM = 't.anneal@0.1,t.ema@0.9,t.steps@20'


def cli():
    res = {}
    r1root, r3root, resroot, logs = SCR / 'r1fits', SCR / 'r3fits', SCR / 'r3fits_resume', SCR / 'logs'
    for d in (r1root, r3root, resroot, logs):
        shutil.rmtree(d, ignore_errors=True)
    # A: round-1 trainer (read-only code, scratch out root) vs B: r3 trainer with snapshots + shadow EMA on
    _, ta = sh([PY, '-B', R1 / 'trainer.py', '--cache', CACHE0, '--arm', 'base', '--seed', '0', '--out-root', r1root,
                '--updates', str(TINY), '--device', 'cpu', '--threads', '3'], cwd=str(R1))
    _, tb = r3_train('base', r3root, updates=TINY)
    fa, fb = fitpath(r1root, 'base'), fitpath(r3root, 'base')
    qa, qb = final_q(fa), final_q(fb)
    ga, gb = npz(fa / 'generated.npz'), npz(fb / 'generated.npz')
    res['seconds'] = {'r1_fit_and_generate': ta, 'r3_fit_and_generate': tb}
    res['r1_vs_r3'] = {'final_models_bitwise': models_equal(qa['policy']['models'], qb['policy']['models']),
                       'optimizers_bitwise': trainer._same(qa['policy']['optimizers'], qb['policy']['optimizers']),
                       'numpy_rng_equal': qa['numpy_rng'] == qb['numpy_rng'], 'curve_equal': curve(fa) == curve(fb),
                       'generated_bitwise_common_arrays': all(np.array_equal(ga[k], gb[k]) for k in ga),
                       'extra_arrays': sorted(set(gb) - set(ga))}
    assert all(v for k, v in res['r1_vs_r3'].items() if k != 'extra_arrays') and res['r1_vs_r3']['extra_arrays'] == ['sq_log']
    cfg = json.loads((fb / 'config.json').read_text())
    snaps = cfg['snapshot_updates']
    res['base_config'] = {k: cfg[k] for k in ('round', 'updates', 'recipe_total_steps', 'snapshot_updates', 'shadow_ema_decays', 'departures',
                                              'default_generation_state', 'batch')}
    assert snaps == [5, 10, 15, 16, 17, 18, 19] and cfg['round'] == 3 and cfg['departures'] == {} and cfg['recipe_total_steps'] == 20000
    s19 = torch.load(fb / 'snap_19.pt', map_location='cpu', weights_only=False)
    res['snapshot_keys'] = sorted(s19)
    res['snapshot_MB'] = round((fb / 'snap_19.pt').stat().st_size / 2 ** 20, 2)
    assert set(s19['shadow']) == {'0.999', '0.9995', '0.9998'} and 'native_ema' not in s19
    assert set(qb['r3_shadow']) == {'0.999', '0.9995', '0.9998'}
    rec = json.loads((fb / 'fit_receipt.json').read_text())
    assert rec['snapshots_missing'] == [] and rec['status'] == 'completed'
    # refuse to overwrite a completed fit; refused out roots
    p, _ = r3_train('base', r3root, check=False, updates=TINY)
    res['refuse_overwrite'] = p.returncode != 0 and 'FileExistsError' in p.stderr
    bad_roots = {}
    for root in (RUNS / 'fits_r2', RUNS / 'fits', RUNS / 'smoke_r2' / 'x', RUNS / 'posthoc_r2', RUNS / 'somewhere_new'):
        p, _ = r3_train('base', root, check=False, updates=2)
        bad_roots[str(root)] = p.returncode != 0 and 'refusing --out-root' in p.stderr
    res['refused_out_roots'] = bad_roots
    p, _ = r3_train('base', r3root, check=False, updates=20001)
    res['updates_above_horizon_refused'] = p.returncode != 0 and 'horizon' in p.stderr
    assert res['refuse_overwrite'] and all(bad_roots.values()) and res['updates_above_horizon_refused']
    # C: annealed native-EMA fit through run_screen.sh (horizon = t.steps, no -u) with modes
    fc = fitpath(r3root, ANNEAL_ARM)
    jobs = SCR / 'jobs.txt'
    jobs.write_text(f'# r3 core test\n{ANNEAL_ARM} hvg2000 0 0 snapens,raw,ema@0.999,snapens@2,snapema@15:0.9995\n')
    p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-o', r3root, '-l', logs, '-n'])
    res['screen_dry_run'] = p.stdout.strip().splitlines()[-6:]
    assert '--updates 20' in p.stdout and '--mode snapema@15:0.9995' in p.stdout
    for line, extra in [(f'{ANNEAL_ARM} hvg2000 9 0', []), ('t.Steps@4 hvg2000 0 0', []), ('base hvg2000 0 0 t.raw', []),
                        ('base hvg2000 0 0', ['-o', RUNS / 'fits_r2']), ('base hvg2000 0 0', ['-o', RUNS / 'newdir']), ('base hvg2000 0 0', ['-t', '6'])]:
        bad = SCR / 'jobs_bad.txt'
        bad.write_text(line + '\n')
        p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', bad, '-n', *extra], check=False)
        assert p.returncode == 2, (line, extra, p.stdout, p.stderr)
    p, tc = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-o', r3root, '-l', logs])
    res['screen_seconds'] = tc
    files = sorted(x.name for x in fc.iterdir())
    res['annealed_fit_files'] = files
    for f in ['generated.npz', 'generated_snapens.npz', 'generated_raw.npz', 'generated_ema@0.999.npz', 'generated_snapens@2.npz',
              'generated_snapema@15:0.9995.npz', 'final.pt', 'snap_5.pt', 'snap_19.pt']:
        assert f in files, f
    cc = json.loads((fc / 'config.json').read_text())
    res['annealed_config'] = {k: cc[k] for k in ('updates', 'recipe_total_steps', 'departures', 'learning_rates', 'default_generation_state',
                                                 'native_ema_decay')}
    assert cc['updates'] == 20 and cc['recipe_total_steps'] == 20 and cc['default_generation_state'] == 'native_ema'
    assert set(cc['departures']) == {'t.anneal@0.1', 't.ema@0.9', 't.steps@20'}
    lrs = [r['G_prior_LRs'][0] for r in curve(fc)]
    res['annealed_curve_G_lrs'] = lrs
    p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-o', r3root, '-l', logs])
    assert '1 skipped' in p.stdout
    # D: interrupted mid-anneal (stop after 13 > 0.6*20, checkpoint every 5) + resume == uninterrupted, bitwise
    fd = fitpath(resroot, ANNEAL_ARM)
    p, _ = r3_train(ANNEAL_ARM, resroot, '--checkpoint-every', '5', '--stop-after', '13', check=False)
    rq = torch.load(fd / 'recovery.pt', map_location='cpu', weights_only=False)
    res['stop_after'] = {'returncode': p.returncode, 'recovery_updates': rq['completed_updates'],
                         'recovery_G_lr_next': trainer.expected_lrs(trainer.recipe(arms.train_options(arms.parse_arm(ANNEAL_ARM))), 13)[0][0],
                         'snaps_before_resume': sorted(x.name for x in fd.glob('snap_*.pt'))}
    del rq
    r3_train(ANNEAL_ARM, resroot, '--checkpoint-every', '5')
    qc, qd = final_q(fc), final_q(fd)
    gc_, gd = npz(fc / 'generated.npz'), npz(fd / 'generated.npz')
    res['resume_mid_anneal'] = {
        'policy_models_bitwise': models_equal(qc['policy']['models'], qd['policy']['models']),
        'native_ema_bitwise': models_equal(qc['policy']['averages'], qd['policy']['averages']),
        'optimizers_bitwise': trainer._same(qc['policy']['optimizers'], qd['policy']['optimizers']),
        'shadow_bitwise': trainer._same(qc['r3_shadow'], qd['r3_shadow']),
        'numpy_rng_equal': qc['numpy_rng'] == qd['numpy_rng'], 'curve_equal': curve(fc) == curve(fd),
        'generated_bitwise': all(np.array_equal(gc_[k], gd[k]) for k in gc_),
        'snapshots_bitwise': all(trainer._same(torch.load(fc / f'snap_{u}.pt', map_location='cpu', weights_only=False)['shadow'],
                                               torch.load(fd / f'snap_{u}.pt', map_location='cpu', weights_only=False)['shadow'])
                                 and models_equal({'g': torch.load(fc / f'snap_{u}.pt', weights_only=False)['generator']},
                                                  {'g': torch.load(fd / f'snap_{u}.pt', weights_only=False)['generator']})
                                 for u in cc['snapshot_updates']),
        'resumed_from': json.loads((fd / 'fit_receipt.json').read_text())['resumed_from']}
    assert p.returncode != 0 and res['stop_after']['recovery_updates'] == 13
    assert all(v for k, v in res['resume_mid_anneal'].items() if k != 'resumed_from') and res['resume_mid_anneal']['resumed_from'] == 13
    # E: generation modes on the annealed t.ema fit
    ctx = trainer.Context(CACHE0, 'cpu', tuple(cc['flags']))
    st = generate.available_states(fc, cc)
    res['states'] = list(st)
    assert st[-1] == 'final' and 'raw' in st and 'ema_0.9995' in st and 'snapema_15_0.999' in st and 'snap_5' in st
    modes = {m: npz(fc / f'generated_{m}.npz') for m in ('snapens', 'raw', 'ema@0.999', 'snapens@2', 'snapema@15:0.9995')}
    se = modes['snapens']
    res['snapens'] = {'state_names': se['state_names'].tolist(),
                      'counts': [int((se['state_index'][0] == i).sum()) for i in range(len(se['state_names']))]}
    assert res['snapens']['state_names'] == ['snap_15', 'snap_16', 'snap_17', 'snap_18', 'snap_19', 'raw'] and res['snapens']['counts'] == [6, 6, 5, 5, 5, 5]
    assert modes['snapens@2']['state_names'].tolist() == ['snap_18', 'snap_19', 'raw']
    assert modes['snapema@15:0.9995']['state_names'].tolist() == ['snapema_15_0.9995']
    diff = {m: (not np.array_equal(a['pb_counts'], gc_['pb_counts'])) and bool(np.isfinite(a['mean_log']).all()) for m, a in modes.items()}
    res['modes_differ_from_default_and_finite'] = diff
    res['native_ema_vs_raw_mean_abs_meanlog_diff'] = float(np.abs(modes['raw']['mean_log'] - gc_['mean_log']).mean())
    assert all(diff.values()) and bool(np.isfinite(gc_['mean_log']).all())
    # CRN: a plan of (default cond, 'final') reproduces generated.npz; 'final' of a t.ema fit = policy averages
    T = generate.targets(ctx)
    sub = [T[0], T[-1]]
    plans, _ = generate.default_plans(ctx, sub)
    models = generate.load_models(ctx, fc, cc, ['final', 'raw', 'ema_0.999', 'snapema_15_0.9995', 'snap_15'])
    arr, _ = generate.generate_arrays(ctx, {'final': models['final']}, plans, T=sub)
    rows = [0, len(T) - 1]
    res['crn_default_bitwise'] = all(np.array_equal(arr[k], gc_[k][rows]) for k in ('pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log'))
    res['final_is_native_ema'] = sd_equal(states(models['final'][0])[0], qc['policy']['averages']['generator'])
    res['raw_is_live'] = sd_equal(states(models['raw'][0])[0], qc['policy']['models']['generator'])
    s15 = torch.load(fc / 'snap_15.pt', map_location='cpu', weights_only=False)
    res['snapema_from_snapshot'] = sd_equal(states(models['snapema_15_0.9995'][0])[0], s15['shadow']['0.9995']['generator'])
    res['snapshot_native_ema_present'] = 'native_ema' in s15 and s15['native_ema']['decay'] == 0.9
    assert res['crn_default_bitwise'] and res['final_is_native_ema'] and res['raw_is_live'] and res['snapema_from_snapshot']
    assert res['snapshot_native_ema_present']
    p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', fc, '--cache', CACHE0, '--mode', 'snapens', '--threads', '3'], check=False)
    assert p.returncode != 0 and 'exists' in p.stderr
    p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', fb, '--cache', CACHE0, '--mode', 'ema@0.99', '--threads', '3'], check=False)
    res['unknown_ema_decay_refused'] = p.returncode != 0 and 'shadow EMA state' in p.stderr
    assert res['unknown_ema_decay_refused']
    res['PASS'] = True
    save('cli', res)
    log('cli', json.dumps({k: v for k, v in res.items() if k in ('seconds', 'r1_vs_r3', 'resume_mid_anneal', 'snapens')}, default=str)[:3000])
    return res


# ============================================================================ round-1 / round-2 fits through r3 generation
OLD_GEN_CODE = """
    import json, numpy as np, trainer, generate
    out = {{}}
    for label, fit, modes in {jobs!r}:
        cfg = json.load(open(fit + '/config.json'))
        ctx = trainer.Context({cache!r}, 'cpu', tuple(cfg['flags']))
        T = generate.targets(ctx)
        sub = [T[0], T[-1]]
        for mode in modes:
            st = generate.available_states(fit, cfg)
            plans = generate.default_plans(ctx, sub)[0] if mode is None else generate.mode_plans(ctx, mode, sub, st)[0]
            used = sorted({{s for _, n in plans for s in n}})
            models = generate.load_models(ctx, fit, cfg, used)
            arr, _ = generate.generate_arrays(ctx, models, plans, T=sub)
            out[label + ':' + str(mode)] = {{k: arr[k] for k in ('pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log')}}
    np.savez({out!r}, **{{k + '|' + a: v for k, d in out.items() for a, v in d.items()}})
"""


def oldfits():
    import arm_pbulk  # noqa: F401
    res = {}
    SCR.mkdir(parents=True, exist_ok=True)
    jobs = [('r1', str(R1_GPU_FIT), [None]), ('r2', str(R2_GPU_FIT), [None, 'snap@19000', 'snapens'])]
    ref_path = SCR / 'oldfits_r2code.npz'
    other_round_python(R2, OLD_GEN_CODE.format(jobs=jobs, cache=str(CACHE0), out=str(ref_path)))
    ref = npz(ref_path)
    ref_path.unlink()
    eq = {}
    for label, fit, modes in jobs:
        fit = Path(fit)
        cfg = json.loads((fit / 'config.json').read_text())
        ctx = trainer.Context(CACHE0, 'cpu', tuple(cfg['flags']))
        T = generate.targets(ctx)
        sub = [T[0], T[-1]]
        st = generate.available_states(fit, cfg)
        res[f'{label}_states'] = list(st)
        for mode in modes:
            plans = generate.default_plans(ctx, sub)[0] if mode is None else generate.mode_plans(ctx, mode, sub, st, updates=cfg['updates'])[0]
            used = sorted({s for _, n in plans for s in n})
            models = generate.load_models(ctx, fit, cfg, used)
            arr, _ = generate.generate_arrays(ctx, models, plans, T=sub)
            key = f'{label}:{mode}'
            eq[key] = all(np.array_equal(arr[a], ref[f'{key}|{a}']) for a in ('pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log'))
            log('oldfits', key, eq[key])
    res['r3_equals_r2_code_cpu'] = eq
    assert all(eq.values()) and res['r1_states'] == ['final'] and res['r2_states'] == ['snap_16000', 'snap_17000', 'snap_18000', 'snap_19000', 'final']
    # CLI with --out: guards and a full mode file for the round-2 fit
    out_dir = SCR / 'oldfits'
    shutil.rmtree(out_dir, ignore_errors=True)
    before = {f: {p.name: p.stat().st_mtime_ns for p in Path(f).iterdir()} for f in (R1_GPU_FIT, R2_GPU_FIT)}
    refusals = {}
    for fit in (R1_GPU_FIT, R2_GPU_FIT):
        for extra in ([], ['--out', fit / 'x.npz'], ['--out', RUNS / 'fits_r2' / 'x.npz'], ['--out', RUNS / 'fits' / 'x.npz'],
                      ['--out', RUNS / 'posthoc_r2' / 'x.npz']):
            p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', CACHE0, '--threads', '3', '--mode', 'raw', *extra], check=False)
            refusals[f'{fit.parent.parent.parent.parent.name}:{extra[-1] if extra else "no --out"}'] = p.returncode != 0
    res['refusals'] = refusals
    assert all(refusals.values()), refusals
    out = out_dir / 'pbulk_snapens.npz'
    _, t = sh([PY, '-B', HERE / 'generate.py', '--fit', R2_GPU_FIT, '--cache', CACHE0, '--threads', '3', '--mode', 'snapens', '--out', out])
    mo = npz(out)
    res['r2_snapens_cli'] = {'seconds': t, 'state_names': mo['state_names'].tolist(),
                             'counts': [int((mo['state_index'][0] == i).sum()) for i in range(len(mo['state_names']))],
                             'receipt_round': json.loads((out_dir / 'pbulk_snapens.receipt.json').read_text())['fit_round']}
    assert res['r2_snapens_cli']['state_names'] == ['snap_16000', 'snap_17000', 'snap_18000', 'snap_19000', 'final']
    assert res['r2_snapens_cli']['counts'] == [7, 7, 6, 6, 6] and res['r2_snapens_cli']['receipt_round'] == 2
    p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', R2_GPU_FIT, '--cache', CACHE0, '--threads', '3', '--mode', 'ema@0.999', '--out',
               out_dir / 'x.npz'], check=False)
    res['r2_fit_has_no_shadow_ema'] = p.returncode != 0 and 'shadow EMA state' in p.stderr
    after = {f: {p.name: p.stat().st_mtime_ns for p in Path(f).iterdir()} for f in (R1_GPU_FIT, R2_GPU_FIT)}
    res['old_fit_folders_untouched'] = before == after
    assert res['r2_fit_has_no_shadow_ema'] and res['old_fit_folders_untouched']
    shutil.rmtree(out_dir, ignore_errors=True)
    res['PASS'] = True
    save('oldfits', res)
    return res


# ============================================================================ real-data drift read-out
def drift():
    """Per-update parameter step of the zero-momentum DualNorm recipe vs the late drift of real round-2 fits.

    For each 2-D weight W [m,n] the DualNorm step is lr * sqrt(max(1, m/n)) * polar(grad) (all singular values ~1,
    smoothing .001), Frobenius norm lr * sqrt(max(1,m/n)) * sqrt(min(m,n)); vectors (biases, log_theta) move by lr in L2;
    each sampled particle row moves by the particle rate.  Over n updates a pure random walk of such steps travels
    sqrt(n) steps, a straight line n steps.  ratio = observed displacement / per-update step.
    """
    res = {}
    fits = sorted(p for p in (RUNS / 'fits_r2').glob('*/hvg2000/fold*/seed0')
                  if (p / 'snap_16000.pt').exists() and (p / 'final.pt').exists() and (p / 'generated.npz').exists())
    lr = 0.012
    for fit in fits:
        snaps = {u: torch.load(fit / f'snap_{u}.pt', map_location='cpu', weights_only=False)['generator'] for u in (16000, 17000, 18000, 19000)}
        fin = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
        snaps[20000] = fin['policy']['models']['generator']
        row = {}
        for name, w in snaps[20000].items():
            if w.ndim == 2:
                m, n = w.shape
                step = lr * math.sqrt(max(1., m / n)) * math.sqrt(min(m, n))
            else:
                step = lr
            d1 = [float((snaps[u + 1000][name] - snaps[u][name]).norm()) for u in (16000, 17000, 18000, 19000)]
            d4 = float((snaps[20000][name] - snaps[16000][name]).norm())
            row[name] = {'per_update_step': step, 'norm': float(w.norm()), 'disp_1000_mean': float(np.mean(d1)), 'disp_4000': d4,
                         'ratio_1000_in_steps': float(np.mean(d1)) / step, 'random_walk_1000': math.sqrt(1000), 'ballistic_1000': 1000,
                         'relative_disp_1000': float(np.mean(d1)) / float(w.norm())}
        res[str(fit.relative_to(RUNS))] = row
    keys = ['out.3.weight', 'out.3.bias', 'out.1.weight', 'donor.2.weight', 'log_theta']
    summ = {k: {s: float(np.median([res[f][k][s] for f in res])) for s in ('ratio_1000_in_steps', 'relative_disp_1000', 'per_update_step', 'norm')}
            for k in keys}
    res['summary_median_over_fits'] = summ
    res['n_fits'] = len(fits)
    save('drift', res)
    log('drift', json.dumps(summ, indent=1))
    return res


def guards():
    """Round-5 repair F1: generate._check_out on real fit folders (no generation, nothing written).

    A fits_r3 fit with out=None or --out inside fits_r3 is refused; --out under smoke_r5 is allowed; a fit folder
    under fits_r5 or outside RUNS (scratch) with out=None is allowed (trainer post-training generation, run_screen.sh
    modes)."""
    fr3 = RUNS / 'fits_r3/pbulk/hvg2000/fold0/seed0'
    cfg = json.loads((fr3 / 'config.json').read_text())
    assert generate.fit_round(cfg) >= 3
    res = {}

    def chk(label, fit, out, ok):
        try:
            generate._check_out(fit, cfg, out)
            got = True
        except ValueError:
            got = False
        res[label] = 'allowed' if got else 'refused'
        assert got == ok, (label, got)

    chk('fits_r3 out=None', fr3, None, False)
    chk('fits_r3 out=fits_r3', fr3, str(fr3 / 'x.npz'), False)
    chk('fits_r3 out=smoke_r5', fr3, str(RUNS / 'smoke_r5/x/y.npz'), True)
    chk('fits_r3 out=results_r3', fr3, str(RUNS / 'results_r3/x.npz'), False)
    chk('fits_r4-like out=None', RUNS / 'fits_r4/pbulk/hvg2000/fold0/seed0', None, False)
    chk('fits_r5 out=None', RUNS / 'fits_r5/pbulk/hvg2000/fold0/seed0', None, True)
    chk('scratch /tmp out=None', Path('/tmp/somefit'), None, True)
    chk('new RUNS folder out=None', RUNS / 'fits_r6/x', None, False)
    save('guards', res)
    log('guards PASS', json.dumps(res))
    return res


def cleanup():
    for d in ('r1fits', 'r3fits', 'r3fits_resume', 'logs', 'oldfits'):
        shutil.rmtree(SCR / d, ignore_errors=True)
    for f in ('jobs.txt', 'jobs_bad.txt', 'parity_round1.pt', 'parity_round2.pt', 'oldfits_r2code.npz'):
        (SCR / f).unlink(missing_ok=True)
    log('cleanup done; kept', sorted(p.name for p in RES.glob('*.json')) if RES.exists() else [])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    steps = {'parity': parity, 'recipes': recipes, 'anneal': anneal, 'capacity': capacity, 'hooks': hooks, 'cli': cli,
             'oldfits': oldfits, 'drift': drift, 'cleanup': cleanup, 'guards': guards}
    if what == 'all':
        for k in ['parity', 'recipes', 'anneal', 'capacity', 'hooks', 'cli', 'oldfits']:
            steps[k]()
    else:
        steps[what]()
