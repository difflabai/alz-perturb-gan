"""Tests for r3/arm_garch.py (g.h, g.zc, g.depth, g.disp).  CPU only, 3 threads, real fold-0 hvg2000 cache (read-only).

    python3 -B r3/test_arm_garch.py flags     # parsing, canonical values, refusals, conflicts, departures in config
    python3 -B r3/test_arm_garch.py neutral   # importing / registering arm_garch leaves base and genemean bitwise unchanged
                                              #   (3 updates, all tensors, RNG states) - compared in two subprocesses
    python3 -B r3/test_arm_garch.py init      # g.depth@2 / g.disp / g.cdisp / combos: released initial tensors, every stream and the
                                              #   update-0 critic step bitwise equal to base; extra params start at 0 and
                                              #   move; g.h@64 / g.zc@16 shapes; 3 updates finite for each arm
    python3 -B r3/test_arm_garch.py probe     # the probe's class factory builds the same classes / parameter names
    python3 -B r3/test_arm_garch.py cli       # tiny CLI fits g.depth@2,g.h@64 / g.cdisp,g.zc@16 / g.disp (6 updates each):
                                              #   interrupted at 3 + resumed == uninterrupted (final.pt models, shadow EMA,
                                              #   curve, generated.npz); mode raw == default; config kwargs
    python3 -B r3/test_arm_garch.py cleanup
    python3 -B r3/test_arm_garch.py all

Scratch: /ml2/alz/runs/improve_20261109/smoke_r3/capacity/garch_test/ (results/*.json kept, fits deleted by cleanup).
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
import trainer  # noqa: E402  (r3)
import arms  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r3/capacity/garch_test'
RES = SCR / 'results'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
           PYTHONDONTWRITEBYTECODE='1')
FAST = 't.bag@16,t.batch@16'


def log(*a):
    print('[test_arm_garch]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', json.loads(json.dumps(obj, default=str)))


def sh(cmd, check=True, **kw):
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p


def sd(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def ctx_of(arm):
    return trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))


# ============================================================================ flags
def flags():
    ok = {}
    for arm in ['g.h@256', 'g.h@64', 'g.zc@16', 'g.zc@2', 'g.depth@2', 'g.depth@3', 'g.disp', 'g.cdisp', 'g.cdisp,g.depth@2', 'g.depth@2,g.h@256',
                'genemean@0.5,g.depth@2,g.h@256,g.zc@16', 'pbulk,g.disp', 'g.dispz', 'g.dispz@0.1', 'g.dispz,pbulk',
                'fixedstage,g.dispz,pbulk', 'fixedstage,g.cdisp,pbulk', 'g.depth@2,g.dispz']:
        ok[arm] = arms.arm_name(arms.parse_arm(arm))
    bad = {}
    for arm in ['g.h@128', 'g.zc@8', 'g.depth@1', 'g.depth@9', 'g.h@256.0', 'g.h@0256', 'g.h', 'g.disp@1', 'g.zc@0',
                'g.depth@2,g.example@32', 'g.h@64,offset', 'g.h@4096', 'g.cdisp,g.disp', 'g.cdisp@1',
                'g.dispz@1', 'g.dispz@0', 'g.dispz@1.5', 'g.disp,g.dispz', 'g.cdisp,g.dispz', 'fixedstage,g.disp,pbulk']:
        try:
            arms.parse_arm(arm)
            bad[arm] = 'ACCEPTED'
        except ValueError as e:
            bad[arm] = str(e)[:120]
    ctx = ctx_of('g.depth@2,g.disp,g.h@64,g.zc@16,' + FAST)
    from types import SimpleNamespace
    cfg = trainer.make_config(ctx, SimpleNamespace(seed=0, cache=str(CACHE0), updates=10, device='cpu', checkpoint_every=2000),
                              trainer.STREAM_ROOT)
    row = {'accepted': ok, 'refused': bad, 'generator_kwargs': cfg['generator_kwargs'],
           'departures': {k: v for k, v in cfg['departures'].items() if k.startswith('g.')}}
    row['PASS'] = (all(v != 'ACCEPTED' for v in bad.values())
                   and ok['genemean@0.5,g.depth@2,g.h@256,g.zc@16'] == 'g.depth@2,g.h@256,g.zc@16,genemean@0.5'
                   and cfg['generator_kwargs'] == {'zd': 8, 'zc': 16, 'h': 64, 'depth': 2, 'disp': True}
                   and set(row['departures']) == {'g.depth@2', 'g.disp', 'g.h@64', 'g.zc@16'})
    save('flags', row)
    log('flags', 'PASS' if row['PASS'] else 'FAIL', json.dumps(row)[:1500])
    assert row['PASS']


# ============================================================================ neutral
NEUTRAL_CODE = """
import sys, json, hashlib, torch
sys.path.insert(0, {here!r}); torch.set_num_threads(3)
import trainer, arms
if {with_garch}:
    import arm_garch
    arms.parse_arm('g.depth@2')      # registry lookup with the plug-in present
    assert 'g.h' in arms._ARMS
res = {{}}
for arm in ('base', 'genemean'):
    ctx = trainer.Context({cache!r}, 'cpu', arms.parse_arm(arm))
    b, g, r, u, s = trainer.build(ctx, 0)
    h = hashlib.sha256()
    def add(m):
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    for m in (b[1], b[2], b[3]): add(m)
    init = h.hexdigest()
    mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(3)]
    for m in (b[1], b[2], b[3]): add(m)
    h.update(torch.get_rng_state().numpy().tobytes())
    for k in sorted(g): h.update(g[k].get_state().numpy().tobytes())
    h.update(json.dumps(r.bit_generator.state, sort_keys=True).encode())
    res[arm] = dict(init=init, final=h.hexdigest(), cls=type(b[1]).__name__,
                    metrics=[{{k: m[k] for k in ('D_total', 'G_total', 'critic_gap', 'G_gradient_norm')}} for m in mets])
print('RESULT' + json.dumps(res))
"""


def neutral():
    out = {}
    for w in (False, True):
        p = sh([PY, '-B', '-c', NEUTRAL_CODE.format(here=str(HERE), with_garch=w, cache=str(CACHE0))], cwd='/tmp')
        out[w] = json.loads([l for l in p.stdout.splitlines() if l.startswith('RESULT')][0][6:])
    row = {'without': out[False], 'with': out[True], 'PASS': out[False] == out[True]
           and out[True]['base']['cls'] == 'CountGenerator' and out[True]['genemean']['cls'] == 'CountGenerator'}
    save('neutral', row)
    log('neutral', 'PASS' if row['PASS'] else 'FAIL', json.dumps(row)[:800])
    assert row['PASS']


# ============================================================================ init
def _rates(b, g, ctx, seed=123):
    G, P = b[1], b[3]
    gen = torch.Generator().manual_seed(seed)
    z = P.z[:4].detach()
    c = ctx.conditions[:4]
    lib = torch.full((4, 16), 300.)
    with torch.no_grad():
        return G.rates(z, c, 16, lib, gen)


def init():
    import arm_garch  # noqa: F401
    rows = {}
    base_ctx = ctx_of(FAST)
    bb, bg, br, bu, bs = trainer.build(base_ctx, 0)
    base_G, base_D, base_P = sd(bb[1]), sd(bb[2]), sd(bb[3])
    base_rng = torch.get_rng_state()
    base_gens = {k: v.get_state() for k, v in bg.items()}
    base_rate = _rates(bb, bg, base_ctx)
    base_m0 = trainer.advance(base_ctx, bb, bg, br, bu, execution_limit=20000)
    for arm in ('g.depth@2', 'g.disp', 'g.depth@3,g.disp', 'g.cdisp', 'g.cdisp,g.depth@2', 'g.dispz', 'g.dispz@0.1', 'g.depth@2,g.dispz'):
        ctx = ctx_of(arm + ',' + FAST)
        b, g, r, u, s = trainer.build(ctx, 0)
        Gs = sd(b[1])
        released = {k: v for k, v in Gs.items() if k in base_G}
        extra = {k: v for k, v in Gs.items() if k not in base_G}
        row = {'class': type(b[1]).__name__, 'mro': [c.__name__ for c in type(b[1]).__mro__[:5]],
               'released_keys_same': set(released) == set(base_G),
               'released_initial_bitwise': sd_equal(released, base_G),
               'extra_params': sorted(extra), 'extra_all_zero': all(bool((v == 0).all()) for v in extra.values()),
               'critic_prior_initial_bitwise': sd_equal(sd(b[2]), base_D) and sd_equal(sd(b[3]), base_P),
               'global_rng_after_build_equal': torch.equal(torch.get_rng_state(), base_rng),
               'generators_after_build_equal': all(torch.equal(g[k].get_state(), base_gens[k]) for k in g),
               'initial_rates_bitwise': torch.equal(_rates(b, g, ctx), base_rate)}
        m0 = trainer.advance(ctx, b, g, r, u, execution_limit=20000)   # also runs the update-0 gradient assertions
        row['update0_critic_bitwise'] = all(m0[k] == base_m0[k] for k in ('D_adversarial', 'D_penalty', 'D_total', 'critic_gap'))
        row['update0_G_adversarial_bitwise'] = m0['G_adversarial'] == base_m0['G_adversarial']
        mets = [m0] + [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(2)]
        moved = {k: float(v.abs().max()) for k, v in sd(b[1]).items() if k in extra}
        row['extra_moved_max_abs'] = moved
        if 'g.dispz' in arm:   # repair TR-1: no condition input; scale = fixed re-parametrisation of the DualNorm step
            G = b[1]
            z = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
            c0 = torch.zeros(4, 7); c3 = c0.clone(); c3[:, 3] = 1.0; cr = torch.randn(4, 7, generator=torch.Generator().manual_seed(2))
            with torch.no_grad():
                d0, d3, dr = G.dispersion(z, c0), G.dispersion(z, c3), G.dispersion(z, cr)
            row['dispersion_independent_of_condition'] = bool(torch.equal(d0, d3) and torch.equal(d0, dr))
            row['dispersion_depends_on_z'] = bool((G.dispersion(z, c0) - G.dispersion(z.flip(0), c0)).abs().max() > 0)
            row['disp_weight_shape'] = list(G.disp_weight.shape)
            row['scale'] = G.garch_dispz
            sc = 0.1 if '@0.1' in arm else 1.0
            row['PASS_dispz'] = (row['dispersion_independent_of_condition'] and row['dispersion_depends_on_z']
                                 and row['disp_weight_shape'] == [2000, 8] and row['scale'] == sc)
        row['finite'] = all(np.isfinite([m['D_total'], m['G_total']]).all() for m in mets)
        row['PASS'] = (row['released_keys_same'] and row['released_initial_bitwise'] and row['extra_all_zero'] and row['critic_prior_initial_bitwise']
                       and row['global_rng_after_build_equal'] and row['generators_after_build_equal'] and row['initial_rates_bitwise']
                       and row['update0_critic_bitwise'] and row['update0_G_adversarial_bitwise'] and row['finite']
                       and all(v > 0 for v in moved.values()) and len(extra) > 0 and row.get('PASS_dispz', True))
        rows[arm] = row
        log('init', arm, 'PASS' if row['PASS'] else 'FAIL', json.dumps(row)[:600])
    for arm, want in (('g.h@64', {'out.3.weight': (2000, 64), 'donor.2.weight': (128, 64), 'cell.0.weight': (64, 8)}),
                      ('g.zc@16', {'cell.0.weight': (128, 16), 'out.3.weight': (2000, 128)})):
        ctx = ctx_of(arm + ',' + FAST)
        b, g, r, u, s = trainer.build(ctx, 0)
        s0 = sd(b[1])
        shapes = {k: tuple(v.shape) for k, v in s0.items()}
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(3)]
        row = {'class': type(b[1]).__name__, 'shapes': {k: shapes[k] for k in want}, 'zc': b[1].zc,
               'finite': all(np.isfinite([m['D_total'], m['G_total']]).all() for m in mets),
               'initial_log_theta_bitwise_base': torch.equal(s0['log_theta'], base_G['log_theta'])}
        row['PASS'] = all(shapes[k] == v for k, v in want.items()) and row['finite'] and row['initial_log_theta_bitwise_base']
        rows[arm] = row
        log('init', arm, 'PASS' if row['PASS'] else 'FAIL', json.dumps(row)[:400])
    save('init', rows)
    assert all(r['PASS'] for r in rows.values())


# ============================================================================ probe factory
def probe():
    import arm_garch
    out = {}
    for arm, kw in (('g.depth@2', dict(depth=2)), ('g.disp', dict(disp=True)), ('g.depth@3,g.disp', dict(depth=3, disp=True)),
                    ('g.cdisp', dict(cdisp=True)), ('g.cdisp,g.depth@2', dict(cdisp=True, depth=2)), ('g.dispz', dict(dispz=1.0)),
                    ('g.depth@2,g.dispz@0.5', dict(depth=2, dispz=0.5))):
        ctx = ctx_of(arm + ',' + FAST)
        b, *_ = trainer.build(ctx, 0)
        cls = arm_garch.make_generator_class(**kw)
        G2 = cls(ctx.mu.cpu(), ctx.theta.cpu(), zd=8, zc=8, h=128, **kw)
        out[arm] = {'trainer_mro': [c.__name__ for c in type(b[1]).__mro__[:4]], 'probe_mro': [c.__name__ for c in cls.__mro__[:4]],
                    'same_param_shapes': {k: tuple(v.shape) for k, v in sd(b[1]).items()} == {k: tuple(v.shape) for k, v in sd(G2).items()}}
        out[arm]['PASS'] = out[arm]['same_param_shapes'] and out[arm]['trainer_mro'] == out[arm]['probe_mro']
    save('probe', out)
    log('probe', json.dumps(out))
    assert all(r['PASS'] for r in out.values())


# ============================================================================ cli
CLI_ARMS = {   # arm -> (expected generator_kwargs, expected extra generator keys)
    'g.depth@2,g.h@64,' + FAST: ({'zd': 8, 'zc': 8, 'h': 64, 'depth': 2}, ['extra.0.bias', 'extra.0.weight']),
    'g.cdisp,g.zc@16,' + FAST: ({'zd': 8, 'zc': 16, 'h': 128, 'cdisp': True}, ['cdisp_bias', 'cdisp_weight']),
    'g.disp,' + FAST: ({'zd': 8, 'zc': 8, 'h': 128, 'disp': True}, ['disp_bias', 'disp_weight']),
    'g.dispz@0.5,' + FAST: ({'zd': 8, 'zc': 8, 'h': 128, 'dispz': 0.5}, ['disp_bias', 'disp_weight']),
}
RELEASED_KEYS = {'log_theta', 'donor.0.weight', 'donor.0.bias', 'donor.2.weight', 'donor.2.bias', 'cell.0.weight', 'cell.0.bias',
                 'out.1.weight', 'out.1.bias', 'out.3.weight', 'out.3.bias'}


def _train(arm, root, *extra, check=True):
    cmd = [PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', '0', '--out-root', root, '--device', 'cpu',
           '--threads', '3', '--updates', '6', *extra]
    return sh(cmd, check=check)


def _npz(p):
    with np.load(p, allow_pickle=False) as q:
        return {k: q[k] for k in q.files}


def cli():
    """Per arm: uninterrupted 6-update CLI fit == fit interrupted after 3 and resumed (bitwise); mode raw == default."""
    SCR.mkdir(parents=True, exist_ok=True)
    rows = {}
    for arm, (want_kw, want_keys) in CLI_ARMS.items():
        ra, rb = SCR / 'fits_a', SCR / 'fits_b'
        for r in (ra, rb):
            shutil.rmtree(r, ignore_errors=True)
        t = time.monotonic()
        _train(arm, ra)
        p = _train(arm, rb, '--stop-after', '3', check=False)
        interrupted = p.returncode != 0 and (next(rb.glob('*/hvg2000/fold0/seed0')) / 'recovery.pt').exists()
        _train(arm, rb)
        fa = next(ra.glob('*/hvg2000/fold0/seed0'))
        fb = next(rb.glob('*/hvg2000/fold0/seed0'))
        qa = torch.load(fa / 'final.pt', map_location='cpu', weights_only=False)
        qb = torch.load(fb / 'final.pt', map_location='cpu', weights_only=False)
        ma, mb = qa['policy']['models'], qb['policy']['models']
        ga, gb = _npz(fa / 'generated.npz'), _npz(fb / 'generated.npz')
        ca = [json.loads(l) for l in (fa / 'learning_curve.jsonl').read_text().splitlines()]
        cb = [json.loads(l) for l in (fb / 'learning_curve.jsonl').read_text().splitlines()]
        for r in ca + cb:
            r.pop('wall_seconds', None)
        cfg = json.loads((fa / 'config.json').read_text())
        sh([PY, '-B', HERE / 'generate.py', '--fit', fa, '--cache', CACHE0, '--mode', 'raw', '--threads', '3'])
        gr = _npz(fa / 'generated_raw.npz')
        row = {'fit': str(fa), 'interrupted_with_recovery': interrupted,
               'models_bitwise': all(sd_equal(ma[k], mb[k]) for k in ('generator', 'critic', 'prior')),
               'shadow_bitwise': trainer._same(qa['r3_shadow'], qb['r3_shadow']),
               'curve_equal': ca == cb, 'generated_bitwise': set(ga) == set(gb) and all(np.array_equal(ga[k], gb[k]) for k in ga),
               'generated_finite': all(np.isfinite(ga[k].astype(np.float64)).all() for k in ('pb_counts', 'mean_log', 'pcs')),
               'raw_mode_equals_default': all(np.array_equal(ga[k], gr[k]) for k in ga),
               'config_generator_kwargs': cfg['generator_kwargs'], 'config_departures': sorted(cfg['departures']),
               'extra_generator_keys': sorted(set(ma['generator']) - RELEASED_KEYS),
               'seconds': time.monotonic() - t}
        row['PASS'] = (row['interrupted_with_recovery'] and row['models_bitwise'] and row['shadow_bitwise'] and row['curve_equal']
                       and row['generated_bitwise'] and row['generated_finite'] and row['raw_mode_equals_default']
                       and row['extra_generator_keys'] == want_keys and cfg['generator_kwargs'] == want_kw)
        rows[arm] = row
        log('cli', arm, 'PASS' if row['PASS'] else 'FAIL', json.dumps(row))
        for r in (ra, rb):
            shutil.rmtree(r, ignore_errors=True)
    save('cli', rows)
    assert all(r['PASS'] for r in rows.values())


def cleanup():
    for p in SCR.iterdir() if SCR.exists() else []:
        if p.name != 'results':
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    log('cleanup done')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    todo = ['flags', 'neutral', 'init', 'probe', 'cli', 'cleanup'] if what == 'all' else [what]
    for w in todo:
        globals()[w]()
