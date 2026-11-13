"""Tests of the round-5 generation mode ``zcal`` (mode_zcal.py).

Run from /tmp with python3 -B (3 threads)::

    test_mode_zcal.py parse        mode grammar (no data)
    test_mode_zcal.py analytic     expected nonzero and its derivative vs Monte Carlo / finite differences (CPU)
    test_mode_zcal.py observed     the calibration target equals the evaluator's train nonzero fraction (CPU, fold 0 cache)
    test_mode_zcal.py leak         NO HELD-OUT DONOR ENTERS THE CALIBRATION (GPU, one round-3 fit):
                                   (a) every file opened during the calibration is traced: none under heldout/;
                                   (b) every donor the calibration generates or reads is a train donor;
                                   (c) the calibration is bitwise unchanged when the held-out side is made
                                       unreadable (generate.targets raises, held-out ids replaced);
                                   (d) the solve reached its tolerance on the genes it could reach
    test_mode_zcal.py identity     with s = 0 forced, zcal@stagehalf reproduces the stored generated_stagehalf.npz
                                   bitwise (all arrays) and the receipt carries mode_extra.zcal (GPU)
    test_mode_zcal.py all

GPU tests use CUDA_VISIBLE_DEVICES (default 0) with memory fraction 0.04 and write only under
runs/improve_20261109/smoke_r5/shape/test/ (deleted at the end).
"""
from __future__ import annotations

import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import builtins  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import trainer  # noqa: E402,F401  (released-module paths)
import arms  # noqa: E402
import generate as GEN  # noqa: E402
import mode_stagedefault  # noqa: E402,F401
import mode_biascal  # noqa: E402,F401
import mode_zcal as MZ  # noqa: E402

RUNS = Path('/ml2/alz/runs/improve_20261109')
FIT = RUNS / 'fits_r3' / 'pbulk' / 'hvg2000' / 'fold0' / 'seed0'
CACHE = RUNS / 'caches' / 'hvg2000' / 'fold0'
TMP = RUNS / 'smoke_r5' / 'shape' / 'test'


def ok(msg):
    print('PASS', msg, flush=True)


def test_parse():
    for text, canon_inner, joint in [('zcal', None, False), ('zcal@stagehalf', 'stagehalf', False),
                                     ('zcal@biascal:stagehalf', 'biascal@stagehalf', True),
                                     ('zcal@biascal:stagehalf:snapens', 'biascal@stagehalf:snapens', True),
                                     ('zcal@snapens', 'snapens', False)]:
        spec, v, _ = arms.parse_mode(text)
        assert spec.name == 'zcal'
        zi, j, inner = MZ._value(v)
        assert zi is False and j == joint, (text, v)
        assert (inner[2] if inner else None) == canon_inner, (text, inner)
    for bad in ('zcal@zcal', 'zcal@zcal:stagehalf', 'zcal@zi:stagehalf', 'zcal@nosuchmode'):
        try:
            arms.parse_mode(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} should be refused')
    ok('parse: zcal, zcal@INNER, zcal@biascal:INNER accepted; zcal@zcal, zi and unknown refused')


def test_analytic():
    g = torch.Generator().manual_seed(1)
    mu = torch.exp(torch.randn(64, 1, 50, generator=g) * 2 - 1)
    theta = torch.exp(torch.randn(50, generator=g))
    nz, d = MZ.expected_nonzero(mu, theta)
    n = 200000
    torch.manual_seed(2)
    rate = torch.distributions.Gamma(theta, theta / mu.expand(64, 1, 50)).sample((n // 64,))
    mc = (torch.poisson(rate) > 0).float().mean((0, 1, 2))
    err = (mc - nz.mean((0, 1))).abs().max().item()
    assert err < 0.006, err
    h = 1e-3
    nzp, _ = MZ.expected_nonzero(mu.double(), theta.double() * np.exp(h))
    nzm, _ = MZ.expected_nonzero(mu.double(), theta.double() * np.exp(-h))
    fd = (nzp - nzm) / (2 * h)
    _, dd = MZ.expected_nonzero(mu.double(), theta.double())
    rel = ((fd - dd).abs() / (dd.abs() + 1e-12)).max().item()
    assert rel < 1e-4, rel
    pl = -torch.expm1(-mu)
    big, _ = MZ.expected_nonzero(mu.double(), torch.full_like(theta, 1e7).double())
    assert (big - pl.double()).abs().max().item() < 1e-5
    ok(f'analytic: expected nonzero vs Monte Carlo max err {err:.4f}; d/dlog theta vs finite differences rel {rel:.1e}; Poisson limit')


def test_observed():
    import evaluate as EV
    ctx = trainer.Context(CACHE, 'cpu', ('pbulk',), allow_nonprotocol_split=False)
    t = MZ.observed_train_nonzero(ctx)
    real = EV.load_real(CACHE, 'hvg2000', 0)
    ref = real.nonzero[~real.is_heldout]
    order = [list(real.donors[~real.is_heldout]).index(d) for d in ctx.ids]
    diff = np.abs(t - ref[order].mean(0)).max()
    assert diff < 1e-12, diff
    ok(f'observed: calibration target = evaluator train nonzero fraction (max diff {diff:.1e}); {len(ctx.ids)} train donors')


def _gpu():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU test: no CUDA device visible (set CUDA_VISIBLE_DEVICES)')
    torch.cuda.set_per_process_memory_fraction(0.04)
    torch.set_num_threads(3)


def _setup(mode):
    cfg = json.loads((FIT / 'config.json').read_text())
    ctx = trainer.Context(CACHE, 'cuda', tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
    T = GEN.targets(ctx)
    states = GEN.available_states(FIT, cfg)
    plans, spec, value = GEN.mode_plans(ctx, mode, T, states, updates=cfg.get('updates'))
    used = [s for s in states if any(s in names for _, names in plans)]
    models = GEN.load_models(ctx, FIT, cfg, used)
    return ctx, models, value, T


def _calibrate(ctx, models):
    st = ctx._zcal
    return MZ.calibrate(ctx, models, st['plans'], joint=True, n_iter=1, log=lambda *_: None)


def test_leak():
    _gpu()
    mode = 'zcal@biascal:stagehalf'
    held = set(json.loads((CACHE / 'manifest.json').read_text())['heldout_donors'])
    # run A with tracing
    ctx, models, value, T = _setup(mode)
    train_ids = set(map(str, ctx.ids))
    assert not (train_ids & held)
    opened, donors_seen = [], set()
    real_open, real_npload = builtins.open, np.load
    real_gen_arrays, real_iter = GEN.generate_arrays, MZ.iterate_chunks

    def t_open(f, *a, **k):
        opened.append(str(f)); return real_open(f, *a, **k)

    def t_npload(f, *a, **k):
        opened.append(str(f)); return real_npload(f, *a, **k)

    def t_gen_arrays(ctx_, models_, plans_=None, *, T=None):
        assert T is not None, 'calibration must pass its own (train) target list'
        donors_seen.update(t[0] for t in T)
        return real_gen_arrays(ctx_, models_, plans_, T=T)

    def t_iter(ctx_, models_, plans_, T_, only=None):
        donors_seen.update(t[0] for t in T_)
        return real_iter(ctx_, models_, plans_, T_, only)
    builtins.open, np.load, GEN.generate_arrays, MZ.iterate_chunks = t_open, t_npload, t_gen_arrays, t_iter
    try:
        info_a = _calibrate(ctx, models)
    finally:
        builtins.open, np.load, GEN.generate_arrays, MZ.iterate_chunks = real_open, real_npload, real_gen_arrays, real_iter
    bad = [p for p in opened if 'heldout' in p]
    assert not bad, bad
    assert donors_seen and donors_seen <= train_ids, donors_seen - train_ids
    ok(f'leak (a,b): {len(opened)} files opened during calibration, none held-out; {len(donors_seen)} donors generated, all train')
    # run B: held-out side unreadable
    ctx_b, models_b, _, _ = _setup(mode)
    ctx_b.heldout_ids = ['NOT_A_DONOR']

    def refuse(*a, **k):
        raise AssertionError('calibration called generate.targets (would read held-out libraries)')
    real_targets = GEN.targets
    GEN.targets = refuse
    try:
        info_b = _calibrate(ctx_b, models_b)
    finally:
        GEN.targets = real_targets
    sa, sb = np.array(info_a['s']), np.array(info_b['s'])
    da, db = np.array(info_a['biascal']['delta']), np.array(info_b['biascal']['delta'])
    assert np.array_equal(sa, sb) and np.array_equal(da, db), (np.abs(sa - sb).max(), np.abs(da - db).max())
    ok('leak (c): calibration bitwise identical with the held-out side made unreadable')
    solve = info_a['solves'][-1]
    assert solve['final_max_abs_resid_free'] < 1e-5, solve
    fin = info_a['final']
    assert fin['nonzero_resid_rms'] < 0.02
    ok(f'leak (d): solve max |F - target| on reachable genes {solve["final_max_abs_resid_free"]:.1e}; '
       f'{solve["n_clipped_high"]} genes at the Poisson bound; sampled train nonzero residual RMS {fin["nonzero_resid_rms"]:.4f}')


def test_identity():
    _gpu()
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / 'identity_zcal@stagehalf.npz'
    real_solve = MZ.solve_dispersion

    def zero_solve(ctx, models, plans, T, target, s0=None, log=None):
        n = len(target)
        return np.zeros(n), {'n_genes': n, 'n_clipped_high': 0, 'n_clipped_low': 0, 'solve_history': [],
                             'final_max_abs_resid_free': None}, np.zeros(n), np.ones(n)
    MZ.solve_dispersion = zero_solve
    try:
        path, info = MZ.run_zcal(FIT, CACHE, 'zcal@stagehalf', out, device='cuda', force=True)
    finally:
        MZ.solve_dispersion = real_solve
    ref = FIT / 'generated_stagehalf.npz'
    with np.load(path) as a, np.load(ref) as b:
        for k in ('pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log', 'conditions', 'state_index'):
            assert np.array_equal(a[k], b[k]), k
    rec = json.loads(out.with_name(out.stem + '.receipt.json').read_text())
    assert rec['mode'] == 'zcal@stagehalf' and 'zcal' in rec['mode_extra'] and len(rec['mode_extra']['zcal']['s']) == 2000
    shutil.rmtree(TMP)
    ok('identity: zcal@stagehalf with s = 0 is bitwise generated_stagehalf.npz; receipt carries mode_extra.zcal')


TESTS = {'parse': test_parse, 'analytic': test_analytic, 'observed': test_observed, 'leak': test_leak, 'identity': test_identity}

if __name__ == '__main__':
    which = sys.argv[1:] or ['all']
    names = list(TESTS) if which == ['all'] else which
    for n in names:
        TESTS[n]()
