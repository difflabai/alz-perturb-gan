"""Tests for r5/shortavg.py (round-5 work item 1).

    cd /tmp; PY="/mnt/nvme/ai/conda/anaconda3/bin/python3 -B"; R5=/ml2/alz/analysis/improve_20261109/r5
    $PY $R5/test_shortavg.py unit          # grammar, init shares, paired statistics (no data)
    $PY $R5/test_shortavg.py modes         # snap@U / snapema@U:D / sblind / biascal plans accepted on a 20,000-update fit (CPU, no generation)
    $PY $R5/test_shortavg.py score         # reproduce results_r3 pbulk~r3#snapens and pbulk~r3 (final), fold 0 seed 0, every headline metric
    $PY $R5/test_shortavg.py gen           # GPU: the in-process generation path reproduces the fit's own CUDA mode files bitwise
                                           #   (snapens, ema@0.9995, final; sblind@final == condmix@1; sblind@snapens == stageblind)
Outputs (gen) under runs/improve_20261109/smoke_r5/shortavg/test (deleted afterwards).
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
sys.dont_write_bytecode = True

import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import shortavg as SA  # noqa: E402

FIT = SA.FITS / 'pbulk' / 'hvg2000' / 'fold0' / 'seed0'
CACHE = SA.CACHES / 'hvg2000' / 'fold0'
RES = SA.RUNS / 'results_r3' / 'fits'


def test_unit():
    assert SA.split_readout('sb:snapema@5000:0.999') == ('sb', 'snapema@5000:0.999')
    assert SA.split_readout('snapema@5000:0.999') == ('', 'snapema@5000:0.999')
    assert SA.gen_mode('', 'snapema@5000:0.999') == 'snapema@5000:0.999'
    assert SA.gen_mode('bc', 'snapema@5000:0.999') == 'biascal@snapema:5000:0.999'
    assert SA.gen_mode('bc', 'snap@10000') == 'biascal@snap:10000'
    assert SA.gen_mode('bc', 'final') == 'biascal'
    assert SA.gen_mode('sb', 'snap@10000') == 'sblind@snap_10000'
    assert SA.gen_mode('bcsb', 'ema@0.9995') == 'biascal@sblind:ema_0.9995'
    assert SA.gen_mode('', 'init') == 'winit' and SA.init_share('init', 20000) == 1.0
    assert SA.gen_mode('db', 'snapema@5000:0.9995') == 'debias@snapema_5000_0.9995'
    assert SA.split_readout('db:snapema@5000:0.9995') == ('db', 'snapema@5000:0.9995')
    assert SA.normalise_base('snap@20000', 20000) == 'final'
    assert SA.normalise_base('snapema@20000:0.9995', 20000) == 'ema@0.9995'
    assert SA.normalise_base('snapema@20000:0.9995', 60000) == 'snapema@20000:0.9995'
    assert abs(SA.init_share('snapema@5000:0.9995', 20000) - 0.0820) < 1e-3
    assert abs(SA.init_share('snapema@5000:0.999', 20000) - 0.00672) < 1e-4
    assert SA.init_share('snap@5000', 20000) == 0.0
    assert len(SA.CANDIDATES) == 9   # 3 single states + 3 x 2 EMAs
    assert SA.parse_arms('pbulk,g.dispz,pbulk,t.kappa@0.25') == ['pbulk', 'g.dispz,pbulk,t.kappa@0.25']
    assert SA.parse_arms('g.dispz,pbulk,t.kappa@0.25') == ['g.dispz,pbulk,t.kappa@0.25']
    # paired statistics on a synthetic score table
    S = {}
    rng = np.random.default_rng(0)
    for f in (0, 1, 2):
        for s in (0, 1, 2, 3):
            base = 0.5 + 0.02 * f + rng.normal(0, 0.01)
            S[('A', f, s, 'snapens')] = {'headline': {'pb_rmse': base}, 'extras': {}}
            S[('A', f, s, 'x')] = {'headline': {'pb_rmse': base - 0.01 + rng.normal(0, 0.002)}, 'extras': {}}
    p = SA.paired(S, 'A', 'x', 'snapens', [0, 1, 2], [0, 1, 2, 3])
    assert p['n_pairs'] == 12 and p['n_folds'] == 3 and p['n_folds_better'] == 3
    assert abs(p['mean_diff'] + 0.01) < 0.003 and 0.0003 < p['seed_se'] < 0.002, p
    p1 = SA.paired(S, 'A', 'x', 'snapens', [0, 1, 2], [1], s2_external=1e-4)
    assert p1['seed_se'] == np.sqrt(3 * 1e-4) / 3 and p1['seed_se_source'].startswith('external')
    print('unit OK')


def _ctx():
    SA.setup_generation('cpu', None)
    import generate as GEN
    from trainer import Context
    cfg = json.loads((FIT / 'config.json').read_text())
    ctx = Context(CACHE, 'cpu', tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
    return GEN, ctx, cfg


def test_modes():
    """Plans only: no model is loaded and nothing is written."""
    GEN, ctx, cfg = _ctx()
    T = GEN.targets(ctx)
    states = GEN.available_states(FIT, cfg)
    assert cfg['updates'] == 20000, cfg['updates']
    for u in (5000, 10000, 15000):
        assert f'snap_{u}' in states
        for d in ('0.999', '0.9995'):
            assert f'snapema_{u}_{d}' in states
    for base in SA.CANDIDATES + SA.REFERENCES:
        for v in SA.VARIANTS + ('bcsb',):
            mode = SA.gen_mode(v, base)
            plans, spec, value = GEN.mode_plans(ctx, mode, T, states, updates=cfg['updates'])
            names = {n for _, ns in plans for n in ns}
            want = {'final'} if base == 'final' else (None if base == 'snapens' else {SA.state_token(base)})
            if want is not None:
                assert names == want, (mode, names)
            if v in ('sb', 'bcsb'):   # stage-blind: every generated donor gets the train stage mixture -> same conditions for all targets
                c0 = plans[0][0][:, 3]
                assert all(np.array_equal(np.sort(c[:, 3]), np.sort(c0)) for c, _ in plans), mode
            if v == '' and base != 'snapens':
                d0, _ = GEN.default_plans(ctx, T)
                assert all(np.array_equal(c, dc) for (c, _), (dc, _) in zip(plans, d0)), mode
    print('modes OK: snap@U and snapema@U:D accept U = 5000, 10000, 15000 on a 20,000-update fit; sblind and biascal compose')


def test_score():
    worst = 0.0
    for unit, fname in (('pbulk~r3#snapens', 'generated_snapens.npz'), ('pbulk~r3', 'generated.npz')):
        ref = json.loads((RES / unit / 'hvg2000' / 'fold0' / 'seed0.json').read_text())
        assert Path(ref['generated_path']) == FIT / fname
        import evaluate as EV
        want = {k: v for k, v in EV.headline(ref).items() if '__' not in k}
        flat, extra, _ = SA.score_file(FIT / fname, 0)
        n = 0
        for k, v in want.items():
            if isinstance(v, (int, float)) and v is not None and not k.startswith('lc_'):   # lc_*: learning curve, not a score
                assert flat.get(k) is not None, k
                d = abs(flat[k] - v)
                worst = max(worst, d)
                assert d < 1e-9, (unit, k, flat[k], v)
                n += 1
        print(f'{unit}: pb_rmse {flat["pb_rmse"]:.10f} (results_r3 {want["pb_rmse"]:.10f}); {n} headline metrics equal; '
              f'dist_pred_to_grand {extra["dist_pred_to_grand_all"]:.4f}')
    print(f'score OK (max abs difference {worst:.2e})')


def test_gen():
    import torch
    SA.setup_generation('cuda', 0.04)
    import generate as GEN
    from trainer import Context
    cfg = json.loads((FIT / 'config.json').read_text())
    ctx = Context(CACHE, 'cuda', tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
    out = SA.OUT / 'test'
    out.mkdir(parents=True, exist_ok=True)
    keys = ['pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log']
    ok = True
    try:
        for mode, ref in (('snapens', 'generated_snapens.npz'), ('ema@0.9995', 'generated_ema@0.9995.npz'),
                          ('raw', 'generated.npz'), ('sblind@final', 'generated_condmix@1.npz'),
                          ('sblind@snapens', 'generated_stageblind.npz'), ('snapens', 'generated_snapens.npz')):
            o = out / (SA.safe(mode) + '.npz')
            GEN.generate_fit(FIT, CACHE, device='cuda', ctx=ctx, mode=mode, out=o, force=True)
            with np.load(o) as a, np.load(FIT / ref) as b:
                same = all(np.array_equal(a[k], b[k]) for k in keys)
                if 'conditions' in b.files:
                    same = same and np.array_equal(a['conditions'], b['conditions'])
            print(f'{mode} vs {ref}: {"bitwise equal" if same else "DIFFERENT"}')
            ok &= same
    finally:
        shutil.rmtree(out, ignore_errors=True)
        del ctx
        torch.cuda.empty_cache()
    assert ok
    print('gen OK')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'unit'
    {'unit': test_unit, 'modes': test_modes, 'score': test_score, 'gen': test_gen}[what]()
