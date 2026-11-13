"""Tests and the offline information probe for ``d.cellset`` (r3/arm_cellcritic.py, work item R5).

Usage (CPU, 3 threads; scratch and results under runs/improve_20261109/smoke_r3/cellcritic/)::

    python3 -B r3/test_arm_cellcritic.py unit      # parsing, features, basis, critic invariance, 3-update fits, neutrality
    python3 -B r3/test_arm_cellcritic.py fit       # CLI tiny fit + interrupted/resumed fit bitwise, generation
    python3 -B r3/test_arm_cellcritic.py cost      # CPU seconds per update vs base
    python3 -B r3/test_arm_cellcritic.py gpu       # ONE short CUDA process (<= 300 updates per arm, --gpu-mem-fraction 0.04)
    python3 -B r3/test_arm_cellcritic.py probe     # offline set-classifier AUCs: does a cell-level view add information?
    python3 -B r3/test_arm_cellcritic.py cleanup   # delete scratch fits, keep result JSONs

Probe design (``probe``).  Fold-0 hvg2000 cache, TRAIN donors only (25
donors; held-out donors are never touched).  The train donors are split by a
fixed permutation (SeedSequence([20261109, crc32('cellprobe')])) into A (13
donors: classifier training) and B (12 donors: classifier test).  For an
existing fit (final state, loaded through r3 generate.load_models on CPU),
fake bags are drawn exactly as in training: donors d uniformly from the
split, 128 real cells of d with replacement give the library sizes, the
condition is ctx.conditions[d], z from the particle prior, rate =
G.rates(z, c, 128, lib), counts = Poisson(rate), and the critic sees the
standardised log1p CP10k of the counts.  Real bags: 128 cells of d.  From
each bag: released 150 features; genemean block (bag mean of every
standardised gene); per-cell 50 PCs; per-cell 64 residual components (the
arm's train-fitted basis).  Classifiers (Adam, 3 restarts, early stopping on
a 20% validation part of the A bags; AUC on B bags):

  released150             MLP 150-128-128-1
  released150+genemean    MLP 2150-128-128-1
  cells_pc50              DeepSets (per-cell MLP 50-64-64, mean+SD pooling, rho 128-128-1)
  cells_pc50_whitened     same on cells standardised WITHIN each bag per coordinate
                          (removes every per-PC bag mean and SD: what is left is
                          cell-to-cell structure beyond the released moment features)
  cells_res64             DeepSets on the 64 residual components only
  cells_all_whitened      DeepSets on PC50 + res64, within-bag standardised
  full                    released150 + genemean + DeepSets(PC50 + res64): the d.cellset@64,genemean critic shape
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '3'
import json
import shutil
import subprocess
import time
import zlib
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
import generate  # noqa: E402
import arm_cellcritic as acc  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r3/cellcritic'
RES = SCR / 'results'
FITS = SCR / 'fits'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
PROBE_FITS = {'base_r1': RUNS / 'fits/base/hvg2000/fold0/seed0',
              'genemean_r2': RUNS / 'fits_r2/genemean/hvg2000/fold0/seed0',
              'pbulk_r2': RUNS / 'fits_r2/pbulk/hvg2000/fold0/seed0'}


def log(*a):
    print('[test_cellcritic]', *a, flush=True)


def save(name, obj):
    RES.mkdir(parents=True, exist_ok=True)
    trainer.dump(RES / f'{name}.json', json.loads(json.dumps(obj, default=str)))


def sh(cmd, check=True, env=ENV):
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


def ctx_of(arm, device='cpu'):
    return trainer.Context(CACHE0, device, arms.parse_arm(arm))


def run_updates(ctx, n, seed=0):
    b, g, r, u, s = trainer.build(ctx, seed)
    mets = [trainer.advance(ctx, b, g, r, u, execution_limit=ctx.topt.steps) for _ in range(n)]
    return b, mets


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# ============================================================================ unit
def unit():
    out = {}
    # parsing
    ok = [arms.arm_name(arms.parse_arm(a)) for a in ('d.cellset', 'd.cellset@64', 'd.cellset@16,genemean', 'd.cellset,pbulk', 'd.cellwhite', 'd.cellwhite@64,genemean')]
    bad = []
    for a in ('d.cellset@0', 'd.cellset@257', 'd.cellset@1.5', 'd.cellset@064', 'd.cellset,d.example', 'd.cellset@x', 'd.cellset,d.cellwhite'):
        try:
            arms.parse_arm(a)
        except ValueError as e:
            bad.append([a, str(e)[:120]])
            continue
        raise AssertionError(f'{a} accepted')
    out['parse'] = {'accepted': ok, 'refused': bad}
    log('parse ok', ok)

    # features: real == fake for the same counts; columns = PCs / residual components
    ctx = ctx_of('d.cellset@16')
    arm = ctx.plugins[0]
    lay = ctx.features.layout()
    blk = lay.block('cells')
    assert blk.cell_dim == 66 and blk.width == 128 * 66 and lay.n_features == 150 + 128 * 66, lay
    rng = np.random.default_rng(0)
    idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128) for i in range(4)]))
    xr = ctx.features.real(idx, np.arange(4))
    cnt = ctx.X[idx]
    rate = cnt.clone().requires_grad_(True)
    from pg_model import critic_features
    cf = critic_features(cnt, rate, ctx.mean, ctx.sd)
    xf = ctx.features.fake(cf, rate, cnt)
    d_rf = float((xr - xf.detach()).abs().max())
    assert d_rf < 1e-5, d_rf
    cells = xr[:, blk.lo:blk.hi].reshape(4, 128, 66)
    assert torch.allclose(cells[..., :50], ctx.trainpcs[idx], atol=1e-5)
    xf[:, blk.lo:blk.hi].sum().backward()
    assert bool(rate.grad.abs().sum() > 0)
    out['features'] = {'real_minus_fake_max': d_rf, 'n_features': lay.n_features, 'layout': lay.as_dict()}
    # residual basis: orthogonal to W, unit variance and zero mean on train cells, deterministic
    V2, c2, _ = acc.fit_residual_basis(ctx.X, ctx.mean, ctx.sd, ctx.W, 16)
    assert torch.equal(V2, arm.V.cpu()) and torch.equal(c2, arm.centre.cpu()), 'basis not deterministic'
    Z = acc._standardize(ctx.X, ctx.mean, ctx.sd)
    proj = (Z @ arm.V - arm.centre).double()
    Qw, _ = torch.linalg.qr(ctx.W.double())
    Vn = arm.V.double() / arm.V.double().norm(dim=0)
    ortho = float((Qw.T @ Vn).abs().max())
    assert ortho < 1e-4, ortho
    assert float(proj.mean(0).abs().max()) < 1e-3 and float((proj.var(0, unbiased=False) - 1).abs().max()) < 1e-3
    out['basis'] = {'max_abs_cos_with_W': ortho, 'train_mean_absmax': float(proj.mean(0).abs().max()),
                    'train_var_minus1_absmax': float((proj.var(0, unbiased=False) - 1).abs().max()), 'info': arm.info['residual_basis']}
    log('features/basis ok', out['basis']['max_abs_cos_with_W'])

    # critic: permutation invariance over cells, deterministic, dD/dx on cells
    torch.manual_seed(0)
    D = acc.CellSetCritic(lay)
    c = ctx.conditions[:4]
    perm = torch.randperm(128)
    xp = xr.clone()
    xp[:, blk.lo:blk.hi] = cells[:, perm].reshape(4, -1)
    y1, y2, y3 = D(xr, c), D(xp, c), D(xr, c)
    assert torch.equal(y1, y3) and float((y1 - y2).abs().max()) < 1e-5
    torch.manual_seed(0)
    Dw = acc.CellSetCritic(lay, whiten=True)
    w1, w2 = Dw(xr, c).detach(), Dw(xp, c).detach()
    assert float((w1 - w2).abs().max()) < 1e-5
    # whitened branch ignores per-bag shifts/scales of every cell coordinate (cell branch only; phi columns unchanged)
    xs = xr.clone()
    xs[:, blk.lo:blk.hi] = (cells * 1.7 + 0.3).reshape(4, -1)
    xs0 = xr.clone()
    assert float((Dw(xs, c) - Dw(xs0, c)).abs().max()) < 1e-4
    out['critic'] = {'perm_diff': float((y1 - y2).abs().max()), 'perm_diff_white': float((w1 - w2).abs().max()),
                     'params': sum(p.numel() for p in D.parameters())}

    # short fits: build + 3 updates (core update-0 check runs inside advance)
    fits = {}
    for a in ('d.cellset', 'd.cellset@16', 'd.cellset@16,genemean', 'd.cellset,t.bag@32,t.batch@16', 'd.cellwhite,genemean'):
        cx = ctx_of(a)
        b, mets = run_updates(cx, 3)
        assert type(b[2]).__name__ == 'CellSetCritic'
        for m in mets:
            assert all(np.isfinite(m[k]) for k in ('D_total', 'G_total', 'critic_gap'))
        ps = mets[0]['penalty_stats']
        fits[a] = {'n_features': cx.features.n_features, 'D_params': sum(p.numel() for p in b[2].parameters()),
                   'critic_gap': [m['critic_gap'] for m in mets], 'D_penalty': [m['D_penalty'] for m in mets],
                   's_thermal_step0': ps.get('s_thermal'), 'mean_grad_norm_step0': ps.get('mean_grad_norm'), 'info': cx.plugins[0].info}
        log('fit ok', a, fits[a]['critic_gap'])
    out['fits'] = fits

    # neutrality: importing this plug-in changes nothing for base / genemean (fresh processes)
    code = ("import sys,hashlib,torch; sys.path.insert(0,%r); torch.set_num_threads(3)\n"
            "import trainer, arms\n"
            "IMPORT\n"
            "res=[]\n"
            "for a in ('base','genemean'):\n"
            "  ctx=trainer.Context(%r,'cpu',arms.parse_arm(a)); b,g,r,u,s=trainer.build(ctx,0)\n"
            "  m=[trainer.advance(ctx,b,g,r,u,execution_limit=20000) for _ in range(3)]\n"
            "  h=hashlib.sha256()\n"
            "  for mod in (b[1],b[2],b[3]):\n"
            "    for k,v in mod.state_dict().items(): h.update(v.cpu().numpy().tobytes())\n"
            "  h.update(repr([(x['D_total'],x['G_total']) for x in m]).encode()); res.append(h.hexdigest())\n"
            "print('HASH', res)\n") % (str(HERE), str(CACHE0))
    hs = []
    for imp in ('', 'import arm_cellcritic'):
        p, _ = sh([PY, '-B', '-c', code.replace('IMPORT', imp)])
        hs.append([l for l in p.stdout.splitlines() if l.startswith('HASH')][0])
    assert hs[0] == hs[1], hs
    out['neutral'] = {'hashes': hs, 'equal': True}
    log('neutrality ok')
    save('unit', out)
    log('UNIT PASS')


# ============================================================================ CLI fit + resume
FIT_ARMS = ('d.cellset@16,t.steps@10', 'd.cellwhite,genemean,t.steps@10')


def fit():
    res = {a: _fit_one(a) for a in FIT_ARMS}
    save('fit', res)
    log('FIT PASS', {a: {k: r[k] for k in ('final_models_bitwise', 'curve_equal', 'generated_bitwise')} for a, r in res.items()})


def _fit_one(FIT_ARM):
    FITS.mkdir(parents=True, exist_ok=True)
    roots = [FITS / 'full', FITS / 'resumed']
    for r in roots:
        shutil.rmtree(r, ignore_errors=True)
    base = [PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', FIT_ARM, '--seed', '0', '--device', 'cpu', '--threads', '3']
    _, t_full = sh(base + ['--out-root', roots[0]])
    p, _ = sh(base + ['--out-root', roots[1], '--stop-after', '5', '--checkpoint-every', '1'], check=False)
    _, t_res = sh(base + ['--out-root', roots[1], '--checkpoint-every', '1'])
    fa, fb = [r / FIT_ARM / 'hvg2000/fold0/seed0' for r in roots]
    qa = torch.load(fa / 'final.pt', map_location='cpu', weights_only=False)
    qb = torch.load(fb / 'final.pt', map_location='cpu', weights_only=False)
    same_models = all(sd_equal(qa['policy']['models'][k], qb['policy']['models'][k]) for k in qa['policy']['models'])
    ca = [json.loads(l) for l in (fa / 'learning_curve.jsonl').read_text().splitlines()]
    cb = [json.loads(l) for l in (fb / 'learning_curve.jsonl').read_text().splitlines()]
    for r in ca + cb:
        r.pop('wall_seconds', None)
    with np.load(fa / 'generated.npz') as A, np.load(fb / 'generated.npz') as B:
        same_gen = all(np.array_equal(A[k], B[k]) for k in A.files)
        finite = bool(np.isfinite(A['mean_log']).all())
    cfg = json.loads((fa / 'config.json').read_text())
    res = {'arm': FIT_ARM, 'stop_rc': p.returncode, 'final_models_bitwise': same_models, 'curve_equal': ca == cb, 'generated_bitwise': same_gen,
           'generated_finite': finite, 'departures': cfg.get('departures'), 'critic_module': cfg.get('critic_module'),
           'critic_layout_n_features': (cfg.get('critic_layout') or {}).get('n_features'), 'seconds_full': t_full, 'seconds_resumed_part': t_res,
           'files': sorted(x.name for x in fa.iterdir())}
    assert same_models and ca == cb and same_gen and finite, res
    return res


# ============================================================================ cost
COST_ARMS = ('base', 'genemean', 'd.cellset', 'd.cellset@64', 'd.cellset@64,genemean')


def _time_arm(arm, device, n, warm=3):
    ctx = ctx_of(arm, device)
    b, g, r, u, s = trainer.build(ctx, 0)
    sync = (lambda: torch.cuda.synchronize()) if device == 'cuda' else (lambda: None)
    for _ in range(warm):
        trainer.advance(ctx, b, g, r, u, execution_limit=ctx.topt.steps)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync()
    t = time.perf_counter()
    gaps = []
    for _ in range(n):
        m = trainer.advance(ctx, b, g, r, u, execution_limit=ctx.topt.steps)
        gaps.append(m['critic_gap'])
    sync()
    dt = (time.perf_counter() - t) / n
    q = {'seconds_per_update': dt, 'n_features': ctx.features.n_features, 'D_params': sum(p.numel() for p in b[2].parameters()),
         'critic_gap_last': gaps[-1], 'updates_timed': n}
    if device == 'cuda':
        q['peak_allocated_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
    del b, g, r, u, ctx
    return q


def cost():
    res = {a: _time_arm(a, 'cpu', 10) for a in COST_ARMS}
    for a in COST_ARMS:
        res[a]['relative_to_base'] = res[a]['seconds_per_update'] / res['base']['seconds_per_update']
        log(a, res[a])
    save('cost_cpu', res)


def gpu():
    """One CUDA process: 30 warm + 150 timed updates per arm (180 <= 300), memory fraction 0.04."""
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
        raise RuntimeError('run without CUDA_VISIBLE_DEVICES=""')
    dev = int(os.environ.get('CELLCRITIC_GPU', '0'))
    torch.cuda.set_device(dev)
    torch.cuda.set_per_process_memory_fraction(0.04, dev)
    res = {}
    for a in ('base', 'd.cellset@64', 'genemean', 'd.cellset@64,genemean'):
        res[a] = _time_arm(a, 'cuda', 150, warm=30)
        torch.cuda.empty_cache()
    for a in res:
        res[a]['relative_to_base'] = res[a]['seconds_per_update'] / res['base']['seconds_per_update']
        log(a, res[a])
    res['note'] = 'GPUs shared with the round-2 screen at 100% utilisation: absolute times are contended; ratios are indicative'
    save('cost_gpu', res)


# ============================================================================ probe
N_TRAIN_BAGS = 1024   # per class, donors A
N_TEST_BAGS = 512     # per class, donors B
PROBE_M = 64


@torch.no_grad()
def _bags(ctx, G, P, donors, n, seed, Vm, centre, K=128):
    """n real and n fake bags over the given train-donor indices (training-style draws). Returns dict of arrays per class."""
    from pg_model import critic_features
    rng = np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(b'cellprobe'), seed]))
    tg = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
    cg = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
    pg = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
    bf = ctx.features.bag_features
    out = {k: {'f150': [], 'gm': [], 'cells': []} for k in ('real', 'fake')}
    for a in range(0, n, 32):
        B = min(32, n - a)
        ds = rng.choice(donors, size=B)
        idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], K, replace=True) for i in ds]))
        lib = ctx.X[idx].sum(-1)
        c = ctx.conditions[torch.as_tensor(ds)]
        z, _ = P.sample(B, generator=tg)
        rate = G.rates(z, c, K, lib, cg)
        counts = torch.poisson(rate, generator=pg)
        for k, zz in (('real', acc._standardize(ctx.X[idx], ctx.mean, ctx.sd)), ('fake', critic_features(counts, rate, ctx.mean, ctx.sd))):
            out[k]['f150'].append(bf(zz @ ctx.W))
            out[k]['gm'].append(zz.mean(1))
            out[k]['cells'].append(torch.cat([zz @ ctx.W, zz @ Vm - centre], -1).half())
    return {k: {f: torch.cat(v[f]) for f in v} for k, v in out.items()}


class _Set(torch.nn.Module):
    def __init__(self, d_cell, d_vec, h=128, hc=64):
        super().__init__()
        self.d_cell, self.d_vec = d_cell, d_vec
        self.psi = torch.nn.Sequential(torch.nn.Linear(d_cell, hc), torch.nn.SiLU(), torch.nn.Linear(hc, hc), torch.nn.SiLU()) if d_cell else None
        self.phi = torch.nn.Sequential(torch.nn.Linear(d_vec, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU()) if d_vec else None
        self.rho = torch.nn.Sequential(torch.nn.Linear((2 * hc if d_cell else 0) + (h if d_vec else 0), h), torch.nn.SiLU(), torch.nn.Linear(h, 1))

    def forward(self, cells, vec):
        f = []
        if self.d_vec:
            f.append(self.phi(vec))
        if self.d_cell:
            e = self.psi(cells)
            mu = e.mean(1)
            f += [mu, ((e - mu[:, None]).square().mean(1) + 1e-6).sqrt()]
        return self.rho(torch.cat(f, 1)).squeeze(1)


def _auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _view(d, name):
    """(cells [N,K,dc] or None, vec [N,dv] or None) for a feature-set name; d = {'f150','gm','cells'} tensors (both classes)."""
    cells = d['cells'].float()
    def white(x):
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-4)
    return {'released150': (None, d['f150']),
            'released150+genemean': (None, torch.cat([d['f150'], d['gm']], 1)),
            'cells_pc50': (cells[..., :50], None),
            'cells_pc50_whitened': (white(cells[..., :50]), None),
            'cells_res64': (cells[..., 50:], None),
            'cells_all_whitened': (white(cells), None),
            'full': (cells, torch.cat([d['f150'], d['gm']], 1)),
            'full_white': (white(cells), torch.cat([d['f150'], d['gm']], 1))}[name]


def _fit_classifier(tr, tests, ytr, seed, epochs=60):
    """Train on tr (A-donor bags); early stopping on a 20% validation part of tr; AUC/BCE on every test set at that epoch."""
    ctr, vtr = tr
    g = torch.Generator().manual_seed(seed)
    n = len(ytr)
    perm = torch.randperm(n, generator=g)
    nval = n // 5
    va, fi = perm[:nval], perm[nval:]

    def norm(x, ref):
        if x is None:
            return None
        dims = tuple(range(x.ndim - 1))
        mu, sd = ref.mean(dims, keepdim=True), ref.std(dims, keepdim=True) + 1e-6
        return (x - mu) / sd
    ctr_n = norm(ctr, None if ctr is None else ctr[fi]); vtr_n = norm(vtr, None if vtr is None else vtr[fi])
    tests_n = {k: (norm(c, None if ctr is None else ctr[fi]), norm(v, None if vtr is None else vtr[fi]), y) for k, ((c, v), y) in tests.items()}
    torch.manual_seed(seed)
    model = _Set(0 if ctr is None else ctr.shape[-1], 0 if vtr is None else vtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    sel = lambda x, i: None if x is None else x[i]
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        order = fi[torch.randperm(len(fi), generator=g)]
        for a in range(0, len(order), 64):
            i = order[a:a + 64]
            loss = lossf(model(sel(ctr_n, i), sel(vtr_n, i)), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc_v = _auc(model(sel(ctr_n, va), sel(vtr_n, va)).numpy(), ytr[va].numpy())
            if auc_v > best[0] + 1e-4:
                r = {'epoch': ep, 'val_auc': auc_v}
                for k, (c, v, y) in tests_n.items():
                    st = model(c, v)
                    r[k] = {'auc': _auc(st.numpy(), y.numpy()), 'bce': float(lossf(st, y))}
                best = (auc_v, r)
    return best[1]


def probe(K=128, fits=None, names=None, tag=''):
    torch.manual_seed(0)
    names = names or ['released150', 'released150+genemean', 'cells_pc50', 'cells_pc50_whitened', 'cells_res64', 'cells_all_whitened', 'full']
    out = {'design': __doc__.split('Probe design')[1].split('"""')[0][:3000], 'fits': {}}
    ctx0 = ctx_of('base')
    Vm, centre, binfo = acc.fit_residual_basis(ctx0.X, ctx0.mean, ctx0.sd, ctx0.W, PROBE_M)
    out['basis'] = binfo
    perm = np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(b'cellprobe')])).permutation(len(ctx0.ids))
    A, Bd = np.sort(perm[:13]), np.sort(perm[13:])
    out['donors_A'] = [ctx0.ids[i] for i in A]; out['donors_B'] = [ctx0.ids[i] for i in Bd]
    only = os.environ.get('PROBE_ONLY')   # optional: one fit label (parallel runs write probe_<label>.json)
    oname = 'probe' + tag if not only else f'probe{tag}_{only}'
    out['bag_cells'] = K
    for label, fit in (fits or PROBE_FITS).items():
        if only and label != only:
            continue
        if not (fit / 'final.pt').exists():
            log('skip', label); continue
        t0 = time.monotonic()
        config = json.loads((fit / 'config.json').read_text())
        ctx = trainer.Context(CACHE0, 'cpu', tuple(config['flags']))
        G, P = generate.load_models(ctx, fit, config, ['final'])['final']
        G.eval(); P.eval()
        tr = _bags(ctx, G, P, A, N_TRAIN_BAGS, 1, Vm, centre, K)
        tsame = _bags(ctx, G, P, A, N_TEST_BAGS, 3, Vm, centre, K)
        te = _bags(ctx, G, P, Bd, N_TEST_BAGS, 2, Vm, centre, K)
        cat = lambda d: {f: torch.cat([d['real'][f], d['fake'][f]]) for f in d['real']}
        trc, tsc, tec = cat(tr), cat(tsame), cat(te)
        ytr = torch.cat([torch.ones(N_TRAIN_BAGS), torch.zeros(N_TRAIN_BAGS)])
        yte = torch.cat([torch.ones(N_TEST_BAGS), torch.zeros(N_TEST_BAGS)])
        res = {}
        for nm in names:
            runs = [_fit_classifier(_view(trc, nm), {'same_donors': (_view(tsc, nm), yte), 'new_donors': (_view(tec, nm), yte)}, ytr, s)
                    for s in range(3)]
            res[nm] = {'runs': runs}
            for k in ('same_donors', 'new_donors'):
                res[nm][k] = {'auc_mean': float(np.mean([r[k]['auc'] for r in runs])), 'auc_sd': float(np.std([r[k]['auc'] for r in runs])),
                              'bce_mean': float(np.mean([r[k]['bce'] for r in runs]))}
            log(label, nm, 'same', round(res[nm]['same_donors']['auc_mean'], 4), 'new', round(res[nm]['new_donors']['auc_mean'], 4),
                'bce same/new', round(res[nm]['same_donors']['bce_mean'], 3), round(res[nm]['new_donors']['bce_mean'], 3))
        # descriptive gaps between real and fake bags in the probe space
        def mstat(d):
            c = d['cells'].float()
            return {'pc_sd_mean': float(c[..., :50].std(1).mean()), 'res_sd_mean': float(c[..., 50:].std(1).mean()),
                    'res_bagmean_rms': float(c[..., 50:].mean(1).square().mean().sqrt())}
        res['descriptive'] = {'real': mstat(te['real']), 'fake': mstat(te['fake'])}
        res['seconds'] = time.monotonic() - t0
        out['fits'][label] = res
        save(oname, out)
        del tr, te, tsame, trc, tec, tsc
    save(oname, out)
    log('PROBE DONE')


def _bag_shape_stats(c):
    """Within-bag structure of per-cell PC coordinates c [N,K,50]: what per-PC means/SDs do not show."""
    x = c.float()
    x = x - x.mean(1, keepdim=True)
    cov = x.transpose(1, 2) @ x / x.shape[1]
    sd = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-8).sqrt()
    cor = cov / (sd[:, :, None] * sd[:, None, :])
    off = cor - torch.diag_embed(cor.diagonal(dim1=1, dim2=2))
    ev = torch.linalg.eigvalsh(cor).flip(-1)
    xs = x / sd[:, None, :]
    return {'mean_abs_offdiag_corr': float(off.abs().sum((1, 2)).mean() / (50 * 49)),
            'top1_corr_eig_frac': float((ev[:, 0] / ev.sum(-1)).mean()),
            'top5_corr_eig_frac': float((ev[:, :5].sum(-1) / ev.sum(-1)).mean()),
            'participation_ratio': float((ev.sum(-1) ** 2 / ev.square().sum(-1)).mean()),
            'mean_kurtosis': float(xs.pow(4).mean(1).mean()), 'mean_abs_skew': float(xs.pow(3).mean(1).abs().mean()),
            'pc1_within_sd': float(sd[:, 0].mean()), 'mean_within_sd': float(sd.mean())}


def describe():
    """Descriptive within-bag statistics of real vs generated bags (all 25 train donors; the probe's fits)."""
    ctx0 = ctx_of('base')
    Vm, centre, _ = acc.fit_residual_basis(ctx0.X, ctx0.mean, ctx0.sd, ctx0.W, 8)
    out = {}
    for label, fit in PROBE_FITS.items():
        config = json.loads((fit / 'config.json').read_text())
        ctx = trainer.Context(CACHE0, 'cpu', tuple(config['flags']))
        G, P = generate.load_models(ctx, fit, config, ['final'])['final']
        G.eval(); P.eval()
        b = _bags(ctx, G, P, np.arange(len(ctx.ids)), 512, 9, Vm, centre)
        out[label] = {k: _bag_shape_stats(b[k]['cells'][..., :50]) for k in ('real', 'fake')}
        log(label, out[label])
    save('describe', out)


def probe_small():
    """Harder probe (16-cell bags, so bag summaries are noisy and AUCs leave 1): does a cell view add to released150+genemean?"""
    probe(K=16, fits={'genemean_r2': PROBE_FITS['genemean_r2'], 'base_r1': PROBE_FITS['base_r1']},
          names=['released150', 'released150+genemean', 'cells_pc50_whitened', 'cells_all_whitened', 'full', 'full_white'], tag='_bag16')


def cleanup():
    shutil.rmtree(FITS, ignore_errors=True)
    for p in SCR.glob('*'):
        if p.name != 'results':
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    log('scratch deleted; kept', sorted(x.name for x in RES.glob('*')))


if __name__ == '__main__':
    {'unit': unit, 'fit': fit, 'cost': cost, 'gpu': gpu, 'probe': probe, 'probe_small': probe_small, 'describe': describe, 'cleanup': cleanup}[sys.argv[1]]()
