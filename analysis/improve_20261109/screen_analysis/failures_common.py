"""failures_common.py - shared helpers for the failures_* scripts (CPU only, read-only inputs).

load_fit(fit_dir, cache_dir): rebuild Context (train counts of the fold's train donors only) and the
generator/prior from final.pt WITHOUT trainer.restore (restore re-derives arm state on the current
device and refuses tiny CPU/CUDA float differences for genemean/offset; the fits ran on CUDA).
gen_pb(...): generated pseudobulk counts [M,G] for one condition vector, mirroring
generate.generate_arrays (P.sample -> G.rates -> Poisson, libraries drawn from the target pool).
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '6')
STUDY = '/ml2/alz/analysis/improve_20261109'
if STUDY not in sys.path:
    sys.path.insert(0, STUDY)

import json
from pathlib import Path

import numpy as np
import torch

import trainer  # noqa: E402  (sets released sys.path, activates the pinned wheel)
import arms  # noqa: E402

torch.set_num_threads(6)
RUNS = Path('/ml2/alz/runs/improve_20261109')
FITS = RUNS / 'fits'
CACHES = RUNS / 'caches'


def fit_path(arm, k, s, panel='hvg2000'):
    return FITS / arm / panel / f'fold{k}' / f'seed{s}'


def load_fit(arm, k, s, panel='hvg2000', ctx=None):
    fd = fit_path(arm, k, s, panel)
    cfg = json.loads((fd / 'config.json').read_text())
    flags = tuple(cfg['flags'])
    if ctx is None or ctx.flags != flags:
        ctx = trainer.Context(CACHES / panel / f'fold{k}', 'cpu', flags)
    q = torch.load(fd / 'final.pt', map_location='cpu', weights_only=False)
    bundle, *_ = trainer.build(ctx, s, stream_root=q['stream_root'])
    G, D, P = bundle[1], bundle[2], bundle[3]
    m = q['policy']['models']
    G.load_state_dict(m['generator'])
    D.load_state_dict(m['critic'])
    P.load_state_dict(m['prior'])
    G.eval(); D.eval(); P.eval()
    return ctx, G, D, P, q, cfg


@torch.no_grad()
def gen_pb(G, P, conds, pool, seed, M=32, K=256, chunk=8):
    seed = int(seed)
    """conds: [7] or [M,7] float32 conditions; pool: library sizes to draw from. Returns pb [M,G] float64."""
    conds = np.asarray(conds, dtype=np.float32)
    if conds.ndim == 1:
        conds = np.repeat(conds[None], M, 0)
    torch.manual_seed(seed)
    lat = torch.Generator().manual_seed(seed + 1)
    cell = torch.Generator().manual_seed(seed + 2)
    cnt = torch.Generator().manual_seed(seed + 3)
    libs = np.random.default_rng(seed + 4).choice(pool, size=(M, K), replace=True).astype(np.float32)
    out = []
    for st in range(0, M, chunk):
        n = min(chunk, M - st)
        z, _ = P.sample(n, generator=lat)
        c = torch.as_tensor(conds[st:st + n])
        rate = G.rates(z, c, K, torch.as_tensor(libs[st:st + n]), cell)
        out.append(torch.poisson(rate, generator=cnt).sum(1).double().numpy())
    return np.concatenate(out)


def log_cp10k(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


def pred_profile(pb):
    """Evaluator prediction: mean over generated donors of log1p CP10k of each pseudobulk."""
    return log_cp10k(pb).mean(0)


def real_fold(k, panel='hvg2000'):
    """Per-donor real pseudobulks (dev donors only, from the fold cache) via evaluate.load_real."""
    import evaluate
    return evaluate.load_real(CACHES / panel / f'fold{k}', panel, k)
