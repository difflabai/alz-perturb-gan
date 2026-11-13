"""Fixed within-donor mean shrinkage followed by the existing train-only zcal.

See ../PROTOCOL_DEFAULT_PASS_20261113.txt. The alpha=0 path is identical to
mode_zcal; alpha=.5 is the single declared candidate. Expected bag gene
totals and per-cell libraries are preserved (up to roundoff), donor latents
are untouched. This is a K-cell bag read-out, not an iid cell generator.
Mode grammar: cellshape@ALPHA:biascal:stagehalf (INNER uses zcal's grammar).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '3')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import trainer  # sets the released module paths
import arms
import mode_zcal as MZ


def shrink_mu(mu, alpha):
    if alpha == 0:
        return mu
    library = mu.sum(-1, keepdim=True)
    donor_proportions = mu.sum(1, keepdim=True) / library.sum(1, keepdim=True).clamp_min(1e-30)
    return (1 - alpha) * mu + alpha * library * donor_proportions


def patch_generator(G, alpha):
    if getattr(G, '_cellshape_alpha', None) is not None:
        raise ValueError('cellshape applied twice')
    original = G.parameters_nb

    def parameters_nb(z, c, k, library, gen=None):
        mu, theta = original(z, c, k, library, gen)
        return shrink_mu(mu, alpha), theta

    G.parameters_nb = parameters_nb
    G._cellshape_alpha = alpha


def parse_value(text):
    number, sep, inner = str(text).partition(':')
    alpha = float(number)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1 or not sep:
        raise ValueError('cellshape requires ALPHA:INNER with 0 <= ALPHA <= 1')
    value, canon = MZ.parse_value(inner)
    return (alpha, value), f'{alpha:g}:{canon}'


def plan(ctx, value, target, rng, states):
    alpha, inner = value
    ctx._cellshape_alpha = alpha
    return MZ.plan(ctx, inner, target, rng, states)


def prepare(ctx, value, state_name, G, P):
    alpha, inner = value
    patch_generator(G, alpha)
    MZ.prepare(ctx, inner, state_name, G, P)
    extra = dict(getattr(ctx, 'mode_receipt_extra', None) or {})
    extra['cellshape'] = {'alpha': alpha, 'preserves': ['expected per-bag gene totals', 'expected cell library'],
                          'scope': 'within generated donor, K=256; donor latent unchanged',
                          'calibration': 'existing train-only zcal (joint with biascal when requested)'}
    ctx.mode_receipt_extra = extra


arms.register_mode('cellshape', plan, value_policy='required', parse_value=parse_value, prepare=prepare,
                   doc=__doc__)


def main():
    import generate as GEN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', default='cellshape@0.5:biascal:stagehalf')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-mem-fraction', type=float, default=.04)
    a = p.parse_args()
    if not a.mode.startswith('cellshape@'):
        raise ValueError('use mode_zcal.py for the reference read-out')
    torch.set_num_threads(3)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    GEN.generate_fit(a.fit, a.cache, mode=a.mode, out=a.out, device=a.device)


if __name__ == '__main__':
    main()
