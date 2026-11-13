"""improve_20261109 generator: Interface-2 ``generated.npz`` from a finished fit.

Standalone::

    python3 generate.py --fit FIT_DIR --cache CACHE_DIR [--device cpu] [--threads 6]

(trainer.py calls ``generate_fit`` itself after training.)

For every target donor t (train donors in cache order, then heldout donors in
cache order when heldout/ exists; train donors only otherwise):

* M = 32 generated donors x K = 256 cells, condition = the arm's condition for
  t (stage only, or stage + covariates of t with TRAIN age mean/SD).
* Donor latents: ``P.sample(n, generator=latent)`` from the trained particle
  prior (released sample path, eval mode, final non-EMA weights; EMA is off in
  the released recipe), rates ``G.rates(z, c, K, lib, cell)`` (Gamma via the
  global torch RNG) and counts ``torch.poisson(rate, generator=count)`` exactly
  as core-comparison generation.iter_pg; chunks of 8 generated donors.
* Libraries: ``numpy.default_rng(library).choice(t's own observed panel library
  sizes, (M, K), replace=True)`` (depth-matched).
* Common random numbers: the five streams (torch global, latent, cell, count,
  library) are spawned from SeedSequence([20261109, 2, inner_fold, j]) where j
  is t's index in folds.json ``development_donors``; they do not depend on arm,
  panel, training seed or fit.
* Zero-total-count generated cells are kept (no repair).  As in
  pg_model.standardize_counts the CP10k denominator is clamp_min(1), so their
  log1p(CP10k) is 0.  Their rate is reported in fit_receipt.json.

Arrays (PROTOCOL Interface 2): targets str[T], is_heldout bool[T], stage int[T],
genes str[G], pb_counts f32[T,M,G], mean_log f32[T,M,G], nonzero f32[T,M,G],
pcs f16[T,M,64,50] (fold space: standardize_counts(counts) @ W / pc_sd, first
64 cells), lib f32[T,M] (mean generated library size).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # read-only source dirs must not get __pycache__

if __name__ == '__main__':
    _t = '6'
    for _i, _a in enumerate(sys.argv):
        if _a == '--threads' and _i + 1 < len(sys.argv):
            _t = sys.argv[_i + 1]
        elif _a.startswith('--threads='):
            _t = _a.split('=', 1)[1]
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = _t

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import trainer
from trainer import Context, restore, dump, sha
from arms import build_conditions, stage_of
from pg_model import standardize_counts

GEN_ROOT = 20261109
M = 32
K = 256
PC_CELLS = 64
CHUNK = 8
GEN_STREAMS = ['global', 'latent', 'cell', 'count', 'library']


def generation_streams(fold, dev_index):
    children = np.random.SeedSequence([GEN_ROOT, 2, int(fold), int(dev_index)]).spawn(len(GEN_STREAMS))
    return {k: int(c.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16) for k, c in zip(GEN_STREAMS, children)}


def targets(ctx):
    """[(donor, is_heldout, raw_c[7], libraries[n_cells])] in output order."""
    libs = ctx.fitlibs.cpu().numpy()
    out = [(str(d), False, ctx.raw_c[i], libs[ctx.by[i]]) for i, d in enumerate(ctx.ids)]
    h = ctx.cache / 'heldout'
    if (h / 'metadata.npz').exists() and (h / 'counts.npz').exists():
        m = np.load(h / 'metadata.npz', allow_pickle=False)
        ids = m['donors'].astype(str)
        if list(ids) != list(ctx.heldout_ids):
            raise ValueError('heldout donors differ from manifest')
        if 'genes' in m.files and not np.array_equal(m['genes'].astype(str), ctx.genes):
            raise ValueError('heldout gene axis differs')
        counts = trainer.load_counts(h / 'counts.npz')
        if counts.shape[1] != ctx.n_genes:
            raise ValueError('heldout gene count differs')
        hl = np.asarray(counts.sum(1)).ravel().astype(np.float32)
        labels = m['donor'].astype(str)
        raw = m['raw_c'].astype(np.float64)
        for i, d in enumerate(ids):
            rows = np.flatnonzero(labels == d)
            if not len(rows) or (hl[rows] <= 0).any():
                raise ValueError('heldout donor without positive-library cells: ' + d)
            out.append((str(d), True, raw[i], hl[rows]))
    return out


@torch.no_grad()
def generate_arrays(ctx, G, P):
    device = ctx.device
    dev_order = {d: j for j, d in enumerate(trainer.load_folds()['development_donors'])}
    T = targets(ctx)
    raws = np.stack([t[2] for t in T])
    conds = build_conditions(raws, ctx.flags, ctx.age_mean, ctx.age_sd)
    Gn = ctx.n_genes
    arr = {'pb_counts': np.zeros((len(T), M, Gn), np.float32), 'mean_log': np.zeros((len(T), M, Gn), np.float32),
           'nonzero': np.zeros((len(T), M, Gn), np.float32), 'pcs': np.zeros((len(T), M, PC_CELLS, 50), np.float16),
           'lib': np.zeros((len(T), M), np.float32)}
    zero_cells = 0
    modes = G.training, P.training
    G.eval(); P.eval()
    try:
        for ti, (donor, _, _, pool) in enumerate(T):
            s = generation_streams(ctx.fold, dev_order[donor])
            torch.manual_seed(s['global'])
            lat = torch.Generator(device=device).manual_seed(s['latent'])
            cell = torch.Generator(device=device).manual_seed(s['cell'])
            count = torch.Generator(device=device).manual_seed(s['count'])
            libs = np.random.default_rng(s['library']).choice(pool, size=(M, K), replace=True).astype(np.float32)
            c = torch.as_tensor(conds[ti], device=device)[None]
            for start in range(0, M, CHUNK):
                n = min(CHUNK, M - start)
                z, _ = P.sample(n, generator=lat)
                lib = torch.as_tensor(libs[start:start + n], device=device)
                rate = G.rates(z, c.expand(n, -1), K, lib, cell)
                counts = torch.poisson(rate, generator=count)
                if not bool(torch.isfinite(counts).all()) or not bool((counts >= 0).all()):
                    raise FloatingPointError('generated count contract failed')
                tot = counts.sum(-1)
                zero_cells += int((tot == 0).sum())
                logcp = torch.log1p(counts * (1e4 / tot.clamp_min(1))[..., None])
                sl = slice(start, start + n)
                arr['pb_counts'][ti, sl] = counts.sum(1).cpu().numpy()
                arr['mean_log'][ti, sl] = logcp.mean(1).cpu().numpy()
                arr['nonzero'][ti, sl] = (counts > 0).float().mean(1).cpu().numpy()
                pcs = standardize_counts(counts[:, :PC_CELLS], ctx.mean, ctx.sd) @ ctx.W
                if not bool(torch.isfinite(pcs).all()):
                    raise FloatingPointError('nonfinite generated projection')
                arr['pcs'][ti, sl] = pcs.cpu().numpy().astype(np.float16)
                arr['lib'][ti, sl] = tot.mean(1).cpu().numpy()
    finally:
        G.train(modes[0]); P.train(modes[1])
    arr.update(targets=np.array([t[0] for t in T]), is_heldout=np.array([t[1] for t in T], dtype=bool),
               stage=stage_of(raws).astype(np.int64), genes=np.asarray(ctx.genes).astype(str))
    stats = {'targets': len(T), 'heldout_targets': int(arr['is_heldout'].sum()), 'M': M, 'K': K,
             'generated_cells': len(T) * M * K, 'zero_count_cells': zero_cells,
             'zero_count_cell_rate': zero_cells / (len(T) * M * K),
             'pcs_float16_max_abs': float(np.abs(arr['pcs'].astype(np.float32)).max()),
             'heldout_dir_present': bool(arr['is_heldout'].any())}
    return arr, stats


def generate_fit(fit, cache, *, device='cpu', ctx=None):
    fit = Path(fit)
    config = json.loads((fit / 'config.json').read_text())
    flags = tuple(config['flags'])
    if ctx is None:
        ctx = Context(cache, device, flags, allow_nonprotocol_split=not config['protocol_split'])
    if sha(ctx.cache / 'manifest.json') != config['cache_manifest_SHA256'] or ctx.flags != flags:
        raise ValueError('cache/arm differ from the fit config')
    q = torch.load(fit / 'final.pt', map_location=ctx.device, weights_only=False)
    if q['status'] != 'completed' or q['completed_updates'] != config['updates']:
        raise ValueError('final.pt is not a completed fit')
    b, *_ = restore(ctx, q, config['seed'])
    del q
    started = time.monotonic()
    arr, stats = generate_arrays(ctx, b[1], b[3])
    out = fit / 'generated.npz'
    tmp = fit / '.generated.partial.npz'
    np.savez_compressed(tmp, **arr)
    tmp.replace(out)
    stats['seconds'] = time.monotonic() - started
    stats['generated_SHA256'] = sha(out)
    stats['generation_streams'] = 'SeedSequence([20261109,2,inner_fold,development_donor_index]) -> global,latent,cell,count,library'
    rp = fit / 'fit_receipt.json'
    receipt = json.loads(rp.read_text()) if rp.exists() else {}
    receipt['generation'] = stats
    receipt['generated_zero_count_cell_rate'] = stats['zero_count_cell_rate']
    receipt['status'] = 'completed'
    receipt['peak_memory_after_generation'] = trainer.peak_memory()
    dump(rp, receipt)
    print(f'[generate] wrote {out} ({stats["targets"]} targets, zero-count rate {stats["zero_count_cell_rate"]:.2e}, {stats["seconds"]:.1f}s)', flush=True)
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(description='Interface-2 generation for a finished improve_20261109 fit')
    p.add_argument('--fit', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--force', action='store_true', help='replace an existing generated.npz')
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    if (Path(a.fit) / 'generated.npz').exists() and not a.force:
        raise FileExistsError('generated.npz exists (use --force to regenerate)')
    generate_fit(a.fit, a.cache, device=a.device)


if __name__ == '__main__':
    main()
