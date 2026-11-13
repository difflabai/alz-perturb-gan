"""improve_20261109 round-2 generation: ``generated.npz`` and named generation modes.

Standalone::

    python3 -B r2/generate.py --fit FIT_DIR --cache CACHE_DIR [--mode NAME[@v]] \
        [--out PATH.npz] [--device cpu] [--threads 3] [--force]

(r2/trainer.py calls ``generate_fit`` itself after training.)

Default generation (no --mode) is round-1 ``../generate.py`` unchanged
(targets, M = 32 x K = 256, depth-matched libraries, common random numbers
from SeedSequence([20261109, 2, inner_fold, j]) with j = index in
``../folds.json`` development_donors, chunks of 8) plus one array:

    sq_log  float16 [T,M,G]: mean over the K cells of (log1p(panel CP10k))^2
            (same CP10k as mean_log, denominator clamp_min(1)).

The condition for target t is ``Context.target_conditions``: round-1
``build_conditions`` for the flags, then the plug-in hooks (a) and (e) (see
r2/arms.py).  For round-1 flags only it is the round-1 condition.

Model loading.  Generator and prior are rebuilt by ``trainer.build`` (same
classes as training) and their state_dicts are loaded from ``final.pt``
(``policy.models.generator/prior``) or from ``snap_{u}.pt``.  Before loading,
final.pt is checked like round-1 ``restore`` does (completed status,
completed_updates = config updates, flags, seed, cache manifest SHA256, and
the arm state recomputed from the cache equal to the checkpoint's, compared
on CPU).  The full policy is not restored: that is what lets a GPU-trained
final.pt generate on CPU (round-1 ``restore`` refuses a device change).
test_core.py checks that this gives bitwise the round-1 ``generate_fit``
output for a CPU fit.

Modes (``--mode NAME``) write ``generated_{NAME}.npz`` (or ``--out``) with
the same arrays plus ``mode`` (str), ``state_names`` (str[S]),
``state_index`` (int8 [T,M], index into state_names per generated donor) and
``conditions`` (float32 [T,M,7]); a small ``*.receipt.json`` is written
next to it.  A mode's plan (r2/arms.py hook e) gives per generated donor a
condition and a model state; generated donors are processed in order m =
0..M-1, contiguous runs of the same state in chunks of 8, with the target's
default streams and library draws (a plan of (default condition, 'final')
for every m reproduces generated.npz bitwise).  Built-in: ``snapens``,
``snap@U``; plug-ins register more.  Modes never retrain.

Round-1 fits (config without ``round`` = 2, e.g. under
runs/improve_20261109/fits/): any generation through this module needs
``--out``, which must lie outside the round-1 fit tree; nothing is written
into the round-1 fit folder.  Round-1 fits have no snapshots, so only
final-state modes apply.

Repair (2026-11-10, review FID-1, common random numbers across devices).
CPU and CUDA torch generators give different latent / cell / count draws, so
a mode file is CRN-paired with a default generation only if both were made
on the same device.  Every receipt written here now records
``generation_device`` (the evaluator reads it to mark cross-device pairs).
Measured: ``--mode condmix@0 --device cuda`` on the CUDA-trained round-1 fit
base/hvg2000/fold0/seed0 reproduces its round-1 generated.npz bitwise (all 9
arrays), so round-1 comparator modes made with ``--device cuda`` are
CRN-paired with the round-1 default and with r2 modes (run_screen.sh
generates r2 modes on the fit's CUDA device).  CPU-made mode files are not.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

if __name__ == '__main__':
    _t = '3'
    for _i, _a in enumerate(sys.argv):
        if _a == '--threads' and _i + 1 < len(sys.argv):
            _t = sys.argv[_i + 1]
        elif _a.startswith('--threads='):
            _t = _a.split('=', 1)[1]
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = _t
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here in sys.path:
        sys.path.remove(_here)
    sys.path.insert(0, _here)

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import trainer
from trainer import Context, dump, sha, _same
import arms
from arms import stage_of, GenTarget
from pg_model import standardize_counts

if Path(trainer.__file__).resolve().parent != Path(__file__).resolve().parent:
    raise ImportError(f'r2 generate imported the wrong trainer module: {trainer.__file__}')

GEN_ROOT = 20261109
M = 32
K = 256
PC_CELLS = 64
CHUNK = 8
GEN_STREAMS = ['global', 'latent', 'cell', 'count', 'library']
ROUND1_FITS = trainer.RUNS / 'fits'


def generation_streams(fold, dev_index):
    children = np.random.SeedSequence([GEN_ROOT, 2, int(fold), int(dev_index)]).spawn(len(GEN_STREAMS))
    return {k: int(c.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16) for k, c in zip(GEN_STREAMS, children)}


def targets(ctx):
    """[(donor, is_heldout, raw_c[7], libraries[n_cells])] in output order (round-1 copy)."""
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


# --------------------------------------------------------------------------- plans
def default_plans(ctx, T):
    """Per target: (cond float32 [M,7], state names [M]) of default generation."""
    raws = np.stack([t[2] for t in T])
    conds = ctx.target_conditions(raws, np.array([t[1] for t in T], dtype=bool))
    return [(np.repeat(conds[i][None], M, 0), ['final'] * M) for i in range(len(T))], conds


def mode_plans(ctx, mode_text, T, states):
    spec, value, _ = arms.parse_mode(mode_text)
    dev_order = trainer.development_order()
    _, conds = default_plans(ctx, T)
    raws = np.stack([t[2] for t in T])
    stages = stage_of(raws)
    plans = []
    for i, (donor, held, raw, _) in enumerate(T):
        tgt = GenTarget(index=i, donor=donor, is_heldout=bool(held), raw_c=np.array(raw, dtype=np.float64), stage=int(stages[i]),
                        dev_index=dev_order[donor], default_cond=conds[i].copy(), M=M, K=K)
        cond, names = spec.plan(ctx, value, tgt, arms.mode_rng(ctx.fold, dev_order[donor], spec.name), tuple(states))
        cond = np.asarray(cond)
        names = [str(s) for s in names]
        if cond.shape != (M, 7) or cond.dtype != np.float32 or not np.isfinite(cond).all() or len(names) != M:
            raise ValueError(f'mode {mode_text} plan must return float32 [{M},7] conditions and {M} state names')
        bad = set(names) - set(states)
        if bad:
            raise ValueError(f'mode {mode_text} uses states {sorted(bad)} not in the fit {list(states)}')
        plans.append((cond, names))
    return plans, spec, value


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_arrays(ctx, models, plans=None, *, T=None):
    """Generate for every target.  models: {state: (G, P)}; plans: per target (cond [M,7], states [M])."""
    device = ctx.device
    dev_order = trainer.development_order()
    T = targets(ctx) if T is None else T
    raws = np.stack([t[2] for t in T])
    if plans is None:
        plans, _ = default_plans(ctx, T)
    Gn = ctx.n_genes
    arr = {'pb_counts': np.zeros((len(T), M, Gn), np.float32), 'mean_log': np.zeros((len(T), M, Gn), np.float32),
           'nonzero': np.zeros((len(T), M, Gn), np.float32), 'pcs': np.zeros((len(T), M, PC_CELLS, 50), np.float16),
           'lib': np.zeros((len(T), M), np.float32), 'sq_log': np.zeros((len(T), M, Gn), np.float16)}
    zero_cells = 0
    saved = {k: (G.training, P.training) for k, (G, P) in models.items()}
    for G, P in models.values():
        G.eval(); P.eval()
    try:
        for ti, (donor, _, _, pool) in enumerate(T):
            s = generation_streams(ctx.fold, dev_order[donor])
            torch.manual_seed(s['global'])
            lat = torch.Generator(device=device).manual_seed(s['latent'])
            cell = torch.Generator(device=device).manual_seed(s['cell'])
            count = torch.Generator(device=device).manual_seed(s['count'])
            libs = np.random.default_rng(s['library']).choice(pool, size=(M, K), replace=True).astype(np.float32)
            cond, names = plans[ti]
            ct = torch.as_tensor(np.ascontiguousarray(cond), device=device)
            runs, a = [], 0
            for m in range(1, M + 1):
                if m == M or names[m] != names[a]:
                    runs.append((a, m, names[a])); a = m
            for a, b, name in runs:
                G, P = models[name]
                for start in range(a, b, CHUNK):
                    n = min(CHUNK, b - start)
                    z, _ = P.sample(n, generator=lat)
                    lib = torch.as_tensor(libs[start:start + n], device=device)
                    rate = G.rates(z, ct[start:start + n], K, lib, cell)
                    counts = torch.poisson(rate, generator=count)
                    if not bool(torch.isfinite(counts).all()) or not bool((counts >= 0).all()):
                        raise FloatingPointError('generated count contract failed')
                    tot = counts.sum(-1)
                    zero_cells += int((tot == 0).sum())
                    logcp = torch.log1p(counts * (1e4 / tot.clamp_min(1))[..., None])
                    sl = slice(start, start + n)
                    arr['pb_counts'][ti, sl] = counts.sum(1).cpu().numpy()
                    arr['mean_log'][ti, sl] = logcp.mean(1).cpu().numpy()
                    arr['sq_log'][ti, sl] = logcp.square().mean(1).cpu().numpy().astype(np.float16)
                    arr['nonzero'][ti, sl] = (counts > 0).float().mean(1).cpu().numpy()
                    pcs = standardize_counts(counts[:, :PC_CELLS], ctx.mean, ctx.sd) @ ctx.W
                    if not bool(torch.isfinite(pcs).all()):
                        raise FloatingPointError('nonfinite generated projection')
                    arr['pcs'][ti, sl] = pcs.cpu().numpy().astype(np.float16)
                    arr['lib'][ti, sl] = tot.mean(1).cpu().numpy()
    finally:
        for k, (G, P) in models.items():
            G.train(saved[k][0]); P.train(saved[k][1])
    arr.update(targets=np.array([t[0] for t in T]), is_heldout=np.array([t[1] for t in T], dtype=bool),
               stage=stage_of(raws).astype(np.int64), genes=np.asarray(ctx.genes).astype(str))
    stats = {'targets': len(T), 'heldout_targets': int(arr['is_heldout'].sum()), 'M': M, 'K': K,
             'generated_cells': len(T) * M * K, 'zero_count_cells': zero_cells,
             'zero_count_cell_rate': zero_cells / (len(T) * M * K),
             'pcs_float16_max_abs': float(np.abs(arr['pcs'].astype(np.float32)).max()),
             'sq_log_float16_max': float(arr['sq_log'].astype(np.float32).max()),
             'heldout_dir_present': bool(arr['is_heldout'].any())}
    return arr, stats


# --------------------------------------------------------------------------- loading
def is_round2(config):
    return config.get('round') == 2


def available_states(fit, config):
    ups = config.get('snapshot_updates', []) if is_round2(config) else []
    return tuple([f'snap_{u}' for u in sorted(ups) if (Path(fit) / f'snap_{u}.pt').exists()] + ['final'])


ARM_STATE_RTOL = 1e-5
ARM_STATE_ATOL = 1e-6
ARM_STATE_DIFFS = []   # accepted float differences of the last checks (written to the receipt)


def _close(a, b, path, diffs):
    """Generation-time arm-state check (integration fix, 2026-11-10).

    Exact for every non-float leaf (shapes, dtypes, ints, strings, bools,
    keys); float leaves (python floats, float tensors/arrays) are accepted
    when |a-b| <= ATOL + RTOL*|b| elementwise.  Reason: round-1 genemean_scale,
    donorpc pc_*, offset_init and similar arm-state values are float32
    reductions on the training device, so a CUDA-trained fit regenerated on
    CPU (or with another thread count) differs in the last bits (measured
    relative 2.6e-9 on fold 0).  Generation never uses the critic features and
    loads every generator parameter/buffer from the checkpoint, so this
    tolerance cannot change generated output.  Exact equality short-circuits,
    so CPU fits pass exactly as before.  trainer.restore (resume) stays exact.
    Accepted differences are appended to ``diffs`` as (path, max abs diff).
    """
    if _same(a, b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], f'{path}/{k}', diffs) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, f'{path}[{i}]', diffs) for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, float) and isinstance(b, float):
        if abs(a - b) <= ARM_STATE_ATOL + ARM_STATE_RTOL * abs(b):
            diffs.append((path, abs(a - b)))
            return True
        return False
    if isinstance(a, (torch.Tensor, np.ndarray)) and isinstance(b, (torch.Tensor, np.ndarray)):
        ta = torch.as_tensor(np.asarray(a) if isinstance(a, np.ndarray) else a).detach().cpu()
        tb = torch.as_tensor(np.asarray(b) if isinstance(b, np.ndarray) else b).detach().cpu()
        if ta.shape != tb.shape or ta.dtype != tb.dtype or not ta.is_floating_point():
            return False
        ta, tb = ta.double(), tb.double()
        if bool(torch.allclose(ta, tb, rtol=ARM_STATE_RTOL, atol=ARM_STATE_ATOL)):
            diffs.append((path, float((ta - tb).abs().max())))
            return True
        return False
    return False


def check_final(ctx, config, q):
    if q['status'] != 'completed' or q['completed_updates'] != config['updates'] or q.get('fixed_endpoint') != config['updates']:
        raise ValueError('final.pt is not a completed fit of the configured length')
    if tuple(q['flags']) != ctx.flags or q['seed'] != config['seed'] or q['fold'] != ctx.fold or q['panel'] != ctx.panel:
        raise ValueError('final.pt arm/seed/fold/panel differ from the context')
    if q['cache_manifest_SHA256'] != sha(ctx.cache / 'manifest.json'):
        raise ValueError('final.pt cache differs')
    if q.get('stream_root') != config['stream_root']:
        raise ValueError('final.pt stream root differs from config')
    diffs = []
    if not _close(q['arm_state'], ctx.arm_state(), 'arm_state', diffs):
        raise ValueError('arm state recomputed from the cache differs from final.pt')
    ARM_STATE_DIFFS.extend(('final',) + d for d in diffs)


def check_snapshot(ctx, config, s, u):
    want = {'kind': 'model_snapshot', 'completed_updates': u, 'flags': list(ctx.flags), 'seed': config['seed'], 'fold': ctx.fold,
            'panel': ctx.panel, 'cache_manifest_SHA256': sha(ctx.cache / 'manifest.json'), 'fixed_endpoint': config['updates'],
            'stream_root': config['stream_root']}
    bad = {k: (s.get(k), v) for k, v in want.items() if s.get(k) != v}
    if bad:
        raise ValueError(f'snapshot snap_{u}.pt does not belong to this fit: {bad}')
    diffs = []
    if not _close(s['arm_state'], ctx.arm_state(), 'arm_state', diffs):
        raise ValueError(f'snap_{u}.pt arm state differs from the cache')
    ARM_STATE_DIFFS.extend((f'snap_{u}',) + d for d in diffs)


def load_models(ctx, fit, config, names):
    """{state name: (G, P)} rebuilt with trainer.build and loaded from final.pt / snap_u.pt (module docstring)."""
    fit = Path(fit)
    b, *_ = trainer.build(ctx, config['seed'], stream_root=config['stream_root'])
    G0, P0 = b[1], b[3]
    del b
    out = {}
    for name in names:
        if name == 'final':
            q = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
            check_final(ctx, config, q)
            gs, ps = q['policy']['models']['generator'], q['policy']['models']['prior']
        else:
            u = int(name.split('_', 1)[1])
            q = torch.load(fit / f'snap_{u}.pt', map_location='cpu', weights_only=False)
            check_snapshot(ctx, config, q, u)
            gs, ps = q['generator'], q['prior']
        G, P = copy.deepcopy(G0), copy.deepcopy(P0)
        G.load_state_dict(gs, strict=True)
        P.load_state_dict(ps, strict=True)
        out[name] = (G, P)
        del q
    return out


def _check_out(fit, config, out):
    fit = Path(fit).resolve()
    if is_round2(config):
        return
    if out is None:
        raise ValueError(f'{fit} is a round-1 fit: give --out outside the round-1 fit tree')
    o = Path(out).resolve()
    for root in (fit, ROUND1_FITS.resolve()):
        if o == root or root in o.parents:
            raise ValueError(f'refusing to write {o} inside the round-1 fit tree {root}')


def generate_fit(fit, cache, *, device='cpu', ctx=None, mode=None, out=None, force=False):
    """Write generated.npz (mode None) or generated_{mode}.npz (or ``out``) for a finished fit."""
    fit = Path(fit)
    config = json.loads((fit / 'config.json').read_text())
    flags = tuple(config['flags'])
    _check_out(fit, config, out)
    if ctx is None:
        ctx = Context(cache, device, flags, allow_nonprotocol_split=not config['protocol_split'])
    if sha(ctx.cache / 'manifest.json') != config['cache_manifest_SHA256'] or ctx.flags != flags:
        raise ValueError('cache/arm differ from the fit config')
    if out is None:
        out = fit / ('generated.npz' if mode is None else f'generated_{mode}.npz')
    out = Path(out)
    if out.suffix != '.npz':
        raise ValueError('--out must end in .npz')
    if out.exists() and not force and mode is not None:
        raise FileExistsError(f'{out} exists (use --force)')
    T = targets(ctx)
    states = available_states(fit, config)
    if mode is None:
        plans, _ = default_plans(ctx, T)
        spec = value = None
    else:
        plans, spec, value = mode_plans(ctx, mode, T, states)
    used = [s for s in states if any(s in names for _, names in plans)]
    started = time.monotonic()
    del ARM_STATE_DIFFS[:]
    models = load_models(ctx, fit, config, used)
    tol = [[n, path, d] for n, path, d in ARM_STATE_DIFFS]
    if spec is not None and spec.prepare is not None:
        with torch.no_grad():
            for name, (G, P) in models.items():
                spec.prepare(ctx, value, name, G, P)
    arr, stats = generate_arrays(ctx, models, plans, T=T)
    if mode is not None:
        arr['mode'] = np.array(mode)
        arr['state_names'] = np.array(used)
        arr['state_index'] = np.array([[used.index(s) for s in names] for _, names in plans], dtype=np.int8)
        arr['conditions'] = np.stack([c for c, _ in plans]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name('.' + out.stem + '.partial.npz')
    np.savez_compressed(tmp, **arr)
    tmp.replace(out)
    stats['seconds'] = time.monotonic() - started
    stats['generated_SHA256'] = sha(out)
    stats['generation_streams'] = 'SeedSequence([20261109,2,inner_fold,development_donor_index]) -> global,latent,cell,count,library'
    stats['states_used'] = used
    stats['fit'] = str(fit.resolve())
    stats['round1_fit'] = not is_round2(config)
    stats['generation_device'] = str(ctx.device)   # repair FID-1: CRN only within one device type
    if tol:  # float arm-state leaves accepted within tolerance (see _close)
        stats['arm_state_tolerance'] = {'rtol': ARM_STATE_RTOL, 'atol': ARM_STATE_ATOL, 'accepted': tol,
                                        'generation_device': str(ctx.device)}
    if mode is None and is_round2(config) and out == fit / 'generated.npz':
        rp = fit / 'fit_receipt.json'
        receipt = json.loads(rp.read_text()) if rp.exists() else {}
        receipt['generation'] = stats
        receipt['generated_zero_count_cell_rate'] = stats['zero_count_cell_rate']
        receipt['status'] = 'completed'
        receipt['peak_memory_after_generation'] = trainer.peak_memory()
        dump(rp, receipt)
    else:
        stats.update(mode=mode, mode_doc=spec.doc if spec else '', mode_module=spec.module if spec else '')
        dump(out.with_name(out.stem + '.receipt.json'), stats)
    print(f'[generate] wrote {out} ({stats["targets"]} targets, states {used}, zero-count rate {stats["zero_count_cell_rate"]:.2e}, '
          f'{stats["seconds"]:.1f}s)', flush=True)
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(description='Round-2 generation for a finished improve_20261109 fit (round 1 or round 2)')
    p.add_argument('--fit', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--mode', default=None, help='generation mode, e.g. snapens, snap@16000, or a plug-in mode')
    p.add_argument('--out', default=None, help='output .npz (required for round-1 fits; default inside the r2 fit)')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--force', action='store_true', help='replace an existing output')
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    _check_out(a.fit, json.loads((Path(a.fit) / 'config.json').read_text()), a.out)
    target = Path(a.out) if a.out else Path(a.fit) / ('generated.npz' if a.mode is None else f'generated_{a.mode}.npz')
    if target.exists() and not a.force:
        raise FileExistsError(f'{target} exists (use --force to regenerate)')
    generate_fit(a.fit, a.cache, device=a.device, mode=a.mode, out=a.out, force=a.force)


if __name__ == '__main__':
    sys.modules.setdefault('generate', sys.modules['__main__'])
    main()
