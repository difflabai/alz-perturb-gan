"""Round-3 generation mode ``biascal[@INNER]`` (PROTOCOL_R3 "Additions" item 2; owner: stage task).

A train-only per-gene bias calibration baked INTO the generator (no retraining): a per-gene shift
delta_g of the output-logit bias (``G.out[-1].bias``, the released CountGenerator head) chosen so that
the generated pseudobulk profile, averaged over the fold's TRAIN target donors, matches their observed
average.  Every generated cell then comes from the shifted generator, so every distribution metric
(pc/cell energies, nonzero, logvar, coverage) must be re-scored under this mode.

Mode text.  ``biascal`` = calibrate the default read-out (default conditions, state 'final').
``biascal@INNER`` = calibrate and generate with the plan of mode INNER, written with ':' in place of
INNER's own '@' (mode values cannot contain '@'):

    biascal@snapens              biascal@stagehalf             biascal@stagehalf:snapens
    biascal@ema:0.9995           biascal@condmix:0.5           biascal@snapema:15000:0.999

INNER's plan (conditions and model state of every generated donor) is used unchanged, with INNER's
own plan random stream, so with delta = 0 ``biascal@X`` reproduces ``generated_X.npz`` bitwise
(tested).  One delta is shared by every model state the plan uses (the ensemble mean is calibrated,
not each snapshot separately), and the calibration runs under the same conditions as the output
(e.g. under stagehalf's half mixture), so the mode composes with snapens, ema@D and the stage modes.

Fixed point (train donors only; held-out donors are never read):
    O_bar = mean over the fold's train donors of log1p CP10k of their observed full pseudobulk
            (summed train counts from the cache, ``ctx.X``; the evaluator's O for train targets);
    P_bar(delta) = mean over the train targets of the evaluator's prediction P[t] = mean over the
            M = 32 generated donors of log1p CP10k(generated pseudobulk), generated with the
            fit's common random numbers (per-target streams, depth-matched libraries: the train
            targets' default generation streams; identical draws in every iteration);
    delta_0 = 0;  delta_{i+1} = delta_i + (O_bar - P_bar(delta_i)) / slope_i  (Newton step per gene),
    slope_i,g = mean over train targets and generated donors of x/(1+x), x = CP10k of the generated
            pseudobulk (the derivative of log1p(x) with respect to log x; softmax coupling ignored),
            clipped to [0.15, 1]; step clipped to +-3;  N_ITER = 3 updates; the iterate with the
            smallest residual RMS is kept (overshoot guard).
Why not the plain unit step (deviation from the protocol's "one or two fixed-point iterations",
measured 2026-11-11 on round-2 genemean@0.25 seed 0): a random +-0.5 logit probe moved P_bar by a
median 0.22 / 0.43 / 0.70 per unit for low / middle / high thirds of genes by expression, so unit
steps stalled (train residual RMS 0.198 -> 0.150 -> 0.151 in fold 1); a per-gene secant slope
from successive iterates diverged (0.146 -> 0.212 -> 0.525).  Cost: N_ITER extra generations of the
train targets (about 3 x 25/37 of one generation).  Every iteration's residual RMS is recorded
(``history``; last entry ``used_residual_rms``).
Residual floor (measured, genemean@0.25 fold 1): after calibration the train residual stays near
0.13 RMS and is almost a COMMON positive offset growing with expression (about +0.05 low, +0.11
middle, +0.19 top third; +0.24 for the top genes) while the generated and observed mean expression
SHARES of the top genes agree within a few percent.  It is a mean-of-log (Jensen) gap: P averages
log1p CP10k over generated 256-cell bags whose per-gene log values scatter more than the observed
full pseudobulks do; a per-gene logit shift cannot move a common offset (CP10k is normalised), so
biascal removes the shape error but not this dispersion term.

Where the work happens.  Hook (e) has no "all states loaded" call, so ``plan`` stores every target's
plan on the context and ``prepare`` (called once per loaded state, in order) collects the loaded
(G, P) pairs; when the last state the plan uses has arrived it runs the fixed point with all of them
and shifts every generator's head bias in place.  The calibration record (delta RMS, residual RMS
per iteration, states) is kept on ``ctx.biascal_info`` and printed; ``run_biascal`` (CLI below) also
writes it next to the output as ``<out stem>.biascal.json``.

Applicability: any generator whose final logits layer is ``G.out[-1]`` with a bias (every released
or round-2/3 class here; arms that add a stage offset after it, e.g. stageanchor / fixedstage, are
fine because the shift is added to every row).  A g. arm that replaces the head must keep that layer
or the mode refuses.

T1b (PROTOCOL_R3 item 3) = T1 with biascal applied; the shift is fitted on train donors only.

CLI (round-1/2 fits need --out outside their trees)::

    python3 -B r3/mode_biascal.py --fit FIT --cache CACHE --mode biascal@snapens --out OUT.npz [--device cuda --gpu-mem-fraction 0.04]
"""
from __future__ import annotations

import os
import sys

if __name__ == '__main__':
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = '3'
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here in sys.path:
        sys.path.remove(_here)
    sys.path.insert(0, _here)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import arms  # noqa: E402
from arms import register_mode  # noqa: E402

NAME = 'biascal'
N_ITER = 3
SLOPE_MIN, SLOPE_MAX, STEP_MAX = 0.15, 1.0, 3.0


def inner_text(value):
    """biascal value text (':' for '@') -> inner mode text, or None."""
    if value is None:
        return None
    name, sep, rest = str(value).partition(':')
    return name + ('@' + rest if sep else '')


def parse_value(text):
    inner = inner_text(text)
    spec, v, canon = arms.parse_mode(inner)
    if spec.name == NAME:
        raise ValueError('biascal cannot wrap biascal')
    return (spec.name, v, canon), canon.replace('@', ':', 1)


def _inner_plan(ctx, value, target, states):
    if value is None:
        return np.repeat(target.default_cond[None], target.M, 0).astype(np.float32), ['final'] * target.M
    name, v, _ = value
    spec = arms._MODES[name]
    return spec.plan(ctx, v, target, arms.mode_rng(ctx.fold, target.dev_index, name), states)


def plan(ctx, value, target, rng, states):
    st = getattr(ctx, '_biascal', None)
    if st is None or target.index == 0:
        st = ctx._biascal = {'plans': {}, 'models': {}, 'done': False}
    cond, names = _inner_plan(ctx, value, target, states)
    names = [str(s) for s in names]
    st['plans'][target.index] = (np.asarray(cond, dtype=np.float32), names)
    return cond, names


def observed_train_profiles(ctx):
    """[n_train, G] log1p CP10k of the observed full pseudobulk of every train TARGET donor (ctx.ids order).

    Round-4 edit: only the cohort-0 train donors (``ctx.target_idx``; all train donors for a round-1
    cache), i.e. the train targets generate.targets lists; external (MADRC) donors are never targets."""
    out = []
    for i in getattr(ctx, 'target_idx', range(len(ctx.by))):
        r = ctx.by[i]
        c = ctx.X[torch.as_tensor(r, device=ctx.X.device)].double().sum(0).cpu().numpy()
        out.append(np.log1p(c * (1e4 / max(c.sum(), 1.0))))
    return np.stack(out)


def predicted_profiles(pb_counts):
    """[T, G] evaluator prediction: mean over M of log1p CP10k of each generated pseudobulk."""
    X = pb_counts.astype(np.float64)
    L = np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))
    return L.astype(np.float32).astype(np.float64).mean(1)


def head(G):
    try:
        lin = G.out[-1]
    except (AttributeError, IndexError, TypeError):
        lin = None
    if not isinstance(lin, torch.nn.Linear) or lin.bias is None:
        raise ValueError('biascal needs a generator whose final logits layer is G.out[-1] (Linear with bias)')
    return lin.bias


def calibrate(ctx, models, plans_by_index, n_iter=None):
    """Run the fixed point on the TRAIN targets; shift every model's head bias in place.  Returns info."""
    import generate as GEN
    n_iter = N_ITER if n_iter is None else int(n_iter)
    # train targets exactly as generate.targets lists them first (no held-out file is opened here)
    libs = ctx.fitlibs.cpu().numpy()
    Tt = [(str(ctx.ids[i]), False, ctx.raw_c[i], libs[ctx.by[i]]) for i in getattr(ctx, 'target_idx', range(len(ctx.ids)))]   # round 4: cohort-0 only
    tr = list(range(len(Tt)))
    plans = [plans_by_index[i] for i in tr]
    used = sorted({s for _, names in plans for s in names})
    if set(used) != set(models):
        raise ValueError(f'biascal: models {sorted(models)} differ from the states the plan uses {used}')
    O_bar = observed_train_profiles(ctx).mean(0)
    biases = {k: head(G) for k, (G, _) in models.items()}
    delta = np.zeros_like(O_bar)
    hist = []
    best = (np.inf, delta.copy())   # (residual RMS, delta) of the best iteration
    for it in range(n_iter + 1):
        arr, _ = GEN.generate_arrays(ctx, models, plans, T=Tt)
        pb = arr['pb_counts'].astype(np.float64)
        P_bar = predicted_profiles(arr['pb_counts']).mean(0)
        resid = O_bar - P_bar
        r_rms = float(np.sqrt(np.mean(resid ** 2)))
        hist.append({'iteration': it, 'residual_rms': r_rms,
                     'residual_mean': float(resid.mean()), 'delta_rms': float(np.sqrt(np.mean(delta ** 2)))})
        print(f'[biascal] iteration {it}: train mean-profile residual RMS {r_rms:.4f} '
              f'(delta RMS {hist[-1]["delta_rms"]:.4f})', flush=True)
        if r_rms < best[0]:
            best = (r_rms, delta.copy())
        if it == n_iter:
            break
        # analytic per-gene slope dP_bar/d delta_g: mean over targets and generated donors of x/(1+x),
        # x = CP10k of the generated pseudobulk (derivative of log1p(x) in log x; softmax coupling ignored)
        x = pb * (1e4 / np.maximum(pb.sum(-1, keepdims=True), 1))
        slope = np.clip((x / (1 + x)).mean((0, 1)), SLOPE_MIN, SLOPE_MAX)
        step_np = np.clip(resid / slope, -STEP_MAX, STEP_MAX)
        delta = delta + step_np
        step = torch.as_tensor(step_np, dtype=torch.float32)
        with torch.no_grad():
            for b in biases.values():
                b.add_(step.to(b.device))
    if best[0] < hist[-1]['residual_rms']:     # keep the best iterate (guard against overshoot)
        back = torch.as_tensor(best[1] - delta, dtype=torch.float32)
        with torch.no_grad():
            for b in biases.values():
                b.add_(back.to(b.device))
        delta = best[1]
    hist.append({'used_residual_rms': float(best[0])})
    info = {'n_iter': n_iter, 'states': used, 'n_train_targets': len(tr), 'history': hist,
            'delta_rms': float(np.sqrt(np.mean(delta ** 2))), 'delta_mean': float(delta.mean()),
            'delta_absmax': float(np.abs(delta).max()), 'delta': delta.astype(np.float32).tolist()}
    return info


def prepare(ctx, value, state_name, G, P):
    st = getattr(ctx, '_biascal', None)
    if st is None or not st['plans']:
        raise ValueError('biascal: prepare called before the plan')
    if st['done']:
        raise ValueError('biascal: calibration already applied for this mode run')
    st['models'][state_name] = (G, P)
    used = {s for _, names in st['plans'].values() for s in names}
    if set(st['models']) == used:
        info = calibrate(ctx, st['models'], st['plans'])
        info['inner_mode'] = None if value is None else value[2]
        ctx.biascal_info = info
        st['done'] = True
        st['models'] = {}


register_mode(NAME, plan, value_policy='optional', parse_value=parse_value, prepare=prepare,
              doc='train-only per-gene shift of the generator output-logit bias so the train-target mean generated pseudobulk '
                  'profile matches the observed train mean (2 fixed-point updates, common random numbers); biascal@INNER '
                  'uses mode INNER\'s plan (":" for its "@"), e.g. biascal@snapens, biascal@stagehalf')


def run_biascal(fit, cache, mode, out, device='cpu', force=False):
    """generate.generate_fit with this mode plus the calibration record next to the output."""
    import generate as GEN
    from trainer import Context
    from pathlib import Path
    cfg = json.loads((Path(fit) / 'config.json').read_text())
    ctx = Context(cache, device, tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
    path, stats = GEN.generate_fit(fit, cache, device=device, ctx=ctx, mode=mode, out=out, force=force)
    info = dict(getattr(ctx, 'biascal_info', {}))
    rec = Path(path).with_name(Path(path).stem + '.biascal.json')
    GEN.dump(rec, {'mode': mode, 'fit': str(Path(fit).resolve()), 'generated': str(path), **info})
    return path, info


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='biascal generation with the calibration record')
    p.add_argument('--fit', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--mode', default='biascal')
    p.add_argument('--out', default=None)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-mem-fraction', type=float, default=None)
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    if not a.mode.split('@', 1)[0] == NAME:
        raise ValueError('--mode must be biascal or biascal@INNER')
    torch.set_num_threads(3)
    if a.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    elif a.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    run_biascal(a.fit, a.cache, a.mode, a.out, a.device, a.force)


if __name__ == '__main__':
    sys.modules.setdefault('mode_biascal', sys.modules['__main__'])
    import trainer  # noqa: F401  (sets the released-module paths before generate is imported)
    main()
