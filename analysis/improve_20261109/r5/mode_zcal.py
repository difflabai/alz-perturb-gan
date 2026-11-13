"""Round-5 generation mode ``zcal[@INNER]`` (PROTOCOL_R5 work item 2, Fix A; owner: count-shape task).

A train-only per-gene calibration of the generator's Gamma-Poisson dispersion, baked into the
generator at generation time (no retraining).  For every gene g a multiplicative factor exp(s_g) is
applied to the generator's dispersion theta (theta' = theta * exp(s_g); for ``g.dispz`` /
``g.disp`` fits theta is donor-dependent and the same factor multiplies every donor's theta), chosen
so that the generated NONZERO FRACTION over the fold's TRAIN target donors equals the observed train
nonzero fraction:

    target_g = mean over the fold's train donors d of (fraction of d's cells with count_g > 0)
               (the evaluator's ``real.nonzero`` averaged over the train targets; held-out donors are
               never read: the calibration only touches ``ctx.X`` / ``ctx.by`` / ``ctx.fitlibs``,
               which ``trainer.Context`` loads from the cache's train split);
    F_g(s)   = mean over the train targets t, their M = 32 generated donors and K = 256 cells of the
               EXPECTED nonzero probability 1 - (1 + mu/theta')^(-theta') of the generator's cell
               (mu = library x softmax proportion, exactly the generator's NB mean), computed with
               the fit's common random numbers (the same latent, cell-noise and library draws as the
               default generation streams of every train target, ``generate.generation_streams``);
               mu does not depend on s, so F_g is a smooth, strictly increasing function of s_g.
    Solve    F_g(s_g) = target_g per gene by safeguarded Newton (analytic dF/ds, bracket kept per
             gene, bisection when Newton leaves the bracket), s in [-S_MAX, S_MAX]; genes whose
             target lies outside [F_g(-S_MAX), F_g(S_MAX)] stay at the bound and are counted
             (``n_clipped_low`` / ``n_clipped_high``).  F_g(+inf) is the Poisson limit: a generator
             cannot produce more nonzeros than Poisson at its own mean.
The expected (not the sampled) nonzero fraction is matched, so the calibration has no Monte-Carlo
noise of its own; the sampled train nonzero residual after calibration is recorded (it is the
Bernoulli noise over 204,800 cells per gene, about 1e-3 or less).

Mode text (same grammar as ``biascal``: INNER written with ':' for its own '@')::

    zcal                      dispersion calibration of the default read-out (state 'final')
    zcal@stagehalf            ... of the stagehalf read-out (snapens states, half-shrunk stage)
    zcal@biascal:stagehalf    JOINT: biascal's per-gene output-bias calibration AND the dispersion
                              calibration, both on the stagehalf plan (the spelling of
                              "biascal+zcal@stagehalf": mode names cannot contain '+')

Joint order (``zcal@biascal:INNER``; fixed here, before any held-out score of this mode).  The two
calibrations interact: the bias shift moves mu (and therefore the nonzero fraction), the dispersion
moves the generated pseudobulk only slightly (theta does not change the NB mean).  Alternation,
starting from delta = 0, s = 0, for it = 1..N_ITER (= 3):
    (1) generate the train targets (common random numbers) under (delta, s); biascal Newton step on
        delta exactly as ``mode_biascal.calibrate`` (same O_bar, P_bar, slope clip [0.15, 1], step
        clip +-3);
    (2) re-solve s for the new delta (the exact expected-nonzero solve above, warm-started).
A last generation of the train targets records both residuals (pseudobulk mean-profile RMS as in
biascal, and the sampled nonzero residual).  The procedure ends on a dispersion solve, so the
expected train nonzero fraction is matched exactly for the final delta; the pseudobulk residual is
the one left after three Newton steps (recorded per iteration in ``history``).  biascal's
"keep the best iterate" guard is not used (reverting delta would undo the matched s); the history
shows whether it would have mattered.  Selection history: N_ITER, S_MAX, the order and the absence of
a zero-inflation term were fixed before any held-out score of this mode; one smoke run (control
pbulk, fold 0, seed 0) was scored before the evaluation batch and nothing was changed after it.

No zero-inflation term (decided from the diagnosis, ``shape_diagnose.py``, before any held-out score of
this mode): at matched mean and depth the generated cells have MORE zeros than real cells, not fewer
(excess zeros over Poisson higher than real in every expression quartile and depth decile), a per-gene
NB fitted by maximum likelihood to the real train cells reproduces their nonzero fraction to 0.004
RMS, and the genes the dispersion cannot reach are on the other side (target ABOVE the generator's
Poisson limit: the generator's cell-to-cell spread of mu is too large).  A structural-zero
probability can only add zeros, so it cannot help any of them.

How it is wired (hook e of ``arms.py``, like ``mode_biascal``): ``plan`` stores every target's plan
on the context (INNER's plan, unchanged, with INNER's own random stream, so with s = 0 (and delta = 0)
the output is bitwise the INNER file); ``prepare`` (called once per loaded state) collects the
models and, when the last state of the plan has arrived, runs the calibration once for all states
(one s and one delta shared by every state: the read-out ensemble is calibrated, not each snapshot)
and patches every generator in place: ``G.parameters_nb`` is wrapped on the instance (theta x
exp(s)), the head bias is shifted for the joint mode.  The calibration record is put on
``ctx.mode_receipt_extra`` (``zcal`` and, for the joint mode, ``biascal`` keys), which
``generate.generate_fit`` writes into the mode receipt under ``mode_extra``.

What the hook cannot express (reported, generate.py not edited): the mode registry finds a mode file
only through ``arm_<name>.py`` or a scan of ``arm_*.py`` (mode_biascal is imported by
``arm_nostage.py``); nothing imports this file, so ``generate.py --mode zcal@...`` raises "unknown
generation mode".  Use the CLI below (it imports this module before calling
``generate.generate_fit``), or add ``import mode_zcal`` to an ``arm_*.py`` file.  Mode names cannot
contain '+', hence the nested spelling.

CLI::

    python3 -B r5/mode_zcal.py --fit FIT --cache CACHE --mode zcal@biascal:stagehalf --out OUT.npz \
        --device cuda --gpu-mem-fraction 0.04
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

NAME = 'zcal'
N_ITER = 3                 # joint alternations (biascal step + dispersion solve)
S_MAX = 6.0                # |log theta factor| bound (theta x e^6 = x403: Poisson for practical purposes)
SOLVE_ITERS = 30
SOLVE_TOL = 2e-6           # max |F - target| over genes (expected nonzero fraction)
K_CELLS = 256              # generate.K (cells per generated donor)


# --------------------------------------------------------------------------- mode text
def inner_text(value):
    if value is None:
        return None
    name, sep, rest = str(value).partition(':')
    return name + ('@' + rest if sep else '')


def parse_value(text):
    """INNER with INNER's '@' written as ':'.  Value = (zi = False, joint, inner (name, v, canon) or None)."""
    t = str(text)
    zi = False
    inner = None
    joint = False
    if t:
        spec, v, canon = arms.parse_mode(inner_text(t))
        if spec.name == NAME:
            raise ValueError('zcal cannot wrap zcal')
        if spec.name == 'biascal':
            joint = True
        inner = (spec.name, v, canon)
    return (zi, joint, inner), (inner[2].replace('@', ':', 1) if inner else '')


def _value(value):
    return (False, False, None) if value is None else value


# --------------------------------------------------------------------------- plan / prepare (hook e)
def _inner_plan(ctx, inner, target, states):
    if inner is None:
        return np.repeat(target.default_cond[None], target.M, 0).astype(np.float32), ['final'] * target.M
    name, v, _ = inner
    spec = arms._MODES[name]
    if name == 'biascal':     # biascal's own plan = its inner plan (we run the bias calibration here)
        import mode_biascal as MB
        return MB._inner_plan(ctx, v, target, states)
    return spec.plan(ctx, v, target, arms.mode_rng(ctx.fold, target.dev_index, name), states)


def plan(ctx, value, target, rng, states):
    st = getattr(ctx, '_zcal', None)
    if st is None or target.index == 0:
        st = ctx._zcal = {'plans': {}, 'models': {}, 'done': False}
    _, _, inner = _value(value)
    cond, names = _inner_plan(ctx, inner, target, states)
    names = [str(s) for s in names]
    st['plans'][target.index] = (np.asarray(cond, dtype=np.float32), names)
    return cond, names


def prepare(ctx, value, state_name, G, P):
    st = getattr(ctx, '_zcal', None)
    if st is None or not st['plans']:
        raise ValueError('zcal: prepare called before the plan')
    if st['done']:
        raise ValueError('zcal: calibration already applied for this mode run')
    st['models'][state_name] = (G, P)
    used = {s for _, names in st['plans'].values() for s in names}
    if set(st['models']) == used:
        zi, joint, inner = _value(value)
        info = calibrate(ctx, st['models'], st['plans'], joint=joint)
        info['inner_mode'] = None if inner is None else inner[2]
        extra = dict(getattr(ctx, 'mode_receipt_extra', None) or {})
        extra['zcal'] = {k: v for k, v in info.items() if k != 'biascal'}
        if 'biascal' in info:
            extra['biascal'] = info['biascal']
        ctx.mode_receipt_extra = extra
        ctx.zcal_info = info
        st['done'] = True
        st['models'] = {}


# --------------------------------------------------------------------------- generator patch
def patch_generator(G, device):
    """Wrap G.parameters_nb on the instance: theta -> theta * exp(s).  Returns the state dict {'s': tensor [G]}."""
    if getattr(G, '_zcal_state', None) is not None:
        return G._zcal_state
    n = len(G.log_theta)
    st = {'s': torch.zeros(n, device=device)}
    orig_nb = G.parameters_nb

    def parameters_nb(z, c, k, library, gen=None):
        mu, theta = orig_nb(z, c, k, library, gen)
        return mu, theta * torch.exp(st['s'])

    G.parameters_nb = parameters_nb
    G._zcal_orig_nb = orig_nb
    G._zcal_state = st
    return st


# --------------------------------------------------------------------------- CRN cell iterator
def train_targets(ctx):
    """Train targets exactly as generate.targets lists them first (no held-out file is opened)."""
    libs = ctx.fitlibs.cpu().numpy()
    return [(str(d), False, ctx.raw_c[i], libs[ctx.by[i]]) for i, d in enumerate(ctx.ids)]


@torch.no_grad()
def iterate_chunks(ctx, models, plans, T, only=None):
    """Yield (ti, start, n, G, z, cond, lib, cell_gen, count_gen) per generation chunk, replaying
    generate.generate_arrays' streams exactly (seeding, latent / cell / library draws, chunking).
    The consumer must draw the chunk's cell noise (G.parameters_nb with cell_gen) before the next
    iteration, as generate_arrays does.  ``only``: optional set of target indices to visit (each target
    has its own streams, so skipping a target does not change any other)."""
    import generate as GEN
    import trainer
    device = ctx.device
    dev_order = trainer.development_order()
    saved = {k: (G.training, P.training) for k, (G, P) in models.items()}
    for G, P in models.values():
        G.eval(); P.eval()
    try:
        for ti, (donor, _, _, pool) in enumerate(T):
            if only is not None and ti not in only:
                continue
            s = GEN.generation_streams(ctx.fold, dev_order[donor])
            torch.manual_seed(s['global'])
            lat = torch.Generator(device=device).manual_seed(s['latent'])
            cell = torch.Generator(device=device).manual_seed(s['cell'])
            count = torch.Generator(device=device).manual_seed(s['count'])
            libs = np.random.default_rng(s['library']).choice(pool, size=(GEN.M, GEN.K), replace=True).astype(np.float32)
            cond, names = plans[ti]
            ct = torch.as_tensor(np.ascontiguousarray(cond), device=device)
            runs, a = [], 0
            for m in range(1, GEN.M + 1):
                if m == GEN.M or names[m] != names[a]:
                    runs.append((a, m, names[a])); a = m
            for a, b, name in runs:
                G, P = models[name]
                for start in range(a, b, GEN.CHUNK):
                    n = min(GEN.CHUNK, b - start)
                    z, _ = P.sample(n, generator=lat)
                    lib = torch.as_tensor(libs[start:start + n], device=device)
                    yield ti, start, n, G, z, ct[start:start + n], lib, cell, count
    finally:
        for k, (G, P) in models.items():
            G.train(saved[k][0]); P.train(saved[k][1])


def base_nb(G, z, c, k, lib, cell):
    """(mu, theta) of the UNPATCHED generator (theta broadcastable to mu)."""
    f = getattr(G, '_zcal_orig_nb', None) or G.parameters_nb
    return f(z, c, k, lib, cell)


def expected_nonzero(mu, theta):
    """P(count > 0) under NB(mu, theta) = 1 - (1 + mu/theta)^-theta, and d/d log theta."""
    x = mu / theta
    l1 = torch.log1p(x)
    p0 = torch.exp(-theta * l1)
    d = p0 * theta * (l1 - x / (1 + x))
    return 1 - p0, d


@torch.no_grad()
def expected_nonzero_pass(ctx, models, plans, T, s):
    """F [G] = mean over all cells of the T targets of the expected nonzero probability under theta x exp(s),
    dF/ds [G], and the Poisson limit F(+inf) [G]."""
    G_n = ctx.n_genes
    F = torch.zeros(G_n, dtype=torch.float64, device=ctx.device)
    D = torch.zeros_like(F)
    Fp = torch.zeros_like(F)
    ncell = 0
    es = torch.exp(torch.as_tensor(s, dtype=torch.float32, device=ctx.device))
    for ti, start, n, G, z, c, lib, cell, count in iterate_chunks(ctx, models, plans, T):
        mu, theta = base_nb(G, z, c, K_CELLS, lib, cell)
        th = theta * es
        nz, d = expected_nonzero(mu, th)
        F += nz.sum((0, 1)).double()
        D += d.sum((0, 1)).double()
        Fp += (-torch.expm1(-mu)).sum((0, 1)).double()
        ncell += mu.shape[0] * mu.shape[1]
    return (F / ncell).cpu().numpy(), (D / ncell).cpu().numpy(), (Fp / ncell).cpu().numpy()


def observed_train_nonzero(ctx):
    """[G] mean over train donors of the fraction of the donor's cells with count > 0 (evaluator real.nonzero)."""
    out = []
    for r in ctx.by:
        x = ctx.X[torch.as_tensor(r, device=ctx.X.device)]
        out.append((x > 0).double().mean(0).cpu().numpy())
    return np.stack(out).mean(0)


def solve_dispersion(ctx, models, plans, T, target, s0=None, log=None):
    """Per-gene safeguarded Newton for F_g(s_g) = target_g (module docstring).  Returns s, info."""
    Gn = len(target)
    lo = np.full(Gn, -S_MAX); hi = np.full(Gn, S_MAX)
    s = np.zeros(Gn) if s0 is None else np.clip(np.asarray(s0, dtype=np.float64), -S_MAX, S_MAX)
    # reachable range at the bounds
    F_lo, _, _ = expected_nonzero_pass(ctx, models, plans, T, lo)
    F_hi, _, F_poi = expected_nonzero_pass(ctx, models, plans, T, hi)
    clip_hi = target >= F_hi
    clip_lo = target <= F_lo
    hist = []
    r = None
    for it in range(SOLVE_ITERS):
        F, D, _ = expected_nonzero_pass(ctx, models, plans, T, s)
        r = F - target
        free = ~(clip_hi | clip_lo)
        amax = float(np.abs(r[free]).max()) if free.any() else 0.0
        hist.append({'iteration': it, 'max_abs_resid_free': amax, 'rms_resid_free': float(np.sqrt(np.mean(r[free] ** 2))) if free.any() else 0.0})
        if log:
            log(f'[zcal] solve {it}: max |F - target| over free genes {amax:.2e}')
        if amax < SOLVE_TOL:
            break
        lo = np.where(free & (r < 0), np.maximum(lo, s), lo)
        hi = np.where(free & (r > 0), np.minimum(hi, s), hi)
        newton = s - r / np.maximum(D, 1e-12)
        ok = (newton > lo) & (newton < hi) & np.isfinite(newton)
        s_new = np.where(ok, newton, 0.5 * (lo + hi))
        s = np.where(free, s_new, s)
    s = np.where(clip_hi, S_MAX, np.where(clip_lo, -S_MAX, s))
    info = {'n_genes': int(Gn), 'n_clipped_high': int(clip_hi.sum()), 'n_clipped_low': int(clip_lo.sum()),
            'solve_history': hist, 'final_max_abs_resid_free': hist[-1]['max_abs_resid_free'] if hist else None,
            'F_at_minus_SMAX_mean': float(F_lo.mean()), 'F_at_plus_SMAX_mean': float(F_hi.mean()),
            'poisson_limit_mean': float(F_poi.mean())}
    return s, info, F_lo, F_hi


def _set_s(states, s):
    t = torch.as_tensor(s, dtype=torch.float32)
    for st in states:
        st['s'].copy_(t.to(st['s'].device))


def calibrate(ctx, models, plans_by_index, *, joint=False, n_iter=None, log=print):
    """Train-only calibration; patches every model in place.  Returns the record (module docstring)."""
    import generate as GEN
    import mode_biascal as MB
    n_iter = N_ITER if n_iter is None else int(n_iter)
    Tt = train_targets(ctx)
    plans = [plans_by_index[i] for i in range(len(Tt))]
    used = sorted({s for _, names in plans for s in names})
    if set(used) != set(models):
        raise ValueError(f'zcal: models {sorted(models)} differ from the states the plan uses {used}')
    states = [patch_generator(G, ctx.device) for G, _ in models.values()]
    target = observed_train_nonzero(ctx)
    info = {'joint_biascal': bool(joint), 'zero_inflation': False, 'states': used, 'n_train_targets': len(Tt),
            'S_MAX': S_MAX, 'target_nonzero_mean': float(target.mean())}
    hist = []

    def record(tag, arr, delta=None, s=None):
        nz = arr['nonzero'].astype(np.float64).mean((0, 1))
        e = {'step': tag, 'nonzero_resid_rms': float(np.sqrt(np.mean((nz - target) ** 2))),
             'nonzero_resid_mean': float((nz - target).mean())}
        if joint:
            P_bar = MB.predicted_profiles(arr['pb_counts']).mean(0)
            res = O_bar - P_bar
            e.update(pb_resid_rms=float(np.sqrt(np.mean(res ** 2))), pb_resid_mean=float(res.mean()))
        if delta is not None:
            e['delta_rms'] = float(np.sqrt(np.mean(delta ** 2)))
        if s is not None:
            e['s_rms'] = float(np.sqrt(np.mean(s ** 2))); e['s_mean'] = float(np.mean(s))
        hist.append(e)
        log(f'[zcal] {tag}: ' + ', '.join(f'{k} {v:.4f}' for k, v in e.items() if isinstance(v, float)))
        return e

    s = np.zeros(ctx.n_genes)
    solve_infos = []
    if joint:
        O_bar = MB.observed_train_profiles(ctx).mean(0)
        biases = {k: MB.head(G) for k, (G, _) in models.items()}
        delta = np.zeros(ctx.n_genes)
        for it in range(n_iter):
            arr, _ = GEN.generate_arrays(ctx, models, plans, T=Tt)
            record(f'iteration {it} (before bias step)', arr, delta, s)
            pb = arr['pb_counts'].astype(np.float64)
            P_bar = MB.predicted_profiles(arr['pb_counts']).mean(0)
            resid = O_bar - P_bar
            x = pb * (1e4 / np.maximum(pb.sum(-1, keepdims=True), 1))
            slope = np.clip((x / (1 + x)).mean((0, 1)), MB.SLOPE_MIN, MB.SLOPE_MAX)
            step_np = np.clip(resid / slope, -MB.STEP_MAX, MB.STEP_MAX)
            delta = delta + step_np
            step = torch.as_tensor(step_np, dtype=torch.float32)
            with torch.no_grad():
                for b in biases.values():
                    b.add_(step.to(b.device))
            s, si, F_lo, F_hi = solve_dispersion(ctx, models, plans, Tt, target, s0=s, log=log)
            _set_s(states, s)
            solve_infos.append({k: v for k, v in si.items() if k != 'solve_history'} | {'n_solve_iters': len(si['solve_history'])})
        info['biascal'] = {'n_iter': n_iter, 'states': used, 'n_train_targets': len(Tt), 'joint_with': 'zcal',
                           'delta_rms': float(np.sqrt(np.mean(delta ** 2))), 'delta_mean': float(delta.mean()),
                           'delta_absmax': float(np.abs(delta).max()), 'delta': delta.astype(np.float32).tolist()}
    else:
        s, si, F_lo, F_hi = solve_dispersion(ctx, models, plans, Tt, target, log=log)
        _set_s(states, s)
        solve_infos.append({k: v for k, v in si.items() if k != 'solve_history'} | {'n_solve_iters': len(si['solve_history'])})
    arr, _ = GEN.generate_arrays(ctx, models, plans, T=Tt)
    final = record('final (after last dispersion solve)', arr, None if not joint else delta, s)
    info.update(history=hist, solves=solve_infos, final=final,
                s_rms=float(np.sqrt(np.mean(s ** 2))), s_mean=float(np.mean(s)), s_min=float(np.min(s)), s_max=float(np.max(s)),
                s=np.asarray(s, dtype=np.float32).tolist(), order=('joint: [generate, biascal Newton step, dispersion solve] x '
                                                                    f'{n_iter}, final generation for the record' if joint else
                                                                    'dispersion solve once, final generation for the record'))
    return info


register_mode(NAME, plan, value_policy='optional', parse_value=parse_value, prepare=prepare,
              doc='train-only per-gene dispersion calibration (theta x exp(s_g)) so the expected generated nonzero fraction over the '
                  'train targets equals the observed train nonzero fraction; zcal@INNER uses INNER\'s plan (":" for its "@"); '
                  'zcal@biascal:INNER = joint with biascal (alternating, ends on the dispersion solve)')


def run_zcal(fit, cache, mode, out, device='cpu', force=False):
    import generate as GEN
    from trainer import Context
    from pathlib import Path
    cfg = json.loads((Path(fit) / 'config.json').read_text())
    ctx = Context(cache, device, tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
    path, stats = GEN.generate_fit(fit, cache, device=device, ctx=ctx, mode=mode, out=out, force=force)
    return path, getattr(ctx, 'zcal_info', {})


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='zcal generation (count-shape calibration) with the calibration record in the receipt')
    p.add_argument('--fit', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--mode', default='zcal')
    p.add_argument('--out', default=None)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-mem-fraction', type=float, default=None)
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    if a.mode.split('@', 1)[0] != NAME:
        raise ValueError('--mode must be zcal or zcal@...')
    torch.set_num_threads(3)
    if a.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    elif a.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    run_zcal(a.fit, a.cache, a.mode, a.out, a.device, a.force)


if __name__ == '__main__':
    sys.modules.setdefault('mode_zcal', sys.modules['__main__'])
    import trainer  # noqa: F401  (sets the released-module paths before generate is imported)
    main()
