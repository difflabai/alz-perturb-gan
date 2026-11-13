"""gm_diagnose.py - F6 diagnosis of the gene-mean critic block (round-1 and round-2 fits, CPU only).

Questions (r2/PROTOCOL_R2.md, F6):

Q1  Does the gene-mean block dominate what the critic sees?
Q2  Does the critic gap plateau, grow or oscillate over training, and is the
    seed-to-seed disagreement of generated per-gene means related to it?
Q3  Is the generator still moving at the end of training (snapshot drift),
    and does averaging late states (``snapens``) help?

Subcommands (all read-only on fits; outputs are small JSON files)::

    python3 -B r2/gm_diagnose.py curves  [--roots R ...] [--arms A ...] [--panels P ...] --out F.json
    python3 -B r2/gm_diagnose.py critic  [--roots ...] [--arms ...] [--panels ...] [--states final,captured,snap] --out F.json
    python3 -B r2/gm_diagnose.py seeds   [--roots ...] [--arms ...] [--panels ...] [--curves F.json] [--critic F.json] --out F.json
    python3 -B r2/gm_diagnose.py drift   [--roots ...] [--arms ...] [--panels ...] [--capture-dir D] --out F.json
    python3 -B r2/gm_diagnose.py all     ... --out-dir D       (curves, critic, seeds, drift -> D/gm_{name}.json)
    python3 -B r2/gm_diagnose.py capture --fits FIT ... --updates 14000 16000 18000
        (round-1 helper: copy generator/critic/prior state dicts out of a RUNNING
        round-1 fit's recovery.pt each time its checkpoint receipt reaches one
        of the updates; read-only on the fit; files go to
        runs/improve_20261109/smoke_r2/gm_diag/capture/{arm}_{panel}_fold{k}_seed{s}_u{u}.pt)

Defaults: roots = runs/improve_20261109/fits and fits_r2; arms = every arm
whose flags are ``base``, contain ``genemean`` / ``genemean@s``, or a flag
starting with ``gm``; panels hvg2000 and wide.  Only fits with config.json
are read; ``critic``/``seeds`` need final.pt / generated.npz (incomplete fits
are used by ``curves`` only and labelled ``complete: false``).

Definitions
-----------
Bags (``critic``): 256 real train bags drawn like training (uniform train
donor, 128 cells with replacement) from numpy SeedSequence([20261109,
crc32('gm_diagnose'), fold]); one fake bag per real bag with that donor's
condition and the real bag's cell libraries, latent from the prior, torch
seeds from the same SeedSequence (fixed per fold, identical for every arm,
seed and state).  Features exactly as the trainer's ``Features.real/fake``.
Blocks: ``released150`` (columns 0-149), ``genemean`` (round-1 block; scale
as stored in the checkpoint), each plug-in block (``flag/name``), ``cond7``
(the 7 condition inputs of the first critic layer).

* ``second_moment_share``: E||x_block||^2 / sum over feature blocks (real, fake).
* ``variance_share``: sum over the block's columns of the across-bag variance /
  total (real, fake).
* ``weight_fro``: Frobenius norm of the block's columns of D.phi[0].weight;
  ``weight_x_sd_share``: share of sum_j ||W[:,j]||^2 sd_j^2 (sd over real bags).
* ``preact_var_share``: share of the across-bag variance of the first-layer
  pre-activation (summed over the 128 units) contributed by the block alone
  (x_block W_block^T), real and fake bags.
* ``input_grad_x_sd_share``: share of sum_j (mean_bags |dD/dx_j| sd_j)^2.
* ``gap``: mean D(real) - mean D(fake) over the 256 bag pairs (low-noise
  version of the learning-curve critic_gap, which is one 32-bag batch).
* ``gap_ig_share[block]``: integrated-gradient attribution of ``gap`` to each
  block: mean over bag pairs of the integral of dD/dx_block along the
  straight line from the fake bag to its paired real bag (16-point midpoint
  rule) dotted with (real - fake)_block, divided by ``gap``
  (``gap_ig_completeness`` = sum over blocks / gap, ~1 when the rule is
  accurate).  This is the clean "which block carries the gap" measure.
* ``gap_swap[block]``: fraction of ``gap`` removed when the fake bags' block is
  replaced by the paired real bag's block.  Caveat: the released features
  contain the per-PC bag means, a linear function of the genemean block, so a
  swap creates bags whose two blocks are mutually inconsistent, which the
  critic may score as very fake (negative values); read ``gap_ig_share``.
* ``push_vs_error``: Pearson over genes between the critic's mean gradient on
  the genemean block at the fake bags (the direction the generator is pushed,
  in standardised log units) and minus the fake-minus-real block mean error
  (``bags``) or minus the generated-minus-real per-gene mean_log error of the
  fit's generated.npz train targets divided by the gene SD (``generated``).
  Positive = the critic pushes each gene toward its real mean.
* ``geometry[block]`` (no critic involved; the 256 real and 256 fake bags of
  that block): ``mean_shift_sq`` = ||mean fake - mean real||^2,
  ``trace_real``/``trace_fake`` = summed across-bag variances and their
  ratio, ``mean_shift_sq_over_trace_real``; ``auc_mean_difference_direction``
  = AUC of real vs fake projected on the mean-difference direction estimated
  on the other half of the bags (separation by a pure mean shift).  Subspace
  metrics, cross-fitted by DONOR halves (top-20 directions of the real bags
  of even- / odd-numbered train donors, evaluated on the other half's real
  bags and on all fake bags, averaged): ``real_var_in_other_half_topk``,
  ``fake_var_in_real_half_topk``, ``mean_shift_in_real_half_topk`` and
  ``auc_residual_outside_real_half_topk`` (AUC of the squared distance from
  that subspace, fake vs held-out-half real bags: separation because fake
  bags leave the directions in which real bags vary).
* ``decomposition`` (every state, also snapshots without a critic; fits
  without a gene-mean block are viewed through a genemean Context of the same
  cache): per-bag mean over 128 cells of the standardised log expression
  (genemean features without the block scale).  Real: 8 bags of each train
  donor; fake: one prior latent per train donor (that donor's condition), 8
  bags per latent with the real bags' libraries.  ``within_trace`` = trace of
  the within-group (bag sampling) covariance, ``between_trace`` = trace of the
  between-donor / between-latent covariance (minus within/8); ratios fake /
  real.  ``real_heldout_donor_means_in_other_half_subspace`` /
  ``fake_latent_means_in_real_half_subspace``: share of the centred
  per-group mean variance inside the top-10 principal directions of the
  OTHER half of the real train donors (cross-fitted over even/odd halves):
  how well donor-level variation of generated gene means follows the
  directions in which real donors vary.

Seeds (``seeds``; generated.npz of complete fits): per fit, per-gene
mean_log averaged over the train targets x M generated donors (``m_f``) and
over held-out targets; real per-donor mean_log from the fold cache (all
cells; train donors from train/counts, held-out from heldout/counts of the
same development fold).  Per (arm, panel, fold): pairwise seed RMS
differences over genes, the Monte Carlo floor of one fit's per-gene mean
(from the two halves m even/odd of M: var(half difference)/4, RMS over
genes) and the ratio; per fit the leave-one-seed-out deviation from the
other seeds' mean and its relation to the fit's final critic gap and late gap
oscillation (Spearman within arm over all fits, fold/panel-centred); errors
(train/held-out RMSE of per-gene mean_log vs real donor mean) of each seed and
of the seed ensemble; ``push_vs_deviation`` (Pearson over genes between the
critic push and minus the fit's deviation from the other seeds, in
standardised units; needs ``--critic``).  ``generated_snapens.npz`` (round-2)
is evaluated as its own unit ``arm#snapens``.

Curves (``curves``; learning_curve.jsonl rows at steps 250k and 250k+1):
2000-update window means of critic_gap, D_adversarial, G_adversarial;
``final_gap`` = mean over the last 2000 updates; OLS ``slope_per_1k`` (and
SE) over the second half; ``noise_sd`` = SD(gap[250k+1] - gap[250k])/sqrt(2)
(per-batch noise; adjacent updates); ``slow_sd`` = sqrt(max(0, var of the
detrended 250-update pair means over the second half - noise_sd^2/2)) (the
part of the fluctuation that is not batch noise: oscillation / drift on a
>=250-update scale) and the lag-1 autocorrelation of those detrended pair
means.  Label: ``grows``/``declines`` when |slope| > 2 SE and the fitted
change over the second half exceeds 1 logit, else ``plateau``; plus
``oscillates`` when slow_sd > 1 logit.

Drift (``drift``): for fits with several model states (round-2 ``snap_u.pt``
+ final; round-1 states captured from recovery.pt by ``capture`` + final),
reduced generation with the round-2 generation code and common random
numbers (generate.generate_arrays; 8 train targets, 2 per stage where
available, M = 32 x K = 256): per state the per-gene mean_log m_u; drift
RMS(m_u - m_final) over genes, relative to the Monte Carlo floor (as above)
and to the seed-to-seed RMS difference of the final states of the same
(arm, panel, fold) on the same targets; the error of each state and of the
average of all states vs the targets' real mean_log (train targets only; a
preview of ``snapens``).  For round-1 captured states that include the
critic the low-noise ``gap`` is also computed per state.

Tolerance: round-1 fits were trained on CUDA, and the genemean scale
recomputed on CPU differs in the 9th digit (r2/generate.py ``check_final``
refuses this exactly; reported to the integrator).  This script accepts a
relative difference < 1e-6 in float arm-state entries and then uses the
checkpoint's own value.

Threads: 3.  Device: CPU.  Nothing is written into any fit folder.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import json
import math
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path('/ml2/alz/runs/improve_20261109')
DIAG = RUNS / 'smoke_r2' / 'gm_diag'
DEFAULT_ROOTS = [RUNS / 'fits', RUNS / 'fits_r2']
CACHES = RUNS / 'caches'
N_BAGS = 256
BAG = 128
DRIFT_TARGETS_PER_STAGE = 2
REL_TOL = 1e-6
IG_STEPS = 16
GEOM_K = 20
DECOMP_REPS = 8
CF_K = 10


def log(*a):
    print(time.strftime('%H:%M:%S'), '[gm_diagnose]', *a, flush=True)


# =========================================================================== capture (round-1 helper; no study imports)
def capture(fits, updates, poll, until):
    """Copy G/D/P state dicts out of round-1 recovery.pt at the given updates (read-only on the fit)."""
    import torch
    torch.set_num_threads(3)
    cap = DIAG / 'capture'
    cap.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + until
    while time.time() < t_end:
        pending = 0
        for fd in fits:
            fd = Path(fd)
            tag = '_'.join(fd.parts[-4:])
            for u in updates:
                out = cap / f'{tag}_u{u}.pt'
                if out.exists():
                    continue
                pending += 1
                try:
                    rc = json.loads((fd / 'checkpoint_receipt.json').read_text())
                except (OSError, ValueError):
                    continue
                if rc.get('status') != 'incomplete' or rc.get('completed_updates') != u:
                    continue
                try:
                    q = torch.load(fd / 'recovery.pt', map_location='cpu', weights_only=False)
                except Exception as exc:  # replaced mid-read: retry at the next poll
                    print('retry', fd, repr(exc), flush=True)
                    continue
                if q['completed_updates'] != u:
                    continue
                keep = {k: q[k] for k in ('arm', 'flags', 'panel', 'fold', 'seed', 'stream_root', 'completed_updates',
                                          'fixed_endpoint', 'cache_manifest_SHA256')}
                keep['models'] = q['policy']['models']
                keep['arm_state'] = q['arm_state']
                keep['source'] = str(fd / 'recovery.pt')
                tmp = out.with_suffix('.tmp')
                torch.save(keep, tmp)
                tmp.replace(out)
                log('captured', out.name)
                del q
        if not pending:
            break
        time.sleep(poll)


# =========================================================================== study imports (lazy)
_M = {}


def mods():
    if not _M:
        if str(HERE) in sys.path:
            sys.path.remove(str(HERE))
        sys.path.insert(0, str(HERE))
        import numpy as np
        import torch
        torch.set_num_threads(3)
        import trainer
        import arms
        import generate
        if Path(trainer.__file__).resolve().parent != HERE:
            raise ImportError('gm_diagnose must use the r2 trainer')
        _M.update(np=np, torch=torch, trainer=trainer, arms=arms, generate=generate)
    return _M


def jsonable(x):
    np = mods()['np']
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    log('wrote', path)


# =========================================================================== fit discovery
def wanted_arm(arm, arms_filter):
    if arms_filter:
        return arm in arms_filter
    if arm == 'base':
        return True
    names = [f.split('@', 1)[0] for f in arm.split(',')]
    return any(n == 'genemean' or n.startswith('gm') for n in names)


def find_fits(roots, arms_filter=None, panels=('hvg2000', 'wide')):
    """[dict(fit, arm, panel, fold, seed, root, round, complete, config)] sorted."""
    out = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for cfgp in sorted(root.glob('*/*/fold*/seed*/config.json')):
            fd = cfgp.parent
            arm, panel = fd.parts[-4], fd.parts[-3]
            if panel not in panels or not wanted_arm(arm, arms_filter):
                continue
            cfg = json.loads(cfgp.read_text())
            rnd = int(cfg.get('round', 1))
            out.append({'fit': fd, 'arm': arm, 'label': arm if rnd == 1 else arm + '~r2', 'panel': panel, 'fold': int(cfg['fold']), 'seed': int(cfg['seed']),
                        'root': str(root), 'round': int(cfg.get('round', 1)), 'config': cfg,
                        'complete': (fd / 'final.pt').exists() and (fd / 'generated.npz').exists()})
    out.sort(key=lambda f: (f['root'], f['arm'], f['panel'], f['fold'], f['seed']))
    return out


def key(f):
    return f"{f['label']}|{f['panel']}|{f['fold']}|{f['seed']}"


# =========================================================================== curves (Q2)
def curve_stats(path, updates):
    np = mods()['np']
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    step = np.array([r['step'] for r in rows])
    gap = np.array([r['critic_gap'] for r in rows], float)
    dadv = np.array([r['D_adversarial'] for r in rows], float)
    gadv = np.array([r['G_adversarial'] for r in rows], float)
    last = int(step.max()) if len(step) else 0
    win = max(updates // 10, 1)
    windows = []
    for lo in range(0, updates, win):
        m = (step > lo) & (step <= lo + win)
        if m.any():
            windows.append({'from': lo, 'to': lo + win, 'n': int(m.sum()), 'gap': float(gap[m].mean()),
                            'D_adversarial': float(dadv[m].mean()), 'G_adversarial': float(gadv[m].mean())})
    # pairs (250k, 250k+1)
    by = {int(s): g for s, g in zip(step, gap)}
    pair_steps = sorted(s for s in by if s % 250 == 0 and s + 1 in by and s > updates * 0.1)
    d = np.array([by[s + 1] - by[s] for s in pair_steps])
    noise_sd = float(d.std(ddof=1) / math.sqrt(2)) if len(d) > 2 else None
    res = {'rows': len(rows), 'last_step': last, 'updates': updates, 'windows': windows, 'noise_sd': noise_sd,
           'final_gap': float(gap[step > updates - win].mean()) if (step > updates - win).any() else None,
           'final_D_adversarial': float(dadv[step > updates - win].mean()) if (step > updates - win).any() else None,
           'final_G_adversarial': float(gadv[step > updates - win].mean()) if (step > updates - win).any() else None,
           'max_window_gap': max(w['gap'] for w in windows) if windows else None}
    half = [s for s in pair_steps if s > updates / 2]
    if len(half) >= 6:
        x = np.array(half, float) / 1000.0
        y = np.array([(by[s] + by[s + 1]) / 2 for s in half])
        A = np.stack([np.ones_like(x), x], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        dof = max(len(y) - 2, 1)
        s2 = float(r @ r / dof)
        se = math.sqrt(s2 / float(((x - x.mean()) ** 2).sum()))
        slow_var = float(r.var(ddof=2)) - (noise_sd ** 2 / 2 if noise_sd else 0.0)
        ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 else None
        change = float(coef[1] * (x.max() - x.min()))
        label = 'plateau'
        if abs(coef[1]) > 2 * se and abs(change) > 1.0:
            label = 'grows' if coef[1] > 0 else 'declines'
        slow_sd = math.sqrt(max(slow_var, 0.0))
        res.update(second_half={'from': int(min(half)), 'to': int(max(half)) + 1, 'n_pairs': len(half),
                                'mean_gap': float(y.mean()), 'slope_per_1k': float(coef[1]), 'slope_se': se,
                                'fitted_change': change, 'resid_sd_pairmeans': math.sqrt(max(s2, 0.0)),
                                'slow_sd': slow_sd, 'lag1_autocorr_detrended': ac1},
                   label=label + (',oscillates' if slow_sd > 1.0 else ''))
    return res


def cmd_curves(args):
    np = mods()['np']
    fits = find_fits(args.roots, args.arms, args.panels)
    per = {}
    for f in fits:
        p = f['fit'] / 'learning_curve.jsonl'
        if not p.exists():
            continue
        st = curve_stats(p, int(f['config']['updates']))
        st.update(complete=f['complete'], round=f['round'], root=f['root'])
        per[key(f)] = st
    agg = {}
    for k, st in per.items():
        arm, panel = k.split('|')[:2]
        if st['last_step'] < st['updates']:
            continue
        a = agg.setdefault(f'{arm}|{panel}', {'n': 0, 'final_gap': [], 'noise_sd': [], 'slow_sd': [], 'slope_per_1k': [],
                                               'lag1': [], 'labels': {}})
        a['n'] += 1
        a['final_gap'].append(st['final_gap'])
        a['noise_sd'].append(st['noise_sd'])
        if 'second_half' in st:
            a['slow_sd'].append(st['second_half']['slow_sd'])
            a['slope_per_1k'].append(st['second_half']['slope_per_1k'])
            a['lag1'].append(st['second_half']['lag1_autocorr_detrended'])
            a['labels'][st['label']] = a['labels'].get(st['label'], 0) + 1
    summary = {}
    for k, a in agg.items():
        for m in ('final_gap', 'noise_sd', 'slow_sd', 'slope_per_1k', 'lag1'):
            a[m] = [x for x in a[m] if x is not None]
        summary[k] = {'n_complete': a['n'], 'labels': a['labels'],
                      **{f'{m}_mean': float(np.mean(a[m])) for m in ('final_gap', 'noise_sd', 'slow_sd', 'slope_per_1k', 'lag1') if a[m]},
                      **{f'{m}_range': [float(np.min(a[m])), float(np.max(a[m]))] for m in ('final_gap', 'slow_sd', 'slope_per_1k') if a[m]}}
        # gap trajectory (window means averaged over complete fits)
        ws = [per[x]['windows'] for x in per if x.startswith(k + '|') and per[x]['last_step'] >= per[x]['updates']]
        if ws:
            n = min(len(w) for w in ws)
            summary[k]['gap_by_window'] = [float(np.mean([w[i]['gap'] for w in ws])) for i in range(n)]
    out = {'per_fit': per, 'summary': summary}
    if args.out:
        write(args.out, out)
    return out


# =========================================================================== model loading
def _close(a, b, path='arm_state'):
    """_same with relative tolerance REL_TOL for floats/float tensors (CUDA-trained vs CPU-recomputed)."""
    np, torch = mods()['np'], mods()['torch']
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f'{path}: keys differ {sorted(set(a) ^ set(b))}')
        for k in a:
            _close(a[k], b[k], f'{path}.{k}')
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f'{path}: lengths differ')
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f'{path}[{i}]')
        return
    if isinstance(a, (torch.Tensor, np.ndarray)) or isinstance(b, (torch.Tensor, np.ndarray)):
        x = torch.as_tensor(np.asarray(a) if isinstance(a, np.ndarray) else a).cpu().double()
        y = torch.as_tensor(np.asarray(b) if isinstance(b, np.ndarray) else b).cpu().double()
        if x.shape != y.shape or not torch.allclose(x, y, rtol=REL_TOL, atol=1e-9):
            raise ValueError(f'{path}: tensors differ')
        return
    if isinstance(a, float) or isinstance(b, float):
        if not math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-12):
            raise ValueError(f'{path}: {a} vs {b}')
        return
    if a != b:
        raise ValueError(f'{path}: {a!r} vs {b!r}')


def adopt_arm_state(ctx, saved):
    """Check saved arm state against the cache recomputation (tolerant) and use the saved genemean scale."""
    _close(saved, ctx.arm_state())
    fs = saved.get('features', {})
    if 'genemean_scale' in fs:
        ctx.features.scale = float(fs['genemean_scale'])


class FitModels:
    """Context + model states of one fit: 'final', 'snap_u' (r2), 'cap_u' (round-1 captured)."""

    def __init__(self, f, capture_dir=None, ctx=None):
        M = mods()
        trainer = M['trainer']
        self.f = f
        cfg = f['config']
        flags = tuple(cfg['flags'])
        self.cfg = cfg
        if ctx is None or ctx.flags != flags or str(ctx.cache) != str(CACHES / f['panel'] / f"fold{f['fold']}"):
            ctx = trainer.Context(CACHES / f['panel'] / f"fold{f['fold']}", 'cpu', flags,
                                  allow_nonprotocol_split=not cfg.get('protocol_split', True))
        self.ctx = ctx
        self.paths = {}
        fd = f['fit']
        if (fd / 'final.pt').exists():
            self.paths['final'] = fd / 'final.pt'
        for u in cfg.get('snapshot_updates', []) if f['round'] == 2 else []:
            if (fd / f'snap_{u}.pt').exists():
                self.paths[f'snap_{u}'] = fd / f'snap_{u}.pt'
        if capture_dir is not None and f['round'] != 2:
            tag = f"{f['arm']}_{f['panel']}_fold{f['fold']}_seed{f['seed']}"
            for p in sorted(Path(capture_dir).glob(tag + '_u*.pt')):
                self.paths[f"cap_{int(p.stem.rsplit('_u', 1)[1])}"] = p

    def states(self):
        def upd(n):
            return int(self.cfg['updates']) if n == 'final' else int(n.split('_', 1)[1])
        return sorted(self.paths, key=upd)

    def update_of(self, name):
        return int(self.cfg['updates']) if name == 'final' else int(name.split('_', 1)[1])

    def load(self, name, critic=True):
        """-> (G, D or None, P) on CPU, eval mode."""
        M = mods()
        torch, trainer = M['torch'], M['trainer']
        q = torch.load(self.paths[name], map_location='cpu', weights_only=False)
        f = self.f
        if q.get('seed') != f['seed'] or q.get('fold') != f['fold'] or q.get('panel') != f['panel'] or list(q.get('flags')) != list(self.ctx.flags):
            raise ValueError(f'{self.paths[name]} does not belong to {f["fit"]}')
        if q.get('cache_manifest_SHA256') != trainer.sha(self.ctx.cache / 'manifest.json'):
            raise ValueError('cache differs from checkpoint')
        if name == 'final' and (q.get('status') != 'completed' or q.get('completed_updates') != int(self.cfg['updates'])):
            raise ValueError('final.pt incomplete')
        if name != 'final' and q.get('completed_updates') != self.update_of(name):
            raise ValueError(f'{name}: completed_updates {q.get("completed_updates")}')
        if 'arm_state' in q:
            adopt_arm_state(self.ctx, q['arm_state'])
        if name == 'final' or name.startswith('cap_'):
            models = q['policy']['models'] if name == 'final' else q['models']
            gs, ds, ps = models['generator'], models.get('critic'), models['prior']
        else:
            gs, ds, ps = q['generator'], None, q['prior']
        b, *_ = trainer.build(self.ctx, f['seed'], stream_root=q.get('stream_root', self.cfg['stream_root']))
        G, D, P = b[1], b[2], b[3]
        del b
        G.load_state_dict(gs, strict=True)
        P.load_state_dict(ps, strict=True)
        if ds is not None and critic:
            D.load_state_dict(ds, strict=True)
            D.eval()
        else:
            D = None
        G.eval(); P.eval()
        return G, D, P


# =========================================================================== critic (Q1)
def bag_rng(fold):
    np = mods()['np']
    ss = np.random.SeedSequence([20261109, zlib.crc32(b'gm_diagnose'), int(fold)])
    a, b = ss.spawn(2)
    return np.random.default_rng(a), int(b.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16)


def make_bags(ctx, G, P, n=N_BAGS):
    """Real and paired fake critic inputs [n, F], conditions [n,7] (fixed per fold)."""
    M = mods()
    np, torch = M['np'], M['torch']
    from pg_model import critic_features
    rng, tseed = bag_rng(ctx.fold)
    ds = rng.integers(len(ctx.ids), size=n)
    idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], BAG, replace=True) for i in ds]))
    feats = ctx.features
    with torch.no_grad():
        real = feats.real(idx, ds)
        c = ctx.conditions[torch.as_tensor(ds)]
        lib = ctx.X[idx].sum(-1)
        torch.manual_seed(tseed)
        lat = torch.Generator().manual_seed(tseed + 1)
        cell = torch.Generator().manual_seed(tseed + 2)
        cnt = torch.Generator().manual_seed(tseed + 3)
        fakes = []
        for st in range(0, n, 32):
            z, _ = P.sample(min(32, n - st), generator=lat)
            rate = G.rates(z, c[st:st + 32], BAG, lib[st:st + 32], cell)
            counts = torch.poisson(rate, generator=cnt)
            fakes.append(feats.fake(critic_features(counts, rate, ctx.mean, ctx.sd), rate, counts))
        fake = torch.cat(fakes)
    return real.double(), fake.double(), c.double(), ds


def blocks_of(ctx):
    feats = ctx.features
    out = {'released150': (0, 150)}
    if feats.kind != 'base':
        out[feats.kind] = (150, feats.n_round1)
    for a, b, lo, hi in feats.blocks:
        out[f'{a.flag}/{b.name}'] = (lo, hi)
    return out


def _auc(pos, neg):
    np = mods()['np']
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def block_geometry(xr, xf, ds, k=GEOM_K):
    """How real and fake bags differ inside one block (scale-free; module docstring, 'geometry').

    Subspace metrics are cross-fitted by DONOR halves (real bags of even- / odd-numbered train donors):
    the top-k directions of one half's real bags are evaluated on the other half's real bags and on
    all fake bags, then the two directions are averaged (an in-sample subspace would make the real
    bags look artificially low-dimensional)."""
    np = mods()['np']
    xr, xf = xr.numpy().astype(np.float64), xf.numpy().astype(np.float64)
    ds = np.asarray(ds)
    mr, mf = xr.mean(0), xf.mean(0)
    cr, cf = xr - mr, xf - mf
    tr_r, tr_f = float((cr ** 2).sum() / (len(xr) - 1)), float((cf ** 2).sum() / (len(xf) - 1))
    shift = mf - mr
    out = {'mean_shift_sq': float(shift @ shift), 'trace_real': tr_r, 'trace_fake': tr_f, 'trace_ratio_fake_real': tr_f / tr_r,
           'mean_shift_sq_over_trace_real': float(shift @ shift) / tr_r}
    aucs = []
    for a, b in ((0, 1), (1, 0)):          # mean-difference direction from one half of the bags, scored on the other
        w = xf[a::2].mean(0) - xr[a::2].mean(0)
        aucs.append(_auc(xf[b::2] @ w, xr[b::2] @ w))
    out['auc_mean_difference_direction'] = float(np.mean(aucs))
    acc = {'real_var_in_other_half_topk': [], 'fake_var_in_real_half_topk': [], 'mean_shift_in_real_half_topk': [],
           'auc_residual_outside_real_half_topk': []}
    for half in (0, 1):
        A, B = (ds % 2) == half, (ds % 2) != half
        if A.sum() < 3 or B.sum() < 3:
            continue
        mA = xr[A].mean(0)
        kk = min(k, int(A.sum()) - 1, xr.shape[1])
        V = np.linalg.svd(xr[A] - mA, full_matrices=False)[2][:kk].T
        rb, fb = xr[B] - xr[B].mean(0), cf
        acc['real_var_in_other_half_topk'].append(float(((rb @ V) ** 2).sum() / (rb ** 2).sum()))
        acc['fake_var_in_real_half_topk'].append(float(((fb @ V) ** 2).sum() / (fb ** 2).sum()))
        acc['mean_shift_in_real_half_topk'].append(float(((shift @ V) ** 2).sum() / max(shift @ shift, 1e-300)))
        r_res = xr[B] - mA
        f_res = xf - mA
        res_r = ((r_res - (r_res @ V) @ V.T) ** 2).sum(1)
        res_f = ((f_res - (f_res @ V) @ V.T) ** 2).sum(1)
        acc['auc_residual_outside_real_half_topk'].append(_auc(res_f, res_r))
    out.update({key_: float(np.mean(v)) for key_, v in acc.items() if v})
    out['k'] = k
    return out


_VIEW = {}


def gm_view_ctx(ctx):
    """A genemean Context of the same cache (for fits without a gene-mean block: 'what genemean's critic would see')."""
    if ctx.features.kind == 'genemean':
        return ctx
    k = str(ctx.cache)
    if k not in _VIEW:
        _VIEW.clear()
        _VIEW[k] = mods()['trainer'].Context(ctx.cache, 'cpu', ('genemean',))
    return _VIEW[k]


def decompose(ctx, G, P, n_rep=DECOMP_REPS, k=GEOM_K):
    """Between/within split of the gene-mean block, real donors vs generated latents (module docstring, 'decomposition')."""
    M = mods()
    np, torch = M['np'], M['torch']
    from pg_model import critic_features
    v = gm_view_ctx(ctx)
    lo, hi = 150, v.features.n_round1
    sc = v.features.scale
    rng, tseed = bag_rng(ctx.fold)
    rng = np.random.default_rng([tseed, 17])
    n = len(v.ids)
    with torch.no_grad():
        real, fake = [], []
        torch.manual_seed(tseed + 17)
        lat = torch.Generator().manual_seed(tseed + 18)
        cell = torch.Generator().manual_seed(tseed + 19)
        cnt = torch.Generator().manual_seed(tseed + 20)
        z, _ = P.sample(n, generator=lat)
        for d in range(n):
            idx = torch.as_tensor(np.stack([rng.choice(v.by[d], BAG, replace=True) for _ in range(n_rep)]))
            real.append((v.Z[idx].mean(1)).double().numpy())
            c = ctx.conditions[[d] * n_rep]
            rate = G.rates(z[[d] * n_rep], c, BAG, v.X[idx].sum(-1), cell)
            counts = torch.poisson(rate, generator=cnt)
            fake.append(critic_features(counts, rate, v.mean, v.sd).mean(1).double().numpy())
    real, fake = np.stack(real), np.stack(fake)            # [n, reps, G] standardised units (no block scale)
    out = {'n_groups': n, 'reps': n_rep, 'block_scale': sc}
    gm_r = real.mean(1)
    k = min(k, n - 1)
    V = np.linalg.svd(gm_r - gm_r.mean(0), full_matrices=False)[2][:k].T
    for nm, x in (('real', real), ('fake', fake)):
        m = x.mean(1)
        within = float(((x - m[:, None]) ** 2).sum(-1).sum(1).mean() / (n_rep - 1))       # trace of within-group cov
        mc = m - m.mean(0)
        between_raw = float((mc ** 2).sum() / (n - 1))
        between = between_raw - within / n_rep                                              # trace of between-group cov
        bin_ = float(((mc @ V) ** 2).sum() / (n - 1))
        out[nm] = {'within_trace': within, 'between_trace': between, 'between_in_real_donor_top%d' % k: bin_ / between_raw,
                   'within_in_real_donor_top%d' % k: float((((x - m[:, None]) @ V) ** 2).sum(-1).sum(1).mean() / (n_rep - 1)) / within}
    # cross-fitted: subspace of half the real donors (even / odd positions), evaluated on the other half and on all latents
    gm_f = fake.mean(1)
    sh_r, sh_f = [], []
    for fit_i, ev_i in ((np.arange(0, n, 2), np.arange(1, n, 2)), (np.arange(1, n, 2), np.arange(0, n, 2))):
        kc = min(CF_K, len(fit_i) - 1)
        Vh = np.linalg.svd(gm_r[fit_i] - gm_r[fit_i].mean(0), full_matrices=False)[2][:kc].T
        for x, lst in ((gm_r[ev_i], sh_r), (gm_f, sh_f)):
            xc = x - x.mean(0)
            lst.append(float(((xc @ Vh) ** 2).sum() / (xc ** 2).sum()))
    out['crossfit_k'] = CF_K
    out['real_heldout_donor_means_in_other_half_subspace'] = float(np.mean(sh_r))
    out['fake_latent_means_in_real_half_subspace'] = float(np.mean(sh_f))
    out['between_ratio_fake_real'] = out['fake']['between_trace'] / out['real']['between_trace']
    out['within_ratio_fake_real'] = out['fake']['within_trace'] / out['real']['within_trace']
    return out


def critic_metrics(ctx, D, real, fake, c, ds):
    M = mods()
    np, torch = M['np'], M['torch']
    D = D.double()
    nf = real.shape[1]
    blocks = blocks_of(ctx)
    W = D.phi[0].weight.detach()          # [h, nf+7]
    res = {'n_features': nf, 'blocks': {k: list(v) for k, v in blocks.items()}}
    with torch.no_grad():
        dr, df = D(real, c), D(fake, c)
    gap = float(dr.mean() - df.mean())
    res['gap'] = gap
    res['D_real_mean'], res['D_fake_mean'] = float(dr.mean()), float(df.mean())
    res['gap_auc'] = float((dr[:, None] > df[None, :]).double().mean())
    xin_r = torch.cat([real, c], 1)
    sd_r = xin_r.std(0)
    allb = dict(blocks, cond7=(nf, nf + 7))
    for nm, x in (('real', real), ('fake', fake)):
        sm = {k: float(x[:, lo:hi].square().sum(1).mean()) for k, (lo, hi) in blocks.items()}
        var = {k: float(x[:, lo:hi].var(0).sum()) for k, (lo, hi) in blocks.items()}
        xin = torch.cat([x, c], 1)
        pre = {k: float((xin[:, lo:hi] @ W[:, lo:hi].T).var(0).sum()) for k, (lo, hi) in allb.items()}
        res[f'second_moment_share_{nm}'] = {k: v / sum(sm.values()) for k, v in sm.items()}
        res[f'variance_share_{nm}'] = {k: v / sum(var.values()) for k, v in var.items()}
        res[f'preact_var_share_{nm}'] = {k: v / sum(pre.values()) for k, v in pre.items()}
        res[f'second_moment_{nm}'] = sm
        xg = xin.clone().requires_grad_(True)
        g, = torch.autograd.grad(D.rho(torch.cat([D.phi(xg), c], 1)).sum(), xg)
        gs = {k: float(((g[:, lo:hi].abs().mean(0) * sd_r[lo:hi]) ** 2).sum()) for k, (lo, hi) in allb.items()}
        res[f'input_grad_x_sd_share_{nm}'] = {k: v / sum(gs.values()) for k, v in gs.items()}
        if nm == 'fake':
            gfake = g[:, :nf].mean(0)
    wf = {k: float(W[:, lo:hi].norm()) for k, (lo, hi) in allb.items()}
    wsd = {k: float((W[:, lo:hi].norm(dim=0) ** 2 * sd_r[lo:hi] ** 2).sum()) for k, (lo, hi) in allb.items()}
    res['weight_fro'] = wf
    res['weight_rms_per_column'] = {k: wf[k] / math.sqrt(hi - lo) for k, (lo, hi) in allb.items()}
    res['weight_x_sd_share'] = {k: v / sum(wsd.values()) for k, v in wsd.items()}
    swap = {}
    for k, (lo, hi) in blocks.items():
        f2 = fake.clone()
        f2[:, lo:hi] = real[:, lo:hi]
        with torch.no_grad():
            g2 = float(dr.mean() - D(f2, c).mean())
        swap[k] = (gap - g2) / gap if gap != 0 else None
    res['gap_swap'] = swap
    # integrated gradients on the straight path fake_i -> real_i (paired bags, same condition):
    # sum over blocks of mean_i int_0^1 grad_block(x(t)) . (real_i - fake_i)_block dt = gap (completeness).
    delta = real - fake
    acc = torch.zeros_like(real)
    for t in (np.arange(IG_STEPS) + 0.5) / IG_STEPS:
        xt = (fake + float(t) * delta).requires_grad_(True)
        gt, = torch.autograd.grad(D(xt, c).sum(), xt)
        acc += gt / IG_STEPS
    contrib = (acc * delta).mean(0)
    ig = {k: float(contrib[lo:hi].sum()) for k, (lo, hi) in blocks.items()}
    res['gap_ig'] = ig
    res['gap_ig_share'] = {k: v / gap for k, v in ig.items()} if gap else None
    res['gap_ig_completeness'] = sum(ig.values()) / gap if gap else None
    res['geometry'] = {k: block_geometry(real[:, lo:hi], fake[:, lo:hi], ds) for k, (lo, hi) in blocks.items()}
    push = None
    if ctx.features.kind == 'genemean':
        lo, hi = blocks['genemean']
        push = gfake[lo:hi].numpy()                                  # dD/d(block) ; block = scale * mean Z
        err = ((fake[:, lo:hi].mean(0) - real[:, lo:hi].mean(0)) / ctx.features.scale).numpy()
        res['push_vs_error_bags'] = float(np.corrcoef(push, -err)[0, 1])
        res['gm_block_error_rms_std_units'] = float(np.sqrt((err ** 2).mean()))
    D.float()
    return res, push


def cmd_critic(args):
    np = mods()['np']
    fits = [f for f in find_fits(args.roots, args.arms, args.panels) if f['complete']]
    which = set(args.states.split(','))
    per = {}
    pushes = {}
    ctx = None
    for f in fits:
        fm = FitModels(f, args.capture_dir, ctx)
        ctx = fm.ctx
        for st in fm.states():
            kind = 'final' if st == 'final' else 'captured' if st.startswith('cap_') else 'snap'
            if kind not in which:
                continue
            G, D, P = fm.load(st)
            m, push = {}, None
            if D is not None:
                real, fake, c, ds = make_bags(ctx, G, P)
                m, push = critic_metrics(ctx, D, real, fake, c, ds)
            if not args.no_decompose:
                m['decomposition'] = decompose(ctx, G, P)
            m['update'] = fm.update_of(st)
            m['has_critic'] = D is not None
            if push is not None and st == 'final':
                m['push_vs_error_generated'] = push_vs_generated(f, ctx, push)
                pushes[key(f)] = push
            per[f'{key(f)}|{st}'] = m
            dc = m.get('decomposition', {})
            log('state', key(f), st, 'between f/r %.2f within f/r %.2f in-real-subspace fake %.2f real %.2f' % (
                dc.get('between_ratio_fake_real', float('nan')), dc.get('within_ratio_fake_real', float('nan')),
                dc.get('fake_latent_means_in_real_half_subspace', float('nan')), dc.get('real_heldout_donor_means_in_other_half_subspace', float('nan'))))
            if D is None:
                continue
            log('critic', key(f), st, 'gap %.2f' % m['gap'], 'preact share real', {k: round(v, 3) for k, v in m['preact_var_share_real'].items()},
                'IG share', {k: round(v, 2) for k, v in (m['gap_ig_share'] or {}).items()}, 'complete %.2f' % (m['gap_ig_completeness'] or 0),
                'push/err', m.get('push_vs_error_bags'), m.get('push_vs_error_generated'))
    summary = {}
    for k, m in per.items():
        arm, panel, _, _, st = k.split('|')
        if st != 'final':
            continue
        s = summary.setdefault(f'{arm}|{panel}', {'n': 0, '_': []})
        s['n'] += 1
        s['_'].append(m)
    for k, s in summary.items():
        L = s.pop('_')
        s['gap_mean'] = float(np.mean([m['gap'] for m in L]))
        s['gap_auc_mean'] = float(np.mean([m['gap_auc'] for m in L]))
        for fld in ('preact_var_share_real', 'preact_var_share_fake', 'variance_share_real', 'variance_share_fake',
                    'second_moment_share_real', 'weight_x_sd_share', 'input_grad_x_sd_share_real', 'input_grad_x_sd_share_fake',
                    'gap_swap', 'gap_ig_share', 'weight_rms_per_column'):
            s[fld] = {b: float(np.mean([m[fld][b] for m in L if m[fld].get(b) is not None])) for b in L[0][fld]}
        s['geometry'] = {b: {m: float(np.mean([x['geometry'][b][m] for x in L])) for m in L[0]['geometry'][b]} for b in L[0]['geometry']}
        dd = [x['decomposition'] for x in L if 'decomposition' in x]
        if dd:
            s['decomposition'] = {m: float(np.mean([x[m] for x in dd])) for m in ('between_ratio_fake_real', 'within_ratio_fake_real',
                                                                                   'fake_latent_means_in_real_half_subspace',
                                                                                   'real_heldout_donor_means_in_other_half_subspace')}
        for fld in ('push_vs_error_bags', 'push_vs_error_generated'):
            v = [m[fld] for m in L if m.get(fld) is not None]
            if v:
                s[fld + '_mean'] = float(np.mean(v))
    out = {'per_state': per, 'summary': summary}
    if args.out:
        write(args.out, out)
        np.savez_compressed(Path(args.out).with_suffix('.push.npz'), **{k.replace('|', '__'): v.astype(np.float32) for k, v in pushes.items()})
    return out


# =========================================================================== real reference and generated summaries
_REAL = {}


def real_meanlog(panel, fold):
    """{donor: per-gene mean over cells of log1p(panel CP10k)} for the fold's train and held-out donors."""
    k = (panel, fold)
    if k in _REAL:
        return _REAL[k]
    M = mods()
    np, trainer = M['np'], M['trainer']
    cache = CACHES / panel / f'fold{fold}'
    out = {}
    for part in ('train', 'heldout'):
        if not (cache / part / 'counts.npz').exists():
            continue
        X = trainer.load_counts(cache / part / 'counts.npz')
        meta = np.load(cache / part / 'metadata.npz', allow_pickle=False)
        lab = meta['donor'].astype(str)
        lib = np.asarray(X.sum(1)).ravel()
        for d in meta['donors'].astype(str):
            rows = np.flatnonzero(lab == d)
            sub = X[rows].multiply(1e4 / np.maximum(lib[rows], 1)[:, None]).tocsr()
            sub.data = np.log1p(sub.data)
            out[d] = np.asarray(sub.mean(0)).ravel().astype(np.float64)
    _REAL[k] = out
    return out


def gen_summary(path):
    """Per-gene mean_log over train targets and held-out targets (float64), MC floor RMS, targets."""
    np = mods()['np']
    with np.load(path, allow_pickle=False) as g:
        ml = g['mean_log'].astype(np.float64)       # [T,M,G]
        held = g['is_heldout'].astype(bool)
        tg = g['targets'].astype(str)
        genes = g['genes'].astype(str)
    out = {'targets': tg, 'is_heldout': held, 'genes': genes, 'per_target': ml.mean(1)}
    for nm, sel in (('train', ~held), ('heldout', held)):
        if not sel.any():
            continue
        x = ml[sel]                                   # [t,M,G]
        out[nm] = x.mean((0, 1))
        h = x[:, 0::2].mean((0, 1)) - x[:, 1::2].mean((0, 1))
        out[nm + '_mc_rms'] = float(np.sqrt((h ** 2 / 4).mean()))
    return out


def push_vs_generated(f, ctx, push):
    np = mods()['np']
    gs = gen_summary(f['fit'] / 'generated.npz')
    real = real_meanlog(f['panel'], f['fold'])
    sel = ~gs['is_heldout']
    r = np.stack([real[d] for d in gs['targets'][sel]]).mean(0)
    err = (gs['train'] - r) / ctx.sd.double().numpy()
    return float(np.corrcoef(push, -err)[0, 1])


def spearman(x, y):
    np = mods()['np']
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return None
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_seeds(args):
    M = mods()
    np = M['np']
    fits = [f for f in find_fits(args.roots, args.arms, args.panels) if f['complete']]
    curves = json.loads(Path(args.curves).read_text())['per_fit'] if args.curves and Path(args.curves).exists() else {}
    pushes = {}
    for cp in ([args.critic] if isinstance(args.critic, str) else (args.critic or [])):
        if Path(cp).with_suffix('.push.npz').exists():
            with np.load(Path(cp).with_suffix('.push.npz')) as z:
                pushes.update({k.replace('__', '|'): z[k].astype(np.float64) for k in z.files})
    units = {}
    for f in fits:
        for name, unit in (('generated.npz', f['label']), ('generated_snapens.npz', f['label'] + '#snapens')):
            p = f['fit'] / name
            if p.exists():
                units.setdefault((unit, f['panel'], f['fold']), []).append((f, gen_summary(p)))
    per_group, per_fit = {}, {}
    for (unit, panel, fold), L in sorted(units.items()):
        real = real_meanlog(panel, fold)
        gs0 = L[0][1]
        tr = [d for d, h in zip(gs0['targets'], gs0['is_heldout']) if not h]
        ho = [d for d, h in zip(gs0['targets'], gs0['is_heldout']) if h]
        r_tr = np.stack([real[d] for d in tr]).mean(0)
        r_ho = np.stack([real[d] for d in ho]).mean(0) if ho else None
        ms = np.stack([g['train'] for _, g in L])
        mh = np.stack([g['heldout'] for _, g in L]) if ho else None
        n = len(L)
        pair = [float(np.sqrt(((ms[i] - ms[j]) ** 2).mean())) for i in range(n) for j in range(i + 1, n)]
        mc = float(np.mean([g['train_mc_rms'] for _, g in L]))
        grp = {'n_seeds': n, 'seeds': [f['seed'] for f, _ in L], 'pair_rms_train': pair,
               'pair_rms_train_mean': float(np.mean(pair)) if pair else None, 'mc_rms_train': mc,
               'pair_over_independent_mc': (float(np.mean(pair)) / (math.sqrt(2) * mc)) if pair else None,
               'rmse_train_each': [float(np.sqrt(((m - r_tr) ** 2).mean())) for m in ms],
               'rmse_train_ensemble': float(np.sqrt(((ms.mean(0) - r_tr) ** 2).mean()))}
        if ho:
            grp['rmse_heldout_each'] = [float(np.sqrt(((m - r_ho) ** 2).mean())) for m in mh]
            grp['rmse_heldout_ensemble'] = float(np.sqrt(((mh.mean(0) - r_ho) ** 2).mean()))
            grp['pair_rms_heldout_mean'] = float(np.mean([np.sqrt(((mh[i] - mh[j]) ** 2).mean()) for i in range(n) for j in range(i + 1, n)])) if n > 1 else None
        per_group[f'{unit}|{panel}|{fold}'] = grp
        if n < 2:
            continue
        sd = None
        for i, (f, g) in enumerate(L):
            others = np.delete(ms, i, 0).mean(0)
            dev = ms[i] - others
            k = key(f)
            row = {'unit': unit, 'panel': panel, 'fold': fold, 'seed': f['seed'],
                   'loo_dev_rms_train': float(np.sqrt((dev ** 2).mean())),
                   'rmse_train': grp['rmse_train_each'][i]}
            cv = curves.get(k)
            if cv:
                row['final_gap_curve'] = cv.get('final_gap')
                row['slow_sd'] = cv.get('second_half', {}).get('slow_sd')
            if k in pushes and '#' not in unit:
                if sd is None:
                    with np.load(CACHES / panel / f'fold{fold}' / 'space.npz') as s:
                        sd = s['sd'].astype(np.float64)
                row['push_vs_deviation'] = float(np.corrcoef(pushes[k], -dev / sd)[0, 1])
            per_fit[f'{k}' + ('#snapens' if '#' in unit else '')] = row
    # relations (within unit, fold/panel-centred)
    rel = {}
    for unit in sorted({r['unit'] for r in per_fit.values()}):
        rows = [r for r in per_fit.values() if r['unit'] == unit]
        out = {'n_fits': len(rows)}
        for xname in ('final_gap_curve', 'slow_sd'):
            R = [r for r in rows if r.get(xname) is not None]
            if len(R) < 4:
                continue
            gm = {}
            for r in R:
                gm.setdefault((r['panel'], r['fold']), []).append(r)
            xs, ys = [], []
            for grp_rows in gm.values():
                if len(grp_rows) < 2:
                    continue
                mx = np.mean([r[xname] for r in grp_rows]); my = np.mean([r['loo_dev_rms_train'] for r in grp_rows])
                xs += [r[xname] - mx for r in grp_rows]; ys += [r['loo_dev_rms_train'] - my for r in grp_rows]
            out[f'spearman_{xname}_vs_loo_dev_centred'] = spearman(xs, ys)
            out[f'n_{xname}'] = len(xs)
        pv = [r['push_vs_deviation'] for r in rows if 'push_vs_deviation' in r]
        if pv:
            out['push_vs_deviation_mean'] = float(np.mean(pv))
            out['push_vs_deviation_range'] = [float(min(pv)), float(max(pv))]
        rel[unit] = out
    # across groups: does a larger mean final gap go with a larger seed disagreement?
    gx, gy = [], []
    for gk, g in per_group.items():
        unit, panel, fold = gk.split('|')
        gaps = [curves.get(f'{unit}|{panel}|{fold}|{s}', {}).get('final_gap') for s in g['seeds']]
        if g['pair_rms_train_mean'] is not None and all(x is not None for x in gaps) and '#' not in unit:
            gx.append(float(np.mean(gaps))); gy.append(g['pair_over_independent_mc'])
    summary = {}
    for unit in sorted({k.split('|')[0] for k in per_group}):
        for panel in ('hvg2000', 'wide'):
            G = [g for k, g in per_group.items() if k.split('|')[0] == unit and k.split('|')[1] == panel]
            if not G:
                continue
            s = {'n_folds': len(G), 'n_seeds': [g['n_seeds'] for g in G]}
            for fld in ('pair_rms_train_mean', 'mc_rms_train', 'pair_over_independent_mc', 'rmse_train_ensemble', 'rmse_heldout_ensemble',
                        'pair_rms_heldout_mean'):
                v = [g[fld] for g in G if g.get(fld) is not None]
                if v:
                    s[fld] = float(np.mean(v))
            for fld in ('rmse_train_each', 'rmse_heldout_each'):
                v = [x for g in G for x in g.get(fld, [])]
                if v:
                    s[fld.replace('_each', '_single_mean')] = float(np.mean(v))
            summary[f'{unit}|{panel}'] = s
    out = {'per_group': per_group, 'per_fit': per_fit, 'relations': rel,
           'across_groups_gap_vs_disagreement': {'n': len(gx), 'spearman': spearman(gx, gy), 'x_mean_final_gap': gx, 'y_pair_over_mc': gy},
           'summary': summary}
    if args.out:
        write(args.out, out)
    return out


# =========================================================================== drift (Q3)
def drift_targets(ctx):
    M = mods()
    np, generate, arms = M['np'], M['generate'], M['arms']
    T = generate.targets(ctx)
    train = [t for t in T if not t[1]]
    st = arms.stage_of(np.stack([t[2] for t in train]))
    pick = []
    for s in range(4):
        pick += [t for t, x in zip(train, st) if x == s][:DRIFT_TARGETS_PER_STAGE]
    return pick


def reduced_generate(ctx, G, P, T):
    M = mods()
    np, generate = M['np'], M['generate']
    plans, _ = generate.default_plans(ctx, T)
    arr, _ = generate.generate_arrays(ctx, {'final': (G, P)}, plans, T=T)
    ml = arr['mean_log'].astype(np.float64)           # [t,M,G]
    h = ml[:, 0::2].mean((0, 1)) - ml[:, 1::2].mean((0, 1))
    return ml.mean((0, 1)), float(np.sqrt((h ** 2 / 4).mean()))


def cmd_drift(args):
    M = mods()
    np = M['np']
    fits = [f for f in find_fits(args.roots, args.arms, args.panels) if f['complete']]
    per = {}
    finals = {}
    ctx = None
    cache_T = {}
    for f in fits:
        fm = FitModels(f, args.capture_dir, ctx)
        ctx = fm.ctx
        states = fm.states()
        ck = (f['panel'], f['fold'])
        if ck not in cache_T:
            cache_T[ck] = drift_targets(ctx)
        T = cache_T[ck]
        multi = len(states) > 1
        if not multi and not args.all_finals:
            continue
        real = real_meanlog(f['panel'], f['fold'])
        r = np.stack([real[t[0]] for t in T]).mean(0)
        ms, mcs, gaps = {}, {}, {}
        for st in states:
            G, D, P = fm.load(st, critic=True)
            ms[st], mcs[st] = reduced_generate(ctx, G, P, T)
            if D is not None:
                real_b, fake_b, c, _ = make_bags(ctx, G, P)
                with M['torch'].no_grad():
                    Dd = D.double()
                    gaps[st] = float(Dd(real_b, c).mean() - Dd(fake_b, c).mean())
        finals[key(f)] = ms['final']
        if not multi:
            continue
        fin = ms['final']
        mc = float(np.mean(list(mcs.values())))
        avg = np.mean([ms[s] for s in states], 0)
        row = {'states': states, 'updates': [fm.update_of(s) for s in states], 'targets': [t[0] for t in T], 'mc_rms': mc,
               'drift_rms_to_final': {s: float(np.sqrt(((ms[s] - fin) ** 2).mean())) for s in states if s != 'final'},
               'step_rms': {f'{a}->{b}': float(np.sqrt(((ms[b] - ms[a]) ** 2).mean())) for a, b in zip(states[:-1], states[1:])},
               'rmse_vs_real_train': {s: float(np.sqrt(((ms[s] - r) ** 2).mean())) for s in states},
               'rmse_state_average': float(np.sqrt(((avg - r) ** 2).mean())),
               'gap_by_state': gaps}
        row['drift_over_independent_mc'] = {s: v / (math.sqrt(2) * mc) for s, v in row['drift_rms_to_final'].items()}
        per[key(f)] = row
        log('drift', key(f), {s: round(v, 4) for s, v in row['drift_rms_to_final'].items()}, 'mc', round(mc, 4),
            'rmse', {s: round(v, 4) for s, v in row['rmse_vs_real_train'].items()}, 'avg', round(row['rmse_state_average'], 4), 'gaps', gaps)
    # seed-to-seed reference on the same reduced targets
    groups = {}
    for k, m in finals.items():
        arm, panel, fold, seed = k.split('|')
        groups.setdefault((arm, panel, fold), []).append(m)
    seedref = {f'{a}|{p}|{k}': float(np.mean([np.sqrt(((L[i] - L[j]) ** 2).mean()) for i in range(len(L)) for j in range(i + 1, len(L))]))
               for (a, p, k), L in groups.items() if len(L) > 1}
    for k, row in per.items():
        arm, panel, fold, _ = k.split('|')
        ref = seedref.get(f'{arm}|{panel}|{fold}')
        row['seed_pair_rms_final'] = ref
        if ref:
            row['drift_over_seed_pair'] = {s: v / ref for s, v in row['drift_rms_to_final'].items()}
    summary = {}
    for k, row in per.items():
        arm, panel = k.split('|')[:2]
        s = summary.setdefault(f'{arm}|{panel}', {'n': 0, '_': []})
        s['n'] += 1
        s['_'].append(row)
    for k, s in summary.items():
        L = s.pop('_')
        firsts = [r['states'][0] for r in L]
        s['first_state'] = sorted(set(firsts))
        s['drift_first_to_final_over_mc'] = float(np.mean([r['drift_over_independent_mc'][r['states'][0]] for r in L]))
        s['drift_last_snapshot_to_final_over_mc'] = float(np.mean([r['drift_over_independent_mc'][r['states'][-2]] for r in L]))
        v = [r['drift_over_seed_pair'][r['states'][0]] for r in L if r.get('drift_over_seed_pair')]
        if v:
            s['drift_first_to_final_over_seed_pair'] = float(np.mean(v))
        byu = {}
        for r in L:
            for st_, u_ in zip(r['states'], r['updates']):
                if st_ == 'final':
                    continue
                b = byu.setdefault(u_, {'mc': [], 'seed': [], 'rmse_minus_final': []})
                b['mc'].append(r['drift_over_independent_mc'][st_])
                if r.get('drift_over_seed_pair'):
                    b['seed'].append(r['drift_over_seed_pair'][st_])
                b['rmse_minus_final'].append(r['rmse_vs_real_train'][st_] - r['rmse_vs_real_train']['final'])
        s['by_update'] = {str(u_): {'n': len(b['mc']), 'drift_over_mc': float(np.mean(b['mc'])),
                                    'drift_over_seed_pair': float(np.mean(b['seed'])) if b['seed'] else None,
                                    'rmse_minus_final': float(np.mean(b['rmse_minus_final']))} for u_, b in sorted(byu.items())}
        s['rmse_final_mean'] = float(np.mean([r['rmse_vs_real_train']['final'] for r in L]))
        s['rmse_state_average_mean'] = float(np.mean([r['rmse_state_average'] for r in L]))
        gg = [r['gap_by_state'] for r in L if len(r['gap_by_state']) > 1]
        if gg:
            s['gap_first_state_mean'] = float(np.mean([g[min(g, key=lambda x: 10 ** 9 if x == 'final' else int(x.split('_')[1]))] for g in gg]))
            s['gap_final_mean'] = float(np.mean([g['final'] for g in gg if 'final' in g]))
    out = {'per_fit': per, 'seed_pair_rms_reduced_targets': seedref, 'summary': summary}
    if args.out:
        write(args.out, out)
    return out


# =========================================================================== CLI
def main(argv=None):
    p = argparse.ArgumentParser(description='F6 genemean diagnosis (see module docstring)')
    p.add_argument('cmd', choices=['capture', 'curves', 'critic', 'seeds', 'drift', 'all'])
    p.add_argument('--roots', nargs='+', default=[str(r) for r in DEFAULT_ROOTS])
    p.add_argument('--arms', nargs='+', default=None)
    p.add_argument('--panels', nargs='+', default=['hvg2000', 'wide'])
    p.add_argument('--states', default='final,captured')
    p.add_argument('--capture-dir', default=str(DIAG / 'capture'))
    p.add_argument('--curves', default=None)
    p.add_argument('--critic', nargs='+', default=None, help='critic JSON file(s) whose .push.npz sidecars give the critic push')
    p.add_argument('--all-finals', action='store_true', help='drift: also generate single-state fits (seed reference)')
    p.add_argument('--no-decompose', action='store_true', help='critic: skip the between/within decomposition')
    p.add_argument('--out', default=None)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--fits', nargs='+')
    p.add_argument('--updates', nargs='+', type=int)
    p.add_argument('--poll', type=float, default=30)
    p.add_argument('--until', type=float, default=3 * 3600)
    a = p.parse_args(argv)
    if a.cmd == 'capture':
        return capture(a.fits, a.updates, a.poll, a.until)
    if a.capture_dir and not Path(a.capture_dir).exists():
        a.capture_dir = None
    if a.cmd == 'all':
        d = Path(a.out_dir or (DIAG / 'results'))
        a.out = str(d / 'gm_curves.json'); cmd_curves(a)
        a.curves = a.out
        a.out = str(d / 'gm_critic.json'); cmd_critic(a)
        a.critic = a.out
        a.out = str(d / 'gm_seeds.json'); cmd_seeds(a)
        a.out = str(d / 'gm_drift.json'); cmd_drift(a)
        return
    {'curves': cmd_curves, 'critic': cmd_critic, 'seeds': cmd_seeds, 'drift': cmd_drift}[a.cmd](a)


if __name__ == '__main__':
    main()
