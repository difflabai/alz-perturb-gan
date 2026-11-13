"""Tests and real-data diagnosis for the round-3 critic-architecture arms (r3/arm_dcritic.py, work item R4).

CPU only, 3 threads per process, real hvg2000 caches (read-only); round-1/2
fits are read (final.pt on CPU), never written.  Run::

    python3 -B r3/test_arm_dcritic.py starve    # R4 diagnosis on round-2 seed-0 fits (+ round-1 base/genemean seed 0):
                                                #   per-block first-layer variance shares, weight norms, gap attribution,
                                                #   penalty gradient budget, DualNorm polar-step shares, push vs error,
                                                #   joined with the held-out PC/cell metrics of results_r2_interim
    python3 -B r3/test_arm_dcritic.py unit      # module-level checks of DTowerCritic / DSumCritic / d.h / d.depth
    python3 -B r3/test_arm_dcritic.py penalty   # what the BISPF penalty sees: invariance of the penalty to internal
                                                #   per-block normalisation (re-parametrisation), and its dependence on
                                                #   x-space scaling; native penalty object on real bags
    python3 -B r3/test_arm_dcritic.py neutral   # absent d. flags: r3 trainer bitwise unchanged (base, genemean, pbulk 3 updates)
    python3 -B r3/test_arm_dcritic.py fits      # tiny CPU fits (training + checkpoint/restore bitwise + config) with every
                                                #   d. arm alone and with genemean@0.5, pbulk and genemean,pbulk
    python3 -B r3/test_arm_dcritic.py cli       # one tiny CLI fit through trainer.py + generation (smoke_r3/critic)
    python3 -B r3/test_arm_dcritic.py short     # <= 300-update GPU fits (fold 0, seed 0, --gpu-mem-fraction 0.04, one process):
                                                #   early gap / penalty engagement / block attribution per critic arm
    python3 -B r3/test_arm_dcritic.py cleanup   # delete scratch (results/*.json kept)
    python3 -B r3/test_arm_dcritic.py all       # unit penalty neutral fits cli cleanup (starve is separate: ~30 min)

Scratch: /ml2/alz/runs/improve_20261109/smoke_r3/critic/ (results/*.json kept).
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
if not (len(sys.argv) > 1 and sys.argv[1] == 'short' and (len(sys.argv) < 3 or sys.argv[2] == 'cuda')):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

torch.set_num_threads(3)
import trainer  # noqa: E402  (r3)
import arms  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHES = RUNS / 'caches'
CACHE0 = CACHES / 'hvg2000/fold0'
SCR = RUNS / 'smoke_r3/critic'
RES = SCR / 'results'
EVAL_R2 = RUNS / 'results_r2_interim/fits'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')


def log(*a):
    print('[test_arm_dcritic]', time.strftime('%H:%M:%S'), *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', json.loads(json.dumps(obj, default=_js)))


def _js(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return str(x)


# ============================================================================ starve (real-data diagnosis)
STARVE_FITS = [  # (root, arm, label)
    ('fits', 'base', 'base'),
    ('fits', 'genemean', 'genemean(r1)'),
    ('fits_r2', 'genemean', 'genemean~r2'),
    ('fits_r2', 'genemean@0.5', 'genemean@0.5'),
    ('fits_r2', 'genemean@0.25', 'genemean@0.25'),
    ('fits_r2', 'gmlite@64', 'gmlite@64'),
    ('fits_r2', 'pbulk', 'pbulk'),
    ('fits_r2', 'genemean,pbulk', 'genemean,pbulk'),
    ('fits_r2', 'donorpc2', 'donorpc2'),
]
POLAR_BATCH = 32


def _polar(Gm):
    U, S, Vh = torch.linalg.svd(Gm, full_matrices=False)
    return U @ Vh


def starve_extras(ctx, D, real, fake, c):
    """Penalty gradient budget, DualNorm polar-step shares and push-vs-error per block (float64, no RNG)."""
    import gm_diagnose as gd
    D = D.double()
    blocks = gd.blocks_of(ctx)
    nf = real.shape[1]
    out = {}
    # (1) input-gradient budget as the BISPF penalty measures it: ||dD/dx|| / sqrt(dim), kappa = 1
    pts = {'real': real, 'fake': fake, 'mid': 0.5 * (real + fake)}
    grads = {}
    for nm, x in pts.items():
        xg = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(D(xg, c).sum(), xg)
        grads[nm] = g
        rms = g.norm(dim=1) / math.sqrt(nf)
        sq = g.square()
        tot = sq.sum(1)
        out[f'grad_rms_{nm}'] = {'mean': float(rms.mean()), 'median': float(rms.median()), 'frac_above_kappa': float((rms > 1).double().mean())}
        out[f'grad_sq_share_{nm}'] = {k: float((sq[:, lo:hi].sum(1) / tot).mean()) for k, (lo, hi) in blocks.items()}
        out[f'grad_rms_per_column_{nm}'] = {k: float((sq[:, lo:hi].sum(1) / (hi - lo)).mean().sqrt()) for k, (lo, hi) in blocks.items()}
    out['column_share'] = {k: (hi - lo) / nf for k, (lo, hi) in blocks.items()}
    # (2) DualNorm step on the first critic layer (adversarial part of the critic loss only; penalty omitted):
    # step = lr * polar(grad); share of ||polar||_F^2 on each block's columns vs share of ||grad||_F^2
    W = D.phi[0].weight
    pol, raw = {k: [] for k in blocks}, {k: [] for k in blocks}
    for st in range(0, len(real), POLAR_BATCH):
        sl = slice(st, st + POLAR_BATCH)
        dr, df = D(real[sl], c[sl]), D(fake[sl], c[sl])
        loss = torch.nn.functional.softplus(-dr).mean() + torch.nn.functional.softplus(df).mean()
        gw, = torch.autograd.grad(loss, W)
        P = _polar(gw)
        for k, (lo, hi) in blocks.items():
            pol[k].append(float(P[:, lo:hi].square().sum() / P[:, :nf].square().sum()))
            raw[k].append(float(gw[:, lo:hi].square().sum() / gw[:, :nf].square().sum()))
    out['polar_step_share_first_layer'] = {k: float(np.mean(v)) for k, v in pol.items()}
    out['grad_share_first_layer'] = {k: float(np.mean(v)) for k, v in raw.items()}
    # (3) generator push through each gene-wide block: mean dD/dx at fake bags vs minus (fake - real) block mean
    pv = {}
    for k, (lo, hi) in blocks.items():
        if hi - lo == ctx.n_genes:
            push = grads['fake'][:, lo:hi].mean(0)
            err = fake[:, lo:hi].mean(0) - real[:, lo:hi].mean(0)
            pv[k] = {'pearson': float(np.corrcoef(push.numpy(), -err.numpy())[0, 1]),
                     'err_rms_block_units': float(err.square().mean().sqrt()),
                     'push_rms': float(push.square().mean().sqrt())}
    out['push_vs_error'] = pv
    D.float()
    return out


def eval_metrics(label_root, arm, fold, seed):
    unit = arm if label_root == 'fits' else arm + '~r2'
    p = EVAL_R2 / unit / 'hvg2000' / f'fold{fold}' / f'seed{seed}.json'
    if not p.exists():
        return None
    q = json.loads(p.read_text())
    h = q['subsets']['all']['heldout']
    pc = q.get('pc_space_rep', {}).get('heldout', {})
    return {'pb_rmse': h['pseudobulk']['rmse'], 'meanlog_rmse_gene': h['mean_log']['rmse_gene'],
            'pc_energy_rep': pc.get('energy_score'), 'cell_energy_rep': pc.get('cell_energy_distance'),
            'pc_coverage80_rep': pc.get('coverage80')}


def starve(seeds=(0,)):
    import gm_diagnose as gd
    gd.mods()
    rows = []
    ctx = None
    for root, arm, label in STARVE_FITS:
        fl = [f for f in gd.find_fits([RUNS / root], [arm], ('hvg2000',)) if f['complete'] and f['seed'] in seeds]
        for f in fl:
            t0 = time.monotonic()
            fm = gd.FitModels(f, None, ctx)
            ctx = fm.ctx
            G, D, P = fm.load('final')
            real, fake, c, ds = gd.make_bags(ctx, G, P)
            m, _ = gd.critic_metrics(ctx, D, real, fake, c, ds)
            m.pop('geometry', None)
            ex = starve_extras(ctx, D, real, fake, c)
            row = {'label': label, 'arm': arm, 'root': root, 'fold': f['fold'], 'seed': f['seed'], **m, **ex,
                   'eval': eval_metrics(root, arm, f['fold'], f['seed'])}
            rows.append(row)
            log('starve', label, f['fold'], f['seed'], 'gap %.2f' % m['gap'],
                'preact real', {k: round(v, 3) for k, v in m['preact_var_share_real'].items()},
                'IG', {k: round(v, 2) for k, v in (m['gap_ig_share'] or {}).items()},
                'polar', {k: round(v, 3) for k, v in ex['polar_step_share_first_layer'].items()},
                'gradRMS fake %.2f' % ex['grad_rms_fake']['mean'], '%.0fs' % (time.monotonic() - t0))
            del G, D, P, real, fake
    summary = starve_summary(rows)
    save('starve', {'per_fit': rows, 'summary': summary})
    return summary


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def starve_summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r['label'], []).append(r)
    out = {}
    for lab, L in by.items():
        blocks = list(L[0]['preact_var_share_real'])
        s = {'n': len(L), 'gap': _mean([r['gap'] for r in L])}
        for fld in ('preact_var_share_real', 'preact_var_share_fake', 'gap_ig_share', 'weight_rms_per_column', 'weight_fro',
                    'polar_step_share_first_layer', 'grad_share_first_layer', 'grad_sq_share_real', 'grad_sq_share_fake',
                    'grad_rms_per_column_fake', 'column_share', 'input_grad_x_sd_share_fake'):
            s[fld] = {b: _mean([(r[fld] or {}).get(b) for r in L]) for b in blocks if any((r[fld] or {}).get(b) is not None for r in L)}
        for nm in ('real', 'fake', 'mid'):
            s[f'grad_rms_{nm}'] = {k: _mean([r[f'grad_rms_{nm}'][k] for r in L]) for k in ('mean', 'frac_above_kappa')}
        pv = {}
        for r in L:
            for k, v in r['push_vs_error'].items():
                pv.setdefault(k, []).append(v['pearson'])
        s['push_vs_error_pearson'] = {k: _mean(v) for k, v in pv.items()}
        ev = [r['eval'] for r in L if r['eval']]
        s['eval'] = {k: _mean([e[k] for e in ev]) for k in ev[0]} if ev else None
        out[lab] = s
    # across fits: released-block first-layer share vs held-out PC/cell metrics (Spearman over all fits)
    from scipy.stats import spearmanr
    xs = [(r['preact_var_share_real']['released150'], r['gap_ig_share']['released150'] if r['gap_ig_share'] else None,
           r['polar_step_share_first_layer']['released150'], r['gap'], r['eval']) for r in rows if r['eval']]
    rel = {}
    for i, nm in enumerate(('preact_share_released', 'ig_share_released', 'polar_share_released', 'gap')):
        for met in ('pc_energy_rep', 'cell_energy_rep', 'pb_rmse', 'meanlog_rmse_gene'):
            a = [x[i] for x in xs if x[i] is not None and x[4][met] is not None]
            b = [x[4][met] for x in xs if x[i] is not None and x[4][met] is not None]
            rel[f'{nm}~{met}'] = {'spearman': float(spearmanr(a, b)[0]), 'n': len(a)}
    return {'per_arm': out, 'across_fits': rel}


CURVE_ARMS = [('fits', 'base'), ('fits_r2', 'genemean'), ('fits_r2', 'genemean@0.5'), ('fits_r2', 'genemean@0.25'),
              ('fits_r2', 'gmlite@64'), ('fits_r2', 'pbulk'), ('fits_r2', 'genemean,pbulk'), ('fits_r2', 'donorpc2'),
              ('fits_r2', 'stagedrop@0.5'), ('fits_r2', 'covdrop@0.5')]


def penalty_curves():
    """Native BISPF penalty statistics logged during training (learning_curve.jsonl rows 250k+1; seed 0, folds 0-2).

    mean_grad_norm = mean RMS input-gradient norm ||dD/dx|| / sqrt(dim) (kappa = 1); ebi_loss = cap + floor terms.
    'late' = updates > 15000.  Read-only on the fits.
    """
    out = {}
    for root, arm in CURVE_ARMS:
        per = []
        for k in range(3):
            p = RUNS / root / arm / 'hvg2000' / f'fold{k}' / 'seed0' / 'learning_curve.jsonl'
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            cfg = json.loads((p.parent / 'config.json').read_text())
            late = [r for r in rows if r['step'] > 15000 and r.get('penalty_stats') and r['step'] % 250 == 1]
            early = [r for r in rows if 1 < r['step'] <= 2001 and r.get('penalty_stats') and r['step'] % 250 == 1]
            if not late:
                continue
            ps = lambda R, k_: float(np.mean([r['penalty_stats'][k_] for r in R]))
            per.append({'fold': k, 'dim': int(cfg['critic_features']), 'late_grad_rms': ps(late, 'mean_grad_norm'),
                        'late_ebi_loss': ps(late, 'ebi_loss'), 'late_floor_pen': ps(late, 'floor_pen'),
                        'late_D_penalty': float(np.mean([r['D_penalty'] for r in late])),
                        'late_critic_gap': float(np.mean([r['critic_gap'] for r in late])),
                        'late_D_adversarial': float(np.mean([r['D_adversarial'] for r in late])),
                        'early_grad_rms': ps(early, 'mean_grad_norm'), 'late_s_thermal': ps(late, 's_thermal')})
        if per:
            m = {k_: float(np.mean([x[k_] for x in per])) for k_ in per[0] if k_ != 'fold'}
            m['abs_grad_norm_late'] = m['late_grad_rms'] * math.sqrt(m['dim'])
            m['abs_cap_kappa1'] = math.sqrt(m['dim'])
            m['kappa_matching_base_abs_cap'] = math.sqrt(150 / m['dim'])
            out[arm] = {'mean': m, 'folds': per}
    return out


# ============================================================================ unit
def ctx_of(arm, cache=CACHE0):
    return trainer.Context(cache, 'cpu', arms.parse_arm(arm))


def unit():
    import arm_dcritic as dc
    from critic import VectorCritic
    res = {}
    # flag parsing
    ok = ['d.tower', 'd.tower@64', 'd.sum', 'd.sum@256', 'd.h@64', 'd.h@256', 'd.depth@1', 'd.depth@3', 'genemean@0.5,d.tower',
          'pbulk,d.sum', 'genemean,pbulk,d.tower']
    bad = ['d.tower@128', 'd.sum@128', 'd.h@128', 'd.depth@2', 'd.h', 'd.depth', 'd.tower@64.0', 'd.tower@8', 'd.h@2048',
           'd.depth@7', 'd.tower,d.sum', 'd.tower,d.h@64', 'd.h@64,d.depth@3', 'd.example,d.tower']
    res['parsed'] = {a: arms.arm_name(arms.parse_arm(a)) for a in ok}
    refused = {}
    for a in bad:
        try:
            arms.parse_arm(a)
            refused[a] = False
        except ValueError as e:
            refused[a] = str(e)[:90]
    res['refused'] = refused
    assert all(refused.values()), refused
    # DeepVectorCritic(depth 2, h 128) == released critics (names, order, initial draws)
    for nf, ref in ((150, lambda: VectorCritic(150, h=128)), (2150, lambda: arms.WideVectorCritic(2150, h=128))):
        torch.manual_seed(11); A = ref()
        torch.manual_seed(11); B = dc.DeepVectorCritic(nf)
        sa, sb = A.state_dict(), B.state_dict()
        same = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
        x, c = torch.randn(5, nf), torch.rand(5, 7)
        res[f'deep_equals_released_{nf}'] = bool(same and torch.equal(A(x, c), B(x, c)))
        assert res[f'deep_equals_released_{nf}']
    # per observation: reference stats, normalisation on independent bags, torch RNG untouched, module checks
    for arm in ('base', 'genemean@0.5', 'pbulk', 'genemean,pbulk', 'gmlite@64'):
        ctx = ctx_of(arm)
        lay = ctx.features.layout()
        rng_before = torch.get_rng_state()
        center, scale, info = dc.block_reference_stats(ctx, lay, 'd.tower')
        rng_untouched = torch.equal(rng_before, torch.get_rng_state())
        c2, s2, _ = dc.block_reference_stats(ctx, lay, 'd.tower')
        deterministic = all(torch.equal(a, b) for a, b in zip(center, c2)) and s2 == scale
        # independent real bags (other seed): normalised blocks have mean per-column variance ~1 and mean ~0
        _, _, other = dc.block_reference_stats(ctx, lay, 'other-seed-check')
        norm = dc._BlockNorm(lay.blocks, center, scale)
        rng = np.random.default_rng(99)
        ds = rng.integers(len(ctx.by), size=64)
        idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], ctx.bag, replace=True) for i in ds]))
        with torch.no_grad():
            xs = norm(ctx.features.real(idx, ds))
        per_block = [{'block': i['block'], 'width': i['columns'][1] - i['columns'][0], 'scale': i['scale'],
                      'mean_col_var_independent_bags': float(xb.var(0).mean()), 'mean_abs_col_mean': float(xb.mean(0).abs().mean()),
                      'scale_other_seed_ratio': o['scale'] / i['scale']} for i, o, xb in zip(info, other, xs)]
        row = {'blocks': per_block, 'torch_rng_untouched': rng_untouched, 'deterministic': deterministic}
        assert rng_untouched and deterministic
        assert all(0.7 < b['mean_col_var_independent_bags'] < 1.4 and 0.85 < b['scale_other_seed_ratio'] < 1.15 for b in per_block), per_block
        # modules: forward shape, determinism, gradient on every block, parameters, buffers
        real = ctx.features.real(idx[:8], ds[:8])
        cnd = ctx.conditions[torch.as_tensor(ds[:8])]
        for name, mk in (('tower', lambda: dc.DTowerCritic(lay, center, scale)), ('sum', lambda: dc.DSumCritic(lay, center, scale)),
                         ('tower64', lambda: dc.DTowerCritic(lay, center, scale, h=64)), ('h256', lambda: dc.DeepVectorCritic(lay.n_features, h=256)),
                         ('depth1', lambda: dc.DeepVectorCritic(lay.n_features, depth=1)), ('depth4', lambda: dc.DeepVectorCritic(lay.n_features, depth=4))):
            torch.manual_seed(3)
            D = mk()
            x = real.detach().clone().requires_grad_(True)
            out = D(x, cnd)
            g, = torch.autograd.grad(out.sum(), x)
            again = torch.equal(out.detach(), D(real, cnd).detach())
            grad_blocks = {b.name: float(g[:, b.lo:b.hi].abs().sum()) for b in lay.blocks}
            row[name] = {'out_shape': list(out.shape), 'deterministic': again, 'n_params': sum(p.numel() for p in D.parameters()),
                         'n_param_tensors': len(list(D.parameters())), 'buffers': len(list(D.buffers())),
                         'all_params_2d_or_1d': all(p.ndim in (1, 2) for p in D.parameters()),
                         'grad_all_blocks_nonzero': all(v > 0 for v in grad_blocks.values())}
            assert out.shape == (8,) and again and row[name]['grad_all_blocks_nonzero'] and row[name]['all_params_2d_or_1d']
        row['released_wide_critic_params'] = sum(p.numel() for p in (VectorCritic(150) if lay.n_features == 150 else arms.WideVectorCritic(lay.n_features)).parameters())
        res[arm] = row
        log('unit', arm, 'scales', [round(b['scale'], 4) for b in per_block], 'indep var', [round(b['mean_col_var_independent_bags'], 3) for b in per_block])
    # per-cell blocks refused by d.tower / d.sum
    lay = arms.CriticLayout(150 + 16 * 8, 7, 128, 32, 16, (arms.BlockSpec(None, 'released', 0, 150), arms.BlockSpec('d.example', 'cellpc8', 150, 278, 8)))
    try:
        dc.DTowerCritic(lay, [torch.zeros(150), torch.zeros(128)], [1., 1.])
        res['cell_block_refused'] = False
    except ValueError:
        res['cell_block_refused'] = True
    assert res['cell_block_refused']
    # arm objects: state/info lazily computed, equal to the module buffers
    ctx = ctx_of('genemean@0.5,d.tower')
    a = [p for p in ctx.plugins if p.name == 'd.tower'][0]
    st = a.state()
    D = a.critic_module(ctx, ctx.features.layout())
    res['state_matches_buffers'] = all(torch.equal(st['center'][i], getattr(D.norm, f'center_{i}')) for i in range(len(st['center'])))
    res['info_blocks'] = a.info['blocks']
    assert res['state_matches_buffers'] and len(a.info['blocks']) == 2
    res['PASS'] = True
    save('unit', res)
    log('unit PASS')
    return res


# ============================================================================ penalty
def _penalty_value(pen, D, real, fake, c):
    p = pen(D, real, fake, c)
    return float(p.detach())


def penalty():
    """What the BISPF penalty sees (native penalty objects from trainer.build, float64; the trained round-2
    genemean@0.5 fold-0 seed-0 critic, 32 real train bags and 32 bags of its generator).

    (a) Re-parametrisation: D_eq(x) = W(norm(x)) with first-layer columns multiplied by s_b and the bias
        shifted by W m is the SAME function of x as the released critic W; the native penalty values are
        equal to round-off.  Hence d.tower's / d.sum's internal normalisation leaves the penalty's
        constraint set unchanged.
    (b) x-space scaling: the same function of the gene statistics observed through a gene block scaled by
        a in x has dD/dx_block multiplied by 1/a: the penalty grows as a shrinks (budget is set in x).
    (c) The penalty does not depend on the critic class beyond dD/dx and the logits: a DTowerCritic goes
        through the same native call and returns finite penalties.
    """
    import arm_dcritic as dc
    res = {}
    ctx = ctx_of('genemean@0.5')
    lay = ctx.features.layout()
    center, scale, _ = dc.block_reference_stats(ctx, lay, 'd.tower')
    # the TRAINED round-2 genemean@0.5 fold-0 seed-0 critic and generator (read-only), 32 real/fake bag pairs
    import gm_diagnose as gd
    gd.mods()
    f = [x for x in gd.find_fits([RUNS / 'fits_r2'], ['genemean@0.5'], ('hvg2000',)) if x['fold'] == 0 and x['seed'] == 0 and x['complete']][0]
    fm = gd.FitModels(f, None, ctx)
    Gt, Dt, Pt = fm.load('final')
    real, fake, c, _ = gd.make_bags(ctx, Gt, Pt, n=32)
    trained_state = {k: v.double().clone() for k, v in Dt.state_dict().items()}
    del Gt, Dt, Pt
    res['critic_weights'] = str(f['fit'] / 'final.pt')

    def fresh_pen(c_=ctx):
        """A native penalty and ITS critic (the penalty only accepts the critic paired with its optimizer or a wrapper)."""
        b, *_ = trainer.build(c_, 0)
        return b[6], b[2].double()
    pen1, Wc = fresh_pen()
    with torch.no_grad():
        Wc.load_state_dict(trained_state)
    W_state = {k: v.clone() for k, v in Wc.state_dict().items()}
    # (a) equivalent normalised critic: same released module class, first-layer columns x s_b, bias + W m,
    # wrapped by the fixed normalisation x' = (x - m)/s (the d.tower / d.sum input map)
    pen2, E = fresh_pen()
    with torch.no_grad():
        E.load_state_dict(W_state)
        W0 = E.phi[0].weight
        m_full = torch.cat([m.double() for m in center] + [torch.zeros(7, dtype=torch.float64)])
        s_full = torch.cat([torch.full((b.width,), float(s), dtype=torch.float64) for b, s in zip(lay.blocks, scale)] + [torch.ones(7, dtype=torch.float64)])
        E.phi[0].bias.add_(W0 @ m_full)
        W0.mul_(s_full)

    class Wrap(torch.nn.Module):
        """Wrapper around a penalty-paired critic: D(x, c) = inner(x * a - shift) ... with fixed buffers."""

        def __init__(self, inner, mult, shift):
            super().__init__()
            self.inner = inner
            self.register_buffer('mult', mult)
            self.register_buffer('shift', shift)

        def forward(self, x, c):
            return self.inner(x * self.mult - self.shift, c)
    Q = Wrap(E, 1.0 / s_full[:lay.n_features], m_full[:lay.n_features] / s_full[:lay.n_features])
    with torch.no_grad():
        d_out = float((Wc(real, c) - Q(real, c)).abs().max())
    xg = real.clone().requires_grad_(True)
    g1, = torch.autograd.grad(Wc(xg, c).sum(), xg)
    xg = real.clone().requires_grad_(True)
    g2, = torch.autograd.grad(Q(xg, c).sum(), xg)
    p1 = _penalty_value(pen1, Wc, real, fake, c)
    p2 = _penalty_value(pen2, Q, real, fake, c)
    res['reparam'] = {'max_abs_logit_diff': d_out, 'max_abs_grad_diff': float((g1 - g2).abs().max()),
                      'penalty_released': p1, 'penalty_equivalent_normalised': p2, 'rel_diff': abs(p1 - p2) / abs(p1)}
    assert d_out < 1e-9 and res['reparam']['rel_diff'] < 1e-8, res['reparam']
    # (b) x-space scaling of the gene block by a, same function of the statistics: D_a(x_a) = W(x_a / a)
    lo, hi = lay.block('genemean').lo, lay.block('genemean').hi
    rows = {}
    for a in (2.0, 1.0, 0.5, 0.25):
        sc = torch.ones(lay.n_features, dtype=torch.float64)
        sc[lo:hi] = a
        pen, Wa = fresh_pen()
        with torch.no_grad():
            Wa.load_state_dict(W_state)
        S = Wrap(Wa, 1.0 / sc, torch.zeros_like(sc))
        xr, xf = real * sc, fake * sc
        xg = xr.clone().requires_grad_(True)
        g, = torch.autograd.grad(S(xg, c).sum(), xg)
        rms = float((g.norm(dim=1) / math.sqrt(lay.n_features)).mean())
        rows[str(a)] = {'penalty': _penalty_value(pen, S, xr, xf, c), 'grad_rms_real': rms,
                        'gene_block_grad_sq_share': float((g[:, lo:hi].square().sum(1) / g.square().sum(1)).mean())}
    res['x_scaling_gene_block'] = rows
    assert rows['0.25']['penalty'] > rows['1.0']['penalty'] > rows['2.0']['penalty']
    # (c) DTowerCritic / DSumCritic as built by the trainer, through their own native penalty
    for nm in ('d.tower', 'd.sum'):
        pen, D = fresh_pen(ctx_of('genemean@0.5,' + nm))
        v = _penalty_value(pen, D, real, fake, c)
        xg = real.clone().requires_grad_(True)
        g, = torch.autograd.grad(D(xg, c).sum(), xg)
        res[f'native_penalty_{nm}_at_init'] = {'penalty': v, 'grad_rms_real': float((g.norm(dim=1) / math.sqrt(lay.n_features)).mean()),
                                               'gene_block_grad_sq_share': float((g[:, lo:hi].square().sum(1) / g.square().sum(1)).mean())}
        assert math.isfinite(v)
    pen0, D0 = fresh_pen()
    xg = real.clone().requires_grad_(True)
    g, = torch.autograd.grad(D0(xg, c).sum(), xg)
    res['released_critic_at_init'] = {'grad_rms_real': float((g.norm(dim=1) / math.sqrt(lay.n_features)).mean()),
                                      'gene_block_grad_sq_share': float((g[:, lo:hi].square().sum(1) / g.square().sum(1)).mean())}
    res['trained_fit_penalty_curves'] = penalty_curves()
    res['PASS'] = True
    save('penalty', res)
    log('penalty PASS', res['reparam'], {k: round(v['penalty'], 5) for k, v in rows.items()})
    return res


# ============================================================================ neutral
NEUTRAL_ARMS = {'base': 3, 'genemean': 3, 'pbulk': 3}
NEUTRAL_REF = """
import sys, torch
sys.path.insert(0, {here!r})
import trainer, arms
res = {{}}
for arm, n in {arms_!r}.items():
    ctx = trainer.Context({cache!r}, 'cpu', arms.parse_arm(arm))
    b, g, r, u, s = trainer.build(ctx, 0)
    init = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
    mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
    fin = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
    res[arm] = dict(init=init, final=fin, gaps=[m['critic_gap'] for m in mets], pens=[m['D_penalty'] for m in mets],
                    torch_rng=torch.get_rng_state(), critic=type(b[2]).__name__)
assert 'arm_dcritic' not in sys.modules, 'reference process imported arm_dcritic'
torch.save(res, {out!r})
"""


def neutral():
    import arm_dcritic  # noqa: F401  (registered in this process; the reference process never imports it)
    ref = SCR / 'neutral_ref.pt'
    code = NEUTRAL_REF.format(here=str(HERE), arms_=NEUTRAL_ARMS, cache=str(CACHE0), out=str(ref))
    subprocess.run([PY, '-B', '-c', code], check=True, env=ENV, cwd=str(SCR))
    R = torch.load(ref, weights_only=False)
    res = {}
    for arm, n in NEUTRAL_ARMS.items():
        ctx = ctx_of(arm)
        b, g, r, u, s = trainer.build(ctx, 0)
        init = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in (b[1], b[2], b[3])]
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
        fin = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in (b[1], b[2], b[3])]
        eq = lambda A, B: all(set(a) == set(b_) and all(torch.equal(a[k], b_[k]) for k in a) for a, b_ in zip(A, B))
        row = {'initial_bitwise': eq(init, R[arm]['init']), 'final_bitwise': eq(fin, R[arm]['final']),
               'gaps_equal': [m['critic_gap'] for m in mets] == R[arm]['gaps'], 'penalties_equal': [m['D_penalty'] for m in mets] == R[arm]['pens'],
               'torch_rng_equal': torch.equal(torch.get_rng_state(), R[arm]['torch_rng']), 'critic': [type(b[2]).__name__, R[arm]['critic']]}
        row['PASS'] = all(v for k, v in row.items() if k != 'critic') and row['critic'][0] == row['critic'][1]
        res[arm] = row
        assert row['PASS'], (arm, row)
    ref.unlink()
    res['PASS'] = True
    save('neutral', res)
    log('neutral PASS', res)
    return res


# ============================================================================ tiny fits
FIT_ARMS = ['d.tower', 'd.sum', 'd.h@64', 'd.depth@3',
            'd.tower,genemean@0.5', 'd.sum,genemean@0.5', 'd.h@256,genemean@0.5', 'd.depth@1,genemean@0.5',
            'd.tower,pbulk', 'd.sum,pbulk',
            'd.tower,genemean,pbulk', 'd.sum,genemean,pbulk', 'd.h@64,genemean,pbulk', 'd.tower@64,genemean,pbulk']
FIT_UPDATES = 3


def fits():
    import argparse
    res = {}
    for arm in FIT_ARMS:
        t0 = time.monotonic()
        flags = arms.parse_arm(arm)
        ctx = ctx_of(arm)
        b, g, r, u, s = trainer.build(ctx, 0)
        D, od = b[2], b[5]
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=ctx.topt.steps) for _ in range(FIT_UPDATES)]
        in_opt = {id(p) for gr in od.param_groups for p in gr['params']}
        q = trainer.snapshot(ctx, b, g, r, u, 0, s, 'incomplete', endpoint=ctx.topt.steps, stream_root=trainer.STREAM_ROOT, wall_seconds=0.)
        cont = [trainer.advance(ctx, b, g, r, u, execution_limit=ctx.topt.steps) for _ in range(2)]
        ctx2 = ctx_of(arm)
        b2, g2, r2, u2, _ = trainer.restore(ctx2, q, 0)
        cont2 = [trainer.advance(ctx2, b2, g2, r2, u2, execution_limit=ctx2.topt.steps) for _ in range(2)]
        same = all(torch.equal(x, y) for m1, m2 in zip((b[1], b[2], b[3]), (b2[1], b2[2], b2[3]))
                   for x, y in zip(m1.state_dict().values(), m2.state_dict().values()))
        ns = argparse.Namespace(seed=0, cache=str(CACHE0), updates=20000, device='cpu', checkpoint_every=2000)
        cfg = trainer.make_config(ctx_of(arm), ns, trainer.STREAM_ROOT)
        dflag = [f for f in flags if f.startswith('d.')][0]
        row = {'critic': type(D).__name__, 'n_features': ctx.features.n_features,
               'layout': [b_.name for b_ in ctx.features.layout().blocks],
               'critic_params': sum(p.numel() for p in D.parameters()),
               'all_critic_params_in_optimizer': all(id(p) in in_opt for p in D.parameters()) and len(in_opt) == len(list(D.parameters())),
               'losses_finite': all(math.isfinite(m['D_total']) and math.isfinite(m['G_total']) for m in mets + cont),
               'gaps': [round(m['critic_gap'], 5) for m in mets], 'penalty_grad_rms_step0': mets[0]['penalty_stats'].get('mean_grad_norm'),
               'restore_bitwise': same and [m['critic_gap'] for m in cont] == [m['critic_gap'] for m in cont2],
               'config_critic_module': cfg['critic_module'], 'config_departure': cfg['departures'].get(dflag),
               'seconds': round(time.monotonic() - t0, 1)}
        assert row['all_critic_params_in_optimizer'] and row['losses_finite'] and row['restore_bitwise'], (arm, row)
        assert cfg['critic_module'] == dflag and dflag in cfg['departures'] and cfg['departures'][dflag]['kind'] == 'critic architecture'
        if dflag.startswith(('d.tower', 'd.sum')):
            assert len(cfg['departures'][dflag]['info']['blocks']) == len(row['layout'])
        res[arm] = row
        log('fit', arm, row['critic'], 'params', row['critic_params'], 'gaps', row['gaps'], '%.0fs' % row['seconds'])
    res['PASS'] = True
    save('fits', res)
    return res


def cli():
    """One tiny CLI fit through trainer.py (training, snapshots, generation) into smoke_r3/critic/fits."""
    out_root = SCR / 'fits'
    arm = 'd.tower,pbulk,t.steps@8'
    fit = out_root / arm / 'hvg2000' / 'fold0' / 'seed0'
    if fit.exists():
        shutil.rmtree(fit)
    t0 = time.monotonic()
    subprocess.run([PY, '-B', str(HERE / 'trainer.py'), '--cache', str(CACHE0), '--arm', arm, '--seed', '0',
                    '--out-root', str(out_root), '--device', 'cpu', '--threads', '3'], check=True, env=ENV, cwd=str(SCR))
    cfg = json.loads((fit / 'config.json').read_text())
    gen = np.load(fit / 'generated.npz')
    q = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
    res = {'files': sorted(p.name for p in fit.iterdir()), 'critic_module': cfg['critic_module'],
           'departures': sorted(cfg['departures']), 'tower_blocks': cfg['departures']['d.tower']['info'].get('blocks'),
           'generated_shapes': {k: list(gen[k].shape) for k in ('pb_counts', 'mean_log')},
           'generated_finite': bool(np.isfinite(gen['mean_log']).all()),
           'critic_state_keys': [k for k in q['policy']['models']['critic'] if 'norm' in k][:4],
           'seconds': round(time.monotonic() - t0, 1)}
    assert res['critic_module'] == 'd.tower' and res['generated_finite'] and 'd.tower' in res['departures']
    res['PASS'] = True
    save('cli', res)
    log('cli PASS', res)
    return res


SHORT_ARMS = ['genemean', 'd.tower,genemean', 'd.sum,genemean', 'd.h@64,genemean', 'genemean,pbulk', 'd.tower,genemean,pbulk',
              'd.sum,genemean,pbulk', 'base', 'd.tower']
SHORT_UPDATES = 300


@torch.no_grad()
def _bags(ctx, G, P, n=256):
    import gm_diagnose as gd
    real, fake, c, ds = gd.make_bags(ctx, G.cpu(), P.cpu(), n=n)
    return real, fake, c


def generic_critic_metrics(ctx, D, real, fake, c, ig_steps=16):
    """Module-agnostic block metrics (any critic): gap, RMS input gradient (penalty units), per-block gradient
    share at real/fake bags, integrated-gradient gap attribution per block (fake -> real straight line)."""
    D = D.double().eval()
    lay = ctx.features.layout()
    nf = real.shape[1]
    with torch.no_grad():
        dr, df = D(real, c), D(fake, c)
    gap = float(dr.mean() - df.mean())
    out = {'gap': gap}
    for nm, x in (('real', real), ('fake', fake)):
        xg = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(D(xg, c).sum(), xg)
        rms = g.norm(dim=1) / math.sqrt(nf)
        out[f'grad_rms_{nm}'] = float(rms.mean())
        out[f'frac_above_kappa_{nm}'] = float((rms > 1).double().mean())
        out[f'grad_sq_share_{nm}'] = {b.name: float((g[:, b.lo:b.hi].square().sum(1) / g.square().sum(1)).mean()) for b in lay.blocks}
    delta = real - fake
    acc = torch.zeros_like(real)
    for t in (np.arange(ig_steps) + 0.5) / ig_steps:
        xt = (fake + float(t) * delta).requires_grad_(True)
        gt, = torch.autograd.grad(D(xt, c).sum(), xt)
        acc += gt / ig_steps
    contrib = (acc * delta).mean(0)
    out['gap_ig_share'] = {b.name: float(contrib[b.lo:b.hi].sum()) / gap for b in lay.blocks} if gap else None
    D.float()
    return out


def short(device='cuda'):
    """<= 300-update GPU fits (fold 0, seed 0; one process; --gpu-mem-fraction 0.04): early critic dynamics per arm.

    Only the first 300 of 20,000 updates: the gap of genemean reaches its ~13 plateau within ~4,000 updates, so
    this shows early direction, not the trained regime.  Nothing is written but results/short.json.
    """
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.04)
    res = {}
    for arm in SHORT_ARMS:
        t0 = time.monotonic()
        ctx = trainer.Context(CACHE0, device, arms.parse_arm(arm))
        b, g, r, u, s = trainer.build(ctx, 0)
        curve = []
        for k in range(SHORT_UPDATES):
            m = trainer.advance(ctx, b, g, r, u, execution_limit=ctx.topt.steps)
            curve.append((m['step'], m['critic_gap'], m['D_adversarial'], (m['penalty_stats'] or {}).get('mean_grad_norm')))
        ctx_cpu = trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))
        G, D, P = b[1], b[2], b[3]
        Gc = trainer.make_generator(ctx_cpu, b[0]); Gc.load_state_dict({k_: v.cpu() for k_, v in G.state_dict().items()})
        bc, *_ = trainer.build(ctx_cpu, 0)
        Dc, Pc = bc[2], bc[3]
        Dc.load_state_dict({k_: v.cpu() for k_, v in D.state_dict().items()}); Pc.load_state_dict({k_: v.cpu() for k_, v in P.state_dict().items()})
        Gc.eval(); Pc.eval()
        real, fake, c = _bags(ctx_cpu, Gc, Pc)
        met = generic_critic_metrics(ctx_cpu, Dc, real, fake, c)
        win = lambda lo, hi: float(np.mean([x[1] for x in curve if lo < x[0] <= hi]))
        res[arm] = {'critic': type(D).__name__, 'gap_window_means': {'1-50': win(0, 50), '51-150': win(50, 150), '151-300': win(150, 300)},
                    'D_adversarial_last50': float(np.mean([x[2] for x in curve[-50:]])),
                    'penalty_grad_rms_logged': [(x[0], x[3]) for x in curve if x[3] is not None], 'final_256_bags': met,
                    'seconds': round(time.monotonic() - t0, 1)}
        log('short', arm, res[arm]['gap_window_means'], 'final gap %.2f rmsF %.3f' % (met['gap'], met['grad_rms_fake']),
            'IG', {k_: round(v, 2) for k_, v in (met['gap_ig_share'] or {}).items()}, '%.0fs' % res[arm]['seconds'])
        del b, g, r, u, ctx, bc
        if device == 'cuda':
            torch.cuda.empty_cache()
        save('short', res)   # after every arm
    return res


def cleanup():
    for p in (SCR / 'fits', SCR / 'neutral_ref.pt'):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    log('cleanup: kept', sorted(x.name for x in SCR.rglob('*') if x.is_file()))


# ============================================================================ main
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    SCR.mkdir(parents=True, exist_ok=True)
    if what == 'starve':
        seeds = tuple(int(s) for s in sys.argv[2].split(',')) if len(sys.argv) > 2 else (0,)
        print(json.dumps(starve(seeds), indent=1, default=_js))
    elif what == 'curves':
        print(json.dumps(penalty_curves(), indent=1))
    elif what == 'all':
        for fn in (unit, penalty, neutral, fits, cli, cleanup):
            fn()
    elif what == 'short':
        short(sys.argv[2] if len(sys.argv) > 2 else 'cuda')
    elif what in ('unit', 'penalty', 'neutral', 'fits', 'cli', 'cleanup'):
        globals()[what]()
    else:
        raise SystemExit(f'unknown subcommand {what}')
