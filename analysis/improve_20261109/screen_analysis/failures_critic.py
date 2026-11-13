"""failures_critic.py - donorpc and covars,genemean,offset: what the critic sees and where the gene-mean error goes.

Per hvg2000 fit (base, genemean, donorpc, covars,genemean,offset; 3 folds x 2 seeds):

A. Feature scaling on 192 REAL train bags (128 cells, uniform donors, fixed rng): mean squared norm of
   the 150 released features and of the added block; for donorpc the per-PC split of bag-level variance
   into between-donor and within-donor (bag sampling) parts, from 8 bags of each train donor.
B. Critic reliance: first critic layer phi[0] column norms times the per-column SD of the real input,
   summed (in quadrature) per block (150 released / added block / 7 condition slots); and the mean
   |d critic / d input| block norms on the same real bags.
C. Gene-mean error geometry from generated.npz (TRAIN targets, so no held-out issue): per-gene mean
   error of mean per-cell log1p CP10k, e_g = mean_t (gen - real), in the critic's standardised units
   e_g / sd_g (exactly what mean_cells(F) carries), and the fraction of ||e/sd||^2 inside
   (i) the donorpc subspace (top-16 donor PCs of the fold, recomputed here exactly as arms._fit_donorpc),
   (ii) the 50-PC span W the released features see.  Also the raw RMS of e and its Q4 (top
   expression quartile) bias.
Writes runs/.../screen_analysis/failures_critic.json.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

import failures_common as fc
import arms

OUT = Path('/ml2/alz/runs/improve_20261109/screen_analysis/failures_critic.json')
ARMS = ['base', 'genemean', 'donorpc', 'covars,genemean,offset']


def donor_subspace(ctx_dp):
    return ctx_dp.features.pc_V.double().numpy()


def part_A_B(ctx, D):
    feats = ctx.features
    rng = np.random.default_rng(99)
    ds = rng.integers(len(ctx.by), size=192)
    idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]))
    x = feats.real(idx).detach()
    c = ctx.conditions[torch.as_tensor(ds)]
    nb = 150
    res = {'n_features': int(x.shape[1]),
           'released_sqnorm': float(x[:, :nb].square().sum(1).mean()),
           'added_sqnorm': float(x[:, nb:].square().sum(1).mean()) if x.shape[1] > nb else 0.0}
    W = D.phi[0].weight.detach()
    xin = torch.cat([x, c], 1)
    sd = xin.std(0)
    contrib = W.norm(dim=0) * sd
    blocks = {'released150': slice(0, nb), 'added': slice(nb, x.shape[1]), 'cond7': slice(x.shape[1], x.shape[1] + 7)}
    res['weight_x_sd_block'] = {k: float(contrib[v].square().sum().sqrt()) for k, v in blocks.items()}
    xr = x.clone().requires_grad_(True)
    D.requires_grad_(False)
    out = D(xr, c).sum()
    g, = torch.autograd.grad(out, xr)
    res['input_grad_block_norm'] = {k: float(g[:, v].norm(dim=1).mean()) for k, v in blocks.items() if k != 'cond7'}
    res['input_grad_x_sd_block'] = {k: float((g[:, v].abs().mean(0) * sd[v]).square().sum().sqrt()) for k, v in blocks.items() if k != 'cond7'}
    if feats.kind == 'donorpc':
        bags = []
        for i in range(len(ctx.by)):
            r2 = np.random.default_rng(1000 + i)
            ii = torch.as_tensor(np.stack([r2.choice(ctx.by[i], 128, replace=True) for _ in range(8)]))
            bags.append(feats.real(ii)[:, nb:].detach().double().numpy())
        bags = np.stack(bags)  # [n_donors, 8, K]
        within = bags.var(1, ddof=1).mean(0)
        between = bags.mean(1).var(0, ddof=1)
        res['donorpc_within_bag_var'] = within.tolist()
        res['donorpc_between_donor_var'] = between.tolist()
        res['donorpc_within_fraction'] = (within / (within + between)).tolist()
    return res


def part_C(arm, k, s, ctx_any, V):
    g = np.load(fc.fit_path(arm, k, s) / 'generated.npz')
    rf_ml, rf_donors = REAL[k]
    tg = list(g['targets'])
    tr = [i for i, d in enumerate(tg) if not g['is_heldout'][i]]
    ho = [i for i, d in enumerate(tg) if g['is_heldout'][i]]
    out = {}
    sdv = ctx_any.sd.double().numpy()
    Wm = ctx_any.W.double().numpy()
    Qw, _ = np.linalg.qr(Wm)
    for name, sel in [('train', tr), ('heldout', ho)]:
        gen = g['mean_log'][sel].astype(np.float64).mean(1)
        real = np.stack([rf_ml[rf_donors.index(tg[i])] for i in sel])
        e = (gen - real).mean(0)
        es = e / sdv
        tot = float(es @ es)
        inV = float(np.sum((es @ V) ** 2))
        inW = float(np.sum((es @ Qw) ** 2))
        meanexpr = real.mean(0)
        q4 = meanexpr >= np.quantile(meanexpr, 0.75)
        out[name] = {'rmse_mean_error': float(np.sqrt((e ** 2).mean())), 'bias': float(e.mean()), 'Q4_bias': float(e[q4].mean()),
                     'std_err_sqnorm': tot, 'frac_in_donorpc_span': inV / tot, 'frac_in_W50_span': inW / tot}
    return out


REAL = {}


def main():
    out = {}
    for k in range(3):
        rf = fc.real_fold(k)
        REAL[k] = (rf.mean_log, list(rf.donors))
        del rf
        ctx_dp = None
        for arm in ARMS:
            ctx = None
            for s in range(2):
                ctx, G, D, P, q, cfg = fc.load_fit(arm, k, s, ctx=ctx)
                if arm == 'donorpc':
                    ctx_dp = ctx
                d = part_A_B(ctx, D)
                out[f'{arm}|{k}|{s}'] = d
            if ctx_dp is None:
                ctx_dp = fc.trainer.Context(fc.CACHES / 'hvg2000' / f'fold{k}', 'cpu', ('donorpc',))
            V = donor_subspace(ctx_dp)
            for s in range(2):
                out[f'{arm}|{k}|{s}']['gene_mean_error'] = part_C(arm, k, s, ctx_dp, V)
                print(arm, k, s, json.dumps(out[f'{arm}|{k}|{s}'])[:2000], flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print('\nsummary (mean over 6 fits)')
    for arm in ARMS:
        L = [out[f'{arm}|{k}|{s}'] for k in range(3) for s in range(2)]
        print(arm, 'released_sqnorm %.1f added_sqnorm %.1f' % (np.mean([x['released_sqnorm'] for x in L]), np.mean([x['added_sqnorm'] for x in L])),
              'Wxsd', {b: round(float(np.mean([x['weight_x_sd_block'][b] for x in L])), 3) for b in L[0]['weight_x_sd_block']},
              'grad x sd', {b: round(float(np.mean([x['input_grad_x_sd_block'][b] for x in L])), 4) for b in L[0]['input_grad_x_sd_block']})
        for sp in ['train', 'heldout']:
            print('   ', sp, {m: round(float(np.mean([x['gene_mean_error'][sp][m] for x in L])), 4) for m in L[0]['gene_mean_error'][sp]})
        if arm == 'donorpc':
            print('    within-donor fraction of bag variance per donor PC', np.round(np.mean([x['donorpc_within_fraction'] for x in L], 0), 2).tolist())
    print('wrote', OUT)


if __name__ == '__main__':
    main()
