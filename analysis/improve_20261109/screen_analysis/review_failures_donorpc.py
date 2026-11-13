"""review_failures_donorpc.py - donorpc / genemean critic-feature scaling and critic reliance (CPU, frozen fits).

Per fold, from the fold's TRAIN cells only (trainer.Context on CPU, arm features recomputed from the cache):
  * donorpc block: for 64 real 128-cell bags per train donor, the 16 donorpc features.  Per PC: within-donor
    (bag sampling) variance vs between-donor variance of the donor-mean features -> share of the feature's
    variance that is sampling noise.  Also checks the recomputed (P-mu)V/sd against the checkpoint arm_state.
  * block magnitudes: E||f150||^2 (released bag features) vs E||extra||^2 on the same real bags.
  * critic reliance: for the frozen final critic of each fit (base, genemean, donorpc, covars,genemean,offset),
    variance over real bags of the first-layer pre-activation contributed by each input block
    (x_block @ W_block^T, summed over the 128 hidden units), as a share of the total.
  * gene-mean error span: e = mean over train targets of (gen mean_log - real mean_log)/sd_g (fold space sd);
    share of ||e||^2 inside the 16-dim donor-PC span V, and inside the 50-dim W span (column space of W).
Output: runs/.../screen_analysis/review_failures_donorpc.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '6'
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109')
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109/screen_analysis')
import numpy as np
import torch

import trainer  # read-only import (sets the pinned release path); nothing is written
from review_failures_common import CACHES, FITS, OUTDIR, load_gen, load_real

torch.set_num_threads(6)
NB = 64


def main():
    out = {}
    for k in range(3):
        cache = CACHES / 'hvg2000' / f'fold{k}'
        ctx = trainer.Context(cache, 'cpu', ('donorpc',))
        f = ctx.features
        q = torch.load(FITS / 'donorpc' / 'hvg2000' / f'fold{k}' / 'seed0' / 'final.pt', map_location='cpu', weights_only=False)
        st = q['arm_state']['features']
        same = all(torch.allclose(getattr(f, n).float(), st[n].float(), atol=1e-4) for n in ('pc_mu', 'pc_sd'))
        Vabs = torch.allclose(f.pc_V.abs(), st['pc_V'].abs(), atol=1e-4)
        rng = np.random.default_rng(777 + k)
        feats, donor = [], []
        for i, rows in enumerate(ctx.by):
            idx = torch.as_tensor(np.stack([rng.choice(rows, 128, replace=True) for _ in range(NB)]))
            with torch.no_grad():
                feats.append(f.real(idx).double().numpy()); donor += [i] * NB
        F = np.concatenate(feats); donor = np.array(donor)
        ex = F[:, 150:]
        dm = np.stack([ex[donor == i].mean(0) for i in range(len(ctx.by))])
        within = np.mean([ex[donor == i].var(0, ddof=1) for i in range(len(ctx.by))], 0)
        between = dm.var(0, ddof=0)
        e = {'checkpoint_arm_state_matches_recompute': bool(same and Vabs),
             'pc_sd': f.pc_sd.numpy().tolist(),
             'within_donor_bag_var_per_pc': within.tolist(), 'between_donor_var_per_pc': between.tolist(),
             'noise_share_per_pc': (within / (within + between)).tolist(),
             'E_norm2_base150': float((F[:, :150] ** 2).sum(1).mean()), 'E_norm2_donorpc16': float((ex ** 2).sum(1).mean())}
        # genemean block magnitude on the same bags (scale from the genemean config of this fold)
        sgm = json.loads((FITS / 'genemean' / 'hvg2000' / f'fold{k}' / 'seed0' / 'config.json').read_text())['arm_info']['genemean_scale']
        Zm = []
        rng = np.random.default_rng(778 + k)
        for i, rows in enumerate(ctx.by):
            idx = torch.as_tensor(np.stack([rng.choice(rows, 128, replace=True) for _ in range(8)]))
            Zm.append(ctx.Z[idx].mean(1).double().numpy())
        Zm = np.concatenate(Zm)
        e['E_norm2_genemean2000'] = float(sgm ** 2 * (Zm ** 2).sum(1).mean())
        # critic reliance per block, on the donorpc real bags (first 150 cols shared by all arms)
        conds = ctx.conditions.double().numpy()[donor]
        rel = {}
        for arm in ['base', 'donorpc', 'genemean', 'covars', 'covars,genemean,offset']:
            rel[arm] = []
            for s in range(2):
                D = torch.load(FITS / arm / 'hvg2000' / f'fold{k}' / f'seed{s}' / 'final.pt', map_location='cpu', weights_only=False)['policy']['models']['critic']
                W = D['phi.0.weight'].double().numpy()
                nfeat = W.shape[1] - 7
                if arm == 'donorpc':
                    X = F
                elif nfeat == 150:
                    X = F[:, :150]
                else:  # genemean arms: rebuild the block for these bags would need ctx.Z of same idx; use Zm-sized subset
                    X = None
                if X is None:
                    rng2 = np.random.default_rng(779 + k)
                    rows_ = []
                    xs = []
                    for i, rows in enumerate(ctx.by):
                        idx = torch.as_tensor(np.stack([rng2.choice(rows, 128, replace=True) for _ in range(8)]))
                        with torch.no_grad():
                            base = f.bag_features(ctx.trainpcs[idx]).double().numpy()
                        xs.append(np.c_[base, sgm * ctx.Z[idx].mean(1).double().numpy()]); rows_ += [i] * 8
                    X = np.concatenate(xs); cc = ctx.conditions.double().numpy()[np.array(rows_)]
                    if 'covars' in arm:
                        cc = trainer.build_conditions(ctx.raw_c, ('covars',), ctx.age_mean, ctx.age_sd)[np.array(rows_)]
                else:
                    cc = conds if 'covars' not in arm else trainer.build_conditions(ctx.raw_c, ('covars',), ctx.age_mean, ctx.age_sd)[donor]
                blocks = {'base150': slice(0, 150), 'extra': slice(150, nfeat), 'cond': slice(nfeat, nfeat + 7)}
                Xa = np.c_[X, cc]
                v = {}
                for b, sl in blocks.items():
                    if sl.stop <= sl.start:
                        continue
                    h = Xa[:, sl] @ W[:, sl].T
                    v[b] = float(h.var(0).sum())
                tot = sum(v.values())
                rel[arm].append({b: x / tot for b, x in v.items()})
        e['critic_first_layer_variance_share'] = rel
        # gene-mean error span (train targets), for base / donorpc / genemean
        R = load_real(k)
        tr = ~R['held']
        sd = ctx.sd.double().numpy()
        Wsp = np.load(cache / 'space.npz')['W'].astype(np.float64)
        Qw, _ = np.linalg.qr(Wsp)
        V = st['pc_V'].double().numpy()
        Qv, _ = np.linalg.qr(V)
        span = {}
        for arm in ['base', 'donorpc', 'genemean']:
            span[arm] = []
            for s in range(2):
                g = load_gen(arm, k, s, donors=list(R['donors']))
                err = ((g['mean_log'][tr].astype(np.float64).mean(1) - R['ml'][tr]) / sd).mean(0)
                n2 = err @ err
                span[arm].append({'share_in_donorpc_span': float(((Qv.T @ err) ** 2).sum() / n2),
                                  'share_in_W50_span': float(((Qw.T @ err) ** 2).sum() / n2),
                                  'norm2': float(n2)})
        e['gene_mean_error_span_train'] = span
        e['random_subspace_expectation'] = {'16': 16 / ctx.n_genes, '50': 50 / ctx.n_genes}
        out[k] = e
        print(k, json.dumps({a: (np.round(b, 3).tolist() if isinstance(b, list) and b and isinstance(b[0], float) else b) for a, b in e.items()}), flush=True)
    (OUTDIR / 'review_failures_donorpc.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
