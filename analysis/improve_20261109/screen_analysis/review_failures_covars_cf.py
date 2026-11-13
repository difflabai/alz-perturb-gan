"""review_failures_covars_cf.py - covariate counterfactuals for the covars arms (CPU generation, frozen fits).

For each held-out donor t of each covars / covars,genemean,offset hvg2000 fit, generate M=32 donors x K=256 cells
(libraries drawn from t's own cells) under four condition choices (stage is always t's own):
  own      t's own (male, APOE4, standardised age)                        -> should reproduce evaluator pb_rmse
  nearest  covariates of the covariate-nearest SAME-STAGE TRAIN donor     -> 'donor lookup' if close to own
  mixed    each generated donor m gets the covariates of a random same-stage train donor (marginalise covariates)
  mean     train-mean male/APOE4 and age z=0 (an off-support point: 'Male' ~0.5)
and also (train targets) own vs mixed, to see how much of the train fit depends on the exact donor key.
Same random streams for all four choices within a target (only the condition changes).
Metric: pb_rmse per donor (RMSE over genes, mean_m log1p CP10k(gen pb) vs log1p CP10k(full real pb)).
Output: runs/.../screen_analysis/review_failures_covars_cf.json
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

import trainer
from review_failures_common import CACHES, FITS, OUTDIR, load_real, logcp, fb

torch.set_num_threads(6)
M, K = 32, 256


@torch.no_grad()
def gen(G, P, C, pool, seed):
    torch.manual_seed(seed)
    lat = torch.Generator().manual_seed(seed + 1); cell = torch.Generator().manual_seed(seed + 2)
    cnt = torch.Generator().manual_seed(seed + 3)
    libs = torch.as_tensor(np.random.default_rng(seed + 4).choice(pool, size=(M, K), replace=True).astype(np.float32))
    out = []
    for a in range(0, M, 8):
        z, _ = P.sample(8, generator=lat)
        rate = G.rates(z, torch.as_tensor(C[a:a + 8], dtype=torch.float32), K, libs[a:a + 8], cell)
        out.append(torch.poisson(rate, generator=cnt).sum(1).double().numpy())
    return np.concatenate(out)


def main():
    res = {}
    for arm in ['covars', 'covars,genemean,offset']:
        flags = tuple(arm.split(','))
        for k in range(3):
            R = load_real(k)
            ctx = trainer.Context(CACHES / 'hvg2000' / f'fold{k}', 'cpu', flags)
            obs = logcp(R['pb'])
            held = R['held']; tr = ~held; st = R['stage']
            cond = trainer.build_conditions(R['raw'], flags, ctx.age_mean, ctx.age_sd)  # train age mean/sd
            tri = np.flatnonzero(tr)
            for s in range(2):
                fd = FITS / arm / 'hvg2000' / f'fold{k}' / f'seed{s}'
                q = torch.load(fd / 'final.pt', map_location='cpu', weights_only=False)
                b, *_ = trainer.build(ctx, s, stream_root=q['stream_root'])
                G, P = b[1], b[3]
                G.load_state_dict(q['policy']['models']['generator']); P.load_state_dict(q['policy']['models']['prior'])
                G.eval(); P.eval()
                rng = np.random.default_rng(4242 + k)
                rows = {'own': [], 'nearest': [], 'mixed': [], 'mean': [], 'train_own': [], 'train_mixed': []}
                for i in range(len(st)):
                    cand = tri[st[tri] == st[i]]
                    others = cand[cand != i]
                    if len(others) == 0:
                        continue
                    pool = R['cells'][i].sum(1)
                    seed = 900000 + 1000 * k + i
                    def score(C):
                        return float(np.sqrt(np.mean((logcp(gen(G, P, C, pool, seed)).mean(0) - obs[i]) ** 2)))
                    own = np.repeat(cond[i][None], M, 0)
                    mixed = cond[rng.choice(others, M)]
                    if held[i]:
                        d2 = ((cond[cand][:, 4:] - cond[i][4:]) ** 2).sum(1)
                        near = np.repeat(cond[cand[np.argmin(d2)]][None], M, 0)
                        mean = own.copy(); mean[:, 4:] = cond[tri][:, 4:].mean(0)
                        rows['own'].append(score(own)); rows['nearest'].append(score(near))
                        rows['mixed'].append(score(mixed)); rows['mean'].append(score(mean))
                    else:
                        rows['train_own'].append(score(own)); rows['train_mixed'].append(score(mixed))
                res[(arm, k, s)] = {a: float(np.mean(v)) for a, v in rows.items()}
                res[(arm, k, s)]['n_heldout_scored'] = len(rows['own'])
                print(arm, k, s, {a: round(v, 3) for a, v in res[(arm, k, s)].items()}, flush=True)
    summ = {arm: {key: fb({(k, s): res[(arm, k, s)][key] for k in range(3) for s in range(2)})
                  for key in ['own', 'nearest', 'mixed', 'mean', 'train_own', 'train_mixed']} for arm in ['covars', 'covars,genemean,offset']}
    print(json.dumps(summ, indent=1))
    (OUTDIR / 'review_failures_covars_cf.json').write_text(json.dumps({'per_fit': {f'{a}|{k}|{s}': v for (a, k, s), v in res.items()},
                                                                        'fold_balanced': summ}, indent=1))


if __name__ == '__main__':
    main()
