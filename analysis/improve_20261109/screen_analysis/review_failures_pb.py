"""review_failures_pb.py - independent recomputation of the train/held-out numbers in ANALYSIS_failures
claims 1, 2, 3 (partly), 5, 6 (library ratio), 7, 8 from generated.npz + caches.

Per hvg2000 fit:
  pb_rmse      per donor RMSE over genes of mean_m log1p CP10k(gen pb_m) vs log1p CP10k(real full pb)
  dev_r        Pearson of (pred - train grand) vs (obs - train grand), per donor, averaged
  ml_rmse      RMSE over genes of e_g = mean over split donors of (gen mean_log - real mean_log)
  ml_q4_bias   mean e_g over top quartile of genes by train-donor real mean_log
  lib_ratio    median over all 37 targets of mean generated library / target real mean library
  ident        TRAIN targets: fraction whose pred deviation correlates best with its OWN real deviation
               among all train donors of the same stage (chance = mean 1/n_stage)
  nn_vs_own    HELD-OUT targets: corr(pred dev, real dev of the covariate-nearest same-stage TRAIN donor)
               minus corr(pred dev, own real dev)
  divM         per target: sum over genes of var over M of log1p CP10k(gen pb); ratio to base (same fold,
               seed, target: common random numbers make cell/Poisson noise identical)
Plus per fold: noise floor = RMSE of mean over 32 bootstrap 256-cell bags of the donor's real cells vs its
full pb (what a perfect generator would score under the evaluator's depth asymmetry), and covariate
uniqueness/age range.
Output: runs/.../screen_analysis/review_failures_pb.json
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109/screen_analysis')
import numpy as np

from review_failures_common import ARMS, OUTDIR, fb, load_gen, load_real, logcp


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main():
    per = {}
    folds = {}
    for k in range(3):
        R = load_real(k)
        held = R['held']; tr = ~held
        obs = logcp(R['pb']); grand = obs[tr].mean(0)
        ref = R['ml'][tr].mean(0); q4 = ref >= np.quantile(ref, .75)
        st = R['stage']
        covz = np.c_[R['raw'][:, 4], R['raw'][:, 5], (R['raw'][:, 6] - R['age_mean']) / R['age_sd']]
        # noise floor
        rng = np.random.default_rng(20261110 + k)
        floor = []
        for i, X in enumerate(R['cells']):
            bags = np.stack([X[rng.integers(0, len(X), 256)].sum(0) for _ in range(32)])
            floor.append(np.sqrt(np.mean((logcp(bags).mean(0) - obs[i]) ** 2)))
        floor = np.array(floor)
        tup = [tuple(np.r_[st[i], R['raw'][i, 4:7]]) for i in np.flatnonzero(tr)]
        ages_tr = R['raw'][tr, 6]; ages_h = R['raw'][held, 6]
        folds[k] = {'n_train': int(tr.sum()), 'n_heldout': int(held.sum()),
                    'unique_stage_male_apoe_age_tuples_train': len(set(tup)),
                    'train_age_range': [float(ages_tr.min()), float(ages_tr.max())],
                    'heldout_ages_outside_train_range': [float(a) for a in ages_h if a < ages_tr.min() or a > ages_tr.max()],
                    'heldout_cov_tuple_seen_in_train': int(sum(tuple(np.r_[st[i], R['raw'][i, 4:7]]) in set(tup) for i in np.flatnonzero(held))),
                    'noise_floor_pb_rmse_train': float(floor[tr].mean()), 'noise_floor_pb_rmse_heldout': float(floor[held].mean()),
                    'stage_counts_train': np.bincount(st[tr], minlength=4).tolist(),
                    'stage_counts_heldout': np.bincount(st[held], minlength=4).tolist()}
        # nearest same-stage train donor in covariate space (for held-out targets)
        tri = np.flatnonzero(tr)
        nn = {}
        for i in np.flatnonzero(held):
            cand = tri[st[tri] == st[i]]
            if len(cand):
                nn[i] = cand[np.argmin(((covz[cand] - covz[i]) ** 2).sum(1))]
        base_div = {}
        for arm in ARMS:
            for s in range(2):
                g = load_gen(arm, k, s, donors=list(R['donors']))
                assert (g['is_heldout'].astype(bool) == held).all() and (g['stage'] == st).all()
                gl = np.stack([logcp(g['pb_counts'][t]) for t in range(len(st))])  # [T,M,G]
                pred = gl.mean(1)
                div = gl.var(1, ddof=1).sum(1)
                if arm == 'base':
                    base_div[s] = div
                e = {}
                for nm, msk in (('train', tr), ('heldout', held)):
                    idx = np.flatnonzero(msk)
                    e[f'pb_rmse_{nm}'] = float(np.mean([np.sqrt(np.mean((pred[i] - obs[i]) ** 2)) for i in idx]))
                    e[f'dev_r_{nm}'] = float(np.mean([corr(pred[i] - grand, obs[i] - grand) for i in idx]))
                    eg = (g['mean_log'][idx].astype(np.float64).mean(1) - R['ml'][idx]).mean(0)
                    e[f'ml_rmse_{nm}'] = float(np.sqrt(np.mean(eg ** 2)))
                    e[f'ml_q4_bias_{nm}'] = float(eg[q4].mean())
                    e[f'divM_ratio_to_base_{nm}'] = float(np.mean(div[idx] / base_div[s][idx]))
                e['lib_ratio_median'] = float(np.median(g['lib'].mean(1) / R['lib']))
                # identification on train targets
                hits, chance = [], []
                for i in tri:
                    cand = tri[st[tri] == st[i]]
                    if len(cand) < 2:
                        continue
                    c = [corr(pred[i] - grand, obs[j] - grand) for j in cand]
                    hits.append(cand[int(np.argmax(c))] == i); chance.append(1 / len(cand))
                e['ident_train'] = float(np.mean(hits)); e['ident_chance'] = float(np.mean(chance))
                d = [corr(pred[i] - grand, obs[nn[i]] - grand) - corr(pred[i] - grand, obs[i] - grand) for i in nn]
                e['nn_minus_own_corr_heldout'] = float(np.mean(d))
                e['nn_corr_heldout'] = float(np.mean([corr(pred[i] - grand, obs[nn[i]] - grand) for i in nn]))
                per[(arm, k, s)] = e
                print(k, s, arm, {a: round(b, 3) for a, b in e.items()}, flush=True)
    keys = list(next(iter(per.values())).keys())
    summ = {arm: {key: fb({(k, s): per[(arm, k, s)][key] for k in range(3) for s in range(2)}) for key in keys} for arm in ARMS}
    paired = {}
    for arm in ARMS[1:]:
        paired[arm] = {}
        for nm in ('train', 'heldout'):
            d = [per[(arm, k, s)][f'pb_rmse_{nm}'] - per[('base', k, s)][f'pb_rmse_{nm}'] for k in range(3) for s in range(2)]
            paired[arm][nm] = {'diffs': d, 'mean': float(np.mean(d)), 'n_worse': int(sum(x > 0 for x in d))}
    print(json.dumps(folds, indent=1))
    for arm in ARMS:
        print(arm.ljust(24), {a: round(b, 3) for a, b in summ[arm].items()})
    for arm in paired:
        print(arm.ljust(24), {nm: (round(v['mean'], 3), v['n_worse']) for nm, v in paired[arm].items()})
    (OUTDIR / 'review_failures_pb.json').write_text(json.dumps(
        {'folds': folds, 'fold_balanced': summ, 'paired_vs_base': paired,
         'per_fit': {f'{a}|{k}|{s}': v for (a, k, s), v in per.items()}}, indent=1))


if __name__ == '__main__':
    main()
