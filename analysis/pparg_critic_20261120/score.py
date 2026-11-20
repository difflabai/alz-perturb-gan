"""Score the declared fold-0 comparison (PROTOCOL.md): offpc,pbulk vs native pbulk control.

python -B score.py --seeds 0 1 --updates U [--out results.json]

Held-out development donors only.  Frozen metrics: analysis/improve_20261109/r5/architecture_score.py
(historical headline + percentile interval summaries).  Per-donor quantities for paired donor
bootstrap CIs are computed here with the same definitions (mean_log per target averaged over the
M generated donors; interval_metrics.bounds / interval_score on PC bag summaries).
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '3')
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R5 = HERE.parents[0] / 'improve_20261109/r5'
RUN = Path(os.environ.get('PPARG_CRITIC_RUN', '/ml2/alz/runs/research_20261120/pparg-critic'))
CACHE = Path('/ml2/alz/runs/improve_20261109/caches/hvg2000/fold0')
ARMS = {'control': 'pbulk', 'variant': 'offpc+pbulk'}
BOOT_REPS = 10000
BOOT_SEED = 20261120


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_bootstrap(x, y, reps=BOOT_REPS, seed=BOOT_SEED, fn=np.mean):
    """Donor bootstrap of fn(x) - fn(y) for paired per-donor values; percentile 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('paired 1-d donor vectors required')
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(reps, len(x)))
    d = np.array([fn(x[i]) - fn(y[i]) for i in idx])
    return dict(estimate=float(fn(x) - fn(y)), lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                frac_below_zero=float((d < 0).mean()), reps=int(reps), seed=int(seed), n_donors=int(len(x)))


def rmse(v):
    return float(np.sqrt(np.mean(v)))


def _modules():
    sys.path.insert(0, str(R5))
    import architecture_score as AS
    import interval_metrics as I
    import baselines as BL
    return AS, I, BL


def per_donor(gen_path, real, held, block, pparg, AS, I, observed):
    gen = AS.E.align_generated(AS.E.load_generated(gen_path), real)
    ml = gen['mean_log'].astype(np.float64).mean(1)                 # [T, G]
    err = ml - real.mean_log
    Y = AS.E.MX.pc_bag_summary(gen['pcs'].astype(np.float32))       # [T, M, 100]
    lo, hi = I.bounds(Y, 'percentile')
    out = {}
    for name, sl in (('sd', slice(50, 100)), ('combined', slice(0, 100)), ('mean', slice(0, 50))):
        y = observed[:, :, sl]
        isc = I.interval_score(lo[:, sl], hi[:, sl], y)            # [R, T, D]
        cov = (y >= lo[:, sl]) & (y <= hi[:, sl])
        out[f'{name}_interval_score'] = isc.mean((0, 2))[held]
        out[f'{name}_coverage'] = cov.mean((0, 2))[held]
    out['pparg_sqerr'] = err[held, pparg] ** 2
    out['pparg_signed'] = err[held, pparg]
    out['block_mse'] = (err[held][:, block] ** 2).mean(1)
    out['block_nonpparg_mse'] = (err[held][:, [b for b in block if b != pparg]] ** 2).mean(1)
    out['allgene_meanlog_mse'] = (err[held] ** 2).mean(1)
    out['pparg_gen'] = ml[held, pparg]
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--out', default=str(HERE / 'results.json'))
    a = p.parse_args(argv)
    AS, I, BL = _modules()
    sel = json.loads((HERE / 'selected_genes.json').read_text())
    real = AS.E.load_real(CACHE)
    genes = list(real.genes)
    block = [genes.index(g) for g in sel['genes']]
    pparg = genes.index('PPARG')
    held = np.flatnonzero(real.is_heldout)
    observed = AS.observed_summaries(real, 64)
    scores_dir = RUN / 'scores'
    # empirical references (stage-blind = required reference; stage-aware shown too)
    refs = {}
    for label in ('resample_blind', 'resample'):
        arc = RUN / 'empirical' / f'{label}_fold0.npz'
        if not arc.exists():
            g, _ = BL.generate(label, real, M=32, K=256, kpc=64)
            AS.E.save_npz(arc, **g)
        frozen = AS.score(arc, CACHE, scores_dir / f'{label}.json')
        refs[label] = dict(frozen=frozen, donors=per_donor(arc, real, held, block, pparg, AS, I, observed), archive_sha256=sha(arc))
    rows = {}
    for readout in ('app', 'raw'):
        for arm, armdir in ARMS.items():
            for s in a.seeds:
                arc = RUN / 'generated' / armdir / 'fold0' / f'seed{s}' / f'{readout}.npz'
                frozen = AS.score(arc, CACHE, scores_dir / f'{armdir}_seed{s}_{readout}.json')
                rows[(readout, arm, s)] = dict(frozen=frozen, donors=per_donor(arc, real, held, block, pparg, AS, I, observed),
                                               archive_sha256=sha(arc))
    keys = ['pparg_sqerr', 'block_mse', 'block_nonpparg_mse', 'allgene_meanlog_mse',
            'sd_interval_score', 'combined_interval_score', 'sd_coverage', 'combined_coverage', 'mean_coverage']
    root = {'rmse': ('pparg_sqerr', 'block_mse', 'block_nonpparg_mse', 'allgene_meanlog_mse')}
    result = dict(protocol='analysis/pparg_critic_20261120/PROTOCOL.md', fold=0, seeds=a.seeds, updates=a.updates,
                  heldout_donors=[str(real.donors[i]) for i in held], heldout_stage=[int(real.stage[i]) for i in held],
                  genes=sel['genes'], readouts={})
    for readout in ('app', 'raw'):
        R = {}
        pooled = {arm: {k: np.mean([rows[(readout, arm, s)]['donors'][k] for s in a.seeds], 0) for k in keys + ['pparg_signed', 'pparg_gen']}
                  for arm in ARMS}
        comp = {}
        for k in keys:
            fn = rmse if k in root['rmse'] else np.mean
            comp[k] = dict(variant=float(fn(pooled['variant'][k])), control=float(fn(pooled['control'][k])),
                           empirical_blind=float(fn(refs['resample_blind']['donors'][k])),
                           empirical_stage=float(fn(refs['resample']['donors'][k])),
                           diff_variant_minus_control=paired_bootstrap(pooled['variant'][k], pooled['control'][k], fn=fn),
                           per_seed={str(s): dict(variant=float(fn(rows[(readout, 'variant', s)]['donors'][k])),
                                                  control=float(fn(rows[(readout, 'control', s)]['donors'][k])))
                                     for s in a.seeds})
        R['paired'] = comp
        R['pparg_by_donor'] = dict(real=[float(real.mean_log[i, pparg]) for i in held],
                                   control=pooled['control']['pparg_gen'].tolist(), variant=pooled['variant']['pparg_gen'].tolist(),
                                   empirical_blind=refs['resample_blind']['donors']['pparg_gen'].tolist())
        hist = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep',
                'pc_coverage80_rep', 'pc_spread_rep')
        R['frozen'] = {}
        for label, getter in [(f'{arm}_seed{s}', (lambda arm=arm, s=s: rows[(readout, arm, s)]['frozen'])) for arm in ARMS for s in a.seeds] + \
                             [(lab, (lambda lab=lab: refs[lab]['frozen'])) for lab in refs]:
            f = getter()
            R['frozen'][label] = dict(historical={h: f['historical'][h] for h in hist},
                                      heldout_intervals={g: {m: f['intervals']['heldout']['percentile'][g][m]
                                                             for m in ('coverage', 'width', 'interval_score')}
                                                         for g in ('mean', 'sd', 'combined')})
        R['archives_sha256'] = {f'{arm}_seed{s}': rows[(readout, arm, s)]['archive_sha256'] for arm in ARMS for s in a.seeds}
        result['readouts'][readout] = R
    result['empirical_archives_sha256'] = {k: v['archive_sha256'] for k, v in refs.items()}
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for readout in ('app', 'raw'):
        print(readout)
        for k, v in result['readouts'][readout]['paired'].items():
            d = v['diff_variant_minus_control']
            print(f"  {k:26s} var {v['variant']:.4f} ctl {v['control']:.4f} blind {v['empirical_blind']:.4f}  diff {d['estimate']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]")


if __name__ == '__main__':
    main()
