"""Leave-donor-out PCA diagnostic: generator under-dispersion vs held-out PCA shrinkage.

The interval targets are within-donor bag SDs of PC coordinates in a basis fit
on the 25 GAN-TRAIN donors' cells (each PC scaled to unit SD over TRAIN cells).
TRAIN donors are therefore in-sample for the PCA, held-out donors are not. This
script refits the same recipe (log CP10k, gene z-score, randomized SVD, 50 PCs,
per-PC TRAIN SD scaling; analysis/improve_20261109/build_cache.py:fit_space)

* leaving each TRAIN donor out in turn and projecting that donor, and
* adding each held-out donor in turn and projecting that donor in-sample,

then reports the mean 64-cell bag SD per PC block, using the same 24 fixed
subsamples as the interval targets. If left-out TRAIN donors shrink like
held-out donors, the held-out deficit on late PCs is consistent with an
out-of-sample PCA effect rather than a donor property. CPU only, read only.

Run rescore_pilot.py first: the generated-vs-real ratios are read from its
rescore.json (--rescore). Without it they are empty and a warning is issued.

    PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
    $PY -B analysis/coverage_calibration_v2_20261120/pca_shrinkage.py [--data-root /ml2/alz]
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.utils.extmath import randomized_svd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R5 = ROOT/'analysis/improve_20261109/r5'
sys.dont_write_bytecode = True
BLOCKS = {'pc01_10': slice(0, 10), 'pc11_30': slice(10, 30), 'pc31_50': slice(30, 50), 'all': slice(0, 50)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_hvg(X):
    X = np.asarray(X, dtype=np.float32)
    return np.log1p(X*(1e4/np.maximum(X.sum(-1, keepdims=True), 1)))


def fit_space(cells, n_pc=50):
    """build_cache.fit_space PCA recipe on a list of per-donor count matrices."""
    X = log_hvg(np.concatenate(cells)); m = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - m)/sd
    sub = np.random.default_rng(0).choice(len(Z), min(20000, len(Z)), replace=False)
    _, _, Vt = randomized_svd(Z[sub], n_components=n_pc, n_iter=5, random_state=0)
    W = Vt.T.copy(); ps = np.maximum((Z @ W).std(0), 1e-6)
    return m.astype(np.float64), sd.astype(np.float64), W.astype(np.float64), ps.astype(np.float64)


def project(counts, space):
    m, sd, W, ps = space
    return ((log_hvg(counts).astype(np.float64) - m)/sd) @ W / ps


def bag_sd(pcs, donor, E, repeats):
    return np.mean([pcs[E.MX.det_subsample(len(pcs), 64, E.MX.stable_seed('pc-summary-rep', donor, r))].std(0)
                    for r in range(repeats)], axis=0)


def blocks(values):
    values = np.asarray(values)
    return {b: float(values[..., s].mean()) for b, s in BLOCKS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default='/ml2/alz', type=Path)
    parser.add_argument('--rescore', default=HERE/'rescore.json', type=Path)
    parser.add_argument('--out', default=HERE/'pca_shrinkage.json', type=Path)
    args = parser.parse_args()
    if not args.rescore.exists():
        warnings.warn(f'{args.rescore} not found; run rescore_pilot.py first. Generated-vs-real '
                      'ratios will be empty in the output.', stacklevel=1)
    sys.path.insert(0, str(R5))
    import evaluate as E
    cache = args.data_root.resolve()/'runs/improve_20261109/caches/hvg2000/fold0'
    real = E.load_real(cache)
    R = E.DEFAULT_PC_REPEATS
    counts = [c.toarray() if hasattr(c, 'toarray') else np.asarray(c) for c in real.cells]
    train = np.flatnonzero(~real.is_heldout); held = np.flatnonzero(real.is_heldout)

    full = fit_space([counts[i] for i in train])
    ref = (real.mean, real.sd, real.W, real.pc_sd)
    reproduction = dict(max_abs_mean=float(np.abs(full[0]-ref[0]).max()),
                        max_abs_sd=float(np.abs(full[1]-ref[1]).max()),
                        max_rel_pc_sd=float(np.abs(full[3]/ref[3]-1).max()),
                        min_abs_loading_cosine=float(np.abs((full[2]*ref[2]).sum(0)).min()))

    cache_basis = {i: bag_sd(real.pcs[i].astype(np.float64), real.donors[i], E, R) for i in range(len(real.donors))}
    loo_train = {}
    for i in train:
        space = fit_space([counts[j] for j in train if j != i])
        loo_train[i] = bag_sd(project(counts[i], space), real.donors[i], E, R)
    in_held = {}
    for i in held:
        space = fit_space([counts[j] for j in train] + [counts[i]])
        in_held[i] = bag_sd(project(counts[i], space), real.donors[i], E, R)

    def mean_of(table, rows):
        return blocks(np.mean([table[i] for i in rows], axis=0))
    real_sd = dict(
        train_in_sample_cache_basis=mean_of(cache_basis, train),
        train_left_out_refit=mean_of(loo_train, train),
        heldout_out_of_sample_cache_basis=mean_of(cache_basis, held),
        heldout_added_in_sample_refit=mean_of(in_held, held))
    per_donor = {str(real.donors[i]): dict(
        split='train' if i in loo_train else 'heldout',
        cache_basis=blocks(cache_basis[i]),
        refit=blocks(loo_train[i] if i in loo_train else in_held[i]))
        for i in range(len(real.donors))}
    generated = {}
    if args.rescore.exists():
        rows = json.loads(args.rescore.read_text())['rows']
        generated = {label: row['generated_bag_sd_by_block'] for label, row in rows.items()}
    ratio = {label: {split: {b: g[split][b]/real_sd[target][b] for b in BLOCKS}
                     for split, target in (('train', 'train_in_sample_cache_basis'),
                                           ('heldout', 'heldout_out_of_sample_cache_basis'))}
             for label, g in generated.items()}
    ratio_vs_left_out = {label: {b: g['train'][b]/real_sd['train_left_out_refit'][b] for b in BLOCKS}
                         for label, g in generated.items()}
    value = dict(
        status='diagnostic_cpu_only', schema=1,
        scope='Fold 0 development cache only; 25 TRAIN and 12 held-out development donors. '
              'Refit PCs are not aligned one-to-one with cache PCs; compare block means, not single PCs.',
        sources=dict(script=sha(__file__), cache_manifest_sha256=sha(cache/'manifest.json')),
        refit_reproduces_cache_basis=reproduction,
        statistic='mean over donors of within-donor 64-cell bag SD (ddof 0), averaged over the 24 fixed '
                  'pc-summary-rep subsamples, per PC block; units are TRAIN-cell PC SDs of the fitting basis',
        real_bag_sd=real_sd, generated_bag_sd=generated,
        generated_over_real=ratio, generated_train_over_left_out_train=ratio_vs_left_out,
        per_donor=per_donor)
    args.out.write_text(json.dumps(value, indent=1, allow_nan=False)+'\n')
    print(json.dumps({k: value[k] for k in ('refit_reproduces_cache_basis', 'real_bag_sd', 'generated_bag_sd',
                                             'generated_over_real', 'generated_train_over_left_out_train')}, indent=1))


if __name__ == '__main__':
    main()
