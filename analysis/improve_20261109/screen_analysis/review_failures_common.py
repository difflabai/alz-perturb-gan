"""review_failures_common.py - independent helpers for the adversarial review of ANALYSIS_failures.

Written from scratch: reads caches (scipy CSR + metadata npz) and generated.npz directly,
does NOT import evaluate.py / metrics.py / failures_*.py.  CPU only, development donors only
(the caches contain only the fold's 37 development donors).
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '6'

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

RUNS = Path('/ml2/alz/runs/improve_20261109')
FITS = RUNS / 'fits'
CACHES = RUNS / 'caches'
OUTDIR = RUNS / 'screen_analysis'
ARMS = ['base', 'offset', 'genemean', 'donorpc', 'covars', 'covars,genemean,offset']
DEV = set(json.loads(Path('/ml2/alz/analysis/improve_20261109/folds.json').read_text())['development_donors'])


def logcp(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1.0)))


def load_real(k, panel='hvg2000'):
    """Per donor (train then heldout, cache order): full pb counts, mean per-cell log1p CP10k,
    mean library, raw_c, stage, is_heldout, and the cell count matrix (dense float32)."""
    cd = CACHES / panel / f'fold{k}'
    out = {'donors': [], 'held': [], 'raw': [], 'pb': [], 'ml': [], 'lib': [], 'cells': []}
    for split in ('train', 'heldout'):
        m = np.load(cd / split / 'metadata.npz', allow_pickle=False)
        ds = m['donors'].astype(str)
        assert set(ds) <= DEV, 'non-development donor in cache'
        lab = m['donor'].astype(str)
        C = sp.load_npz(cd / split / 'counts.npz').tocsr()
        for j, d in enumerate(ds):
            X = C[np.flatnonzero(lab == d)].toarray().astype(np.float64)
            out['donors'].append(d); out['held'].append(split == 'heldout')
            out['raw'].append(m['raw_c'][j].astype(np.float64))
            out['pb'].append(X.sum(0)); out['ml'].append(logcp(X).mean(0)); out['lib'].append(X.sum(1).mean())
            out['cells'].append(X.astype(np.float32))
    r = {k2: (np.array(v) if k2 != 'cells' else v) for k2, v in out.items()}
    r['stage'] = np.rint(r['raw'][:, 3] * 3).astype(int)
    sp_ = np.load(cd / 'space.npz')
    r['age_mean'] = float(sp_['age_mean']); r['age_sd'] = float(sp_['age_sd'])
    return r


def load_gen(arm, k, s, panel='hvg2000', donors=None):
    f = np.load(FITS / arm / panel / f'fold{k}' / f'seed{s}' / 'generated.npz', allow_pickle=False)
    g = {x: f[x] for x in ('pb_counts', 'mean_log', 'lib', 'is_heldout', 'stage')}
    g['targets'] = f['targets'].astype(str)
    if donors is not None:
        order = np.array([list(g['targets']).index(d) for d in donors])
        for x in ('pb_counts', 'mean_log', 'lib', 'is_heldout', 'stage', 'targets'):
            g[x] = g[x][order]
    return g


def fb(vals):
    """vals: dict (k,s)->float. Seeds averaged within fold, then mean over folds."""
    folds = [np.mean([vals[(k, s)] for s in range(2) if (k, s) in vals]) for k in range(3)]
    return float(np.mean(folds))
