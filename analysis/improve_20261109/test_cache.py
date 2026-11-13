"""Tests for the improve_20261109 Interface 1 caches (component: cache).

Run: CUDA_VISIBLE_DEVICES='' /mnt/nvme/ai/conda/anaconda3/bin/python3 -B test_cache.py
Reads only files under /ml2/alz/runs/improve_20261109/caches, folds.json and the
gene axis file; never opens donor count files.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import hashlib, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

HERE = Path('/ml2/alz/analysis/improve_20261109')
ROOT = Path('/ml2/alz/runs/improve_20261109/caches')
GENES_FILE = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt/counts/genes.npy')
FOLDS = json.loads((HERE / 'folds.json').read_text())
DEV = set(FOLDS['development_donors']); SEALED = set(FOLDS['sealed_evaluation_donors_fold0'])
AXIS = np.load(GENES_FILE, allow_pickle=False)
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg); print('FAIL', msg, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def log_cp10k(C):
    X = C.toarray().astype(np.float32)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


def test_guard():
    sys.path.insert(0, str(HERE))
    import build_cache as bc
    for d in sorted(SEALED) + ['H19.33.004', 'H20.33.002', 'not-a-donor']:
        try:
            bc.check_donor_allowed(d); check(False, f'guard accepted forbidden donor {d}')
        except PermissionError:
            pass
    for d in sorted(DEV):
        bc.check_donor_allowed(d)
    try:
        bc.read_donors(['H20.33.001', 'H21.33.006']); check(False, 'read_donors accepted a sealed donor')
    except PermissionError:
        pass
    check(bc.COUNT_FILES_OPENED == [], 'guard test opened a count file')
    print('guard ok', flush=True)


def test_cache(panel, k):
    d = ROOT / panel / f'fold{k}'; tag = f'{panel}/fold{k}'
    man = json.loads((d / 'manifest.json').read_text()); f = FOLDS['inner_folds'][k]
    for rel, h in man['files_SHA256'].items():
        check(sha(d / rel) == h, f'{tag} sha {rel}')
    present = {str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()} - {'manifest.json'}
    check(present == set(man['files_SHA256']), f'{tag} file set differs from manifest')
    tr, he = man['train_donors'], man['heldout_donors']
    check(tr == f['train_donors'] and he == f['heldout_donors'], f'{tag} donors differ from folds.json')
    check(not set(tr) & set(he), f'{tag} train/heldout overlap')
    check(set(tr) | set(he) == DEV, f'{tag} donors not the development set')
    check(not (set(tr) | set(he)) & SEALED, f'{tag} sealed donor present')
    g = man['donor_guard']
    check(set(g['count_files_read_by_this_process']) <= DEV and g['forbidden_donors_read'] == []
          and g['all_read_donors_are_development_donors'] is True, f'{tag} guard record')

    S = np.load(d / 'space.npz', allow_pickle=False)
    G = len(S['hvg'])
    check(set(S.files) >= {'genes', 'hvg', 'mean', 'sd', 'W', 'pc_sd', 'age_mean', 'age_sd'}, f'{tag} space keys')
    check(G == man['n_genes'], f'{tag} n_genes')
    check(G == 2000 if panel == 'hvg2000' else G >= 5000, f'{tag} panel size')
    check(S['W'].shape == (G, 50) and S['pc_sd'].shape == (50,) and S['mean'].shape == (G,) and S['sd'].shape == (G,), f'{tag} space shapes')
    check(len(set(S['hvg'].tolist())) == G and S['hvg'].min() >= 0 and S['hvg'].max() < len(AXIS), f'{tag} hvg index')
    check(np.array_equal(AXIS[S['hvg']], S['genes']), f'{tag} genes != axis[hvg]')
    for key in ['mean', 'sd', 'W', 'pc_sd', 'age_mean', 'age_sd']:
        check(np.isfinite(S[key]).all(), f'{tag} nonfinite {key}')
    check((S['sd'] > 0).all() and (S['pc_sd'] > 0).all() and float(S['age_sd']) > 0, f'{tag} nonpositive scale')
    if panel == 'wide':
        fg = man['forced_genes']
        sel = set(S['genes'].tolist())
        check(set(fg['present_on_axis']) | set(fg['missing_from_axis']) == set(fg['requested']), f'{tag} forced accounting')
        check(all(x in sel for x in fg['already_in_hvg'] + fg['appended']), f'{tag} forced gene not in panel')
        check(not any(x in sel for x in fg['dropped_zero_train_counts']), f'{tag} dropped forced gene in panel')
        check(G == 5000 + len(fg['appended']), f'{tag} wide size accounting')

    genes = {}
    for split, donors in [('train', tr), ('heldout', he)]:
        M = np.load(d / split / 'metadata.npz', allow_pickle=False)
        C = sp.load_npz(d / split / 'counts.npz')
        n = len(M['cell'])
        check(sp.isspmatrix_csr(C) and C.dtype == np.float32, f'{tag} {split} counts not float32 CSR')
        check(C.shape == (n, G), f'{tag} {split} counts shape')
        check(np.isfinite(C.data).all() and (C.data >= 0).all() and np.equal(C.data, np.round(C.data)).all(), f'{tag} {split} counts not nonnegative integers')
        check((np.asarray(C.sum(1)).ravel() > 0).all(), f'{tag} {split} zero-panel cell present')
        check(M['donor'].shape == (n,) and M['type'].shape == (n,) and M['raw_c'].shape == (len(donors), 7), f'{tag} {split} metadata shapes')
        check(list(M['donors']) == donors, f'{tag} {split} donors order')
        check(set(M['donor'].tolist()) == set(donors), f'{tag} {split} per-cell donors')
        check(not set(M['donor'].tolist()) & SEALED and set(M['donor'].tolist()) <= DEV, f'{tag} {split} forbidden donor in cells')
        check(len(set(M['cell'].tolist())) == n, f'{tag} {split} duplicate cells')
        check(M['type'].min() >= 0 and M['type'].max() < 6, f'{tag} {split} type range')
        rc = M['raw_c']
        check(np.isfinite(rc).all(), f'{tag} {split} raw_c finite')
        check(np.array_equal(np.rint(rc[:, 3] * 3).astype(int), [FOLDS['stage_by_donor'][x] for x in donors]), f'{tag} {split} ADNC slot')
        check(((rc[:, :4] >= 0) & (rc[:, :4] <= 1)).all() and np.isin(rc[:, 4], [0, 1]).all() and np.isin(rc[:, 5], [0, 1, 2]).all(), f'{tag} {split} raw_c ranges')
        check((rc[:, 6] > 40).all() and (rc[:, 6] < 120).all(), f'{tag} {split} raw age slot')
        per = man['cells_per_donor'][split]
        check(all(per[x]['cells_kept'] == int((M['donor'] == x).sum()) for x in donors), f'{tag} {split} manifest cell counts')
        check(np.array_equal(M['genes'], S['genes']), f'{tag} {split} genes differ from space')
        genes[split] = M['genes']
        if split == 'train':
            check(np.isclose(float(S['age_mean']), rc[:, 6].mean()) and np.isclose(float(S['age_sd']), rc[:, 6].std()), f'{tag} age scaler not from train donors')
            Z = (log_cp10k(C) - S['mean']) / S['sd']
            check(np.abs(Z.mean(0)).max() < 1e-3, f'{tag} standardized train mean not ~0')
            P = (Z @ S['W']) / S['pc_sd']
            sd = P.std(0)
            check(np.abs(sd - 1).max() < 1e-3, f'{tag} train PC sd not unit: max dev {np.abs(sd - 1).max():.2e}')
            check(np.abs(P.mean(0)).max() < 1e-3, f'{tag} train PC mean not ~0')
            WtW = S['W'].T.astype(np.float64) @ S['W']
            check(np.abs(WtW - np.eye(50)).max() < 1e-4, f'{tag} W not orthonormal')
            print(f'{tag} train cells={n} G={G} PC sd range [{sd.min():.6f},{sd.max():.6f}]', flush=True)
        else:
            Ph = ((log_cp10k(C) - S['mean']) / S['sd']) @ S['W'] / S['pc_sd']
            check(np.isfinite(Ph).all(), f'{tag} heldout projection nonfinite')
            print(f'{tag} heldout cells={n} PC sd range [{Ph.std(0).min():.3f},{Ph.std(0).max():.3f}]', flush=True)
    check(np.array_equal(genes['train'], genes['heldout']), f'{tag} gene axes differ between train and heldout')


def main():
    test_guard()
    for panel in ['hvg2000', 'wide']:
        for k in range(3):
            test_cache(panel, k)
    # hvg2000 HVGs must be the first 2000 of the wide ranking (same rule, same train donors)
    for k in range(3):
        a = np.load(ROOT / 'hvg2000' / f'fold{k}' / 'space.npz')['hvg']
        b = np.load(ROOT / 'wide' / f'fold{k}' / 'space.npz')['hvg']
        check(np.array_equal(a, b[:2000]), f'fold{k} hvg2000 is not the prefix of wide')
    fid = ROOT / 'fidelity_legacy_fold0.json'
    if fid.exists():
        r = json.loads(fid.read_text())
        for key in ['hvg_identical_set_and_order', 'genes_identical', 'cells_identical_order', 'counts_identical']:
            check(r[key] is True, f'fidelity {key}')
        check(r['mean']['bitwise_equal'] and r['sd']['bitwise_equal'], 'fidelity mean/sd')
        check(r['W']['subspace_min_principal_cosine'] > 0.99999 and r['W']['max_abs_diff_after_sign'] < 1e-3, 'fidelity W tolerance')
    total = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    print(f'total cache bytes {total}')
    if FAILS:
        print(f'{len(FAILS)} FAILURES'); sys.exit(1)
    print('ALL CACHE TESTS PASSED')


if __name__ == '__main__':
    main()
