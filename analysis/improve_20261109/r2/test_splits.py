"""Tests for the round-2 donor splits (F9): r2/folds_r2.json and hvg2000 caches for folds 3-8.

Run (CPU only, 3 threads):
  cd /tmp; /mnt/nvme/ai/conda/anaconda3/bin/python3 -B \
      /ml2/alz/analysis/improve_20261109/r2/test_splits.py {folds|guard|caches|rebuild|trainer|all}

  folds    folds_r2.json: regenerates it with make_folds_r2.generate() and compares text;
           schema equal to ../folds.json (top-level values kept, folds 0-2 identical);
           each repeat partitions the 37 donors; held-out sizes {12,12,13}; per-stage
           held-out counts differ by <= 1 across a repeat (stratification); every train split
           has >= 2 Not AD donors and every held-out split >= 1; pairwise held-out overlap
           matrices across repeats and the minimal co-held-out pair count (58) are asserted.
  guard    build_cache_r2 (configured) still refuses sealed / unknown donors before any file
           access, and its out-root guard refuses folds 0-2 into the real cache root.
  caches   round-1 test_cache.test_cache checks, adapted to folds_r2.json, for hvg2000 folds
           0-8 (folds 0-2 read-only): manifest SHA256s and file set, donors equal to the
           fold, donor guard record, space shapes/finiteness, genes == axis[hvg], counts
           float32 CSR nonnegative integers with no zero-panel cells, metadata shapes and
           order, ADNC slot, raw_c ranges, train-donor age scaler, standardised train mean ~0,
           unit train PC sd, orthonormal W, finite held-out projection. Extra for folds 3-8:
           manifest r2_build provenance (folds_r2.json SHA256, repeat) and the same top-level
           manifest keys as round 1; per-cell counts of every donor on genes shared with
           fold 0 are identical to fold 0 (same cells, same order, same values). Writes the
           gene-set overlap with folds 0-2 and disk use to results/caches.json.
  rebuild  builds folds 0 and 3 through build_cache_r2.py into a scratch root (subprocess,
           CACHE_THREADS=1), asserts fold 0 equals the round-1 cache byte for byte (every
           data file SHA256 and every manifest key except wall_seconds, peak_RSS_KiB,
           source_pins (folds file path) and r2_build) and fold 3 equals the built fold 3
           byte for byte (determinism), then deletes the scratch root.
  trainer  r2/trainer.load_folds() accepts folds_r2.json; validate_split accepts every
           fold 0-8 cache; trainer.Context(cache, 'cpu', ('base',)) loads folds 3-8.

Results go to /ml2/alz/runs/improve_20261109/smoke_r2/splits/results/{name}.json (small).
Never opens donor count files except through build_cache_r2.py in `rebuild` (development
donors only, guarded).
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp

R2 = Path(__file__).resolve().parent
R1 = R2.parent
PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = Path('/ml2/alz/runs/improve_20261109')
ROOT = RUNS / 'caches'
RESULTS = RUNS / 'smoke_r2' / 'splits' / 'results'
SCRATCH = RUNS / 'smoke_r2' / 'splits_scratch'
GENES_FILE = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt/counts/genes.npy')
R1F = json.loads((R1 / 'folds.json').read_text())
Q = json.loads((R2 / 'folds_r2.json').read_text())
DEV = set(Q['development_donors'])
SEALED = set(Q['sealed_evaluation_donors_fold0'])
STAGES = ['Not AD', 'Low', 'Intermediate', 'High']
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print('FAIL', msg, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'{name}.json').write_text(json.dumps(obj, indent=1) + '\n')


def log_cp10k(C):
    X = C.toarray().astype(np.float32)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


# ---------------------------------------------------------------- folds
def test_folds():
    sys.path.insert(0, str(R2))
    import make_folds_r2 as mk
    text = json.dumps(mk.generate(), indent=1) + '\n'
    check((R2 / 'folds_r2.json').read_text() == text, 'folds_r2.json differs from make_folds_r2.generate()')
    for k, v in R1F.items():
        if k != 'inner_folds':
            check(Q[k] == v, f'top-level key {k} differs from folds.json')
    check(set(Q) - set(R1F) == {'r2_rule', 'repeats', 'r2_note'}, 'unexpected top-level keys')
    check(len(Q['inner_folds']) == 9, 'need 9 inner folds')
    for k in range(3):
        a, b = R1F['inner_folds'][k], Q['inner_folds'][k]
        check({x: b[x] for x in a} == a and b['repeat'] == 0, f'inner fold {k} not identical to round 1')
    dev = Q['development_donors']
    sbd = Q['stage_by_donor']
    assign = {}
    table = []
    for k, f in enumerate(Q['inner_folds']):
        check(f['inner_fold'] == k and f['repeat'] == k // 3, f'fold {k} index/repeat')
        check(list(f) == ['inner_fold', 'heldout_donors', 'train_donors', 'repeat'], f'fold {k} key order')
        tr, he = f['train_donors'], f['heldout_donors']
        check(not set(tr) & set(he) and set(tr) | set(he) == DEV and len(tr) + len(he) == 37, f'fold {k} partition')
        check(tr == [d for d in dev if d in set(tr)] and he == [d for d in dev if d in set(he)], f'fold {k} list order')
        check(not (set(tr) | set(he)) & SEALED, f'fold {k} sealed donor')
        ts = [sum(sbd[d] == s for d in tr) for s in range(4)]
        hs = [sum(sbd[d] == s for d in he) for s in range(4)]
        check(ts[0] >= 2 and hs[0] >= 1, f'fold {k} Not AD constraint')
        table.append({'inner_fold': k, 'repeat': k // 3, 'n_train': len(tr), 'n_heldout': len(he),
                      'train_stage': dict(zip(STAGES, ts)), 'heldout_stage': dict(zip(STAGES, hs))})
        for d in he:
            assign.setdefault(k // 3, {})[d] = k % 3
    vec = {}
    for r in range(3):
        check(set(assign[r]) == DEV, f'repeat {r} held-out sets do not partition the donors')
        a = np.array([assign[r][d] for d in dev])
        vec[r] = a
        check(sorted(np.bincount(a, minlength=3).tolist()) == [12, 12, 13], f'repeat {r} held-out sizes')
        for s in range(4):
            c = np.bincount(a[[sbd[d] == s for d in dev]], minlength=3)
            check(c.max() - c.min() <= 1, f'repeat {r} stage {s} not stratified')
        rep = Q['repeats'][r]
        check(rep['seed'] == 20261109 + 100000 * r + (rep['candidate_index'] or 0), f'repeat {r} seed record')
        check(np.array_equal(mk.assign(dev, sbd, rep['seed']), a), f'repeat {r} not reproduced from its seed')
    overl = {}
    for r in range(3):
        for q in range(r):
            m = mk.overlap(vec[r], vec[q])
            pairs = int((m * (m - 1) // 2).sum())
            overl[f'{r}_vs_{q}'] = {'matrix_rows_repeat_%d_cols_repeat_%d' % (r, q): m.tolist(), 'co_heldout_pairs': pairs,
                                    'jaccard_heldout': [[round(len({d for d in dev if assign[r][d] == i} & {d for d in dev if assign[q][d] == j})
                                                               / len({d for d in dev if assign[r][d] == i} | {d for d in dev if assign[q][d] == j}), 3)
                                                         for j in range(3)] for i in range(3)]}
            check(pairs == 58 and m.max() == 5 and m.min() == 4, f'repeats {r} vs {q} not maximally different: {m.tolist()}')
            if q == 0 or r == 2:
                check(Q['repeats'][r]['heldout_overlap_vs_repeat'][str(q)]['matrix_rows_this_repeat_cols_other'] == m.tolist(),
                      f'repeat {r} recorded overlap vs {q}')
    # co-held-out frequency of donor pairs over the 3 repeats
    co = {}
    for i, d in enumerate(dev):
        for e in dev[i + 1:]:
            n = sum(assign[r][d] == assign[r][e] for r in range(3))
            co[n] = co.get(n, 0) + 1
    res = {'folds': table, 'heldout_overlap': overl, 'donor_pairs_by_times_held_out_together': {str(k): v for k, v in sorted(co.items())},
           'repeats': Q['repeats']}
    save('folds', res)
    print(json.dumps({'overlap': {k: v['co_heldout_pairs'] for k, v in overl.items()}, 'pairs': res['donor_pairs_by_times_held_out_together']}))
    print('folds ok', flush=True)


# ---------------------------------------------------------------- guard
def test_guard():
    sys.path.insert(0, str(R2))
    import build_cache_r2 as b2
    b2.configure(SCRATCH / 'guard_never_written')
    bc = b2.bc
    check(bc.FOLDS_JSON is not None and len(bc.FOLDS_JSON['inner_folds']) == 9, 'wrapper did not rebind folds')
    for d in sorted(SEALED) + ['H19.33.004', 'H20.33.002', 'not-a-donor']:
        try:
            bc.check_donor_allowed(d)
            check(False, f'guard accepted forbidden donor {d}')
        except PermissionError:
            pass
    try:
        bc.read_donors(['H20.33.001', 'H21.33.006'])
        check(False, 'read_donors accepted a sealed donor')
    except PermissionError:
        pass
    check(bc.COUNT_FILES_OPENED == [], 'guard test opened a count file')
    check(not (SCRATCH / 'guard_never_written').exists(), 'configure wrote output')
    for args in (['--fold', '0'], ['--fold', '3', '--out-root', str(ROOT / 'hvg2000')], ['--fold', '3', '--out-root', '/ml2/alz/runs/improve_20261109/fits'],
                 ['--fold', '9']):
        p = subprocess.run([PY, '-B', str(R2 / 'build_cache_r2.py')] + args, capture_output=True, text=True, cwd='/tmp')
        check(p.returncode != 0, f'build_cache_r2 accepted {args}')
    p = subprocess.run([PY, '-B', str(R2 / 'build_cache_r2.py'), '--fold', '3'], capture_output=True, text=True, cwd='/tmp')
    check(p.returncode != 0 and 'FileExistsError' in p.stderr, 'build_cache_r2 did not refuse an existing fold 3')
    p = subprocess.run([PY, '-B', str(R2 / 'build_cache_r2.py'), '--fold', '3'], capture_output=True, text=True, cwd='/tmp',
                       env=dict(os.environ, CACHE_THREADS='6'))
    check(p.returncode != 0, 'build_cache_r2 accepted CACHE_THREADS=6')
    print('guard ok', flush=True)


# ---------------------------------------------------------------- caches
def test_cache(root, k, axis):
    panel = 'hvg2000'
    d = root / panel / f'fold{k}'
    tag = f'{panel}/fold{k}'
    man = json.loads((d / 'manifest.json').read_text())
    f = Q['inner_folds'][k]
    for rel, h in man['files_SHA256'].items():
        check(sha(d / rel) == h, f'{tag} sha {rel}')
    present = {str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()} - {'manifest.json'}
    check(present == set(man['files_SHA256']), f'{tag} file set differs from manifest')
    tr, he = man['train_donors'], man['heldout_donors']
    check(tr == f['train_donors'] and he == f['heldout_donors'], f'{tag} donors differ from folds_r2.json')
    check(man['inner_fold'] == k and man['panel'] == panel, f'{tag} manifest identity')
    check(not set(tr) & set(he) and set(tr) | set(he) == DEV, f'{tag} donors not a partition of the development set')
    check(not (set(tr) | set(he)) & SEALED, f'{tag} sealed donor present')
    g = man['donor_guard']
    check(set(g['count_files_read_by_this_process']) <= DEV and g['forbidden_donors_read'] == []
          and g['all_read_donors_are_development_donors'] is True, f'{tag} guard record')
    if k >= 3:
        r1man = json.loads((root / panel / 'fold0' / 'manifest.json').read_text())
        check(set(man) - set(r1man) == {'r2_build'} and set(r1man) <= set(man), f'{tag} manifest keys differ from round 1')
        rb = man['r2_build']
        check(rb['folds_SHA256'] == sha(R2 / 'folds_r2.json') and rb['repeat'] == k // 3 and rb['threads'] <= 3, f'{tag} r2_build provenance')
        check(man['source_pins'].get(str(R2 / 'folds_r2.json')) == sha(R2 / 'folds_r2.json'), f'{tag} folds pin')
        check(man['recipe'] == r1man['recipe'] and man['pc_convention'] == r1man['pc_convention'], f'{tag} recipe text differs')
        check(man['source_pins'][str(R1 / 'build_cache.py')] == r1man['source_pins'][str(R1 / 'build_cache.py')], f'{tag} build_cache.py pin differs from round 1')
    S = np.load(d / 'space.npz', allow_pickle=False)
    G = len(S['hvg'])
    check(set(S.files) >= {'genes', 'hvg', 'mean', 'sd', 'W', 'pc_sd', 'age_mean', 'age_sd'}, f'{tag} space keys')
    check(G == man['n_genes'] == 2000, f'{tag} n_genes')
    check(S['W'].shape == (G, 50) and S['pc_sd'].shape == (50,) and S['mean'].shape == (G,) and S['sd'].shape == (G,), f'{tag} space shapes')
    check(len(set(S['hvg'].tolist())) == G and S['hvg'].min() >= 0 and S['hvg'].max() < len(axis), f'{tag} hvg index')
    check(np.array_equal(axis[S['hvg']], S['genes']), f'{tag} genes != axis[hvg]')
    for key in ['mean', 'sd', 'W', 'pc_sd', 'age_mean', 'age_sd']:
        check(np.isfinite(S[key]).all(), f'{tag} nonfinite {key}')
    check((S['sd'] > 0).all() and (S['pc_sd'] > 0).all() and float(S['age_sd']) > 0, f'{tag} nonpositive scale')
    out = {'train_cells': None, 'heldout_cells': None}
    for split, donors in [('train', tr), ('heldout', he)]:
        M = np.load(d / split / 'metadata.npz', allow_pickle=False)
        C = sp.load_npz(d / split / 'counts.npz')
        n = len(M['cell'])
        out[f'{split}_cells'] = n
        check(sp.isspmatrix_csr(C) and C.dtype == np.float32, f'{tag} {split} counts not float32 CSR')
        check(C.shape == (n, G), f'{tag} {split} counts shape')
        check(np.isfinite(C.data).all() and (C.data >= 0).all() and np.equal(C.data, np.round(C.data)).all(), f'{tag} {split} counts not nonnegative integers')
        check((np.asarray(C.sum(1)).ravel() > 0).all(), f'{tag} {split} zero-panel cell present')
        check(M['donor'].shape == (n,) and M['type'].shape == (n,) and M['raw_c'].shape == (len(donors), 7), f'{tag} {split} metadata shapes')
        check(list(M['donors']) == donors, f'{tag} {split} donors order')
        check(set(M['donor'].tolist()) == set(donors), f'{tag} {split} per-cell donors')
        check(len(set(M['cell'].tolist())) == n, f'{tag} {split} duplicate cells')
        check(M['type'].min() >= 0 and M['type'].max() < 6, f'{tag} {split} type range')
        rc = M['raw_c']
        check(np.isfinite(rc).all(), f'{tag} {split} raw_c finite')
        check(np.array_equal(np.rint(rc[:, 3] * 3).astype(int), [Q['stage_by_donor'][x] for x in donors]), f'{tag} {split} ADNC slot')
        check(((rc[:, :4] >= 0) & (rc[:, :4] <= 1)).all() and np.isin(rc[:, 4], [0, 1]).all() and np.isin(rc[:, 5], [0, 1, 2]).all(), f'{tag} {split} raw_c ranges')
        check((rc[:, 6] > 40).all() and (rc[:, 6] < 120).all(), f'{tag} {split} raw age slot')
        per = man['cells_per_donor'][split]
        check(all(per[x]['cells_kept'] == int((M['donor'] == x).sum()) for x in donors), f'{tag} {split} manifest cell counts')
        check(np.array_equal(M['genes'], S['genes']) and np.array_equal(M['hvg'], S['hvg']), f'{tag} {split} genes differ from space')
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
            out['train_pc_sd_range'] = [float(sd.min()), float(sd.max())]
        else:
            Ph = ((log_cp10k(C) - S['mean']) / S['sd']) @ S['W'] / S['pc_sd']
            check(np.isfinite(Ph).all(), f'{tag} heldout projection nonfinite')
    return man, S, out


def donor_rows(root, k):
    """donor -> (cell names, CSR counts, genes) over both splits of a cache."""
    d = root / 'hvg2000' / f'fold{k}'
    res = {}
    for split in ('train', 'heldout'):
        M = np.load(d / split / 'metadata.npz', allow_pickle=False)
        C = sp.load_npz(d / split / 'counts.npz').tocsr()
        for x in M['donors']:
            rows = np.where(M['donor'] == x)[0]
            res[str(x)] = (M['cell'][rows], C[rows], M['genes'])
    return res


def test_caches():
    axis = np.load(GENES_FILE, allow_pickle=False)
    hvg = {}
    info = {}
    for k in range(9):
        man, S, out = test_cache(ROOT, k, axis)
        hvg[k] = S['hvg']
        out.update({'total_bytes': man['total_bytes'], 'repeat': k // 3,
                    'train_stage_donors': {s: v['donors'] for s, v in man['stage_counts']['train'].items()},
                    'heldout_stage_donors': {s: v['donors'] for s, v in man['stage_counts']['heldout'].items()},
                    'dropped_zero_panel_cells': {s: len(v) for s, v in man['dropped_zero_panel_cells'].items()}})
        info[k] = out
        print(f'fold{k} checked: {json.dumps(out)}', flush=True)
    # per-donor counts on genes shared with fold 0 are identical
    ref = donor_rows(ROOT, 0)
    for k in range(3, 9):
        other = donor_rows(ROOT, k)
        check(set(other) == set(ref), f'fold{k} donor set')
        for x, (cells, C, genes) in other.items():
            rcells, rC, rgenes = ref[x]
            shared, ia, ib = np.intersect1d(genes, rgenes, return_indices=True)
            if np.array_equal(cells, rcells):
                same = (C[:, ia] != rC[:, ib]).nnz == 0
            else:  # zero-panel drops may differ between panels: compare common cells
                cm, ca, cb = np.intersect1d(cells, rcells, return_indices=True)
                same = (C[ca][:, ia] != rC[cb][:, ib]).nnz == 0 and len(cm) >= .99 * max(len(cells), len(rcells))
            check(same, f'fold{k} donor {x} counts differ from fold 0 on shared genes')
    overlap = {}
    for k in range(3, 9):
        a = set(hvg[k].tolist())
        overlap[f'fold{k}'] = {f'fold{j}': {'shared_genes': len(a & set(hvg[j].tolist())),
                                            'jaccard': round(len(a & set(hvg[j].tolist())) / len(a | set(hvg[j].tolist())), 4)}
                               for j in range(9) if j != k}
    allr1 = set(hvg[0].tolist()) | set(hvg[1].tolist()) | set(hvg[2].tolist())
    common_all = set.intersection(*[set(v.tolist()) for v in hvg.values()])
    union_all = set.union(*[set(v.tolist()) for v in hvg.values()])
    for k in range(3, 9):
        overlap[f'fold{k}']['in_union_of_folds_0_2'] = len(set(hvg[k].tolist()) & allr1)
    sizes = {f'fold{k}': sum(p.stat().st_size for p in (ROOT / 'hvg2000' / f'fold{k}').rglob('*') if p.is_file()) for k in range(9)}
    res = {'per_fold': {f'fold{k}': v for k, v in info.items()}, 'gene_overlap': overlap,
           'genes_common_to_all_9_folds': len(common_all), 'genes_in_union_of_9_folds': len(union_all),
           'genes_in_union_of_folds_0_2': len(allr1), 'disk_bytes': sizes,
           'disk_bytes_new_folds_3_8': sum(sizes[f'fold{k}'] for k in range(3, 9))}
    save('caches', res)
    print(json.dumps({k: {j: (v['shared_genes'] if isinstance(v, dict) else v) for j, v in o.items()
                          if j in ('fold0', 'fold1', 'fold2', 'in_union_of_folds_0_2')} for k, o in overlap.items()}))
    print(json.dumps({x: res[x] for x in ('genes_common_to_all_9_folds', 'genes_in_union_of_9_folds', 'genes_in_union_of_folds_0_2', 'disk_bytes_new_folds_3_8')}))
    print('caches ok', flush=True)


# ---------------------------------------------------------------- rebuild
def test_rebuild():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    sroot = SCRATCH / 'caches'
    res = {}
    try:
        p = subprocess.run([PY, '-B', str(R2 / 'build_cache_r2.py'), '--fold', '0,3', '--out-root', str(sroot)],
                           capture_output=True, text=True, cwd='/tmp', env=dict(os.environ, CACHE_THREADS='1'))
        check(p.returncode == 0, 'scratch rebuild failed: ' + p.stderr[-2000:])
        volatile = {'wall_seconds', 'peak_RSS_KiB', 'source_pins', 'r2_build'}
        for k in (0, 3):
            a = json.loads((ROOT / 'hvg2000' / f'fold{k}' / 'manifest.json').read_text())
            b = json.loads((sroot / 'hvg2000' / f'fold{k}' / 'manifest.json').read_text())
            same_files = a['files_SHA256'] == b['files_SHA256']
            recomputed = {rel: sha(sroot / 'hvg2000' / f'fold{k}' / rel) for rel in b['files_SHA256']}
            diff_keys = sorted(x for x in set(a) | set(b) if a.get(x) != b.get(x))
            check(same_files and recomputed == a['files_SHA256'], f'fold{k} scratch rebuild files differ')
            check(set(diff_keys) <= volatile, f'fold{k} manifest differs beyond volatile keys: {diff_keys}')
            pins_a = {x: v for x, v in a['source_pins'].items() if not x.endswith(('folds.json', 'folds_r2.json'))}
            pins_b = {x: v for x, v in b['source_pins'].items() if not x.endswith(('folds.json', 'folds_r2.json'))}
            check(pins_a == pins_b, f'fold{k} source pins (other than the folds file) differ')
            res[f'fold{k}'] = {'files_SHA256_equal': same_files, 'manifest_keys_differing': diff_keys,
                               'files': sorted(b['files_SHA256'])}
    finally:
        if SCRATCH.exists():
            shutil.rmtree(SCRATCH)
    check(not SCRATCH.exists(), 'scratch not deleted')
    res['scratch_deleted'] = not SCRATCH.exists()
    save('rebuild', res)
    print(json.dumps(res))
    print('rebuild ok', flush=True)


# ---------------------------------------------------------------- trainer
def test_trainer():
    sys.path.insert(0, str(R2))
    import trainer
    q = trainer.load_folds()
    check(len(q['inner_folds']) == 9, 'trainer.load_folds did not return folds_r2.json')
    for k in range(9):
        c = ROOT / 'hvg2000' / f'fold{k}'
        man = json.loads((c / 'manifest.json').read_text())
        check(trainer.validate_split(man, man['train_donors'], man['heldout_donors'], False) is True, f'validate_split fold {k}')
        bad = list(man['train_donors'])[1:]
        try:
            trainer.validate_split(man, bad, man['heldout_donors'], False)
            check(False, f'validate_split accepted wrong donors for fold {k}')
        except ValueError:
            pass
    for k in range(3, 9):
        ctx = trainer.Context(ROOT / 'hvg2000' / f'fold{k}', 'cpu', ('base',))
        check(ctx.fold == k and ctx.protocol_split and ctx.n_genes == 2000 and list(ctx.ids) == q['inner_folds'][k]['train_donors'],
              f'Context fold {k}')
        check(ctx.space_age is not None and np.isclose(ctx.space_age[0], ctx.age_mean), f'Context fold {k} age scaler')
        del ctx
    print('trainer ok', flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    tests = {'folds': test_folds, 'guard': test_guard, 'caches': test_caches, 'rebuild': test_rebuild, 'trainer': test_trainer}
    if which == 'all':
        bad = []
        for name in tests:
            p = subprocess.run([PY, '-B', str(Path(__file__).resolve()), name], cwd='/tmp')
            if p.returncode:
                bad.append(name)
        print('FAILED: ' + ', '.join(bad) if bad else 'ALL SPLIT TESTS PASSED')
        sys.exit(1 if bad else 0)
    tests[which]()
    if FAILS:
        print(f'{len(FAILS)} FAILURES')
        sys.exit(1)
    print(f'{which}: PASS')


if __name__ == '__main__':
    main()
