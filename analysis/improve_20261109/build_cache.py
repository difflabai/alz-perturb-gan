"""Interface 1 data caches for improve_20261109 (component: cache).

Writes /ml2/alz/runs/improve_20261109/caches/{panel}/fold{k}/ for panels
`hvg2000` and `wide` and inner folds 0..2 of folds.json, and a fidelity check
against the legacy core-comparison fold-0 cache.

Recipe (mirrors bench_core.fit_space and core-comparison fold_data.py):
  * raw counts per donor from rebuilt/counts/{donor}.npz (36,601-gene axis
    rebuilt/counts/genes.npy), receipts and SHA256 verified, integer/finite/
    nonnegative checked; cell metadata from mg_meta.json (streamed; development-donor rows only); every cell must be
    "Used in analysis" with a known microglia supertype; donor covariates must
    be constant over cells; per-donor cell count must equal candidate_cohort.json
    and be >= 100 (bench_core.load_dataset rule).
  * HVG rule on train donors only: per-cell CP10k (float32), log1p, gene mean
    and variance over cells, score = var/max(mean,.001) * (mean>.01), stable
    descending argsort, first n_hvg genes.
  * cells whose selected-panel total is zero are dropped (legacy train rule;
    the protocol applies it to heldout as well), recorded in the manifest.
  * X = log1p(CP10k over the panel), mean, sd = std + 1e-6 (float32), Z;
    randomized_svd(Z[rng(0).choice(n, min(20000,n))], 50, n_iter=5,
    random_state=0); W = Vt.T; pc_sd = max(std(Z@W), 1e-6). Hence the
    standardized train features projected with W/pc_sd have unit variance
    (ddof=0) per PC.
  * age_mean/age_sd over train *donors* (raw age, one value per donor).
  * raw_c [n_donors,7] = bench_core.encode_condition (raw age in slot 6).
  * type_classifier.pt: bench_core.TypeClassifier fitted on train PCs (extra
    file kept for bench_core.Space compatibility; not part of Interface 1).
  * wide: first 5,000 genes of the same ranking, then the forced AD/microglia
    genes present on the axis that are not already in the top 5,000 are
    appended in the listed order, unless their train count total is zero
    (then dropped and recorded).

Donor guard: only the 37 development donors of folds.json may ever be opened;
`check_donor_allowed` raises for anything else, and every count file opened is
recorded and asserted in the manifest. The counts directory is never globbed.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = os.environ.get('CACHE_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True  # never write __pycache__ into read-only source dirs

import argparse, gc, hashlib, json, resource, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.utils.extmath import randomized_svd

THREADS = int(os.environ.get('CACHE_THREADS', '1'))
assert 1 <= THREADS <= 6
torch.set_num_threads(THREADS)

HERE = Path('/ml2/alz/analysis/improve_20261109')
OUT_ROOT = Path('/ml2/alz/runs/improve_20261109/caches')
FOLDS = HERE / 'folds.json'
B = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
RAW = B / 'rebuilt'
COUNTS_DIR = RAW / 'counts'
GENES_FILE = COUNTS_DIR / 'genes.npy'
LEGACY = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/caches/fold0')
LEGACY_CALIBRATION = ['H20.33.032', 'H21.33.001', 'H21.33.023', 'H21.33.025']

sys.path.insert(0, str(B))
import bench_core  # noqa: E402  (read-only; bytecode writing disabled above)
from bench_core import encode_condition, log_hvg, TypeClassifier, TYPES, STAGES  # noqa: E402

FORCED_GENES = ['APOE', 'TREM2', 'INPP5D', 'ACSL1', 'SORL1', 'SPI1', 'DGAT1', 'DGAT2', 'ABCA1', 'BIN1',
                'CD33', 'TYROBP', 'NLRP3', 'PLCG2', 'ABI3', 'MS4A6A', 'MS4A4A', 'CD2AP', 'PICALM', 'CLU',
                'CR1', 'ABCA7', 'GRN', 'TMEM119', 'P2RY12', 'CX3CR1', 'CSF1R', 'SPP1', 'GPNMB', 'LPL',
                'CD74', 'HLA-DRA']
PANELS = {'hvg2000': {'n_hvg': 2000, 'forced': []}, 'wide': {'n_hvg': 5000, 'forced': FORCED_GENES}}
N_PC = 50


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n'); tmp.replace(p)


FOLDS_JSON = json.loads(FOLDS.read_text())
DEV = list(FOLDS_JSON['development_donors'])
SEALED = list(FOLDS_JSON['sealed_evaluation_donors_fold0'])
COUNT_FILES_OPENED = []  # every donor whose count/name file was opened in this process


def validate_folds():
    assert len(DEV) == 37 == len(set(DEV)), 'need 37 unique development donors'
    assert not set(DEV) & set(SEALED), 'development and sealed donors overlap'
    assert set(FOLDS_JSON['stage_by_donor']) == set(DEV)
    held = []
    for k, f in enumerate(FOLDS_JSON['inner_folds']):
        assert f['inner_fold'] == k
        tr, he = f['train_donors'], f['heldout_donors']
        assert len(set(tr)) == len(tr) and len(set(he)) == len(he)
        assert not set(tr) & set(he) and set(tr) | set(he) == set(DEV)
        held += he
    assert sorted(held) == sorted(DEV), 'heldout sets must partition the development donors'


def check_donor_allowed(donor):
    """Raise unless `donor` is one of the 37 development donors. Called before any file access."""
    if not isinstance(donor, str) or donor not in DEV or donor in SEALED:
        raise PermissionError(f'donor {donor!r} is not a development donor of improve_20261109; reading it is forbidden')


_META = None
_COHORT = None


def _cell_meta():
    """Per-cell metadata of development-donor cells only (fixer revision).

    mg_meta.json ({'cols': [...], 'rows': {cell name: [values]}}) and
    candidate_cohort.json are single shared files that also hold sealed and
    other donors.  They are streamed with ijson (use_float=True, the same float
    parsing as json.loads) and each row is kept only when its key is a cell name
    from a development donor's own _names.npy (cohort: only development donor
    keys); every other row is discarded as it is parsed and is never stored or
    inspected.  (The bytes of a shared file necessarily pass through the parser.)
    """
    global _META, _COHORT
    if _META is None:
        import ijson
        keep = set()
        for d in DEV:
            check_donor_allowed(d)
            COUNT_FILES_OPENED.append(d)  # its names file
            keep.update(str(n) for n in np.load(COUNTS_DIR / f'{d}_names.npy', allow_pickle=False).tolist())
        with open(RAW / 'mg_meta.json', 'rb') as fh:
            cols = next(ijson.items(fh, 'cols', use_float=True))
        with open(RAW / 'mg_meta.json', 'rb') as fh:
            rows = {k: v for k, v in ijson.kvitems(fh, 'rows', use_float=True) if k in keep}
        if set(rows) != keep:
            raise ValueError('mg_meta.json lacks rows for development-donor cells')
        _META = {'cols': cols, 'rows': rows}
        dev = set(DEV)
        with open(RAW / 'candidate_cohort.json', 'rb') as fh:
            _COHORT = {k: v for k, v in ijson.kvitems(fh, 'donors', use_float=True) if k in dev}
        if set(_COHORT) != dev:
            raise ValueError('candidate_cohort.json lacks a development donor')
    return _META, _COHORT


def read_donors(donors):
    """Read raw counts + metadata for an ordered list of development donors."""
    donors = list(donors)
    for d in donors:
        check_donor_allowed(d)
    if len(set(donors)) != len(donors):
        raise ValueError('duplicate donor')
    meta, cohort = _cell_meta()
    cols, rows = meta['cols'], meta['rows']
    genes = np.load(GENES_FILE, allow_pickle=False); genes_sha = sha(GENES_FILE)
    mats, names, labels, types, raw_c, sources = [], [], [], [], [], []
    for d in donors:
        check_donor_allowed(d)
        p = COUNTS_DIR / f'{d}.npz'; nf = COUNTS_DIR / f'{d}_names.npy'
        rec = json.loads(p.with_suffix('.receipt.json').read_text())
        COUNT_FILES_OPENED.append(d)
        if sha(p) != rec['counts_sha256'] or sha(nf) != rec['names_sha256'] or not rec['raw_counts_validated']:
            raise ValueError(f'raw source receipt differs for {d}')
        if genes_sha != rec['genes_sha256']:
            raise ValueError('raw gene axis differs')
        c = sp.load_npz(p).tocsr(); ns = np.load(nf, allow_pickle=False)
        if c.shape != (len(ns), len(genes)) or not np.isfinite(c.data).all() or (c.data < 0).any() \
                or not np.equal(c.data, np.floor(c.data)).all():
            raise ValueError(f'invalid raw count matrix for {d}')
        if len(ns) != cohort[d] or len(ns) < 100:
            raise ValueError(f'donor cell count rule violated for {d}')
        recs = [dict(zip(cols, rows[str(n)])) for n in ns]
        if any(r['Donor ID'] != d or r['Used in analysis'] != 'True' or r['Supertype'] not in TYPES for r in recs):
            raise ValueError(f'cell cohort or donor mismatch for {d}')
        cc = np.stack([encode_condition(r) for r in recs])
        if not np.equal(cc, cc[0]).all():
            raise ValueError(f'conflicting donor covariates for {d}')
        stage = int(round(cc[0][3] * 3))
        if stage != FOLDS_JSON['stage_by_donor'][d]:
            raise ValueError(f'stage of {d} differs from folds.json')
        mats.append(c); names.extend(ns.tolist()); labels.extend([d] * len(ns))
        types.extend(TYPES.index(r['Supertype']) for r in recs); raw_c.append(cc[0])
        sources.append({'donor': d, 'counts_SHA256': rec['counts_sha256'], 'names_SHA256': rec['names_sha256'],
                        'cells': int(len(ns)), 'stage': stage})
    C = sp.vstack(mats, format='csr'); del mats
    if len(set(names)) != len(names):
        raise ValueError('duplicate cell identity')
    return {'counts': C, 'cell': np.asarray(names), 'donor': np.asarray(labels),
            'type': np.asarray(types, dtype=np.int64), 'donors': np.asarray(donors),
            'raw_c': np.stack(raw_c).astype(np.float64), 'genes': genes, 'sources': sources}


def subset(data, donors):
    """Rows of an already-read dataset for an ordered donor list (order follows `donors`)."""
    for d in donors:
        check_donor_allowed(d)
    pos = {d: i for i, d in enumerate(data['donors'])}
    rows = np.concatenate([np.where(data['donor'] == d)[0] for d in donors])
    return {'counts': data['counts'][rows], 'cell': data['cell'][rows], 'donor': data['donor'][rows],
            'type': data['type'][rows], 'donors': np.asarray(donors),
            'raw_c': data['raw_c'][[pos[d] for d in donors]], 'genes': data['genes'],
            'sources': [s for d in donors for s in data['sources'] if s['donor'] == d]}


def hvg_ranking(counts):
    """bench_core.fit_space HVG score and stable ranking (train cells only)."""
    C = counts.astype(np.float32)
    scale = 1e4 / np.maximum(np.asarray(C.sum(1)).ravel(), 1)
    L = sp.diags(scale) @ C; L.data = np.log1p(L.data)
    mean = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
    score = (var / np.maximum(mean, .001)) * (mean > .01)
    return np.argsort(-score, kind='stable'), score


def select_panel(train, panel):
    cfg = PANELS[panel]; genes = train['genes']
    order, score = hvg_ranking(train['counts'])
    n_positive = int((score > 0).sum())
    if n_positive < cfg['n_hvg']:
        raise ValueError(f'only {n_positive} genes have positive HVG score (< {cfg["n_hvg"]})')
    hvg = list(order[:cfg['n_hvg']])
    gene_tot = np.asarray(train['counts'].sum(0)).ravel()
    idx = {g: i for i, g in enumerate(genes.tolist())}
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    forced = {'requested': list(cfg['forced']), 'present_on_axis': [], 'missing_from_axis': [],
              'already_in_hvg': [], 'appended': [], 'dropped_zero_train_counts': [], 'detail': {}}
    top = set(hvg)
    for g in cfg['forced']:
        if g not in idx:
            forced['missing_from_axis'].append(g); continue
        i = idx[g]; forced['present_on_axis'].append(g)
        info = {'axis_index': i, 'hvg_rank': int(rank[i]), 'hvg_score': float(score[i]),
                'train_total_count': float(gene_tot[i])}
        if i in top:
            forced['already_in_hvg'].append(g); info['status'] = 'in_top_hvg'
        elif gene_tot[i] <= 0:
            forced['dropped_zero_train_counts'].append(g); info['status'] = 'dropped_zero_train_counts'
        else:
            hvg.append(i); top.add(i); forced['appended'].append(g); info['status'] = 'appended'
        forced['detail'][g] = info
    hvg = np.asarray(hvg, dtype=np.int64)
    assert len(set(hvg.tolist())) == len(hvg)
    return hvg, forced, n_positive


def fit_space(train, hvg, n_pc=N_PC):
    selected = train['counts'][:, hvg].tocsr()
    keep = np.asarray(selected.sum(1)).ravel() > 0
    if set(train['donor'][keep]) != set(train['donors']):
        raise ValueError('a train donor lost all cells')
    selected = selected[keep]
    X = log_hvg(selected); m = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - m) / sd; del X
    sub = np.random.default_rng(0).choice(len(Z), min(20000, len(Z)), replace=False)
    _, _, Vt = randomized_svd(Z[sub], n_components=n_pc, n_iter=5, random_state=0)
    W = Vt.T.copy(); ZW = Z @ W; ps = np.maximum(ZW.std(0), 1e-6)
    pc = (ZW / ps).astype(np.float32); del Z, ZW
    age = train['raw_c'][:, -1]
    if not np.isfinite(age).all() or age.std() == 0:
        raise ValueError('undefined train age scaler')
    torch.manual_seed(0)
    clf = TypeClassifier(n_pc); clf.fit(pc, train['type'][keep])
    space = {'hvg': hvg, 'mean': m, 'sd': sd, 'W': W, 'pc_sd': ps,
             'age_mean': float(age.mean()), 'age_sd': float(age.std())}
    for k in ['mean', 'sd', 'W', 'pc_sd']:
        if not np.isfinite(space[k]).all():
            raise ValueError(f'nonfinite {k}')
    return space, selected, keep, pc, clf


def split_stats(part, keep):
    per = {}
    for d in part['donors']:
        mk = part['donor'] == d
        per[str(d)] = {'cells_raw': int(mk.sum()), 'cells_kept': int((mk & keep).sum()),
                       'cells_dropped_zero_panel': int((mk & ~keep).sum()),
                       'stage': int(FOLDS_JSON['stage_by_donor'][str(d)])}
    stage = {}
    for s in range(4):
        ds = [d for d, v in per.items() if v['stage'] == s]
        stage[STAGES[s]] = {'donors': len(ds), 'cells_kept': sum(per[d]['cells_kept'] for d in ds)}
    return per, stage


def write_split(folder, part, keep, counts, genes, hvg):
    folder.mkdir(parents=True, exist_ok=True)
    C = sp.csr_matrix(counts, dtype=np.float32); C.sort_indices()
    assert np.equal(C.data, np.round(C.data)).all() and (C.data >= 0).all()
    sp.save_npz(folder / 'counts.npz', C, compressed=True)
    np.savez_compressed(folder / 'metadata.npz', cell=part['cell'][keep], donor=part['donor'][keep],
                        type=part['type'][keep], donors=part['donors'], raw_c=part['raw_c'],
                        genes=genes, hvg=hvg)


def build(data, panel, k, overwrite=False):
    t0 = time.monotonic()
    f = FOLDS_JSON['inner_folds'][k]
    out = OUT_ROOT / panel / f'fold{k}'
    if out.exists():
        if not overwrite:
            raise FileExistsError(f'{out} exists; pass --overwrite')
        for p in sorted(out.rglob('*'), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    train = subset(data, f['train_donors']); held = subset(data, f['heldout_donors'])
    hvg, forced, n_positive = select_panel(train, panel)
    space, train_counts, keep_tr, pc, clf = fit_space(train, hvg)
    genes = train['genes'][hvg]
    held_counts = held['counts'][:, hvg].tocsr()
    keep_he = np.asarray(held_counts.sum(1)).ravel() > 0
    held_counts = held_counts[keep_he]
    if set(held['donor'][keep_he]) != set(held['donors']):
        raise ValueError('a heldout donor lost all cells')
    np.savez_compressed(out / 'space.npz', genes=genes, **space)
    torch.save(clf.state_dict(), out / 'type_classifier.pt')
    write_split(out / 'train', train, keep_tr, train_counts, genes, hvg)
    write_split(out / 'heldout', held, keep_he, held_counts, genes, hvg)
    per_tr, st_tr = split_stats(train, keep_tr); per_he, st_he = split_stats(held, keep_he)
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    sizes = {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob('*')) if p.is_file()}
    read = sorted(set(COUNT_FILES_OPENED))
    forbidden_read = sorted(set(read) - set(DEV)) + sorted(set(read) & set(SEALED))
    assert not forbidden_read
    man = {
        'study': 'improve_20261109', 'component': 'cache', 'panel': panel, 'inner_fold': k,
        'train_donors': f['train_donors'], 'heldout_donors': f['heldout_donors'],
        'n_genes': int(len(hvg)), 'n_pcs': N_PC, 'n_hvg_ranked': PANELS[panel]['n_hvg'],
        'n_genes_positive_hvg_score': n_positive,
        'forced_genes': forced,
        'cells': {'train_kept': int(keep_tr.sum()), 'heldout_kept': int(keep_he.sum()),
                  'train_raw': int(len(keep_tr)), 'heldout_raw': int(len(keep_he))},
        'cells_per_donor': {'train': per_tr, 'heldout': per_he},
        'stage_counts': {'train': st_tr, 'heldout': st_he},
        'dropped_zero_panel_cells': {'train': train['cell'][~keep_tr].tolist(),
                                     'heldout': held['cell'][~keep_he].tolist()},
        'age_scaler': {'age_mean': space['age_mean'], 'age_sd': space['age_sd'], 'scope': 'train donors, one value per donor, ddof=0'},
        'pc_convention': 'pcs = ((log1p(CP10k_panel) - mean)/sd) @ W / pc_sd; unit variance (ddof=0) per PC over kept train cells',
        'donor_guard': {
            'count_files_read_by_this_process': read,
            'all_read_donors_are_development_donors': set(read) <= set(DEV),
            'forbidden_donors_read': forbidden_read,
            'assertion': 'No count/name file of any donor outside the 37 development donors was opened; '
                         'check_donor_allowed raises before any file access; the counts directory was never globbed.',
            'metadata_note': 'mg_meta.json and candidate_cohort.json are single shared files; they are streamed with ijson and '
                             'only rows keyed by cell names of development donors (and development-donor cohort entries) are '
                             'kept; all other rows are discarded as parsed and never stored or inspected (fixer revision).'},
        'files_SHA256': files, 'file_bytes': sizes, 'total_bytes': int(sum(sizes.values())),
        'recipe': {
            'source': 'bench_core.fit_space / core-comparison fold_data.fit_space_only_training',
            'hvg_rule': 'CP10k float32, log1p, score=var/max(mean,.001)*(mean>.01), stable argsort desc, top n_hvg; train donors only',
            'wide_rule': 'top 5000 by the same ranking, then forced genes not already selected appended in listed order unless zero train counts',
            'zero_panel_rule': 'cells with zero total counts over the panel are dropped in train and heldout',
            'scaling': 'mean, sd=std+1e-6 of log1p(CP10k over panel) on kept train cells (float32)',
            'pca': 'sklearn randomized_svd(Z[rng(0).choice(n,min(20000,n))], 50, n_iter=5, random_state=0); W=Vt.T; pc_sd=max(std(Z@W),1e-6)',
            'raw_c': 'bench_core.encode_condition: Braak/6, Thal/5, CERAD/3, ADNC/3, Male, APOE4 dose, raw age',
            'counts_dtype': 'scipy CSR float32, integer valued',
            'type_classifier': 'bench_core.TypeClassifier on train PCs (extra, for Space compatibility)'},
        'source_pins': {str(p): sha(p) for p in [Path(__file__).resolve(), FOLDS, B / 'bench_core.py', GENES_FILE,
                                                 RAW / 'mg_meta.json', RAW / 'candidate_cohort.json']},
        'raw_sources': {'train': train['sources'], 'heldout': held['sources']},
        'versions': {'numpy': np.__version__, 'scipy': sp.__name__ and __import__('scipy').__version__,
                     'sklearn': __import__('sklearn').__version__, 'torch': torch.__version__, 'python': sys.version.split()[0]},
        'threads': THREADS, 'wall_seconds': time.monotonic() - t0,
        'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_initialized': torch.cuda.is_initialized(),
    }
    dump(out / 'manifest.json', man)
    print(json.dumps({'panel': panel, 'fold': k, 'n_genes': man['n_genes'], 'cells': man['cells'],
                      'forced_appended': len(forced['appended']), 'total_bytes': man['total_bytes'],
                      'wall_seconds': round(man['wall_seconds'], 1)}), flush=True)
    return man


def fidelity():
    """Rebuild the legacy fold-0 hvg2000 space on the 33 optimization donors and compare."""
    opt = [d for d in DEV if d not in LEGACY_CALIBRATION]
    lman = json.loads((LEGACY / 'manifest.json').read_text())
    assert len(opt) == 33 and sorted(opt) == sorted(lman['train_donors'])
    order = list(lman['train_donors'])  # legacy concatenation order
    data = read_donors(order)
    hvg, _, _ = select_panel(data, 'hvg2000')
    space, counts, keep, pc, clf = fit_space(data, hvg)
    L = np.load(LEGACY / 'space.npz'); LM = np.load(LEGACY / 'train/metadata.npz')
    LC = sp.load_npz(LEGACY / 'train/counts_hvg.npz').tocsr(); LP = np.load(LEGACY / 'train/real_pcs.npy')
    r = {'donors': order, 'n_donors': len(order)}
    r['hvg_identical_set_and_order'] = bool(np.array_equal(hvg, L['hvg']))
    r['hvg_set_identical'] = bool(set(hvg.tolist()) == set(L['hvg'].tolist()))
    r['genes_identical'] = bool(np.array_equal(data['genes'][hvg], LM['genes']))
    r['cells_identical_order'] = bool(np.array_equal(data['cell'][keep], LM['cell']))
    r['donor_identical'] = bool(np.array_equal(data['donor'][keep], LM['donor']))
    r['type_identical'] = bool(np.array_equal(data['type'][keep], LM['type']))
    r['n_cells'] = [int(keep.sum()), int(LC.shape[0])]
    r['dropped_zero_panel'] = int((~keep).sum())
    ours = sp.csr_matrix(counts, dtype=np.float64); theirs = sp.csr_matrix(LC, dtype=np.float64)
    r['counts_identical'] = bool(ours.shape == theirs.shape and (ours != theirs).nnz == 0)
    r['legacy_counts_dtype'] = str(LC.dtype)
    for k in ['mean', 'sd', 'pc_sd']:
        a, b = np.asarray(space[k], np.float64), np.asarray(L[k], np.float64)
        r[k] = {'bitwise_equal': bool(np.array_equal(space[k], L[k])), 'dtype': [str(space[k].dtype), str(L[k].dtype)],
                'max_abs_diff': float(np.abs(a - b).max()), 'max_rel_diff': float((np.abs(a - b) / np.maximum(np.abs(b), 1e-12)).max())}
    W, LW = np.asarray(space['W'], np.float64), np.asarray(L['W'], np.float64)
    sign = np.sign((W * LW).sum(0)); Ws = W * sign
    cos = (W * LW).sum(0) / np.linalg.norm(W, axis=0) / np.linalg.norm(LW, axis=0)
    sv = np.linalg.svd(W.T @ LW, compute_uv=False)
    r['W'] = {'bitwise_equal': bool(np.array_equal(space['W'], L['W'])), 'dtype': [str(space['W'].dtype), str(L['W'].dtype)],
              'sign_flips': int((sign < 0).sum()), 'max_abs_diff_after_sign': float(np.abs(Ws - LW).max()),
              'min_abs_column_cosine': float(np.abs(cos).min()),
              'per_column_abs_cosine_below_0.9999': [int(i) for i in np.where(np.abs(cos) < .9999)[0]],
              'subspace_min_principal_cosine': float(sv.min())}
    r['age'] = {'ours': [space['age_mean'], space['age_sd']], 'legacy': [float(L['age_mean']), float(L['age_sd'])]}
    pc_s = pc * sign.astype(np.float32)
    r['train_pcs_vs_legacy_real_pcs_max_abs_diff_after_sign'] = float(np.abs(pc_s - LP).max())
    r['raw_c'] = {'slot3_ADNC_equal': bool(np.allclose(data['raw_c'][:, 3], LM['raw_c'][:, 3])),
                  'slot6_age_equal': bool(np.array_equal(data['raw_c'][:, 6], LM['raw_c'][:, 6])),
                  'note': 'legacy raw_c is stage-only (slots 0,1,2,4,5 zero by design); this study uses full encode_condition'}
    r['versions'] = {'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
                     'sklearn': __import__('sklearn').__version__, 'torch': torch.__version__,
                     'python_executable': sys.executable}
    r['legacy_versions'] = lman.get('versions')
    r['count_files_read'] = sorted(set(COUNT_FILES_OPENED))
    return r


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--fidelity', action='store_true')
    a.add_argument('--fidelity-out', type=Path, default=OUT_ROOT / 'fidelity_legacy_fold0.json')
    a.add_argument('--panel', choices=['hvg2000', 'wide', 'all'], default='all')
    a.add_argument('--fold', choices=['0', '1', '2', 'all'], default='all')
    a.add_argument('--overwrite', action='store_true')
    args = a.parse_args()
    validate_folds()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if args.fidelity:
        r = fidelity()
        if not str(args.fidelity_out.resolve()).startswith(('/ml2/alz/runs/improve_20261109/', '/tmp/')):
            raise PermissionError('fidelity output outside the study run directory')
        dump(args.fidelity_out, r); print(json.dumps(r, indent=1))
        return
    data = read_donors(DEV)
    panels = list(PANELS) if args.panel == 'all' else [args.panel]
    folds = range(3) if args.fold == 'all' else [int(args.fold)]
    for panel in panels:
        for k in folds:
            build(data, panel, k, overwrite=args.overwrite); gc.collect()


if __name__ == '__main__':
    main()
