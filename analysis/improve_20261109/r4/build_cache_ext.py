"""Round-4 caches ``hvg2000x``: SEA-AD fold k plus the external MADRC (GSE268599 ITG) microglia as extra TRAIN donors.

CLI (from /tmp; at most 3 CPU threads, no GPU)::

    python3 -B r4/build_cache_ext.py [--fold 0..8|all|confirm|K,K,..] [--overwrite] [--out-root DIR]

    --fold all      inner folds 0-2 (unchanged meaning)
    --fold confirm  inner folds 3-8 (r4/PROTOCOL_CONFIRM.md; folds of ../r2/folds_r2.json = r4/folds_r2.json)
    --out-root DIR  write DIR/fold{k} instead of caches/hvg2000x/fold{k}; DIR must lie under
                    runs/improve_20261109/smoke_r4/ or /tmp/ (scratch rebuilds, e.g. the fold 0-2 identity check
                    of test_cache_ext.py rebuild).  The real root is the default.

Extension to inner folds 3-8 (confirmation task, 2026-11-12): exactly the fold 0-2 recipe below, applied to
``caches/hvg2000/fold{k}`` k = 3..8 (built by r2/build_cache_r2.py from r2/folds_r2.json).  Nothing in the
recipe changed; the only additions are (i) the SEA-AD train and held-out donor sets of hvg2000 fold k are
checked against folds_r2.json inner fold k (and folds 0-2 also against ../folds.json) before anything is
written, and (ii) for k >= 3 the manifest's ``source_pins`` also pin r4/folds_r2.json.  Fold 0-2 output is
unchanged apart from the builder's own SHA256 in ``source_pins`` and the run-time fields (wall_seconds,
peak_RSS_KiB): ``test_cache_ext.py rebuild --folds 0,1,2`` rebuilt folds 0-2 into a scratch root on 2026-11-12
and every data file (train/*.npz included; numpy writes zip members with the fixed 1980 timestamp) was
byte-identical (SHA256) to the live caches/hvg2000x/fold{0,1,2}.

Writes ``/ml2/alz/runs/improve_20261109/caches/hvg2000x/fold{k}/`` (r4/PROTOCOL_EXT.md "Interfaces";
round-1 Interface 1 of ../PROTOCOL.md):

* ``space.npz``, ``type_classifier.pt``, ``heldout/counts.npz``, ``heldout/metadata.npz``: BYTE copies of
  ``caches/hvg2000/fold{k}/`` (same genes, scaling, PCA, held-out donors; checked by SHA256).  Nothing is
  refitted: gene selection, scaling and PCA stay those of the SEA-AD train donors of the fold.
* ``train/counts.npz`` (scipy CSR float32, integer valued, sorted indices): the rows of
  ``hvg2000/fold{k}/train/counts.npz`` unchanged and in the same order (the SEA-AD train donors of the
  fold, protocol order), FOLLOWED by every external donor of ``ext/GSE268599/donors.json`` in sorted donor
  order (MADRC-01 .. MADRC-32 without MADRC-27, which has no ITG library), each donor's cells in the
  store's row order, restricted to the panel (``space['hvg']`` indexes the same 36,601-gene axis: the
  external ``counts/genes.npy`` is checked equal to the SEA-AD rebuilt ``counts/genes.npy``).
  Zero-panel-count cells are dropped with the round-1 rule (panel total == 0) and counted per donor.
  CELL CAP (v2, review finding D2): each external donor then keeps at most EXT_CAP = 1000 cells, chosen by
  ``metrics.det_subsample(n, 1000, metrics.stable_seed('ext-cap', donor))`` over its nonzero-panel cells (the
  exact rule ``evaluate_ext.load_ext_entries`` uses, so cache-fed baselines see the same cells).  MADRC
  microglia yield falls with stage (kept cells rho -0.43 with ADNC; Not AD donors up to 6,777 cells), and
  every cell-weighted pooled quantity (generator mu/theta initialisation, pooled count baselines, cohort
  statistics) would otherwise be dominated by Not AD MADRC donors.  SEA-AD development donors have
  117-984 cells, so the cap never binds for them.  ``manifest['external_cell_cap']`` records the rule and
  per-donor counts before and after.
* ``train/metadata.npz``: ``cell``, ``donor`` (per cell), ``type`` (SEA-AD supertype index for SEA-AD
  cells as in hvg2000; -1 for external cells, which carry no SEA-AD supertype), ``donors`` (SEA-AD train
  donors then external donors), ``raw_c`` [n_donors, 7], ``genes``, ``hvg`` (as hvg2000) and the new
  ``cohort`` (int64 per donor: 0 = SEA-AD MTG, 1 = MADRC ITG) and ``raw_c_slots012`` (str per donor:
  how slots 0-2 were filled).
* ``manifest.json``: every hvg2000 manifest key (``train_donors`` = the SEA-AD train donors of the
  protocol fold, unchanged) plus ``external_train_donors``, ``cohort_by_donor``, per-donor cell counts of
  the external donors, cells/donors by stage and cohort, median panel library size by cohort, zero-panel
  drop fractions, source SHA256 pins, identity checks against hvg2000.

raw_c of external donors (bench_core.encode_condition semantics, slot by slot):

    slot 0 Braak/6, slot 2 CERAD/3 (v2, review finding D6): the PUBLISHED Braak_index/6 and CERAD_index/3
            from donors.json (v1 wrote 0, which reads as 'no pathology' for 13 High donors).
    slot 1 Thal/5: Thal is NOT published for MADRC; it is IMPUTED as the midpoint of thal_range_from_A
            (A0 -> 0, A1 -> 1.5, A2 -> 3, A3 -> 4.5) / 5, the same rule as evaluate_ext.ext_raw_c.
            ``raw_c_slots012`` records 'published_braak_cerad_thal_mid_A' for these donors and
            'encode_condition' for SEA-AD donors (rows copied unchanged).  No study condition reads
            slots 0-2 today (round-1 ``build_conditions`` writes 0 into condition slots 0-2; the cohort arm
            writes condition slot 1); an arm that ever reads raw_c slot 1 must treat it as imputed.
    slot 3 ADNC/3: ADNC_index/3 (NIA-AA 2012 from the published A, B, C; same STAGES order).
    slot 4 Male, slot 5 APOE4 dose: from donors.json (male, APOE4_dose).
    slot 6 raw age: exact age when published; '90+' (age_90plus_censored) is RECOMPUTED PER FOLD as the
            mean raw age of the fold's SEA-AD TRAIN donors aged >= 90 (the store's 94.35 is the same rule
            over all 37 development donors, which would use held-out ages).  Age enters no condition
            unless a covariate arm is on.

Donor guard: no SEA-AD count file is opened at all (the SEA-AD train rows come from the hvg2000 cache);
external counts are read only from ext/GSE268599/counts/ and verified against donors.json SHA256s.
GSE264648 is not read here.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = os.environ.get('CACHE_THREADS', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as MX  # noqa: E402  (r4 copy, byte-identical to r3/metrics.py)

HERE = Path(__file__).resolve().parent
RUNS = Path('/ml2/alz/runs/improve_20261109')
SRC_ROOT = RUNS / 'caches' / 'hvg2000'
OUT_ROOT = RUNS / 'caches' / 'hvg2000x'
EXT = RUNS / 'ext' / 'GSE268599'
EXT_DONORS = EXT / 'donors.json'
SEAAD_GENES = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt/counts/genes.npy')
FOLDS = HERE.parent / 'folds.json'
FOLDS_R2 = HERE / 'folds_r2.json'   # inner folds 0-8 (copy of ../r2/folds_r2.json)
SCRATCH_PREFIXES = (str(RUNS / 'smoke_r4') + '/', '/tmp/')
ALL_FOLDS = tuple(range(9))
STAGES = ['Not AD', 'Low', 'Intermediate', 'High']
PANEL = 'hvg2000x'
SOURCE_PANEL = 'hvg2000'
COPIED = ['space.npz', 'type_classifier.pt', 'heldout/counts.npz', 'heldout/metadata.npz']
COHORT_NAMES = {0: 'SEA-AD MTG', 1: 'MADRC ITG (GSE268599)'}
VERSION = 'hvg2000x-v2'
EXT_CAP = 1000


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n')
    tmp.replace(p)


def load_external():
    """Ordered external donor list and their donors.json records (ITG, cohort_code 1)."""
    q = json.loads(EXT_DONORS.read_text())
    donors = sorted(q['donors'])
    folds = json.loads(FOLDS.read_text())
    forbidden = set(folds['development_donors']) | set(folds['sealed_evaluation_donors_fold0'])
    for d in donors:
        r = q['donors'][d]
        if d in forbidden or not d.startswith('MADRC-'):
            raise ValueError(f'external donor id {d!r} collides with SEA-AD or is malformed')
        if r['cohort_code'] != 1 or not str(r['region']).startswith('ITG'):
            raise ValueError(f'{d}: not a MADRC ITG donor')
        if r['ADNC'] != STAGES[r['ADNC_index']]:
            raise ValueError(f'{d}: ADNC index inconsistent')
    return donors, q


def check_fold_donors(k, smani):
    """hvg2000 fold k donor sets must be inner fold k of folds_r2.json (and of ../folds.json for k < 3)."""
    q = json.loads(FOLDS_R2.read_text())
    r1 = json.loads(FOLDS.read_text())
    if set(q['development_donors']) != set(r1['development_donors']) or \
            set(q['sealed_evaluation_donors_fold0']) != set(r1['sealed_evaluation_donors_fold0']):
        raise ValueError('folds_r2.json development/sealed donors differ from folds.json')
    srcs = [q] + ([r1] if k < len(r1['inner_folds']) else [])
    for f in srcs:
        fk = f['inner_folds'][k]
        if int(fk['inner_fold']) != k or set(smani['train_donors']) != set(fk['train_donors']) or \
                set(smani['heldout_donors']) != set(fk['heldout_donors']):
            raise ValueError(f'hvg2000 fold{k} donors differ from inner fold {k} of the folds file')
    if int(smani['inner_fold']) != k or smani['panel'] != SOURCE_PANEL:
        raise ValueError(f'hvg2000 fold{k} manifest inner_fold/panel mismatch')


def ext_age(rec, train_raw_c):
    """Raw age for slot 6 (docstring): exact, or the fold's SEA-AD train mean over ages >= 90 for '90+'."""
    if rec['age_90plus_censored']:
        old = train_raw_c[:, 6][train_raw_c[:, 6] >= 90]
        if not len(old):
            raise ValueError('no SEA-AD train donor aged >= 90 to code 90+')
        return float(old.mean()), int(len(old))
    a = float(rec['age'])
    if str(rec['age_raw']).strip() != str(int(a)) and str(rec['age_raw']).strip() != str(a):
        raise ValueError(f'age {a} differs from age_raw {rec["age_raw"]!r}')
    return a, None


def read_external(donors, q, hvg):
    """{donor: (panel counts CSR float32 [cells,G] (all cells), names, record, source)}."""
    genes = np.load(EXT / 'counts' / 'genes.npy', allow_pickle=False)
    seaad = np.load(SEAAD_GENES, allow_pickle=False)
    if not np.array_equal(genes, seaad):
        raise ValueError('external gene axis differs from the SEA-AD rebuilt axis')
    out = {}
    for d in donors:
        r = q['donors'][d]
        pc, pn = EXT / r['counts_file'], EXT / r['names_file']
        if sha(pc) != r['counts_sha256'] or sha(pn) != r['names_sha256']:
            raise ValueError(f'{d}: external store SHA256 differs from donors.json')
        c = sp.load_npz(pc).tocsr()
        names = np.load(pn, allow_pickle=False).astype(str)
        if c.shape != (len(names), len(genes)) or c.shape[0] != r['cells_kept']:
            raise ValueError(f'{d}: count matrix shape {c.shape} vs names {len(names)} / cells_kept {r["cells_kept"]}')
        if (c.data < 0).any() or not np.issubdtype(c.dtype, np.integer):
            raise ValueError(f'{d}: counts must be nonnegative integers')
        full_lib = np.asarray(c.sum(1)).ravel()
        p = c[:, hvg].tocsr().astype(np.float32)
        out[d] = (p, names, r, {'donor': d, 'counts_SHA256': r['counts_sha256'], 'names_SHA256': r['names_sha256'],
                                'cells': int(c.shape[0]), 'stage': int(r['ADNC_index']),
                                'median_full_library': float(np.median(full_lib))})
    return out


def build(k, ext_data, donors, q, overwrite=False, out_root=None):
    t0 = time.monotonic()
    src = SRC_ROOT / f'fold{k}'
    out = Path(out_root or OUT_ROOT) / f'fold{k}'
    if out.exists():
        if not overwrite:
            raise FileExistsError(f'{out} exists; pass --overwrite')
        shutil.rmtree(out)
    smani = json.loads((src / 'manifest.json').read_text())
    check_fold_donors(k, smani)
    for rel in COPIED:   # source files must still be what the hvg2000 manifest recorded
        if sha(src / rel) != smani['files_SHA256'][rel]:
            raise ValueError(f'hvg2000 fold{k} {rel} differs from its manifest')
    (out / 'train').mkdir(parents=True)
    (out / 'heldout').mkdir(parents=True)
    for rel in COPIED:
        shutil.copyfile(src / rel, out / rel)
        if sha(out / rel) != smani['files_SHA256'][rel]:
            raise ValueError(f'copy of {rel} is not byte-identical')
    with np.load(src / 'space.npz', allow_pickle=False) as s:
        hvg = s['hvg'].astype(np.int64)
        genes = s['genes']
    sm = np.load(src / 'train' / 'metadata.npz', allow_pickle=False)
    sm = {kk: sm[kk] for kk in sm.files}
    sc = sp.load_npz(src / 'train' / 'counts.npz').tocsr()
    if list(sm['donors'].astype(str)) != list(smani['train_donors']):
        raise ValueError('hvg2000 train donor order differs from its manifest')
    if not np.array_equal(sm['genes'], genes) or not np.array_equal(sm['hvg'], hvg):
        raise ValueError('hvg2000 metadata genes/hvg differ from space')
    train_raw_c = sm['raw_c'].astype(np.float64)

    mats, cells, dl, types, raw_rows, per_ext, sources = [sc], [sm['cell']], [sm['donor']], [sm['type']], [], {}, []
    age_rule = {}
    for d in donors:
        p, names, r, source = ext_data[d]
        lib = np.asarray(p.sum(1)).ravel()
        keep = lib > 0
        if not keep.any():
            raise ValueError(f'{d} lost all cells on the panel')
        nz = np.where(keep)[0]
        if EXT_CAP and len(nz) > EXT_CAP:
            nz = nz[MX.det_subsample(len(nz), EXT_CAP, MX.stable_seed('ext-cap', d))]
        keep_c = np.zeros_like(keep)
        keep_c[nz] = True
        mats.append(p[nz])
        cells.append(names[nz]); dl.append(np.full(len(nz), d)); types.append(np.full(len(nz), -1, np.int64))
        age, n_old = ext_age(r, train_raw_c)
        if n_old is not None:
            age_rule[d] = {'age_raw': r['age_raw'], 'coded_age': age, 'n_seaad_train_donors_ge90': n_old}
        lo, hi = r['thal_range_from_A']
        raw_rows.append([r['Braak_index'] / 6, (lo + hi) / 2 / 5, r['CERAD_index'] / 3, r['ADNC_index'] / 3,
                         float(r['male']), float(r['APOE4_dose']), age])
        per_ext[d] = {'cells_raw': int(len(keep)), 'cells_nonzero_panel': int(keep.sum()), 'cells_kept': int(len(nz)),
                      'cells_dropped_zero_panel': int((~keep).sum()), 'cells_dropped_cap': int(keep.sum() - len(nz)),
                      'stage': int(r['ADNC_index']), 'cohort': 1,
                      'median_panel_library': float(np.median(lib[nz])),
                      'dropped_cells': names[~keep].tolist()}
        sources.append(source)
    C = sp.vstack(mats, format='csr').astype(np.float32)
    C.sort_indices()
    assert np.equal(C.data, np.round(C.data)).all() and (C.data >= 0).all()
    n_sea = sc.shape[0]
    # SEA-AD rows identical (values, structure, order)
    top = C[:n_sea]
    if (top != sc.astype(np.float32)).nnz or not np.array_equal(top.indptr, sc.indptr):
        raise ValueError('SEA-AD train rows changed')
    cell = np.concatenate(cells)
    if len(set(cell.tolist())) != len(cell):
        raise ValueError('duplicate cell names')
    donor = np.concatenate(dl)
    ids = list(sm['donors'].astype(str)) + donors
    raw_c = np.concatenate([train_raw_c, np.asarray(raw_rows, np.float64)])
    cohort = np.array([0] * len(sm['donors']) + [1] * len(donors), np.int64)
    slots012 = np.array(['encode_condition'] * len(sm['donors']) + ['published_braak_cerad_thal_mid_A'] * len(donors))
    sp.save_npz(out / 'train' / 'counts.npz', C, compressed=True)
    np.savez_compressed(out / 'train' / 'metadata.npz', cell=cell, donor=donor, type=np.concatenate(types),
                        donors=np.asarray(ids), raw_c=raw_c, genes=genes, hvg=hvg, cohort=cohort, raw_c_slots012=slots012)

    # ---------------------------------------------------------------- statistics
    lib_all = np.asarray(C.sum(1)).ravel()
    is_ext_cell = np.arange(C.shape[0]) >= n_sea
    held = sp.load_npz(out / 'heldout' / 'counts.npz')
    stage_cohort = {}
    stage_of = {d: int(round(raw_c[i, 3] * 3)) for i, d in enumerate(ids)}
    cells_of = {d: int(n) for d, n in zip(*np.unique(donor, return_counts=True))}
    for coh in (0, 1):
        tab = {}
        for s in range(4):
            ds = [d for i, d in enumerate(ids) if cohort[i] == coh and stage_of[d] == s]
            tab[STAGES[s]] = {'donors': len(ds), 'cells_kept': int(sum(cells_of[d] for d in ds))}
        stage_cohort[COHORT_NAMES[coh]] = tab
    stage_cohort['pooled'] = {STAGES[s]: {kk: sum(stage_cohort[COHORT_NAMES[c]][STAGES[s]][kk] for c in (0, 1))
                                          for kk in ('donors', 'cells_kept')} for s in range(4)}
    ext_raw = sum(v['cells_raw'] for v in per_ext.values())
    ext_drop = sum(v['cells_dropped_zero_panel'] for v in per_ext.values())
    ext_capd = sum(v['cells_dropped_cap'] for v in per_ext.values())
    stats = {
        'cells_donors_by_stage_and_cohort': stage_cohort,
        'heldout_SEAAD': smani['stage_counts']['heldout'],
        'median_panel_library_per_cell': {
            'SEA-AD train': float(np.median(lib_all[~is_ext_cell])), 'MADRC train': float(np.median(lib_all[is_ext_cell])),
            'SEA-AD heldout': float(np.median(np.asarray(held.sum(1)).ravel()))},
        'median_of_donor_median_panel_library': {
            'SEA-AD train': float(np.median([np.median(lib_all[donor == d]) for d in sm['donors'].astype(str)])),
            'MADRC train': float(np.median([v['median_panel_library'] for v in per_ext.values()]))},
        'zero_panel_cells_dropped': {
            'MADRC': {'raw': ext_raw, 'dropped': ext_drop, 'fraction': ext_drop / ext_raw},
            'SEA-AD train (from hvg2000)': {'raw': smani['cells']['train_raw'],
                                            'dropped': smani['cells']['train_raw'] - smani['cells']['train_kept']}},
        'external_cell_share_of_train_cells': float(is_ext_cell.mean()),
        'external_donor_share_of_train_donors': float(cohort.mean()),
    }
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    sizes = {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob('*')) if p.is_file()}
    man = {kk: smani[kk] for kk in smani if kk not in ('files_SHA256', 'file_bytes', 'total_bytes', 'source_pins', 'versions',
                                                         'threads', 'wall_seconds', 'peak_RSS_KiB', 'cuda_initialized', 'component')}
    man.update({
        'component': 'cache_ext', 'version': VERSION, 'panel': PANEL, 'source_panel': SOURCE_PANEL, 'inner_fold': k,
        'train_donors': list(smani['train_donors']),
        'external_train_donors': donors,
        'external_dataset': 'GEO GSE268599 (MADRC, Serrano-Pozo 2024 / Wachter 2024), ITG microglia/PVM, selection '
                            + str(q.get('selection_version')),
        'cohort_codes': {str(c): n for c, n in COHORT_NAMES.items()},
        'cohort_by_donor': {d: int(c) for d, c in zip(ids, cohort)},
        'train_donor_order': 'SEA-AD train donors of the fold (hvg2000 order) followed by external donors (sorted ids)',
        'external_cell_cap': {'cap': EXT_CAP, 'rule': "metrics.det_subsample(n_nonzero_panel, cap, "
                                                       "metrics.stable_seed('ext-cap', donor)) after the zero-panel drop "
                                                       "(same rule as evaluate_ext.load_ext_entries)",
                              'cells_dropped_by_cap': ext_capd,
                              'donors_capped': sorted(d for d, v in per_ext.items() if v['cells_dropped_cap'])},
        'cells': {**smani['cells'], 'external_train_raw': ext_raw, 'external_train_kept': ext_raw - ext_drop - ext_capd,
                  'train_total_kept': int(C.shape[0])},
        'cells_per_donor': {**smani['cells_per_donor'], 'external': per_ext},
        'stage_counts': {**smani['stage_counts'], 'external': stage_cohort[COHORT_NAMES[1]],
                         'train_pooled': stage_cohort['pooled']},
        'statistics': stats,
        'external_raw_c': {
            'slots': 'Braak/6, Thal/5, CERAD/3, ADNC/3, Male, APOE4 dose, raw age (bench_core.encode_condition order)',
            'slots_0_2': 'external: published Braak_index/6, Thal IMPUTED as mid(thal_range_from_A)/5, published '
                         'CERAD_index/3 (no study condition reads them); SEA-AD rows unchanged',
            'slot_3': 'ADNC_index/3 from donors.json (NIA-AA 2012 from published A, B, C)',
            'slots_4_5': 'male, APOE4_dose from donors.json',
            'slot_6': "exact age, or '90+' -> mean raw age of this fold's SEA-AD train donors aged >= 90",
            'age_90plus': age_rule},
        'identity_with_source': {
            'byte_identical_files': {rel: smani['files_SHA256'][rel] for rel in COPIED},
            'seaad_train_rows_identical': True, 'seaad_train_rows': int(n_sea),
            'source_manifest_SHA256': sha(src / 'manifest.json')},
        'files_SHA256': files, 'file_bytes': sizes, 'total_bytes': int(sum(sizes.values())),
        'source_pins': {str(p): sha(p) for p in [Path(__file__).resolve(), FOLDS, EXT_DONORS, EXT / 'counts' / 'genes.npy',
                                                 SEAAD_GENES, src / 'manifest.json'] + ([FOLDS_R2] if k >= 3 else [])},
        'raw_sources': {**smani['raw_sources'], 'external': sources},
        'donor_guard': {**smani['donor_guard'],
                        'this_builder': 'opened no SEA-AD count or name file (SEA-AD rows copied from the hvg2000 cache); '
                                        'external counts only from ext/GSE268599/counts with donors.json SHA256 checks'},
        'versions': {'numpy': np.__version__, 'scipy': __import__('scipy').__version__, 'python': sys.version.split()[0]},
        'threads': int(os.environ['OMP_NUM_THREADS']), 'wall_seconds': time.monotonic() - t0,
        'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    dump(out / 'manifest.json', man)
    print(json.dumps({'fold': k, 'cells': man['cells'], 'by_stage_cohort': stage_cohort,
                      'median_panel_library': stats['median_panel_library_per_cell'],
                      'zero_panel_dropped': stats['zero_panel_cells_dropped'], 'total_bytes': man['total_bytes']}), flush=True)
    return man


def parse_folds(text):
    """'all' -> 0-2 (unchanged meaning), 'confirm' -> 3-8, else a comma list of folds 0-8."""
    if text == 'all':
        return [0, 1, 2]
    if text == 'confirm':
        return [3, 4, 5, 6, 7, 8]
    out = []
    for x in text.split(','):
        if not x.strip().isdigit() or int(x) not in ALL_FOLDS:
            raise ValueError(f'bad --fold {text!r} (want 0..8, all, confirm or a comma list)')
        out.append(int(x))
    return sorted(set(out))


def check_out_root(out_root):
    """The real root, or a scratch root under smoke_r4/ or /tmp/."""
    o = Path(out_root).resolve()
    if o == OUT_ROOT.resolve():
        return o
    if not any(str(o).startswith(p) for p in SCRATCH_PREFIXES):
        raise PermissionError(f'refusing --out-root {o}: only {OUT_ROOT} or a scratch root under {SCRATCH_PREFIXES}')
    return o


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--fold', default='all', help="0..8, 'all' (= 0,1,2), 'confirm' (= 3..8) or a comma list")
    a.add_argument('--overwrite', action='store_true')
    a.add_argument('--out-root', default=str(OUT_ROOT))
    args = a.parse_args(argv)
    out_root = check_out_root(args.out_root)
    donors, q = load_external()
    folds = parse_folds(args.fold)
    cache = {}
    mans = {}
    for k in folds:
        with np.load(SRC_ROOT / f'fold{k}' / 'space.npz', allow_pickle=False) as s:
            hvg = s['hvg'].astype(np.int64)
        key = hvg.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = read_external(donors, q, hvg)
        mans[k] = build(k, cache[key], donors, q, overwrite=args.overwrite, out_root=out_root)
    return mans


if __name__ == '__main__':
    main()
