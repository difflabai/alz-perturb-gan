"""Tests for the hvg2000x caches (r4/build_cache_ext.py) and the round-4 trainer split checks.

CPU only, 3 threads.  Run from /tmp::

    python3 -B r4/test_cache_ext.py [integrity|context|all|rebuild|fit20] [--folds 0,1,2]

--folds    folds to test (default 0,1,2 = the original behaviour; 'confirm' = 3..8; e.g. 0,1,2,3,4,5,6,7,8).
           Fold donor sets are checked against r4/folds_r2.json (inner folds 0-8; folds 0-2 equal ../folds.json).

integrity  for folds 0-2: space.npz, type_classifier.pt, heldout/* byte-identical to hvg2000 (SHA256 of the
           live hvg2000 files, not only of its manifest); genes identical; the SEA-AD train block of
           train/counts.npz and of every metadata array identical (values, CSR structure, order) to
           hvg2000; the external block equal to the panel columns of ext/GSE268599/counts/{donor}.npz
           recomputed here (zero-panel rule applied and counted, then the v2 cap of 1,000 cells per donor
           with the evaluate_ext rule det_subsample(n, 1000, stable_seed('ext-cap', donor))), donors in sorted
           order; cohort array; external raw_c (v2: slot 0 = Braak_index/6, slot 1 = mid(thal_range_from_A)/5
           (imputed), slot 2 = CERAD_index/3, slot 3 = ADNC_index/3, slots 4-5 = male / APOE4 dose, slot 6 =
           exact age or the fold's SEA-AD train >= 90 mean); manifest SHA256s; SEA-AD donor sets equal
           to folds.json; no external id among SEA-AD ids.
context    trainer.Context loads every fold (protocol split True, 31 external donors validated,
           target_idx = the SEA-AD block); validate_external refuses an external donor that is held out,
           a SEA-AD id posing as external, and a stage mismatch; Context refuses a cohort block in the
           wrong order (checked on the in-memory arrays).
rebuild    (confirmation task) the EDITED builder rebuilds the given folds (default 0,1,2) into a scratch root
           smoke_r4/confirm_cache_rebuild/ (never over the live caches) and every file is compared with the live
           caches/hvg2000x/fold{k}: space.npz, type_classifier.pt, heldout/* byte-identical (SHA256);
           train/counts.npz and train/metadata.npz: same zip members, and per member identical CRC32,
           uncompressed and compressed size, compression type and decompressed bytes, and whole-file SHA256
           (numpy writes zip members with the fixed 1980 timestamp, so the files are byte-identical); manifest.json
           equal except wall_seconds, peak_RSS_KiB, the builder's own SHA256 pin and the SHA256 of the two train
           .npz files (timestamps).  The scratch root is deleted afterwards.
fit20      (confirmation task) a 20-update CPU fit of the r4 trainer on hvg2000x fold 3 (arm g.dispz,pbulk,
           seed 0, --threads 3) into smoke_r4/confirm_fit20 (deleted afterwards): the trainer's split validation
           accepts a fold-3..8 cache.
Results: /ml2/alz/runs/improve_20261109/smoke_r4/test_cache_ext.json (keys 'integrity'/'context' for folds 0-2,
'integrity_folds345678' etc. for other fold sets, 'rebuild_folds012', 'fit20')
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import scipy.sparse as sp
import metrics as MX

RUNS = Path('/ml2/alz/runs/improve_20261109')
SRC = RUNS / 'caches/hvg2000'
DST = RUNS / 'caches/hvg2000x'
EXT = RUNS / 'ext/GSE268599'
OUT = RUNS / 'smoke_r4/test_cache_ext.json'
FOLDS_R1 = json.loads((HERE.parent / 'folds.json').read_text())
FOLDS = json.loads((HERE / 'folds_r2.json').read_text())     # inner folds 0-8; 0-2 checked equal to folds.json below
for _k in range(len(FOLDS_R1['inner_folds'])):
    _a, _b = FOLDS_R1['inner_folds'][_k], FOLDS['inner_folds'][_k]
    assert set(_a['train_donors']) == set(_b['train_donors']) and set(_a['heldout_donors']) == set(_b['heldout_donors'])
assert FOLDS['development_donors'] == FOLDS_R1['development_donors'] or set(FOLDS['development_donors']) == set(FOLDS_R1['development_donors'])
assert set(FOLDS['sealed_evaluation_donors_fold0']) == set(FOLDS_R1['sealed_evaluation_donors_fold0'])
CAP = 1000
SCRATCH = RUNS / 'smoke_r4/confirm_cache_rebuild'
FIT20 = RUNS / 'smoke_r4/confirm_fit20'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def integrity(folds=(0, 1, 2)):
    ext = json.loads((EXT / 'donors.json').read_text())['donors']
    donors = sorted(ext)
    res = {}
    for k in folds:
        s, d = SRC / f'fold{k}', DST / f'fold{k}'
        r = {}
        r['byte_identical'] = {rel: sha(s / rel) == sha(d / rel) for rel in
                               ('space.npz', 'type_classifier.pt', 'heldout/counts.npz', 'heldout/metadata.npz')}
        sm, dm = np.load(s / 'train/metadata.npz'), np.load(d / 'train/metadata.npz')
        sc, dc = sp.load_npz(s / 'train/counts.npz').tocsr(), sp.load_npz(d / 'train/counts.npz').tocsr()
        n, ns = sc.shape[0], len(sm['donors'])
        top = dc[:n]
        r['seaad_rows_identical'] = bool(top.shape == sc.shape and top.dtype == sc.dtype and np.array_equal(top.indptr, sc.indptr)
                                         and np.array_equal(top.indices, sc.indices) and np.array_equal(top.data, sc.data))
        r['seaad_metadata_identical'] = {key: bool(np.array_equal(dm[key][:n] if key in ('cell', 'donor', 'type') else dm[key][:ns], sm[key]))
                                         for key in ('cell', 'donor', 'type', 'donors', 'raw_c')}
        r['genes_hvg_identical'] = bool(np.array_equal(dm['genes'], sm['genes']) and np.array_equal(dm['hvg'], sm['hvg']))
        with np.load(d / 'space.npz') as sp_:
            hvg = sp_['hvg']
            r['space_genes_equal_metadata'] = bool(np.array_equal(sp_['genes'], dm['genes']))
        ids = dm['donors'].astype(str)
        r['donor_order'] = bool(list(ids) == list(sm['donors'].astype(str)) + donors)
        f = FOLDS['inner_folds'][k]
        man = json.loads((d / 'manifest.json').read_text())
        r['manifest_sets'] = bool(set(man['train_donors']) == set(f['train_donors']) and set(man['heldout_donors']) == set(f['heldout_donors'])
                                  and man['train_donors'] == json.loads((s / 'manifest.json').read_text())['train_donors']
                                  and man['external_train_donors'] == donors and man['panel'] == 'hvg2000x' and man['inner_fold'] == k)
        r['no_seaad_id_external'] = not (set(donors) & (set(FOLDS['development_donors']) | set(FOLDS['sealed_evaluation_donors_fold0'])))
        r['cohort'] = bool(list(dm['cohort']) == [0] * ns + [1] * len(donors))
        r['manifest_sha'] = all(sha(d / rel) == h for rel, h in man['files_SHA256'].items())
        # external block recomputed from the store
        rows, ok, dropped, capped, raw = n, True, 0, 0, dm['raw_c']
        old = sm['raw_c'][:, 6][sm['raw_c'][:, 6] >= 90].mean()
        raw_ok = True
        for j, dn in enumerate(donors):
            e = ext[dn]
            c = sp.load_npz(EXT / e['counts_file']).tocsr()[:, hvg].astype(np.float32)
            names = np.load(EXT / e['names_file']).astype(str)
            nzm = np.asarray(c.sum(1)).ravel() > 0
            dropped += int((~nzm).sum())
            keep = np.where(nzm)[0]
            if len(keep) > CAP:
                keep = keep[MX.det_subsample(len(keep), CAP, MX.stable_seed('ext-cap', dn))]
            capped += int(nzm.sum()) - len(keep)
            m = len(keep)
            blk = dc[rows:rows + m]
            ok &= bool((blk != c[keep]).nnz == 0 and np.array_equal(dm['cell'][rows:rows + m], names[keep])
                       and (dm['donor'][rows:rows + m] == dn).all() and (dm['type'][rows:rows + m] == -1).all())
            rows += m
            rc = raw[ns + j]
            age = old if e['age_90plus_censored'] else float(e['age'])
            lo, hi = e['thal_range_from_A']
            raw_ok &= bool(np.array_equal(rc[:3], np.array([e['Braak_index'] / 6, (lo + hi) / 2 / 5, e['CERAD_index'] / 3]))
                           and rc[3] == e['ADNC_index'] / 3 and rc[4] == e['male']
                           and rc[5] == e['APOE4_dose'] and rc[6] == age)
        r['external_rows_recomputed_equal'] = bool(ok and rows == dc.shape[0])
        r['external_raw_c'] = raw_ok
        r['zero_panel_dropped_external'] = dropped
        r['cap_dropped_external'] = capped
        r['cap_manifest_equal'] = bool(man['external_cell_cap']['cap'] == CAP
                                       and man['external_cell_cap']['cells_dropped_by_cap'] == capped)
        r['max_cells_per_external_donor'] = int(max(np.unique(dm['donor'][n:], return_counts=True)[1]))
        r['cap_respected'] = r['max_cells_per_external_donor'] <= CAP
        r['manifest_zero_panel'] = man['statistics']['zero_panel_cells_dropped']['MADRC']
        r['counts_integer_float32'] = bool(dc.dtype == np.float32 and np.array_equal(dc.data, np.round(dc.data)) and (dc.data >= 0).all())
        r['stats'] = {'cells_by_stage_cohort': man['statistics']['cells_donors_by_stage_and_cohort'],
                      'median_panel_library': man['statistics']['median_panel_library_per_cell'],
                      'external_cell_share': man['statistics']['external_cell_share_of_train_cells']}
        flat = [v for kk, v in r.items() if isinstance(v, bool)] + list(r['byte_identical'].values()) + list(r['seaad_metadata_identical'].values())
        r['PASS'] = bool(all(flat) and dropped == man['statistics']['zero_panel_cells_dropped']['MADRC']['dropped'])
        res[f'fold{k}'] = r
        print(f'[test_cache_ext] integrity fold{k}', r['PASS'], flush=True)
    res['PASS'] = all(v['PASS'] for v in res.values())
    return res


def context(folds=(0, 1, 2)):
    import trainer
    import arms
    res = {}
    for k in folds:
        ctx = trainer.Context(DST / f'fold{k}', 'cpu', arms.parse_arm('base'))
        ns = len(FOLDS['inner_folds'][k]['train_donors'])
        res[f'fold{k}'] = {'protocol_split': ctx.protocol_split, 'n_external': ctx.external_validation['n_external'],
                           'target_idx_is_seaad_block': list(ctx.target_idx) == list(range(ns)),
                           'n_train': len(ctx.ids), 'cells': int(ctx.X.shape[0])}
        res[f'fold{k}']['PASS'] = bool(ctx.protocol_split and ctx.external_validation['n_external'] == 31
                                       and res[f'fold{k}']['target_idx_is_seaad_block'])
        if k == folds[0]:
            man = dict(ctx.manifest)
            ext_ids, ext_raw = ctx.ids[ns:], ctx.raw_c[ns:]
            refusals = {}
            for label, mm, ids_, raw_ in [
                    ('external_heldout', {**man, 'heldout_donors': man['heldout_donors'] + [str(ext_ids[0])]}, ext_ids, ext_raw),
                    ('seaad_id_as_external', {**man, 'external_train_donors': [FOLDS['development_donors'][0]] + list(man['external_train_donors'][1:])},
                     np.array([FOLDS['development_donors'][0]] + list(ext_ids[1:])), ext_raw),
                    ('stage_mismatch', man, ext_ids, np.vstack([ext_raw[:1] * np.array([1, 1, 1, 0, 1, 1, 1]) + np.array([0, 0, 0, (round(ext_raw[0, 3] * 3) + 1) % 4 / 3, 0, 0, 0]), ext_raw[1:]]))]:
                try:
                    trainer.validate_external(mm, ids_, raw_)
                    refusals[label] = False
                except ValueError:
                    refusals[label] = True
            res['refusals'] = refusals
            res['refusals']['PASS'] = all(refusals.values())
        del ctx
        print(f'[test_cache_ext] context fold{k}', res[f'fold{k}']['PASS'], flush=True)
    res['PASS'] = all(v['PASS'] for v in res.values())
    return res


def _zip_members(p):
    import zipfile
    with zipfile.ZipFile(p) as z:
        return {i.filename: (i.CRC, i.file_size, i.compress_size, i.compress_type, z.read(i.filename), i.date_time)
                for i in z.infolist()}


VOLATILE_MANIFEST = ('wall_seconds', 'peak_RSS_KiB')
TIMESTAMPED = ('train/counts.npz', 'train/metadata.npz')


def rebuild(folds=(0, 1, 2)):
    """Rebuild into SCRATCH with the edited builder and compare with the live caches (docstring)."""
    import shutil
    import build_cache_ext as B
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    res = {}
    try:
        B.main(['--fold', ','.join(map(str, folds)), '--out-root', str(SCRATCH)])
        builder = str(Path(B.__file__).resolve())
        for k in folds:
            live, new = DST / f'fold{k}', SCRATCH / f'fold{k}'
            r = {}
            lf = sorted(str(p.relative_to(live)) for p in live.rglob('*') if p.is_file())
            nf = sorted(str(p.relative_to(new)) for p in new.rglob('*') if p.is_file())
            r['same_file_list'] = lf == nf
            r['byte_identical'] = {rel: sha(live / rel) == sha(new / rel) for rel in lf if rel != 'manifest.json'}
            for rel in TIMESTAMPED:
                a, b = _zip_members(live / rel), _zip_members(new / rel)
                same = set(a) == set(b) and all(a[m][:5] == b[m][:5] for m in a)
                r[f'{rel}_members_identical'] = bool(same)
                r[f'{rel}_timestamps_differ'] = sorted({str(a[m][5]) + ' vs ' + str(b[m][5]) for m in a if m in b and a[m][5] != b[m][5]})
            ma, mb = json.loads((live / 'manifest.json').read_text()), json.loads((new / 'manifest.json').read_text())
            diffs = []
            for key in sorted(set(ma) | set(mb)):
                if key in VOLATILE_MANIFEST:
                    continue
                va, vb = ma.get(key), mb.get(key)
                if key == 'source_pins':
                    va = {kk: v for kk, v in va.items() if kk != builder}
                    vb = {kk: v for kk, v in vb.items() if kk != builder}
                if key == 'files_SHA256':
                    va = {kk: v for kk, v in va.items() if kk not in TIMESTAMPED}
                    vb = {kk: v for kk, v in vb.items() if kk not in TIMESTAMPED}
                if va != vb:
                    diffs.append(key)
            r['manifest_differences_outside_volatile'] = diffs
            r['builder_sha_live_vs_edited'] = [ma['source_pins'].get(builder), mb['source_pins'].get(builder)]
            r['PASS'] = bool(r['same_file_list'] and not diffs
                             and all(v for rel, v in r['byte_identical'].items() if rel not in TIMESTAMPED)
                             and all(r[f'{rel}_members_identical'] for rel in TIMESTAMPED))
            res[f'fold{k}'] = r
            print(f'[test_cache_ext] rebuild fold{k}', r['PASS'], flush=True)
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    res['PASS'] = all(v['PASS'] for v in res.values())
    return res


def fit20(fold=3):
    """20-update CPU fit of the r4 trainer on hvg2000x fold `fold` (split validation for folds 3-8)."""
    import shutil
    import subprocess
    if FIT20.exists():
        shutil.rmtree(FIT20)
    cmd = ['/mnt/nvme/ai/conda/anaconda3/bin/python3', '-B', str(HERE / 'trainer.py'), '--cache', str(DST / f'fold{fold}'),
           '--arm', 'g.dispz,pbulk', '--seed', '0', '--out-root', str(FIT20), '--updates', '20', '--device', 'cpu',
           '--threads', '3']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '3', 'MKL_NUM_THREADS': '3'}
    try:
        p = subprocess.run(cmd, cwd='/tmp', env=env, capture_output=True, text=True)
        fd = FIT20 / 'g.dispz,pbulk' / 'hvg2000x' / f'fold{fold}' / 'seed0'
        rec = json.loads((fd / 'fit_receipt.json').read_text()) if (fd / 'fit_receipt.json').exists() else None
        cfg = json.loads((fd / 'config.json').read_text()) if (fd / 'config.json').exists() else {}
        res = {'fold': fold, 'returncode': p.returncode, 'stderr_tail': p.stderr[-1500:],
               'status': rec.get('status') if rec else None, 'protocol_split': cfg.get('protocol_split'),
               'n_train_donors': len(cfg.get('train_donors', [])), 'generated': (fd / 'generated.npz').exists()}
        res['PASS'] = bool(p.returncode == 0 and res['status'] == 'completed' and res['protocol_split'] is True
                           and res['generated'])
    finally:
        shutil.rmtree(FIT20, ignore_errors=True)
    print('[test_cache_ext] fit20', res['PASS'], flush=True)
    return res


def _parse_folds(text):
    if text == 'confirm':
        return (3, 4, 5, 6, 7, 8)
    return tuple(int(x) for x in text.split(','))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--folds')]
    folds = (0, 1, 2)
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith('--folds='):
            folds = _parse_folds(a.split('=', 1)[1])
        elif a == '--folds':
            folds = _parse_folds(sys.argv[i + 2])
            args = [x for x in args if x != sys.argv[i + 2]]
    what = args[0] if args else 'all'
    tag = '' if folds == (0, 1, 2) else '_folds' + ''.join(map(str, folds))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    ran = []
    if what in ('integrity', 'all'):
        out['integrity' + tag] = integrity(folds); ran.append('integrity' + tag)
    if what in ('context', 'all'):
        out['context' + tag] = context(folds); ran.append('context' + tag)
    if what == 'rebuild':
        out['rebuild_folds' + ''.join(map(str, folds))] = rebuild(folds); ran.append('rebuild_folds' + ''.join(map(str, folds)))
    if what == 'fit20':
        out['fit20'] = fit20(folds[0] if tag else 3); ran.append('fit20')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str) + '\n')
    ok = all(out[k].get('PASS') for k in ran)
    print('[test_cache_ext]', 'PASS' if ok else 'FAIL', {kk: out[kk].get('PASS') for kk in ran}, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
