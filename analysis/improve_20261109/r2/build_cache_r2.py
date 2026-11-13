"""Round-2 hvg2000 caches for inner folds 3-8 of r2/folds_r2.json (F9).

Usage (CPU only, 1 thread by default, at most 3 via CACHE_THREADS):
  /mnt/nvme/ai/conda/anaconda3/bin/python3 -B r2/build_cache_r2.py --fold 3,4,5,6,7,8
  /mnt/nvme/ai/conda/anaconda3/bin/python3 -B r2/build_cache_r2.py --fold 0 \
      --out-root /ml2/alz/runs/improve_20261109/smoke_r2/splits_scratch/caches   # reproduction check

Thin wrapper around the frozen round-1 ../build_cache.py, imported read-only.
Nothing in the recipe changes: the same read_donors (receipt/SHA checks, cell
cohort rules, donor guard), select_panel (train-only HVG ranking), fit_space
(train-only scaling, PCA, age scaler, type classifier), write_split and
build() run unmodified. Only four module globals are rebound before use:

  FOLDS       -> r2/folds_r2.json   (recorded in manifest source_pins)
  FOLDS_JSON  -> its parsed content (inner folds 0-8; validate_folds() and the
                 per-donor stage check then run against it)
  OUT_ROOT    -> /ml2/alz/runs/improve_20261109/caches (or --out-root)
  dump        -> round-1 dump, plus one extra manifest key 'r2_build'
                 (wrapper and generator SHA256, folds_r2.json SHA256, repeat,
                 thread count). Every round-1 manifest key is unchanged; the
                 extra key is not part of files_SHA256 (the manifest never is).

Threads: round-1 caches were built with CACHE_THREADS=1; the default here is
also 1, which reproduces round-1 fold 0 byte for byte (test_splits.py
rebuild0). CACHE_THREADS is capped at 3 by this wrapper.

Guards: panel hvg2000 only (the wide panel is not built in round 2); into the
real cache root only folds 3-8 may be written and existing folders are never
overwritten; folds 0-2 may be built only into a scratch root under
runs/improve_20261109/smoke_r2/ or /tmp/. The donor guard of build_cache.py
(only the 37 development donors may be opened) is unchanged and its record is
asserted in each manifest.
"""
import os
import sys

sys.dont_write_bytecode = True
_threads = os.environ.get('CACHE_THREADS', '1')
if not _threads.isdigit() or not 1 <= int(_threads) <= 3:
    raise SystemExit('CACHE_THREADS must be 1, 2 or 3')
os.environ['CACHE_THREADS'] = _threads
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import gc
import hashlib
import json
from pathlib import Path

R2 = Path(__file__).resolve().parent
R1 = R2.parent
FOLDS_R2 = R2 / 'folds_r2.json'
CACHE_ROOT = Path('/ml2/alz/runs/improve_20261109/caches')
SCRATCH_PREFIXES = ('/ml2/alz/runs/improve_20261109/smoke_r2/', '/tmp/')
NEW_FOLDS = (3, 4, 5, 6, 7, 8)
PANEL = 'hvg2000'

sys.path.insert(0, str(R1))
import build_cache as bc  # noqa: E402  (frozen round-1 module, read-only)
sys.path.remove(str(R1))


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


_R1_DUMP = bc.dump


def configure(out_root):
    """Rebind the round-1 module globals (see module docstring)."""
    q = json.loads(FOLDS_R2.read_text())
    r1 = json.loads((R1 / 'folds.json').read_text())
    assert q['development_donors'] == r1['development_donors'] == bc.DEV
    assert q['sealed_evaluation_donors_fold0'] == r1['sealed_evaluation_donors_fold0'] == bc.SEALED
    assert q['stage_by_donor'] == r1['stage_by_donor']
    for k in range(3):
        a, b = r1['inner_folds'][k], q['inner_folds'][k]
        assert (a['inner_fold'], a['train_donors'], a['heldout_donors']) == (b['inner_fold'], b['train_donors'], b['heldout_donors'])
    assert [f['inner_fold'] for f in q['inner_folds']] == list(range(9))
    bc.FOLDS = FOLDS_R2
    bc.FOLDS_JSON = q
    bc.OUT_ROOT = Path(out_root)
    extra = {'wrapper': str(Path(__file__).resolve()), 'wrapper_SHA256': _sha(Path(__file__).resolve()),
             'generator': str(R2 / 'make_folds_r2.py'), 'generator_SHA256': _sha(R2 / 'make_folds_r2.py'),
             'folds_file': str(FOLDS_R2), 'folds_SHA256': _sha(FOLDS_R2), 'threads': int(_threads),
             'recipe': 'round-1 build_cache.py unmodified (imported); only FOLDS, FOLDS_JSON, OUT_ROOT, dump rebound'}

    def dump(p, obj):
        if Path(p).name == 'manifest.json' and isinstance(obj, dict) and 'inner_fold' in obj:
            obj = dict(obj)
            obj['r2_build'] = dict(extra, repeat=int(q['inner_folds'][obj['inner_fold']]['repeat']))
        return _R1_DUMP(p, obj)

    bc.dump = dump
    validate_folds_r2(q)
    return q


def validate_folds_r2(q):
    """round-1 validate_folds, applied per repeat (each repeat's held-out sets partition the 37 donors).

    build_cache.validate_folds asserts that the held-out sets of ALL inner folds partition the
    development donors, which is true only within one repeat, so it is not called on folds_r2.json."""
    dev, sealed = list(q['development_donors']), list(q['sealed_evaluation_donors_fold0'])
    assert len(dev) == 37 == len(set(dev)), 'need 37 unique development donors'
    assert not set(dev) & set(sealed), 'development and sealed donors overlap'
    assert set(q['stage_by_donor']) == set(dev)
    held = {}
    for k, f in enumerate(q['inner_folds']):
        assert f['inner_fold'] == k and f['repeat'] == k // 3
        tr, he = f['train_donors'], f['heldout_donors']
        assert len(set(tr)) == len(tr) and len(set(he)) == len(he)
        assert not set(tr) & set(he) and set(tr) | set(he) == set(dev)
        held.setdefault(f['repeat'], []).extend(he)
    for r, h in held.items():
        assert sorted(h) == sorted(dev), f'heldout sets of repeat {r} must partition the development donors'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='3,4,5,6,7,8', help='comma-separated inner folds')
    ap.add_argument('--out-root', type=Path, default=CACHE_ROOT)
    args = ap.parse_args()
    folds = [int(x) for x in args.fold.split(',')]
    assert len(set(folds)) == len(folds) and all(0 <= k <= 8 for k in folds)
    out_root = args.out_root.resolve()
    if out_root == CACHE_ROOT:
        if not set(folds) <= set(NEW_FOLDS):
            raise PermissionError('only inner folds 3-8 may be written into the real cache root')
    elif not str(out_root).startswith(SCRATCH_PREFIXES) or str(out_root).startswith(str(CACHE_ROOT)):
        raise PermissionError('--out-root must be the cache root or a scratch folder under smoke_r2/ or /tmp/')
    for k in folds:
        if (out_root / PANEL / f'fold{k}').exists():
            raise FileExistsError(f'{out_root / PANEL / f"fold{k}"} exists; never overwritten')
    configure(out_root)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    data = bc.read_donors(bc.DEV)
    for k in folds:
        man = bc.build(data, PANEL, k, overwrite=False)
        assert man['donor_guard']['forbidden_donors_read'] == []
        assert man['train_donors'] == bc.FOLDS_JSON['inner_folds'][k]['train_donors']
        gc.collect()


if __name__ == '__main__':
    main()
