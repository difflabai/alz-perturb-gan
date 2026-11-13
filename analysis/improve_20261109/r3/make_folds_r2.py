"""Round-2 donor splits (F9): write r2/folds_r2.json with inner folds 0-8.

Usage:
  /mnt/nvme/ai/conda/anaconda3/bin/python3 -B r2/make_folds_r2.py [--check]

--check regenerates the splits in memory and asserts that the existing
folds_r2.json equals them (no write).

Rule (reverse-engineered from ../folds.json and verified here bit for bit):
  rng = numpy.random.default_rng(seed)
  for stage in 0,1,2,3 (ADNC Not AD, Low, Intermediate, High):
      ds  = development donors of that stage, in ../folds.json order
      p   = rng.permutation(ds)
      off = int(rng.integers(3))
      donor p[i] is held out in split j = (i + off) % 3
  With seed 20261109 this gives exactly round-1 inner folds 0, 1, 2
  (held-out set of inner fold j = donors assigned to j).

Repeats:
  repeat 0 = round-1 folds 0-2 (seed 20261109), copied verbatim from
             ../folds.json (same donors, same list order).
  repeat r in {1, 2}: candidate seeds 20261109 + 100000*r + i, i = 0..19999.
      Admissible: held-out sizes 12, 12, 13 in some order (as round 1); every
      training split has >= 2 Not AD donors and every held-out split >= 1
      (always true for 4 Not AD donors under per-stage cyclic assignment;
      asserted). Score = sum over earlier repeats q of the number of donor
      pairs held out together in both r and q, i.e. sum_ij C(n_ij, 2) of the
      3x3 held-out overlap matrix n_ij. Lowest score wins; ties are broken by
      the largest overlap entry, then by the smallest i. Inner folds of
      repeat r are 3r + j for j = 0, 1, 2 (j = assignment index).
  For 37 donors in 3 x 3 cells the minimum possible score against one
  earlier repeat is 58 (eight cells of 4 and one of 5). Both new repeats
  reach that minimum against every earlier repeat (the chosen seeds and the
  overlap matrices are recorded in the file under "repeats").

Schema: identical to ../folds.json (all top-level keys and values kept,
inner folds 0-2 identical, train/heldout lists in ../folds.json development
order), plus per fold "repeat" and top-level "r2_rule", "repeats" and
"r2_note". The trainer (r2/trainer.py load_folds) checks folds 0-2, the
development set and the sealed set against round 1.

No donor data are read: only ../folds.json (donor ids and ADNC stage).
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

R2 = Path(__file__).resolve().parent
R1_FOLDS = R2.parent / 'folds.json'
OUT = R2 / 'folds_r2.json'
BASE_SEED = 20261109
SEED_STRIDE = 100000
N_CANDIDATES = 20000
N_REPEATS = 3          # repeat 0 = round 1
STAGES = ['Not AD', 'Low', 'Intermediate', 'High']


def assign(dev, stage_by_donor, seed):
    """Held-out split index (0..2) per development donor, in `dev` order."""
    rng = np.random.default_rng(seed)
    a = {}
    for st in range(4):
        ds = [d for d in dev if stage_by_donor[d] == st]
        p = rng.permutation(ds)
        off = int(rng.integers(3))
        for i, d in enumerate(p):
            a[str(d)] = (i + off) % 3
    return np.array([a[d] for d in dev], dtype=np.int64)


def overlap(a, b):
    return np.array([[int(np.sum((a == i) & (b == j))) for j in range(3)] for i in range(3)])


def co_heldout_pairs(a, b):
    m = overlap(a, b)
    return int((m * (m - 1) // 2).sum())


def admissible(a, stages):
    n = np.bincount(a, minlength=3)
    if sorted(n.tolist()) != [12, 12, 13]:
        return False
    for j in range(3):
        held0 = int(np.sum((a == j) & (stages == 0)))
        train0 = int(np.sum((a != j) & (stages == 0)))
        if held0 < 1 or train0 < 2:
            return False
    return True


def folds_of(a, dev, repeat):
    out = []
    for j in range(3):
        out.append({'inner_fold': 3 * repeat + j,
                    'heldout_donors': [d for d, x in zip(dev, a) if x == j],
                    'train_donors': [d for d, x in zip(dev, a) if x != j],
                    'repeat': repeat})
    return out


def stage_counts(donors, sbd):
    return {STAGES[s]: sum(1 for d in donors if sbd[d] == s) for s in range(4)}


def generate():
    r1 = json.loads(R1_FOLDS.read_text())
    dev = list(r1['development_donors'])
    sbd = r1['stage_by_donor']
    stages = np.array([sbd[d] for d in dev])
    # repeat 0: verify the rule reproduces round 1, then copy round 1 verbatim
    a0 = assign(dev, sbd, BASE_SEED)
    for j, f in enumerate(r1['inner_folds']):
        assert f['inner_fold'] == j
        assert set(f['heldout_donors']) == {d for d, x in zip(dev, a0) if x == j}, 'rule does not reproduce round 1'
        assert f['heldout_donors'] == [d for d in dev if d in set(f['heldout_donors'])]
        assert f['train_donors'] == [d for d in dev if d in set(f['train_donors'])]
    assert admissible(a0, stages)
    chosen = [a0]
    repeats = [{'repeat': 0, 'inner_folds': [0, 1, 2], 'seed': BASE_SEED, 'candidate_index': None,
                'source': 'round-1 ../folds.json inner folds 0-2 (copied verbatim; rule reproduces them)'}]
    for r in range(1, N_REPEATS):
        best = None
        n_adm = 0
        for i in range(N_CANDIDATES):
            seed = BASE_SEED + SEED_STRIDE * r + i
            a = assign(dev, sbd, seed)
            if not admissible(a, stages):
                continue
            n_adm += 1
            key = (sum(co_heldout_pairs(q, a) for q in chosen), max(int(overlap(q, a).max()) for q in chosen), i)
            if best is None or key < best[0]:
                best = (key, seed, a)
        (score, mx, i), seed, a = best
        chosen.append(a)
        repeats.append({'repeat': r, 'inner_folds': [3 * r, 3 * r + 1, 3 * r + 2], 'seed': int(seed),
                        'candidate_index': int(i), 'candidates_searched': N_CANDIDATES,
                        'candidates_admissible': n_adm, 'score_co_heldout_pairs_vs_earlier': int(score),
                        'max_overlap_cell_vs_earlier': int(mx)})
    # cross-repeat overlap tables
    for r in range(1, N_REPEATS):
        repeats[r]['heldout_overlap_vs_repeat'] = {
            str(q): {'matrix_rows_this_repeat_cols_other': overlap(chosen[r], chosen[q]).tolist(),
                     'co_heldout_pairs': co_heldout_pairs(chosen[r], chosen[q])} for q in range(r)}
    folds = [dict(f, repeat=0) for f in r1['inner_folds']]
    for r in range(1, N_REPEATS):
        folds += folds_of(chosen[r], dev, r)
    q = dict(r1)
    q['inner_folds'] = folds
    q['r2_rule'] = ('Repeat r (0,1,2) uses the round-1 rule: rng=numpy.random.default_rng(seed); per ADNC stage 0..3, '
                    'p=rng.permutation(stage donors in development order), off=int(rng.integers(3)), donor p[i] held out in '
                    'split (i+off)%3; inner fold = 3r + split. Repeat 0 = round 1 (seed 20261109). Repeats 1-2: candidate '
                    'seeds 20261109+100000*r+i, i<20000, admissible if held-out sizes are {12,12,13} and every train split '
                    'has >=2 Not AD donors and every held-out split >=1; chosen by minimum number of donor pairs held out '
                    'together in this repeat and any earlier repeat, ties by max overlap cell then smallest i. '
                    'Generator: r2/make_folds_r2.py.')
    q['repeats'] = repeats
    q['r2_note'] = ('Inner folds 0-2 are identical to ../folds.json. Within a repeat the held-out sets partition the 37 '
                    'development donors; folds of different repeats share donors, so evidence across repeats is correlated.')
    return q


def summary(q):
    sbd = q['stage_by_donor']
    rows = []
    for f in q['inner_folds']:
        rows.append({'inner_fold': f['inner_fold'], 'repeat': f['repeat'], 'n_train': len(f['train_donors']),
                     'n_heldout': len(f['heldout_donors']), 'train_stage': stage_counts(f['train_donors'], sbd),
                     'heldout_stage': stage_counts(f['heldout_donors'], sbd)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    q = generate()
    text = json.dumps(q, indent=1) + '\n'
    if args.check:
        if OUT.read_text() != text:
            raise SystemExit('folds_r2.json differs from regenerated splits')
        print('folds_r2.json matches the generator')
    else:
        if OUT.exists() and OUT.read_text() != text:
            raise SystemExit(f'{OUT} exists and differs; refusing to overwrite')
        tmp = OUT.with_name('.' + OUT.name + '.tmp')
        tmp.write_text(text)
        tmp.replace(OUT)
        print('wrote', OUT)
    for r in q['repeats']:
        print(json.dumps({k: v for k, v in r.items() if k != 'source'}))
    for row in summary(q):
        print(json.dumps(row))


if __name__ == '__main__':
    main()
