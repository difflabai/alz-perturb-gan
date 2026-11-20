"""Re-score the 2026-11-17 EMPF coverage pilot with block, stage, donor-CI and v2 calibration readouts.

CPU only, no model or generator execution: it reads the pilot's retained
generated archives and the frozen fold-0 development cache, checks them
against the committed pilot comparison, reproduces the pilot's equal-weight
aggregates, and adds new readouts. Pinned pilot files are hashed before and
after and never written.

    PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
    $PY -B analysis/coverage_calibration_v2_20261120/rescore_pilot.py [--data-root /ml2/alz]
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT/'analysis/coverage_empf_20261117'
R5 = ROOT/'analysis/improve_20261109/r5'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import calibration_v2 as C2  # noqa: E402
import metrics_blocks as MB  # noqa: E402

LABELS = {'baseline': ('runs/microglia_empf/verification/final_generated.npz', 'cellshape@0.5:biascal:stagehalf'),
          'quarter': ('runs/coverage_empf_20261117/quarter_generated.npz', 'cellshape@0.25:biascal:stagehalf')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pinned_hashes(data_root):
    """Hash every pinned pilot source/input (code from this tree, data from data_root)."""
    pin = json.loads((PILOT/'source_manifest.json').read_text())
    got = {}
    for name, expected in pin['execution_sources'].items():
        got[name] = sha(ROOT/name)
        if got[name] != expected:
            raise ValueError(f'pinned pilot source differs: {name}')
    for name, expected in pin['inputs'].items():
        got[name] = sha(data_root/name)
        if got[name] != expected:
            raise ValueError(f'pinned pilot input differs: {name}')
    got['analysis/coverage_empf_20261117/source_manifest.json'] = sha(PILOT/'source_manifest.json')
    return got


def split_half(C, low, high, observed0, donors, halves, train):
    """The pilot's two fixed 6/6 swaps with the pinned schema-1 calibrator."""
    lo, hi = low.copy(), high.copy(); ranks = []
    for swap in (0, 1):
        cal, test = halves[swap], halves[1-swap]
        record = C.fit_sd(low[cal], high[cal], observed0[cal], donors[cal], train)
        lo[test], hi[test] = C.apply_sd(low[test], high[test], record, donors[test])
        ranks.append(record['rank'])
    return lo, hi, ranks


def loo(low, high, observed0, donors, held, train, asymmetric=False):
    lo, hi = low.copy(), high.copy()
    lo[held], hi[held], records = C2.loo_sd(low[held], high[held], observed0[held], donors[held],
                                            train, asymmetric=asymmetric)
    return lo, hi, records


def contrasts(pairs, observed, held):
    """Paired donor-bootstrap CIs for each (name, base bounds, candidate bounds)."""
    out = {}
    for name, (bl, bh), (cl, ch) in pairs:
        row = {}
        for group in ('sd', 'combined'):
            for block in ('all', 'pc01_10', 'pc11_30', 'pc31_50'):
                key = f'{group}_{block}'
                a = MB.per_donor(bl[held], bh[held], observed[:, held], 'interval_score', group, block)
                b = MB.per_donor(cl[held], ch[held], observed[:, held], 'interval_score', group, block)
                row[f'{key}_interval_score_reduction'] = MB.paired_bootstrap(a, b)
                a = MB.per_donor(bl[held], bh[held], observed[:, held], 'coverage', group, block)
                b = MB.per_donor(cl[held], ch[held], observed[:, held], 'coverage', group, block)
                row[f'{key}_coverage_difference'] = MB.paired_bootstrap(a, b, kind='difference')
        out[name] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default='/ml2/alz', type=Path,
                        help='checkout holding the local runs/ tree (read only)')
    parser.add_argument('--out', default=HERE/'rescore.json', type=Path)
    args = parser.parse_args()
    data = args.data_root.resolve()
    before = pinned_hashes(data)
    sys.path.insert(0, str(R5))
    A = load_module('architecture_score', R5/'architecture_score.py')
    E = A.E
    C = load_module('pilot_calibration', PILOT/'calibration.py')
    committed = json.loads((PILOT/'completion/comparison.json').read_text())
    cache = data/'runs/improve_20261109/caches/hvg2000/fold0'
    real = E.load_real(cache)
    observed = A.observed_summaries(real, 64)
    donors, held = real.donors, real.is_heldout
    train = donors[~held].tolist()
    order = sorted(np.flatnonzero(held), key=lambda i: E.MX.stable_seed(
        'interval-calibration-split', 20261113, str(donors[i])))
    halves = np.array_split(order, 2)
    pc_variance = np.asarray(real.pc_sd, dtype=float)**2
    held_idx = np.flatnonzero(held)

    bounds, rows, reproduction = {}, {}, {}
    for label, (rel, mode) in LABELS.items():
        path = data/rel
        if sha(path) != committed['rows'][label]['generated_sha256']:
            raise ValueError(f'{label}: archive differs from the committed pilot comparison')
        generated = E.align_generated(E.load_generated(path), real)
        sample = E.MX.pc_bag_summary(generated['pcs'].astype(np.float32))
        low, high = np.quantile(sample, [.1, .9], axis=1)
        split_lo, split_hi, ranks = split_half(C, low, high, observed[0], donors, halves, train)
        loo_lo, loo_hi, loo_records = loo(low, high, observed[0], donors, held, train)
        asym_lo, asym_hi, asym_records = loo(low, high, observed[0], donors, held, train, asymmetric=True)
        bounds[label] = dict(raw=(low, high), split=(split_lo, split_hi), loo=(loo_lo, loo_hi),
                             loo_asym=(asym_lo, asym_hi))
        # Reproduce the pilot's own equal-weight aggregates from the same arrays.
        diff = 0.
        for key, (lo, hi), mask in [('raw_heldout', (low, high), held),
                                    ('calibrated_heldout', (split_lo, split_hi), held),
                                    ('raw_train', (low, high), ~held)]:
            ours = MB.summarize(lo, hi, observed, mask)
            for group in MB.GROUPS:
                for metric in MB.METRICS:
                    diff = max(diff, abs(ours[group]['all'][metric] - committed['rows'][label][key][group][metric]))
        if diff > 5e-7:
            raise ValueError(f'{label}: pilot aggregates not reproduced (max abs diff {diff})')
        reproduction[label] = dict(max_abs_difference=diff, compared='raw_heldout, calibrated_heldout, raw_train '
                                   'x mean/sd/combined x coverage/below/above/width/interval_score',
                                   split_half_ranks=ranks, split_half_n_calibration=[6, 6])
        # Generated within-bag SD per PC (mean over donors and draws), for the PCA diagnostic.
        gen_sd = sample[..., 50:]
        rows[label] = dict(
            generated_sha256=committed['rows'][label]['generated_sha256'], mode=mode,
            heldout={name: MB.summarize(lo, hi, observed, held, pc_variance)
                     for name, (lo, hi) in bounds[label].items()},
            train_raw=MB.summarize(low, high, observed, ~held, pc_variance),
            heldout_by_stage={name: MB.per_stage(lo, hi, observed, held, real.stage)
                              for name, (lo, hi) in bounds[label].items()},
            train_raw_by_stage=MB.per_stage(low, high, observed, ~held, real.stage),
            loo_calibration=dict(n_calibration_donors=loo_records[0]['n_calibration_donors'],
                                 rank=loo_records[0]['rank'], nominal_coverage=loo_records[0]['nominal_coverage'],
                                 asymmetric_rank=asym_records[0]['rank'],
                                 asymmetric_nominal_coverage=asym_records[0]['nominal_coverage']),
            generated_bag_sd_by_block={split: {b: float(gen_sd[mask][..., MB.BLOCKS[b]].mean())
                                               for b in MB.BLOCKS}
                                       for split, mask in (('train', ~held), ('heldout', held))})

    # End-to-end invariance: perturbing one evaluation donor's observed bags
    # must leave that donor's own calibrated bounds unchanged. (It is a
    # calibration donor for the other half/other donors, whose bounds may move;
    # that is recorded as a sensitivity check that the perturbation took effect.)
    invariance = {}
    for label in LABELS:
        low, high = bounds[label]['raw']
        for victim in (halves[0][0], halves[1][-1]):
            changed = observed[0].copy(); changed[victim, 50:] = changed[victim, 50:] * 3 + 5
            s_lo, s_hi, _ = split_half(C, low, high, changed, donors, halves, train)
            l_lo, l_hi, _ = loo(low, high, changed, donors, held, train)
            a_lo, a_hi, _ = loo(low, high, changed, donors, held, train, asymmetric=True)
            ok_split = (np.array_equal(s_lo[victim], bounds[label]['split'][0][victim]) and
                        np.array_equal(s_hi[victim], bounds[label]['split'][1][victim]))
            others_changed = not (np.array_equal(s_lo, bounds[label]['split'][0]) and
                                  np.array_equal(s_hi, bounds[label]['split'][1]))
            ok_loo = all(np.array_equal(x[victim], y[victim]) for x, y in
                         [(l_lo, bounds[label]['loo'][0]), (l_hi, bounds[label]['loo'][1]),
                          (a_lo, bounds[label]['loo_asym'][0]), (a_hi, bounds[label]['loo_asym'][1])])
            if not (ok_split and ok_loo):
                raise AssertionError(f'{label}: evaluation donor {donors[victim]} changed its own bounds')
            invariance[f'{label}:{donors[victim]}'] = dict(split_half_own_bounds_unchanged=ok_split,
                                                           loo_own_bounds_unchanged=ok_loo,
                                                           split_half_other_half_bounds_changed=others_changed)

    b, q = bounds['baseline'], bounds['quarter']
    ci = contrasts([('quarter_vs_baseline_raw', b['raw'], q['raw']),
                    ('quarter_vs_baseline_split_calibrated', b['split'], q['split']),
                    ('quarter_vs_baseline_loo_calibrated', b['loo'], q['loo']),
                    ('baseline_split_calibrated_vs_raw', b['raw'], b['split']),
                    ('quarter_split_calibrated_vs_raw', q['raw'], q['split']),
                    ('baseline_loo_calibrated_vs_raw', b['raw'], b['loo']),
                    ('quarter_loo_calibrated_vs_raw', q['raw'], q['loo']),
                    ('quarter_loo_asymmetric_vs_symmetric', q['loo'], q['loo_asym'])], observed, held)
    stage_names = ('Not AD', 'Low', 'Intermediate', 'High')
    after = pinned_hashes(data)
    if after != before:
        raise ValueError('pinned pilot files changed during the rescore')
    value = dict(
        status='rescored_pilot_cpu_only', schema=1,
        scope='Re-analysis of one native EMPF fold/seed (fold 0, seed 0): 25 GAN TRAIN donors and 12 held-out '
              'development donors already used in development selection. No new generation or training. '
              'Bootstrap CIs resample these 12 donors only; they do not include fold/seed or training variability.',
        sources=dict(rescore_pilot=sha(__file__), calibration_v2=sha(HERE/'calibration_v2.py'),
                     metrics_blocks=sha(HERE/'metrics_blocks.py'),
                     cache_manifest_sha256=sha(cache/'manifest.json')),
        pinned_pilot_sha256=before, pinned_pilot_unchanged=True,
        reproduction_of_pilot_aggregates=reproduction,
        heldout_donors=dict(n=int(held.sum()),
                            by_stage={n: int((held & (real.stage == s)).sum()) for s, n in enumerate(stage_names)},
                            split_halves=[donors[h].tolist() for h in halves],
                            split_half_stage_counts=[{n: int((real.stage[h] == s).sum()) for s, n in enumerate(stage_names)}
                                                     for h in halves]),
        train_donors_by_stage={n: int((~held & (real.stage == s)).sum()) for s, n in enumerate(stage_names)},
        pc_variance_weights=(pc_variance/pc_variance.sum()).tolist(),
        blocks={k: [v.start+1, v.stop] for k, v in MB.BLOCKS.items()},
        readouts=dict(raw='frozen M32 10/90 percentile bounds',
                      split='pilot schema-1 split-half SD calibration (n=6, rank 6 = max, nominal 6/7)',
                      loo='v2 leave-one-donor-out symmetric SD calibration over 12 held-out donors (n=11, rank 10, nominal 10/12)',
                      loo_asym='v2 leave-one-donor-out separate lower/upper SD corrections at alpha/2 each '
                               '(n=11, rank 11 = max, union-bound nominal 10/12)'),
        rows=rows, invariance_checks=invariance, donor_bootstrap=ci)
    args.out.write_text(json.dumps(value, indent=1, allow_nan=False)+'\n')
    show = {k: {r: {g: {m: round(rows[k]['heldout'][r][g]['all'][m], 6) for m in ('coverage', 'interval_score')}
                    for g in ('sd', 'combined')} for r in bounds[k]} for k in rows}
    print(json.dumps(show, indent=1))


if __name__ == '__main__':
    main()
