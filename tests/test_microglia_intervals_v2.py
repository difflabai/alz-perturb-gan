"""Schema-2 SD calibration: honest nominal levels, donor separation and block readouts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'analysis/coverage_calibration_v2_20261120'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


C2 = load('calibration_v2', STUDY/'calibration_v2.py')
MB = load('metrics_blocks', STUDY/'metrics_blocks.py')
IM = load('pinned_interval_metrics', ROOT/'analysis/improve_20261109/r5/interval_metrics.py')


def ids(prefix, n):
    return [f'{prefix}{i}' for i in range(n)]


def synthetic(rng, n, scale=1.):
    """Exchangeable donors: fixed generator bounds, donor SDs drawn iid."""
    low, high = np.zeros((n, 100)), np.ones((n, 100))
    low[:, 50:], high[:, 50:] = .8, 1.1
    y = np.full((n, 100), .5)
    y[:, 50:] = rng.lognormal(0., .35 * scale, size=(n, 50))
    return low, high, y


def test_six_donors_at_alpha_point_two_put_the_rank_at_the_maximum():
    low, high, y = synthetic(np.random.default_rng(0), 6)
    with pytest.warns(C2.RankAtMaximumWarning, match='nominal level is 0.8571'):
        record = C2.fit_sd(low, high, y, ids('cal', 6), ['train'])
    assert record['rank'] == record['n_calibration_donors'] == 6 and record['rank_is_maximum']
    assert record['nominal_coverage'] == pytest.approx(6/7)
    score = np.maximum(0, np.maximum(low[:, 50:] - y[:, 50:], y[:, 50:] - high[:, 50:]))
    assert np.array_equal(record['sd_correction_lower'], score.max(0).tolist())
    # Eleven leave-one-out calibration donors reach a sub-maximal rank.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        record = C2.fit_sd(*synthetic(np.random.default_rng(1), 11), ids('cal', 11), ['train'])
    assert (record['rank'], record['rank_is_maximum']) == (10, False)
    assert record['nominal_coverage'] == pytest.approx(10/12)


def test_rank_uses_exact_integer_quantile():
    assert C2.conformal_rank(9, .2) == (8, .8)
    assert C2.conformal_rank(11, .2)[0] == 10
    with pytest.raises(ValueError, match='insufficient'):
        C2.conformal_rank(3, .2)


@pytest.mark.parametrize('n,asymmetric', [(6, False), (11, False), (19, True)])
def test_exchangeable_donor_simulation_reaches_the_nominal_level(n, asymmetric):
    rng = np.random.default_rng(20261120 + n)
    trials, covered, nominal = 3000, [], None
    for t in range(trials):
        low, high, y = synthetic(rng, n + 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', C2.RankAtMaximumWarning)
            record = C2.fit_sd(low[:n], high[:n], y[:n], ids('cal', n), ['train'], asymmetric=asymmetric)
        nominal = record['nominal_coverage']
        lo, hi = C2.apply_sd(low[n:], high[n:], record, ['test'])
        covered.append(((y[n:, 50:] >= lo[:, 50:]) & (y[n:, 50:] <= hi[:, 50:])).mean())
    se = np.sqrt(nominal * (1 - nominal) / trials)  # conservative: coordinates are independent too
    assert np.mean(covered) >= nominal - 3 * se
    # Uncalibrated bounds badly undercover in this setup, so the test has power.
    low, high, y = synthetic(rng, 2000)
    assert ((y[:, 50:] >= low[:, 50:]) & (y[:, 50:] <= high[:, 50:])).mean() < .5


def test_leave_one_donor_out_reaches_the_nominal_level():
    rng = np.random.default_rng(7)
    covered = []
    for _ in range(400):
        low, high, y = synthetic(rng, 12)
        lo, hi, records = C2.loo_sd(low, high, y, ids('held', 12), ['train'])
        covered.append(((y[:, 50:] >= lo[:, 50:]) & (y[:, 50:] <= hi[:, 50:])).mean())
    assert records[0]['rank'] == 10 and records[0]['nominal_coverage'] == pytest.approx(10/12)
    assert np.mean(covered) >= 10/12 - .01


def test_evaluation_donor_observations_never_move_its_own_bounds():
    """Replaces the pilot's tautological assert: perturb, rebuild everything, compare."""
    rng = np.random.default_rng(3)
    low, high, y = synthetic(rng, 12)
    donors = ids('held', 12)
    halves = [np.arange(6), np.arange(6, 12)]

    def split(observed):
        lo, hi = low.copy(), high.copy()
        for cal, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', C2.RankAtMaximumWarning)
                record = C2.fit_sd(low[cal], high[cal], observed[cal], np.asarray(donors)[cal], ['train'])
            lo[test], hi[test] = C2.apply_sd(low[test], high[test], record, np.asarray(donors)[test])
        return lo, hi

    def pooled(observed):
        out = C2.pooled_loo_sd([dict(fit='a', low=low[:6], high=high[:6], observed=observed[:6],
                                     donors=donors[:6], training_donors=['train']),
                                dict(fit='b', low=low[6:], high=high[6:], observed=observed[6:],
                                     donors=donors[:5] + [donors[11]], training_donors=['x'])])
        return np.r_[out[0]['low'], out[1]['low']], np.r_[out[0]['high'], out[1]['high']]

    for victim in (0, 8):
        changed = y.copy(); changed[victim, 50:] = changed[victim, 50:] * 50 + 3
        for build in (split, lambda o: C2.loo_sd(low, high, o, donors, ['train'])[:2],
                      lambda o: C2.loo_sd(low, high, o, donors, ['train'], asymmetric=True)[:2], pooled):
            before, after = build(y), build(changed)
            assert np.array_equal(before[0][victim], after[0][victim])
            assert np.array_equal(before[1][victim], after[1][victim])
            assert not (np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]))
    # Pooled: a donor identity repeated in another fit is never its own calibrator.
    changed = y.copy(); changed[0, 50:] += 100
    assert np.array_equal(pooled(y)[1][6], pooled(changed)[1][6])


def test_training_donors_are_rejected_for_calibration_and_evaluation():
    low, high, y = synthetic(np.random.default_rng(4), 11)
    with pytest.raises(ValueError, match='calibration donor was used in GAN training'):
        C2.fit_sd(low, high, y, ids('cal', 11), ['cal3'])
    record = C2.fit_sd(low, high, y, ids('cal', 11), ['train0', 'train1'])
    with pytest.raises(ValueError, match='evaluation donor was used in GAN training'):
        C2.apply_sd(low[:2], high[:2], record, ['test', 'train1'])
    with pytest.raises(ValueError, match='overlap'):
        C2.apply_sd(low[:1], high[:1], record, ['cal0'])
    with pytest.raises(ValueError, match='GAN training'):
        C2.pooled_loo_sd([dict(fit='a', low=low, high=high, observed=y, donors=ids('cal', 11),
                               training_donors=['cal5'])])
    tampered = dict(record, nominal_coverage=.8)
    with pytest.raises(ValueError, match='invalid'):
        C2.apply_sd(low[:1], high[:1], tampered, ['test'])


def test_pooled_records_rank_nominal_and_training_rows():
    rng = np.random.default_rng(6)
    low, high, y = synthetic(rng, 12)
    blocks = [dict(fit='a', low=low[:6], high=high[:6], observed=y[:6], donors=ids('d', 6),
                   training_donors=['t0', 'd7']),
              dict(fit='b', low=low[6:], high=high[6:], observed=y[6:], donors=ids('d', 12)[6:],
                   training_donors=['t0'])]
    out = C2.pooled_loo_sd(blocks)
    first = out[0]['records'][0]
    assert (first['donor'], first['n_calibration_rows'], first['n_calibration_identities']) == ('d0', 11, 11)
    assert (first['rank'], first['rank_is_maximum']) == (10, False)
    assert abs(first['nominal_coverage'] - 10/12) < 1e-12
    assert first['n_rows_training_donor_of_fit'] == 1 and not first['training_rows_excluded']
    assert out[1]['records'][0]['n_rows_training_donor_of_fit'] == 0
    excluded = C2.pooled_loo_sd(blocks, exclude_training=True)
    assert excluded[0]['records'][0]['n_calibration_rows'] == 10
    assert excluded[0]['records'][0]['training_rows_excluded']
    assert excluded[1]['records'][0]['n_calibration_rows'] == 11
    # Excluding d7 from fit a's calibration can only change fit a's bounds.
    assert np.array_equal(excluded[1]['high'], out[1]['high'])
    with pytest.warns(C2.RankAtMaximumWarning):
        small = C2.pooled_loo_sd([dict(blocks[0], low=low[:7], high=high[:7], observed=y[:7],
                                       donors=ids('d', 7), training_donors=['t0'])])
    assert small[0]['records'][0]['rank'] == small[0]['records'][0]['n_calibration_rows'] == 6
    assert abs(small[0]['records'][0]['nominal_coverage'] - 6/7) < 1e-12


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_calibration_leaves_means_alone_and_respects_sd_support():
    low, high, y = synthetic(np.random.default_rng(5), 11)
    for asymmetric in (False, True):
        record = C2.fit_sd(low, high, y, ids('cal', 11), ['train'], asymmetric=asymmetric)
        lo, hi = C2.apply_sd(low[:3], high[:3], record, ids('test', 3))
        assert np.array_equal(lo[:, :50], low[:3, :50]) and np.array_equal(hi[:, :50], high[:3, :50])
        assert np.all(lo[:, 50:] >= 0) and np.all(hi[:, 50:] >= high[:3, 50:])


def test_block_readout_reproduces_the_pinned_equal_weight_summary():
    rng = np.random.default_rng(6)
    samples = np.abs(rng.normal(size=(9, 32, 100)))
    observed = np.abs(rng.normal(size=(24, 9, 100)))
    mask = np.arange(9) % 3 != 0
    low, high = IM.bounds(samples, 'percentile')
    ours = MB.summarize(low, high, observed, mask, pc_variance=np.linspace(5, 1, 50))
    pinned = IM.summarize(samples, observed, mask, 'percentile')
    for group in MB.GROUPS:
        for metric in MB.METRICS:
            assert ours[group]['all'][metric] == pytest.approx(pinned[group][metric], abs=1e-12)
        # Blocks partition the 50 PCs equally weighted by coordinate count.
        parts = sum(ours[group][b]['interval_score'] * n for b, n in (('pc01_10', 10), ('pc11_30', 20), ('pc31_50', 20)))
        assert parts / 50 == pytest.approx(ours[group]['all']['interval_score'])
    flat = MB.summarize(low, high, observed, mask, pc_variance=np.ones(50))
    assert flat['sd']['variance_weighted']['coverage'] == pytest.approx(flat['sd']['all']['coverage'])
    per = MB.per_donor(low[mask], high[mask], observed[:, mask], 'interval_score', 'sd')
    assert per.mean() == pytest.approx(ours['sd']['all']['interval_score'])


def test_paired_donor_bootstrap():
    base = np.array([1., 2., 3., 4.])
    result = MB.paired_bootstrap(base, base * .8, B=2000)
    assert result['estimate'] == pytest.approx(.2)
    assert result['ci'] == pytest.approx([.2, .2]) and result['donors_candidate_lower'] == 4
    noisy = MB.paired_bootstrap(base, base[::-1], B=2000)
    assert noisy['ci'][0] < 0 < noisy['ci'][1]


def test_pinned_pilot_sources_are_unchanged():
    pin = json.loads((ROOT/'analysis/coverage_empf_20261117/source_manifest.json').read_text())
    for name, digest in pin['execution_sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


def test_committed_rescore_reproduces_pilot_and_records_block_results():
    value = json.loads((STUDY/'rescore.json').read_text())
    comparison = json.loads((ROOT/'analysis/coverage_empf_20261117/completion/comparison.json').read_text())
    for label in ('baseline', 'quarter'):
        assert value['reproduction_of_pilot_aggregates'][label]['max_abs_difference'] < 1e-9
        for name, key in (('raw', 'raw_heldout'), ('split', 'calibrated_heldout')):
            for group in ('sd', 'combined'):
                ours = value['rows'][label]['heldout'][name][group]['all']['interval_score']
                assert round(ours, 6) == round(comparison['rows'][label][key][group]['interval_score'], 6)
    rows = value['rows']
    assert round(rows['baseline']['heldout']['raw']['sd']['pc01_10']['coverage'], 3) == .246
    assert round(rows['quarter']['heldout']['raw']['sd']['pc01_10']['coverage'], 3) == .448
    assert round(rows['baseline']['heldout']['raw']['sd']['pc31_50']['coverage'], 3) == .750
    assert round(rows['quarter']['heldout']['raw']['sd']['pc31_50']['coverage'], 3) == .614
    ci = value['donor_bootstrap']['quarter_vs_baseline_raw']['sd_all_interval_score_reduction']
    assert round(ci['estimate'], 3) == .172 and ci['ci'][0] > 0
    assert all(all(v.values()) for v in value['invariance_checks'].values())
