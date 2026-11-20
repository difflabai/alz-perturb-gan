"""Tests for the native EMPF confirmation scoring and analysis.

Run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=3 python -B -m pytest -p no:cacheprovider -q test_confirm.py
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R5 = ROOT/'analysis/improve_20261109/r5'
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


S = load('confirm_score', HERE/'score.py')
AN = load('confirm_analyze', HERE/'analyze.py')
C = load('pilot_calibration', ROOT/'analysis/coverage_empf_20261117/calibration.py')
I = load('pilot_interval_metrics', R5/'interval_metrics.py')
W = load('pilot_run', ROOT/'analysis/coverage_empf_20261117/run.py')
MX = load('r5_metrics_for_tests', R5/'metrics.py')


# --------------------------------------------------------------------------- planted gate cases
def metrics(sd_cov=.6, comb_cov=.7, sd_is=.8, comb_is=.9, **historical):
    base = dict(pb_rmse=.465, meanlog_rmse_gene=.07, nonzero_rmse_gene=.0155, logvar_rmse_gene=.19,
                pc_energy_rep=1.86, cell_energy_rep=.604, pc_spread_rep=1.1, pc_coverage80_rep=.69,
                pbdist_spread=1.14, pbdist_coverage80=.79)
    base.update(historical)
    value = dict(base)
    for group, cov, score in (('sd', sd_cov, sd_is), ('combined', comb_cov, comb_is), ('mean', .8, 1.08)):
        value.update({f'{group}_coverage': cov, f'{group}_interval_score': score, f'{group}_width': .3,
                      f'{group}_below': (1-cov)/2, f'{group}_above': (1-cov)/2})
    return value


FOLDS = [k for k, _ in S.CONFIRMATION]


def planted(**candidate):
    a = [metrics() for _ in FOLDS]
    b = [metrics(**candidate) for _ in FOLDS]
    return a, b


def test_confirmation_set_is_the_declared_17_fits_and_three_repeats():
    assert len(S.CONFIRMATION) == 17 and S.PILOT not in S.CONFIRMATION
    assert S.CONFIRMATION == AN.CONFIRMATION and S.REPEATS == AN.REPEATS
    assert sorted(FOLDS.count(k) for k in range(9)) == [1] + [2]*8
    assert sum(FOLDS.count(k) for k in AN.REPEATS['folds0-2_without_pilot']) == 5


def test_planted_pass():
    a, b = planted(sd_cov=.65, comb_cov=.72, sd_is=.7, comb_is=.85)
    q = AN.question1(a, b, FOLDS, complete=True)
    assert q['status'] == 'PASS' and all(q['gates_on_means'].values()) and q['repeat_condition']
    assert list(q['gates_on_means']) == list(AN.GATE_ORDER) and len(AN.GATE_ORDER) == 10


def test_incomplete_is_never_reported_as_pass():
    a, b = planted(sd_cov=.65, comb_cov=.72, sd_is=.7, comb_is=.85)
    q = AN.question1(a[:5], b[:5], FOLDS[:5], complete=False)
    assert q['status'] == 'INCOMPLETE' and q['provisional_on_scored_fits'] == 'FAIL'  # repeats 2/3 have no fits


@pytest.mark.parametrize('base,candidate,expected', [
    (.60, .65, True),    # below 80%, moves up: closer
    (.70, .95, False),   # overshoots further than before (.15 > .10): not closer
    (.60, .95, True),    # overshoots but closer (.15 < .20)
    (.95, .85, True),    # above 80%, moves down: closer
    (.95, .60, False),   # crosses to farther below
    (.90, .70, False),   # equal distance is not closer
    (.70, .70, False),   # unchanged is not closer
])
def test_coverage_closer_to_80_from_both_sides(base, candidate, expected):
    a = [metrics(sd_cov=base, comb_cov=base) for _ in FOLDS]
    b = [metrics(sd_cov=candidate, comb_cov=candidate, sd_is=.7, comb_is=.85) for _ in FOLDS]
    gates = AN.gates_on_means(a, b)
    assert gates['sd_coverage'] is expected and gates['combined_coverage'] is expected
    assert S.closer(candidate, base) is expected
    row = lambda cov, s: dict(raw_heldout={g: dict(coverage=cov, interval_score=s) for g in ('sd', 'combined')},
                              historical=metrics())
    assert S.pilot_gates(row(base, .9), row(candidate, .8))['sd_coverage'] is expected


@pytest.mark.parametrize('metric,limit', list(AN.GUARDS.items()))
def test_profile_and_distribution_guards_at_their_thresholds(metric, limit):
    base = metrics()[metric]
    ok = AN.gates_on_means(*planted(sd_cov=.65, comb_cov=.72, sd_is=.7, comb_is=.85,
                                    **{metric: base*(1+limit*.9)}))
    bad = AN.gates_on_means(*planted(sd_cov=.65, comb_cov=.72, sd_is=.7, comb_is=.85,
                                     **{metric: base*(1+limit*1.1)}))
    assert ok[metric] and not bad[metric]
    assert all(v for k, v in bad.items() if k != metric)


@pytest.mark.parametrize('key', ['sd_is', 'comb_is'])
def test_proper_score_gate_fails_when_not_lower(key):
    values = dict(sd_cov=.65, comb_cov=.72, sd_is=.7, comb_is=.85)
    values[key] = .8 if key == 'sd_is' else .9   # equal to the base: not lower
    q = AN.question1(*planted(**values), FOLDS, complete=True)
    assert q['status'] == 'FAIL'


def test_one_repeat_regression_fails_even_when_pooled_mean_passes():
    a = [metrics() for _ in FOLDS]
    b = []
    for k in FOLDS:
        if k in (6, 7, 8):   # third repeat: worse proper score, but pooled mean still lower
            b.append(metrics(sd_cov=.65, comb_cov=.72, sd_is=.85, comb_is=.85))
        else:
            b.append(metrics(sd_cov=.65, comb_cov=.72, sd_is=.6, comb_is=.85))
    q = AN.question1(a, b, FOLDS, complete=True)
    assert q['all_ten_gates'] and not q['repeat_condition'] and q['status'] == 'FAIL'
    assert q['repeats_both_proper_scores_lower']['folds6-8']['sd'] is False


def test_paired_se_and_improvement_counts():
    a = [metrics(sd_is=1.) for _ in range(4)]
    b = [metrics(sd_is=v) for v in (.9, .8, 1.1, .8)]
    r = AN.paired(a, b, 'sd_interval_score')
    d = np.array([-.1, -.2, .1, -.2])
    assert r['mean_diff'] == pytest.approx(d.mean())
    assert r['paired_se'] == pytest.approx(d.std(ddof=1)/2)
    assert r['n_improved'] == 3 and r['n'] == 4
    c = AN.paired([metrics(sd_cov=.7)]*2, [metrics(sd_cov=.85), metrics(sd_cov=.95)], 'sd_coverage')
    assert c['n_improved'] == 1


def test_question2_calibration_rule_pooled_and_per_repeat():
    raw = [metrics(sd_cov=.7) for _ in FOLDS]
    good = [metrics(sd_cov=.85, comb_cov=.82, sd_is=.62, comb_is=.85) for _ in FOLDS]
    assert AN.question2(raw, good, FOLDS, True)['status'] == 'CONFIRMED'
    over = [metrics(sd_cov=.89, comb_cov=.82, sd_is=.62, comb_is=.85) for _ in FOLDS]  # .09 above 80% vs .10 below
    assert AN.question2(raw, over, FOLDS, True)['status'] == 'CONFIRMED'
    too_wide = [metrics(sd_cov=.95, comb_cov=.82, sd_is=.62, comb_is=.85) for _ in FOLDS]
    assert AN.question2(raw, too_wide, FOLDS, True)['status'] == 'NOT_CONFIRMED'
    worse_score = [metrics(sd_cov=.85, comb_cov=.82, sd_is=.62, comb_is=.95) for _ in FOLDS]
    assert AN.question2(raw, worse_score, FOLDS, True)['status'] == 'NOT_CONFIRMED'
    mixed = [metrics(sd_cov=.85, comb_cov=.82, sd_is=.62 if k < 6 else .9, comb_is=.85) for k in FOLDS]
    q = AN.question2(raw, mixed, FOLDS, True)
    assert q['pooled']['sd_interval_score_lower'] and q['status'] == 'NOT_CONFIRMED'
    assert AN.question2(raw, good, FOLDS, False)['status'] == 'INCOMPLETE'


# --------------------------------------------------------------------------- split rule
def fold_manifest(fold):
    return json.loads((ROOT/f'runs/improve_20261109/caches/hvg2000/fold{fold}/manifest.json').read_text())


@pytest.mark.parametrize('fold', range(9))
def test_per_fold_split_rule(fold):
    m = fold_manifest(fold)
    donors = np.array(m['train_donors']+m['heldout_donors'])
    held = np.r_[np.zeros(len(m['train_donors']), bool), np.ones(len(m['heldout_donors']), bool)]
    halves = S.split_halves(donors, held, MX.stable_seed)
    n = len(m['heldout_donors'])
    assert [len(h) for h in halves] == ([6, 6] if n == 12 else [7, 6])
    ids = [set(donors[h]) for h in halves]
    assert not ids[0] & ids[1] and ids[0] | ids[1] == set(m['heldout_donors'])
    assert not (ids[0] | ids[1]) & set(m['train_donors'])
    order = sorted(m['heldout_donors'], key=lambda d: MX.stable_seed('interval-calibration-split', 20261113, d))
    assert list(donors[halves[0]]) == order[:len(halves[0])]
    # the rule depends on donor IDs only, not on the order donors are listed in
    perm = np.random.default_rng(fold).permutation(len(donors))
    again = S.split_halves(donors[perm], held[perm], MX.stable_seed)
    assert [set(donors[perm][h]) for h in again] == ids
    for h in halves:   # n=6 or 7 calibration donors give a finite rank (ceil((n+1)*.8) <= n)
        assert int(np.ceil((len(h)+1)*.8)) <= len(h)


def test_fold0_split_matches_the_pilot_calibration_files():
    m = fold_manifest(0)
    donors = np.array(m['train_donors']+m['heldout_donors'])
    held = np.r_[np.zeros(25, bool), np.ones(12, bool)]
    halves = S.split_halves(donors, held, MX.stable_seed)
    for swap in (0, 1):
        pilot = json.loads((ROOT/f'analysis/coverage_empf_20261117/completion/calibration/baseline_half{swap}.json').read_text())
        assert list(donors[halves[swap]]) == pilot['calibration_donors']


# --------------------------------------------------------------------------- leakage
def synthetic(n_train=5, n_held=13, seed=0):
    rng = np.random.default_rng(seed)
    donors = np.array([f'T{i}' for i in range(n_train)] + [f'H{i}' for i in range(n_held)])
    held = np.r_[np.zeros(n_train, bool), np.ones(n_held, bool)]
    n = len(donors)
    low = rng.normal(size=(n, 100)); high = low + rng.uniform(.1, 1, size=(n, 100))
    low[:, 50:] = rng.uniform(.1, .5, size=(n, 50)); high[:, 50:] = low[:, 50:] + rng.uniform(.05, .3, size=(n, 50))
    observed = rng.normal(size=(3, n, 100)); observed[:, :, 50:] = np.abs(observed[:, :, 50:])
    return donors, held, low, high, observed


def test_changing_a_test_donors_counts_does_not_change_its_calibration():
    donors, held, low, high, observed = synthetic()
    halves = S.split_halves(donors, held, MX.stable_seed)
    lo, hi, records = S.crossfit(low, high, observed, donors, held, halves, C)
    for swap, (cal, test, record) in enumerate(records):
        changed = observed.copy(); changed[:, test] = 1e6 + changed[:, test]
        lo2, hi2, records2 = S.crossfit(low, high, changed, donors, held, halves, C)
        assert records2[swap][2] == record                          # the record used on these test donors
        assert np.array_equal(lo2[test], lo[test]) and np.array_equal(hi2[test], hi[test])
    # mean bounds unchanged; SD lower support respected
    assert np.array_equal(lo[:, :50], low[:, :50]) and np.all(lo[:, 50:] >= 0)


def test_changing_a_test_donors_cells_does_not_change_its_calibration_end_to_end():
    """Count-level: perturb one evaluation donor's cells -> PC summaries -> calibration."""
    sys.path.insert(0, str(R5))
    A = load('r5_architecture_score_for_tests', R5/'architecture_score.py')
    rng = np.random.default_rng(1)
    donors, held, low, high, _ = synthetic(n_train=4, n_held=12, seed=1)

    class Real:
        pass
    real = Real(); real.donors = donors; real.pcs = [rng.normal(size=(80, 50)).astype(np.float32) for _ in donors]
    observed = A.observed_summaries(real, 64)
    halves = S.split_halves(donors, held, A.E.MX.stable_seed)
    lo, hi, records = S.crossfit(low, high, observed, donors, held, halves, C)
    target = int(halves[1][0])                     # evaluated with the record fitted on half 0
    real.pcs[target] = real.pcs[target]*50 + 7     # wildly different cells for that donor
    observed2 = A.observed_summaries(real, 64)
    assert not np.allclose(observed2[:, target], observed[:, target])
    lo2, hi2, records2 = S.crossfit(low, high, observed2, donors, held, halves, C)
    assert records2[0][2] == records[0][2]
    assert np.array_equal(lo2[target], lo[target]) and np.array_equal(hi2[target], hi[target])


def test_calibration_donors_are_never_in_the_gans_train_set():
    donors, held, low, high, observed = synthetic()
    halves = S.split_halves(donors, held, MX.stable_seed)
    for cal, test, record in S.crossfit(low, high, observed, donors, held, halves, C)[2]:
        assert not set(record['calibration_donors']) & set(record['training_donors'])
        assert set(record['training_donors']) == set(donors[~held])
    leaked = [np.r_[halves[0][:-1], 0], halves[1]]          # donor 0 is a TRAIN donor
    with pytest.raises(ValueError, match='held-out'):
        S.crossfit(low, high, observed, donors, held, leaked, C)
    record = C.fit_sd(low[halves[0]], high[halves[0]], observed[0, halves[0]], donors[halves[0]], donors[~held])
    with pytest.raises(ValueError, match='overlap'):
        C.apply_sd(low[halves[0]], high[halves[0]], record, donors[halves[0]])
    with pytest.raises(ValueError, match='GAN training'):
        C.fit_sd(low[halves[0]], high[halves[0]], observed[0, halves[0]], donors[halves[0]], [donors[halves[0][0]]])


def test_crossfit_reproduces_the_pilot_loop_arithmetic():
    """score.crossfit is the pilot's analyze() loop: compare with an inline copy of it."""
    donors, held, low, high, observed = synthetic(n_held=12)
    halves = S.split_halves(donors, held, MX.stable_seed)
    lo, hi, _ = S.crossfit(low, high, observed, donors, held, halves, C)
    train = donors[~held].tolist(); lo2, hi2 = low.copy(), high.copy()
    for swap in (0, 1):
        cal, test = halves[swap], halves[1-swap]
        record = C.fit_sd(low[cal], high[cal], observed[0, cal], donors[cal], train)
        lo2[test], hi2[test] = C.apply_sd(low[test], high[test], record, donors[test])
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    mask = held
    assert W.scores(lo, hi, observed, mask, I) == W.scores(lo2, hi2, observed, mask, I)


def test_production_leakage_guard_fails_on_a_leaky_calibration(monkeypatch):
    """Review F3: the guard in crossfit() must FAIL when the evaluation donors' own counts reach
    their calibration. Plant a leaky pass whose calibration rows are shifted by the evaluation
    donors' mean observed summary (donor identities stay disjoint, so the pilot calibrator's
    identity checks pass, and the pilot's in-loop check is not run) and check crossfit refuses it."""
    donors, held, low, high, observed = synthetic()
    halves = S.split_halves(donors, held, MX.stable_seed)
    honest = S.crossfit_once

    def leaky(low, high, observed, donors, is_heldout, halves, C, pilot_check=True):
        donors = np.asarray(donors).astype(str); train = donors[~np.asarray(is_heldout)].tolist()
        lo, hi = low.copy(), high.copy(); records = []
        for swap in (0, 1):
            cal, test = halves[swap], halves[1-swap]
            rows = observed[0, cal] + np.abs(observed[:, test]).mean(axis=(0, 1))   # leak
            record = C.fit_sd(low[cal], high[cal], rows, donors[cal], train)
            lo[test], hi[test] = C.apply_sd(low[test], high[test], record, donors[test])
            records.append((cal, test, record))
        return lo, hi, records
    # leaky still runs when called alone: nothing inside it notices the leak
    leaky(low, high, observed, donors, held, halves, C)
    monkeypatch.setattr(S, 'crossfit_once', leaky)
    with pytest.raises(AssertionError, match='evaluation donor counts changed'):
        S.crossfit(low, high, observed, donors, held, halves, C)
    monkeypatch.setattr(S, 'crossfit_once', honest)
    lo, hi, _ = S.crossfit(low, high, observed, donors, held, halves, C)
    lo0, hi0, _ = S.crossfit(low, high, observed, donors, held, halves, C, check_leakage=False)
    assert np.array_equal(lo, lo0) and np.array_equal(hi, hi0)   # the guard never changes a number


def test_reproduction_compares_the_generated_archive_sha():
    """Review F4: only the source path and the binding's experiment-manifest SHA are ignored."""
    src = (HERE/'score.py').read_text()
    assert "provenance = ('/source',)" in src
    assert "ignore=('/binding/experiment_manifest_sha256',)" in src
    assert 'savez writes zip timestamps' not in src


# --------------------------------------------------------------------------- secondary references (review F1)
def cache_sha(fold):
    import hashlib
    return hashlib.sha256((ROOT/f'runs/improve_20261109/caches/hvg2000/fold{fold}/manifest.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('fold', range(9))
@pytest.mark.parametrize('seed', (0, 1))
def test_every_secondary_reference_resolves_pinned_on_the_fold_cache(fold, seed):
    for loader in (AN.previous_default, AN.empirical_blind):
        ref, problem = loader(fold, seed, cache_sha(fold))
        assert ref is not None, problem
        if fold < 3:
            assert ref['path'].startswith('runs/improve_20261109/results_r5/covariance_challenge_20261114/')
            assert ref['pin'] == 'analysis/covariance_challenge_20261114/artifact_audit_complete_20261116.json'
            alt, problem = loader(fold, seed, cache_sha(fold), 'empirical_challenge')
            assert alt is not None and alt['path'].startswith('runs/improve_20261109/results_r5/empirical_challenge_20261114/')
        else:
            assert loader(fold, seed, cache_sha(fold), 'empirical_challenge')[0] is None
        assert loader(fold, seed, cache_sha((fold+1) % 9))[0] is None   # other fold's donors are refused


def test_fold0_references_equal_the_owner_empf_selection_evidence():
    """The covariance-challenge rescorings are the references in empf_comparison_20261117/completion."""
    summary = json.loads((ROOT/'analysis/empf_comparison_20261117/completion/comparison.json').read_text())['summary']
    for loader, key in ((AN.previous_default, 'current32'), (AN.empirical_blind, 'empirical32')):
        row = loader(0, 0, cache_sha(0))[0]['row']
        assert row['historical']['pb_rmse'] == summary[key]['profile_RMSE']
        assert row['raw_heldout']['sd']['interval_score'] == summary[key]['held_sd_interval_score']
        assert row['raw_heldout']['combined']['interval_score'] == summary[key]['held_combined_interval_score']


def test_previous_default_stream_to_stream_spread():
    sts = AN.stream_to_stream([(k, s) for k in (0, 1, 2) for s in (0, 1)])
    assert sts['n'] == 6 and not sts['notes']
    d = sts['spread']['pb_rmse']['differences']
    assert d[0] == pytest.approx(.469424 - .472544, abs=1e-6)   # fold 0 seed 0: namespace 1 minus namespace 6
    assert sts['spread']['pb_rmse']['max_abs_difference'] == max(abs(x) for x in d)


# --------------------------------------------------------------------------- coverage diagnostics and fold SE (F2, F5)
def test_question2_reports_per_fit_distance_when_fits_cancel():
    raw = [metrics(sd_cov=.7) for _ in FOLDS]
    cal = [metrics(sd_cov=.7 if i % 2 else .9, comb_cov=.82, sd_is=.62, comb_is=.85) for i, _ in enumerate(FOLDS)]
    q = AN.question2(raw, cal, FOLDS, True)
    d = q['coverage_distance_mean_of_per_fit']['pooled']['sd']
    assert d['raw'] == pytest.approx(.1) and d['calibrated'] == pytest.approx(.1)
    assert set(q['coverage_distance_mean_of_per_fit']['repeats']) == set(AN.REPEATS)
    assert 'MEAN coverage' in q['coverage_rule']


def test_fold_level_se_averages_seeds_first():
    a = [metrics(sd_is=1.) for _ in range(4)]
    b = [metrics(sd_is=v) for v in (.9, .7, 1.1, 1.3)]
    r = AN.paired(a, b, 'sd_interval_score', folds=[3, 3, 4, 4])
    assert r['fold_level']['n_folds'] == 2
    assert r['fold_level']['mean_of_fold_means'] == pytest.approx(0.)
    means = np.array([-.2, .2])
    assert r['fold_level']['fold_se'] == pytest.approx(means.std(ddof=1)/np.sqrt(2))
    assert 'fold_level' not in AN.paired(a, b, 'sd_interval_score')


def test_train_partition_is_carried_and_paired():
    row = dict(historical=dict(pb_rmse=.5),
               raw_heldout={g: {k: .5 for k in AN.INTERVAL_KEYS} for g in ('mean', 'sd', 'combined')},
               raw_train={g: {k: .7 for k in AN.INTERVAL_KEYS} for g in ('mean', 'sd', 'combined')},
               calibrated_heldout={g: {k: .6 for k in AN.INTERVAL_KEYS} for g in ('mean', 'sd', 'combined')})
    f = AN.flat(row)
    assert f['train_sd_coverage'] == .7 and f['sd_coverage'] == .5
    assert 'train_sd_coverage' not in AN.flat(row, True)
    a = [metrics() | {k: .7 for k in AN.TRAIN_KEYS} for _ in FOLDS]
    b = [metrics(sd_cov=.65, comb_cov=.72, sd_is=.7, comb_is=.85) | {k: .6 for k in AN.TRAIN_KEYS} for _ in FOLDS]
    q = AN.question1(a, b, FOLDS, True)
    assert {r['metric'] for r in q['paired']} >= set(AN.TRAIN_KEYS)
    assert q['status'] == 'PASS'   # TRAIN values are reported, never gated


def test_compare_tree_reports_exactness_and_max_difference():
    a = dict(x=1., y=[1, 2., dict(z=3.)], s='p', fingerprint=dict(generated_sha256='a'))
    b = copy.deepcopy(a)
    assert S.compare_tree(a, b)['exact'] and S.compare_tree(a, b)['max_abs_difference'] == 0
    b['y'][2]['z'] = 3.5; b['fingerprint']['generated_sha256'] = 'b'
    r = S.compare_tree(a, b, ignore=('/fingerprint/generated_sha256',))
    assert not r['exact'] and r['max_abs_difference'] == .5 and r['mismatched'] == ['/y[2]/z']
    assert r['differing_ignored_provenance_fields'] == ['/fingerprint/generated_sha256']


# --------------------------------------------------------------------------- integration: pilot reproduction
REPRO = S.OUT/'scores/fold0/seed0'


@pytest.mark.skipif(not (REPRO/'score.json').exists(), reason='pilot reproduction has not been run yet')
def test_pilot_reproduction_is_exact():
    report = json.loads((REPRO/'reproduction.json').read_text())
    assert report['all_exact'] and report['max_abs_difference'] == 0
    assert report['readout_B_arrays_identical_to_pilot_quarter_generated']
    assert report['readout_B_archive_byte_identical_to_pilot_quarter_generated']
    ignored = {f for v in report.values() if isinstance(v, dict) for f in v.get('differing_ignored_provenance_fields', [])}
    assert ignored <= {'/source', '/binding/experiment_manifest_sha256'}
    # independent recomputation from the files on disk
    pilot = ROOT/'analysis/coverage_empf_20261117/completion'
    for old, new in (('baseline_raw_score.json', 'A_raw_score.json'), ('quarter_raw_score.json', 'B_raw_score.json'),
                     *[(f'calibration/{o}_half{s}.json', f'calibration/{n}_half{s}.json')
                       for o, n in (('baseline', 'A'), ('quarter', 'B')) for s in (0, 1)]):
        e, g = json.loads((pilot/old).read_text()), json.loads((REPRO/new).read_text())
        for drop in ('fingerprint', 'source', 'binding'):
            e.pop(drop, None); g.pop(drop, None)
        assert e == g, old
    score = json.loads((REPRO/'score.json').read_text())
    comparison = json.loads((pilot/'comparison.json').read_text())
    assert score['gates_B_vs_A'] == comparison['candidate_gates']
    for old, new in (('baseline', 'A'), ('quarter', 'B')):
        for key in ('historical', 'raw_heldout', 'calibrated_heldout', 'raw_train', 'raw_rank_heldout'):
            assert score['rows'][new][key] == comparison['rows'][old][key], (old, key)


@pytest.mark.skipif(not (REPRO/'score.json').exists(), reason='pilot reproduction has not been run yet')
def test_analyze_end_to_end_on_replicated_pilot_scores(tmp_path, monkeypatch):
    """Plumbing check: 17 copies of the pilot score must reproduce the pilot's own gates, zero paired SE,
    and the INCOMPLETE label must appear as soon as one fit is missing."""
    import shutil
    for k, s in (S.PILOT, *S.CONFIRMATION):
        dst = tmp_path/f'scores/fold{k}/seed{s}'
        shutil.copytree(REPRO, dst)
    monkeypatch.setattr(AN, 'OUT', tmp_path)
    monkeypatch.setattr(AN, 'check_sources', lambda: (json.loads(AN.PIN.read_text()), []))
    report = AN.analyze('cpu')
    pilot_gates = json.loads((REPRO/'score.json').read_text())['gates_B_vs_A']
    q = report['question1_B_vs_A']
    assert report['status'] == 'COMPLETE' and q['gates_on_means'] == {k: pilot_gates[k] for k in AN.GATE_ORDER}
    assert all(r['paired_se'] < 1e-12 for r in q['paired'] if r['metric'] in AN.GATE_ORDER)
    assert all(r['n_improved'] in (0, 17) for r in q['paired'] if r['n_improved'] is not None)
    assert q['status'] == ('PASS' if all(pilot_gates.values()) else 'FAIL')
    assert report['question2']['A']['status'] == report['question2']['B']['status'] == 'CONFIRMED'
    # fold-0 scores copied to other folds must be rejected by the reference loaders (different cache)
    assert report['secondary']['previous_default']['n'] == 1 and len(report['secondary']['previous_default']['notes']) == 16
    shutil.rmtree(tmp_path/'scores/fold8/seed1')
    partial = AN.analyze('cpu')
    assert partial['status'] == 'INCOMPLETE' and partial['missing_confirmation_fits'] == ['fold8/seed1']
    assert partial['question1_B_vs_A']['status'] == 'INCOMPLETE'
