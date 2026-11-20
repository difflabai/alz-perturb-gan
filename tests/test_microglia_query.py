"""CPU-only tests for the query tool and its NumPy helpers, using tiny synthetic fixtures."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = load('microglia_query', ROOT / 'scripts/microglia_query.py')
sys.path.insert(0, str(ROOT / 'analysis/query_tool_20261120'))
C = load('query_common', ROOT / 'analysis/query_tool_20261120/common.py')


def make_bundle(tmp, name='b', shift=0.0, genes=('AAA', 'BBB', 'CCC'), reps=3):
    rng = np.random.default_rng(0)
    G = len(genes)
    stage_mean = np.zeros((reps, 4, G))
    stage_mean[:, :, 0] = np.array([0.1, 0.2, 0.3, 0.6]) + shift      # gene AAA rises
    stage_mean[:, :, 0] += rng.normal(0, .01, (reps, 4))
    stage_mean[:, :, 1] = 1.0
    stage_mean[:, :, 2] = np.array([0.5, 0.4, 0.3, 0.2])
    within = stage_mean.copy(); within[:, 1] = np.nan                    # Low stage invalid
    D = 16
    obs_stage = np.repeat(np.arange(4), 4)
    sums = np.zeros((D, 7, G)); n = np.full((D, 7), 50)
    for d in range(D):
        sums[d, :, 0] = (0.1 + 0.2 * obs_stage[d]) * n[d]
        sums[d, :, 1] = 1.0 * n[d]
        sums[d, :, 2] = 0.3 * n[d]
    lo = np.stack([stage_mean.mean(0) - .2, np.full((4, G), .1)])
    hi = np.stack([stage_mean.mean(0) + .2, np.full((4, G), .5)])
    path = tmp / f'{name}.npz'
    np.savez_compressed(
        path, genes=np.array(genes), stage_mean_rep=stage_mean, within_rep=within,
        within_cells=np.zeros((reps, 4, 2), int), interval_lo=lo, interval_hi=hi,
        correction=np.stack([np.full(G, .05), np.full(G, .3)]),
        crossfit_coverage_raw=np.full((2, G), .7), crossfit_coverage_cal=np.full((2, G), .8),
        obs_donors=np.array([f'd{i}' for i in range(D)]), obs_stage=obs_stage, obs_heldout=np.zeros(D, bool),
        obs_mean_log=sums[:, 0] / 50, obs_subtype_sums=sums.astype(np.float32), obs_subtype_n=n,
        empirical_stage_blind_mean=np.full(G, .4), empirical_interval_lo=np.zeros((2, G)),
        empirical_interval_hi=np.ones((2, G)), heldout_rmse=np.array([.1, .5, .1]), empirical_rmse=np.full(G, .2),
        rmse_cut=np.array(.3), pc_gap=np.full((4, G), .1), pc_outside_share=np.full((4, G), .2),
        pc_captured=np.full(G, .5), detection=np.array([.5, .5, .001]),
        flag_poorly_fit=np.array([False, True, False]), flag_pc_blind_spot=np.array([False, False, True]),
        flag_low_expression=np.array([False, False, True]))
    meta = dict(schema='microglia-query-bundle/1', fit=f'fit-{name}', readout='r', fold=0,
                weights_sha256={'final.pt': 'x'}, replicates=list(range(1, reps + 1)), generation_devices=['cuda'],
                units='u', stage_mean_definition='s', within_subtype_definition='w', interval_definition='i',
                calibration={}, flags={})
    path.with_suffix('.json').write_text(json.dumps(meta))
    return path


def test_stage_parsing():
    assert Q.stage_index('Not AD') == 0 and Q.stage_index('high') == 3 and Q.stage_index('2') == 2
    with pytest.raises(ValueError):
        Q.stage_index('severe')


def test_query_contrast_flags_and_intervals(tmp_path):
    B = Q.Bundle(make_bundle(tmp_path))
    rows = Q.query([B], ['AAA', 'BBB', 'CCC', 'APOE'], 0, 3, n_boot=200)
    by = {r['gene']: r for r in rows}
    assert by['APOE']['flags']['not_in_panel'] and not by['APOE']['in_panel']
    c = by['AAA']['generated']['contrast']
    expected = (B.z['stage_mean_rep'][:, 3, 0] - B.z['stage_mean_rep'][:, 0, 0]).mean()
    assert c['estimate'] == pytest.approx(expected)
    assert c['fraction_replicates_positive'] == 1.0 and c['between_fit_sd'] is None
    assert 'unmeasured' in c['uncertainty_scope']
    assert by['BBB']['flags']['poorly_fit'] and by['CCC']['flags']['pc_blind_spot'] and by['CCC']['flags']['low_expression']
    iv = by['AAA']['per_fit'][0]['donor_intervals_80']['High']
    assert iv['mean']['calibrated'][0] == pytest.approx(max(iv['mean']['raw'][0] - .05, 0))
    assert iv['mean']['calibrated'][1] == pytest.approx(iv['mean']['raw'][1] + .05)
    assert iv['sd']['calibrated'][0] == 0.0                      # 0.1 - 0.3 clipped at zero support
    obs = by['AAA']['per_fit'][0]['observed_reference']['pooled']
    assert obs['contrast'] == pytest.approx(0.6)
    lo, hi = obs['ci95_donor_bootstrap']
    assert lo <= 0.6 + 1e-9 and hi >= 0.6 - 1e-9
    assert by['AAA']['per_fit'][0]['stage_blind_empirical_reference']['contrast'] == 0.0


def test_within_subtype_invalid_stage_is_null(tmp_path):
    B = Q.Bundle(make_bundle(tmp_path))
    row = Q.query([B], ['AAA'], 1, 3, n_boot=50)[0]
    assert row['generated']['within_subtype']['contrast'] is None
    assert row['generated']['contrast'] is not None


def test_ensemble_between_fit(tmp_path):
    b1 = Q.Bundle(make_bundle(tmp_path, 'one'))
    b2 = Q.Bundle(make_bundle(tmp_path, 'two', shift=0.0, genes=('AAA', 'BBB', 'DDD')))
    b2.z['stage_mean_rep'][:, 3, 0] += .2
    row = {r['gene']: r for r in Q.query([b1, b2], ['AAA', 'CCC'], 0, 3, n_boot=50)}
    c = row['AAA']['generated']['contrast']
    assert len(c['per_fit']) == 2 and c['between_fit_sd'] > 0 and 'between-fit' in c['uncertainty_scope']
    assert row['CCC']['flags']['not_in_panel_of_some_fits'] and row['CCC']['in_panel_by_fit'] == [True, False]


def test_cli_json_has_no_nan(tmp_path):
    path = make_bundle(tmp_path)
    out = tmp_path / 'q.json'
    assert Q.main(['--genes', 'AAA', 'ZZZ', '--stages', 'Not AD', 'High', '--bundle', str(path),
                   '--n-boot', '50', '--out', str(out)]) == 0
    value = json.loads(out.read_text())
    assert value['schema'] == 'microglia-query/1' and value['query']['contrast'] == 'stage_b minus stage_a'
    assert [g['gene'] for g in value['genes']] == ['AAA', 'ZZZ']
    with pytest.raises(SystemExit):
        Q.main(['--genes', 'AAA', '--stages', 'High', 'high', '--bundle', str(path)])


def test_pilot_rule_correction_and_leakage_guards():
    low = np.zeros((6, 2)); high = np.ones((6, 2))
    y = np.array([[0.5, 2.0], [0.5, 1.5], [1.2, 0.5], [0.5, 0.5], [0.5, 0.5], [-0.3, 0.5]])
    rec = C.fit_correction(low, high, y, [f'd{i}' for i in range(6)], ['t1'])
    assert rec['rank'] == 6                                       # ceil(7*0.8)
    np.testing.assert_allclose(rec['correction'], [0.3, 1.0])
    with pytest.raises(ValueError):
        C.fit_correction(low, high, y, ['t1'] + [f'd{i}' for i in range(5)], ['t1'])
    with pytest.raises(ValueError):
        C.fit_correction(low[:3], high[:3], y[:3], ['a', 'b', 'c'], [])  # rank 4 > n 3
    lo, hi = C.apply_correction(np.array([.1]), np.array([.2]), np.array([.5]), nonnegative=True)
    assert lo[0] == 0 and hi[0] == pytest.approx(.7)


def test_contrast_rules_match_frozen_definitions():
    stage = np.array([0, 0, 0, 3, 3, 3, 3])
    n = np.full((7, 7), 30); n[0, 2] = 5                         # one Not AD donor below 20 cells in group 2
    sums = np.zeros((7, 7, 1))
    sums[:, :, 0] = np.where(stage[:, None] == 3, 2.0, 1.0) * n
    eff, used = C.within_contrast_observed(sums, n, stage, groups=[2, 4])
    assert used == [4] and eff[0] == pytest.approx(1.0)          # group 2 has only 2 eligible Not AD donors
    syn_sums = np.zeros((4, 7, 1)); syn_n = np.full((4, 7), 150); syn_n[1, 4] = 20
    syn_sums[:, :, 0] = np.arange(4)[:, None] * syn_n
    st = np.array([0, 1, 2, 3])
    means, counts = C.synthetic_stage_means(syn_sums, syn_n, st)
    assert np.isnan(means[1, 0]) and means[3, 0] == pytest.approx(3.0) and counts[1, 1] == 20


def test_generated_sd_matches_bag_sd():
    rng = np.random.default_rng(1)
    L = rng.gamma(2., 1., size=(256, 5))
    m, s = C.bag_summary(L)
    np.testing.assert_allclose(C.generated_sd(L.mean(0), (L ** 2).mean(0)), s, rtol=1e-10)


def test_mc_interval_uses_student_t(tmp_path):
    B = Q.Bundle(make_bundle(tmp_path))
    c = Q.query([B], ['AAA'], 0, 3, n_boot=20)[0]['generated']['contrast']
    assert 'approx_ci95' not in c
    n = c['replicates']
    assert c['mc_interval95_t_df'] == n - 1
    half = (c['mc_interval95'][1] - c['mc_interval95'][0]) / 2
    assert half == pytest.approx(Q.t975(n - 1) * c['standard_error_total'])
    assert Q.t975(7) == pytest.approx(2.365) and Q.t975(100) == 1.96 and Q.t975(0) is None
    assert 'not an inferential CI' in c['mc_interval95_note']


def test_stagehalf_readout_note(tmp_path, capsys):
    path = make_bundle(tmp_path)
    meta = json.loads(path.with_suffix('.json').read_text())
    meta['readout'] = 'cellshape@0.5:biascal:stagehalf'
    path.with_suffix('.json').write_text(json.dumps(meta))
    out = tmp_path / 'q.json'
    assert Q.main(['--genes', 'AAA', '--stages', '0', '3', '--bundle', str(path), '--n-boot', '20', '--out', str(out)]) == 0
    value = json.loads(out.read_text())
    assert 'halved' in value['model']['fits'][0]['readout_note']
    assert any('halved' in s for s in value['scope'])
    assert 'halved' in capsys.readouterr().err
