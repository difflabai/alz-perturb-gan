"""Guard against misleading final-recipe selection, using analytic examples."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from compare_architecture_recipes import BIASCAL, COUNT_SHAPE, EXPECTED, compare
import compare_architecture_recipes as recipes


def reference(recipe):
    rows = {}
    for fold, seed in EXPECTED:
        rows[(fold, seed)] = {
            'fold': fold, 'M': 32, 'generation_device': 'cpu', 'generation_recipe': recipe,
            'fingerprint': {'cache_sha256': f'fold{fold}', 'score_code_sha256': 'score',
                            'interval_code_sha256': 'interval'},
            'historical': {'pb_rmse': .5, 'meanlog_rmse_gene': .1, 'nonzero_rmse_gene': .03,
                           'logvar_rmse_gene': .2, 'pc_energy_rep': 2., 'cell_energy_rep': .8,
                           'pc_spread_rep': 1.2},
            'intervals': {split: {'percentile': {group: {'coverage': .6, 'width': .4,
                'interval_score': 2.} for group in ('mean', 'sd', 'combined')}}
                for split in ('train', 'heldout')},
        }
    return rows


def example():
    distribution = reference(COUNT_SHAPE)
    profile = reference(BIASCAL)
    candidate = deepcopy(distribution)
    for row in candidate.values():
        row['intervals']['heldout']['percentile']['sd'].update(coverage=.7, interval_score=1.8)
    return candidate, distribution, profile


def test_analytic_paired_seed_noise_and_training_deterioration_remain_visible():
    candidate, distribution, profile = example()
    for (fold, seed), row in candidate.items():
        row['intervals']['heldout']['percentile']['sd']['interval_score'] += (-.1, .1)[seed]
        row['intervals']['train']['percentile']['sd'].update(coverage=.1, interval_score=3.)
    result = compare(candidate, distribution, profile)
    block = result['distribution_comparison']['sd_interval_score']
    # Six fold means of -.2; within-fold differences are -.3 and -.1.
    assert block['mean_diff'] == pytest.approx(-.2)
    assert block['se_seed_noise'] == pytest.approx(np.sqrt(6 * .02 / 2) / 6)
    assert result['passes_numeric_final_recipe_rule']
    assert result['means']['candidate']['intervals']['train']['sd']['coverage'] == pytest.approx(.1)
    assert result['means']['candidate']['intervals']['train']['sd']['interval_score'] == 3.


def test_profile_budget_cannot_compound_after_previous_promotion():
    candidate, distribution, profile = example()
    for row in distribution.values():
        row['historical']['pb_rmse'] = .5024
    for row in candidate.values():
        row['historical']['pb_rmse'] = .504
    # The new cost is <.5% relative to the distribution default, but .8%
    # relative to the original profile reference. Only the latter is allowed.
    assert (.504 - .5024) / .5024 < .005
    result = compare(candidate, distribution, profile)
    assert result['profile_allowance'] == .0025
    assert not result['gates']['profile_budget_from_original_reference']
    assert not result['passes_numeric_final_recipe_rule']


def test_small_mean_profile_cost_can_still_exceed_uncertainty_allowance():
    candidate, distribution, profile = example()
    for (fold, seed), row in candidate.items():
        row['historical']['pb_rmse'] = .5024 + (-.0004, .0004)[seed]
    result = compare(candidate, distribution, profile)
    assert result['profile_comparison_against_original_reference']['mean_diff'] < .0025
    assert result['profile_cost_upper_2seedSE'] > .0025
    assert not result['gates']['profile_budget_from_original_reference']


def test_one_bad_fold_cannot_hide_inside_a_small_overall_profile_cost():
    candidate, distribution, profile = example()
    for (fold, seed), row in candidate.items():
        if fold == 8:
            row['historical']['pb_rmse'] = .503
    result = compare(candidate, distribution, profile)
    assert result['profile_cost_upper_2seedSE'] < result['profile_allowance']
    assert result['profile_relative_cost_by_fold'][8] > .005
    assert not result['gates']['profile_budget_from_original_reference']


def test_overcoverage_cannot_cancel_undercoverage():
    candidate, distribution, profile = example()
    for key in EXPECTED:
        seed = key[1]
        distribution[key]['intervals']['heldout']['percentile']['sd']['coverage'] = (.7, .9)[seed]
        candidate[key]['intervals']['heldout']['percentile']['sd']['coverage'] = (.6, 1.)[seed]
    result = compare(candidate, distribution, profile)
    assert result['means']['candidate']['intervals']['heldout']['sd']['coverage'] == pytest.approx(.8)
    assert result['distribution_comparison']['sd_coverage_distance']['mean_diff'] == pytest.approx(.1)
    assert not result['gates']['sd_coverage_distance']


def test_five_lower_folds_are_insufficient_if_one_repeat_worsens():
    candidate, distribution, profile = example()
    for (fold, seed), row in candidate.items():
        row['intervals']['heldout']['percentile']['sd']['interval_score'] = (
            1. if fold < 6 else 1.99 if fold < 8 else 2.1)
    result = compare(candidate, distribution, profile)
    assert result['distribution_comparison']['sd_interval_score']['n_folds_lower'] == 5
    assert result['distribution_comparison']['sd_interval_score']['mean_diff'] < 0
    assert result['sd_score_repeat_differences']['2'] > 0
    assert not result['gates']['sd_score']


def test_overall_sd_coverage_improvement_cannot_hide_a_worse_repeat():
    candidate, distribution, profile = example()
    for (fold, seed), row in candidate.items():
        row['intervals']['heldout']['percentile']['sd']['coverage'] = .78 if fold < 6 else .59
    result = compare(candidate, distribution, profile)
    assert result['distribution_comparison']['sd_coverage_distance']['mean_diff'] < 0
    assert result['sd_coverage_distance_repeat_differences']['2'] > 0
    assert result['gates']['sd_score']
    assert not result['gates']['sd_coverage_distance']


def test_distribution_regression_cannot_be_offset_by_better_primary_targets():
    candidate, distribution, profile = example()
    for row in candidate.values():
        row['historical']['logvar_rmse_gene'] += .02
    result = compare(candidate, distribution, profile)
    assert result['gates']['sd_score'] and result['gates']['sd_coverage_distance']
    assert result['gates']['profile_budget_from_original_reference']
    assert not result['distribution_guards']['logvar_rmse_gene']
    assert not result['passes_numeric_final_recipe_rule']


def test_missing_seed_cannot_be_silently_dropped():
    candidate, distribution, profile = example()
    del candidate[(8, 1)]
    with pytest.raises(ValueError, match='exactly folds3-8,seeds0/1'):
        compare(candidate, distribution, profile)


@pytest.mark.parametrize('change', ('device', 'M', 'recipe', 'cache', 'score_code', 'nan'))
def test_incompatible_or_nonfinite_scores_cannot_pass(change):
    candidate, distribution, profile = example()
    row = candidate[(3, 0)]
    if change == 'device':
        row['generation_device'] = 'cuda'
    elif change == 'M':
        row['M'] = 128
    elif change == 'recipe':
        row['generation_recipe'] = 'cellshape@0.25:biascal:stagehalf'
    elif change == 'cache':
        row['fingerprint']['cache_sha256'] = 'different donors'
    elif change == 'score_code':
        row['fingerprint']['score_code_sha256'] = 'different scorer'
    else:
        row['historical']['meanlog_rmse_gene'] = float('nan')
    with pytest.raises(ValueError):
        compare(candidate, distribution, profile)


def test_identical_recipe_cannot_qualify_as_an_improvement():
    distribution = reference(COUNT_SHAPE)
    result = compare(deepcopy(distribution), distribution, reference(BIASCAL))
    assert not result['gates']['sd_score']
    assert not result['gates']['sd_coverage_distance']
    assert not result['passes_numeric_final_recipe_rule']


@pytest.mark.parametrize('changed', ('evidence', 'score'))
def test_frozen_references_cannot_be_rebased_or_overwritten(tmp_path, monkeypatch, changed):
    artifacts = []
    for label, recipe in (('default', BIASCAL), ('count_shape', COUNT_SHAPE)):
        for (fold, seed), row in reference(recipe).items():
            path = tmp_path/label/f'fold{fold}/seed{seed}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(row).encode()
            path.write_bytes(raw)
            artifacts.append({'label': label, 'fold': fold, 'seed': seed,
                              'score_sha256': hashlib.sha256(raw).hexdigest()})
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps({'artifact_audit': {'artifacts': artifacts}}))
    monkeypatch.setattr(recipes, 'BASE', tmp_path)
    monkeypatch.setattr(recipes, 'EVIDENCE', evidence)
    monkeypatch.setattr(recipes, 'EVIDENCE_SHA256', hashlib.sha256(evidence.read_bytes()).hexdigest())
    distribution, profile = recipes.fixed_references()
    assert set(distribution) == set(profile) == EXPECTED
    path = evidence if changed == 'evidence' else tmp_path/'default/fold3/seed0.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed'):
        recipes.fixed_references()
