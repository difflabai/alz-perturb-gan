"""Regression checks for the NeuroKO evaluation and release metadata mapping."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import torch

from alztest.neuroko import LAYOUT, evaluate, normalize_counts, open_screen, project_fold, select_targets


def test_particle_search_uses_both_cohorts_and_changes_only_particle_rate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from neuroko_particle_rates import particle_arms, select_particle_rate
    from neuroko_rate_noise import NOISY_PRIOR

    trials = [arm[2] for arm in particle_arms(NOISY_PRIOR)]
    assert [t["lr"] * t["prior_lr_mult"] for t in trials] == pytest.approx([.0287, .0406, .0574])
    for trial in trials:
        assert all(trial[k] == v for k, v in NOISY_PRIOR.items() if k not in ("name", "prior_lr_mult"))
    first = dict(low=[.50], high=[.89], baseline=[.95], empf=[.90])
    second = dict(low=[.94], high=[.87], baseline=[.88], empf=[.92])
    result = select_particle_rate(dict(first=first, second=second), ("low", "high"), ("baseline", "empf"))
    assert result["winner"] == "high"
    assert result["passes"]
    second["high"] = [.88]
    assert not select_particle_rate(dict(first=first, second=second), ("low", "high"), ("baseline", "empf"))["passes"]


@pytest.mark.parametrize("fault", (None, "anchor", "batches"))
def test_extended_curves_check_replayed_anchor_and_intermediate_batches(tmp_path, monkeypatch, fault):
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_convergence as study

    published = json.loads((study.NOTES / "neuroko_v12_original_control_benchmark_first.json").read_text())
    arms = published["protocol"]["arms"]
    energy = {role: values[0] for role, values in published["result"]["scores"].items()}
    group = study.GROUPS[0]

    def fake_fit(args, stage, seed, group, gpu, requested, *, options):
        arm, = requested
        role = next(name for name, value in arms.items() if value == arm)
        version, recipe, trial = arm
        rates = dict(generator=trial["lr"], critic=trial["lr"] * trial["d_lr_mult"],
                     particles=trial["lr"] * trial["prior_lr_mult"])
        trace = [dict(updates=n, mean_energy_distance=energy[role], training_batches_sha256=str(n))
                 for n in study.STEPS]
        if role == "matched_v11" and fault == "anchor":
            trace[0]["mean_energy_distance"] += .01
        if role == "matched_v11" and fault == "batches":
            trace[1]["training_batches_sha256"] = "different"
        row = dict(updates=3600, constant_lr_verified_updates=3600, particlegan=version, recipe=recipe,
                   seed=seed, deterministic=True, score_groups=[group], fit_groups=study.GROUPS[1:],
                   fixed_lr_optimizer_steps=dict(generator_and_particles=3600, critic=3600),
                   initial_rates=rates, final_rates=rates, visible_devices=str(gpu), trace=trace,
                   means=dict(energy_distance=energy[role]), projected_data_sha256="cells",
                   initial_parameters_sha256="initial", training_batches_sha256="3600")
        base = study.receipt(args.output, stage, *arm, seed, group)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_stem(base.stem + "-u3600-trace1200").write_text(json.dumps(row))

    monkeypatch.setattr(study, "fit", fake_fit)
    args = SimpleNamespace(output=tmp_path)
    if fault:
        with pytest.raises(ValueError, match="Invalid fixed-rate curve|identical projected cells"):
            study.run_fold(args, arms, 22, group, energy)
    else:
        cell = study.run_fold(args, arms, 22, group, energy)
        assert cell["curves"].keys() == arms.keys()
        assert cell["fingerprints"]["training_batches"] == ["1200", "2400", "3600"]


def test_fresh_duration_reports_all_pairs_at_fixed_endpoint(tmp_path, monkeypatch):
    import json
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_convergence as study

    def example(args, arms, seed, group, original):
        assert original is None
        scores = dict(v12=1., tuned_v11=.95, matched_v11=1.05, matched_empf=1.1,
                      tuned_empf=.9 if seed < 840 else 1.1)
        curves = {role: [dict(updates=n, mean_energy_distance=value if n == 3600 else 1.,
                              effective_rank=18., spread_ratio=1.) for n in study.STEPS]
                  for role, value in scores.items()}
        return dict(seed=seed, group=group, curves=curves)

    monkeypatch.setattr(study, "run_fold", example)
    monkeypatch.setattr(sys, "argv", ["study", "--fresh", "--output", str(tmp_path),
                                       "--data", str(tmp_path), "--wheels", str(tmp_path)])
    study.main()
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert protocol["seeds"] == list(range(838, 844))
    assert len(result["cells"]) == 18
    summary = result["endpoint_comparisons"]["tuned_empf"]
    assert summary["pairs_won"] == 12 and summary["seed_means_won"] == 4
    assert summary["mean_v12_minus_control"] == pytest.approx(-1 / 30)
    assert not result["leads_all_controls_on_average"]


def test_transport_grid_selects_using_remaining_seeds(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_transport_grid as study
    from neuroko_rate_noise import NOISY_PRIOR, WIDE_NOISY_PRIOR

    baseline = dict(NOISY_PRIOR, name="original-v11-critic-49", d_lr_mult=.0049 / .014)
    arms = study.transport_arms(baseline)
    leader, runner_up = map(study.key, arms[:2])
    cohorts = []

    def scores(args, requested, seeds):
        first = len(seeds) == 1
        cohorts.append((tuple(seeds), len(requested)))
        values = {study.key(arm): [1.] * (3 * len(seeds)) for arm in requested}
        values[leader] = [.5 if first else 1.01] * (3 * len(seeds))
        values[runner_up] = [.9 if first else .95] * (3 * len(seeds))
        return values

    monkeypatch.setattr(study, "evaluate", scores)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, second_screen=None, stage="search")
    winner, grid = study.transport_grid(args, baseline, WIDE_NOISY_PRIOR, [22, 23, 24])
    assert cohorts == [((22,), 12), ((251,), 12), ((23, 24), 6), ((252, 253), 6)]
    assert winner == arms[1][2]
    assert len({(a[2]["rhbi_tau"], a[2]["rhbi_eps"]) for a in grid["candidates"]}) == 8
    assert all(v == baseline[k] for k, v in winner.items() if k not in ("name", "rhbi_eps", "rhbi_tau"))


def test_transport_metrics_exclude_slack_row_and_weight_by_matched_mass(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from neuroko_transport_plans import plan_metrics

    plan = torch.tensor([[1., 1., 2.], [4., 0., 0.], [0., 0., 9.]])
    slack, count = plan_metrics(plan)
    assert slack.item() == pytest.approx(2 / 8)
    assert count.item() == pytest.approx((2 * 2 + 4 * 1) / 6)
    plan[:2, :2] = 0
    slack, count = plan_metrics(plan)
    assert (slack.item(), count.item()) == (1., 0.)


def test_published_release_comparison_starts_without_prior_run_directory(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import neuroko_rate_noise as study

    args = SimpleNamespace(output=tmp_path, shared_screen=None, second_screen=None,
                           original_v11_motion=root / "notes/neuroko_original_v11_critic_motion.json")
    seen = []

    def scores(local, arms, seeds):
        assert local.output == tmp_path
        seen.append(list(seeds))
        return {study.key(a): [1 + i * .01] * 9 for i, a in enumerate(arms)}

    monkeypatch.setattr(study, "evaluate", scores)
    study.compare_published_rates(args)
    assert seen == [[22, 23, 24], [251, 252, 253]]
    arms = json.loads((tmp_path / "comparison_protocol.json").read_text())["arms"]
    assert arms["matched_v11"][:2] == arms["tuned_v11"][:2] == ["0.11.1", "unified_v11"]
    assert arms["matched_empf"][:2] == arms["tuned_empf"][:2] == ["0.10.0", "empf"]
    assert arms["v12"][2] == arms["matched_v11"][2]
    for role, (_, _, t) in arms.items():
        critic = {"tuned_v11": .0046, "tuned_empf": .021}.get(role, .0049)
        assert (t["lr"], t["lr"] * t["d_lr_mult"], t["lr"] * t["prior_lr_mult"]) == pytest.approx(
            (.014, critic, .0406))
        assert t["prior_sigma"] == (.25 if role == "tuned_empf" else .15)
        assert t["prior_reg"] == (.041 if role == "tuned_v11" else .1)
    result = json.loads((tmp_path / "second_comparison.json").read_text())
    assert result["comparisons"]["tuned_empf"]["pairs_won"] == 9


def test_critic_weight_interaction_requires_both_cohorts(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    def scores(local, arms, seeds):
        corner_scores = [1., .95, .98, .9 if seeds[0] == 22 else .99]
        return {study.key(a): [v]*9 for a, v in zip(arms, corner_scores + [.96]*4, strict=True)}

    monkeypatch.setattr(study, "evaluate", scores)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, second_screen=None, stage="search")
    baseline = dict(study.NOISY_PRIOR, name="original-v11-critic-49", d_lr_mult=.0049/.014)
    winner, plan = study.critic_weight_interaction(args, baseline)
    result = json.loads((tmp_path / "critic_weight_scores.json").read_text())
    assert winner is None and not result["passes"]
    assert result["cohorts"]["first"]["passes"] and not result["cohorts"]["second"]["passes"]
    assert result["cohorts"]["first"]["paired_interaction"] == pytest.approx([-.03]*9)
    assert plan["cells"]["upper_half"][2]["d_lr_mult"] * .014 == pytest.approx(.0071)


def test_prior_spread_target_is_shared_through_qualification(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from neuroko_rate_noise import spread_target
    from neuroko_role_factorial import paired
    from neuroko_stratified import control_arms, key

    trial, spread = spread_target(root / "notes/neuroko_v12_prior_weight_geometry.json")
    assert trial["prior_reg_target_std"] == round(1 / spread, 2) == .89
    candidates, labels = control_arms([trial])
    final = {key(a): a for a in candidates}
    for check, label in zip(paired(trial), ("selected_v12", "matched_v11", "matched_empf"), strict=True):
        for arm in (check, final[labels[label]]):
            assert arm[2]["prior_reg_target_std"] == .89
            assert arm[2]["prior_reg"] == .1
            assert arm[2]["prior_sigma"] == .15


@pytest.mark.parametrize("original_v11_energy,expected_cohorts", [(.8, 1), (1.2, 2)])
@pytest.mark.parametrize("study_name", ["balanced-particle-prior", "original-v11-critic"])
def test_native_release_ranking_gates_the_fresh_matched_replication(tmp_path, monkeypatch,
                                                                    original_v11_energy, expected_cohorts, study_name):
    import json
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    trial = dict(study.NOISY_PRIOR, name="motion-critic-46-prior41", prior_reg=.041, d_lr_mult=.0046/.014)
    if study_name == "original-v11-critic":
        trial = dict(study.NOISY_PRIOR, name="original-v11-critic-49", d_lr_mult=.0049/.014)
    arms = study.paired(trial)
    study.write_frozen(tmp_path / "search.json", dict(arms=arms, screen_seeds=[22, 23, 24]))
    scores = {study.key(arm): [1.0 + i / 10] * 9 for i, arm in enumerate(arms)}
    study.write_frozen(tmp_path / "screen.json", dict(scores=scores, passes=False))
    initial = {study.key(arms[0]): scores[study.key(arms[0])]}
    study.write_frozen(tmp_path / "preliminary.json",
                       dict(scores={**initial, study.key(study.paired(study.NOISY_PRIOR)[0]): [1.1] * 9}))
    cohorts = []

    def compare(args, destination, matches, seeds, rule, *, published):
        assert matches == arms
        source = initial if study_name == "original-v11-critic" else scores
        assert published == (source if not cohorts else None)
        cohorts.append((destination.name, seeds))
        return {**{n: {"energy": v[0]} for n, v in scores.items()},
                "0.11.1/v11": {"energy": original_v11_energy}, "0.10.0/empf": {"energy": 1.1}}

    monkeypatch.setattr(study, "original_release_scores", compare)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, study=study_name)
    study.replicate_calibrated_releases(args)
    start = 524 if study_name == "original-v11-critic" else 452
    assert cohorts == [("development", [22, 23, 24]), ("fresh", list(range(start, start + 6)))][:expected_cohorts]
    assert json.loads((tmp_path / "screen.json").read_text())["passes"] is False


def test_calibrated_output_noise_reaches_all_matched_arms(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from neuroko_rate_noise import calibrated_output_noise, paired
    from neuroko_stratified import control_arms, key

    trial, variance = calibrated_output_noise(root / "notes/neuroko_v12_particle_spread.json")
    assert trial["output_noise_std"] == round(variance**.5, 2) == .35
    controls, labels = control_arms([trial])
    arms = {key(a): a for a in controls}
    for early, label in zip(paired(trial), ("selected_v12", "matched_v11", "matched_empf"), strict=True):
        for arm in (early, arms[labels[label]]):
            assert (arm[2]["output_noise_std"], arm[2]["prior_sigma"], arm[2]["prior_reg"]) == (.35, .15, .1)


@pytest.mark.parametrize("study_name,sigma", [("original-v11-critic", .15),
                                              ("original-v11-critic-wide-noise", .25),
                                              ("original-v11-critic-narrow-noise", .09)])
def test_original_critic_probe_gates_expensive_controls_on_training_energy(tmp_path, monkeypatch, study_name, sigma):
    import json
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import neuroko_rate_noise as study

    args = SimpleNamespace(study=study_name, stage="search", output=tmp_path,
                           shared_screen=None, original_v11_motion=root / "notes/neuroko_original_v11_critic_motion.json",
                           motion_result=root / "notes/neuroko_v12_critic_motion.json",
                           motion_check=root / "notes/neuroko_v12_critic_motion_check.json",
                           gradient_result=root / "notes/neuroko_v12_particle_balance.json")
    candidate, ratios = study.original_v11_critic_rate(args.original_v11_motion)
    assert candidate["lr"] * candidate["d_lr_mult"] == pytest.approx(.0049)
    assert ratios[.0046] < 1 < ratios[.021]

    def screen(config, arms, seeds):
        assert [a[1] for a in arms] == ["unified_v12", "unified_v12", "unified_v11", "empf"]
        assert [a[2]["prior_sigma"] for a in arms] == [sigma, .15, .25, .25]
        expected_critics = [.0049, .021 if sigma == .15 else .0049, .021, .021]
        assert [a[2]["lr"] * a[2]["d_lr_mult"] for a in arms] == pytest.approx(expected_critics)
        return {study.key(arm): [1.0 - i / 100] * 9 for i, arm in enumerate(arms)}

    monkeypatch.setattr(study, "evaluate", screen)
    study.compare_prior_weight(args)
    assert json.loads((tmp_path / "selection.json").read_text())["advances"] is False
    assert not (tmp_path / "screen.json").exists()


def test_critic_rate_interpolation_checks_the_same_v11_reference(tmp_path, monkeypatch):
    import json

    paper_watch = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(paper_watch / "scripts"))
    from neuroko_rate_noise import calibrated_critic

    baseline = paper_watch / "notes/neuroko_v12_critic_motion.json"
    check = paper_watch / "notes/neuroko_v12_critic_motion_check.json"
    trial, calibration = calibrated_critic(baseline, check)
    assert calibration["lower_critic_lr"] == .0035
    assert trial["lr"] * trial["d_lr_mult"] == calibration["fixed_critic_lr"] == .0046
    different_probe = json.loads(check.read_text())
    different_probe["cells"][0]["probe_indices_sha256"] = "different"
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(different_probe))
    with pytest.raises(ValueError, match="identical v11 probes"):
        calibrated_critic(baseline, incompatible)


@pytest.mark.parametrize("passes", [True, False])
def test_critic_interval_gates_on_endpoints_and_tuned_controls(tmp_path, monkeypatch, passes):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    calibrated = dict(study.NOISY_PRIOR, name="original-v11-critic-49", d_lr_mult=.0049/.014)
    args = SimpleNamespace(output=tmp_path, stage="search", second_screen=tmp_path / "second",
                           shared_screen=tmp_path / "first")
    observed = []

    def evaluate(config, arms, seeds):
        observed.append((config.output, seeds))
        if seeds == [251, 252, 253]:
            assert [arm[1] for arm in arms] == ["unified_v12", "unified_v12", "unified_v11", "empf"]
            assert [arm[2]["prior_sigma"] for arm in arms] == [.15, .15, .25, .25]
            values = [1.] * 4
        else:
            rates = [arm[2]["lr"] * arm[2]["d_lr_mult"] for arm in arms[:3]]
            assert rates == pytest.approx([.0101, .0071, .0146])
            assert all(dict(t, name=calibrated["name"], d_lr_mult=calibrated["d_lr_mult"]) == calibrated
                       for _, _, t in arms[:3])
            values = [1., .90, .95, .94, .96, .93, .92 if passes else .89]
        return {study.key(arm): [v] * 9 for arm, v in zip(arms, values, strict=True)}

    monkeypatch.setattr(study, "evaluate", evaluate)
    chosen, plan = study.critic_interval(args, calibrated)
    assert observed == [(args.second_screen, plan["second"]), (args.shared_screen, plan["seeds"])]
    assert (chosen and chosen["name"]) == ("interval-critic-71" if passes else None)
    assert (args.output / "selection.json").exists() == (not passes)
    args.stage = "confirm"
    if passes:
        assert study.critic_interval(args, calibrated) == (chosen, plan)
    else:
        with pytest.raises(ValueError, match="No interior critic rate"):
            study.critic_interval(args, calibrated)
    assert len(observed) == 2


def test_prior_weight_uses_the_paired_particle_gradient_change(tmp_path, monkeypatch):
    import json

    paper_watch = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(paper_watch / "scripts"))
    from neuroko_rate_noise import NOISY_PRIOR, balanced_prior

    trial = dict(NOISY_PRIOR, name="motion-critic-46", d_lr_mult=.0046 / .014)
    path = paper_watch / "notes/neuroko_v12_particle_balance.json"
    balanced, scale = balanced_prior(path, trial)
    assert balanced["prior_reg"] == round(.1 / scale, 3) == .041
    assert balanced["lr"] * balanced["d_lr_mult"] == .0046
    bad = json.loads(path.read_text())
    bad["cells"][0]["v12_calibrated"]["ratios_by_update"].pop()
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Incomplete paired particle gradients"):
        balanced_prior(partial, trial)


def test_centroid_probe_averages_targets_and_requires_paired_support(tmp_path, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from neuroko_centroid_gradient import score_trace
    from neuroko_rate_noise import (CENTROID_WEIGHTS, GROUPS, NOISY_PRIOR, REPLICATION_SEEDS,
                                   centroid_weight_summary, centroid_weight_trial, paired)

    trace = [{"condition": label, "cosines": {str(w): float(label == 1) for w in CENTROID_WEIGHTS}}
             for label in (0, 1, 1, 2, 3, 4)]
    assert all(score == .25 for score in score_trace(trace).values())
    baseline = dict(NOISY_PRIOR, name="original-v11-critic-49", d_lr_mult=.0049/.014)
    plan = dict(weights=list(CENTROID_WEIGHTS), steps=64, rates=dict(generator=.014, critic=.0049, particles=.0406),
                prior_sigma=.15, prior_reg=.1)
    cells = [dict(seed=s, group=g, mean_cosine={str(w): 0. for w in CENTROID_WEIGHTS},
                  first_step_relative_errors={"0.5": 0., "1.0": 0., "4.0": 0.})
             for s in REPLICATION_SEEDS for g in GROUPS]
    assert centroid_weight_summary(cells)["best_weight"] == 2.
    for c in cells[:5]:
        c["mean_cosine"]["0.5"] = .5
    assert centroid_weight_summary(cells)["selected_weight"] is None
    cells[5]["mean_cosine"]["0.5"] = .5
    result = dict(protocol=plan, cells=cells, **centroid_weight_summary(cells))
    path = tmp_path / "gradient.json"
    path.write_text(json.dumps(result))
    trial = centroid_weight_trial(path, baseline)
    v12, v11, empf = paired(trial)
    assert v12[2]["rhbi_relativistic_weight"] == v11[2]["rhbi_relativistic_weight"] == .5
    assert "rhbi_relativistic_weight" not in empf[2]
    assert all(a[2]["lr"] * a[2]["d_lr_mult"] == .0049 for a in (v12, v11, empf))
    cells[0]["first_step_relative_errors"]["0.5"] = .2
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="direct gradient check"):
        centroid_weight_trial(path, baseline)


def test_generator_calibration_requires_improved_replay_and_preserves_other_rates(tmp_path, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from neuroko_rate_noise import GROUPS, NOISY_PRIOR, REPLICATION_SEEDS, generator_motion_trial

    baseline = dict(NOISY_PRIOR, d_lr_mult=.35, rhbi_relativistic_weight=.5)
    plan = dict(fixed_rates=dict(generator=.014, critic=.0049, particles=.0406), steps=64,
                releases={"0.11.1": "unified_v11", "0.12.0": "unified_v12"}, seeds=REPLICATION_SEEDS,
                groups=list(GROUPS), prior_reg=.1, prior_sigma=.15, relativistic_weight=.5)

    def check(lr, ratio):
        motion = {"0.12.0": {"rms": ratio}, "0.11.1": {"rms": 1.}}
        pairs = [dict(seed=s, group=g, ratio=ratio, motion=motion) for s in REPLICATION_SEEDS for g in GROUPS]
        return dict(generator_lr=lr, median_ratio=ratio, pairs=pairs)

    result = dict(protocol=plan, initial=check(.014, 1.4), checked=check(.01, .98), candidate_lr=.01,
                  agreeing_pairs=9, qualifies=True)
    source = tmp_path / "motion.json"
    source.write_text(json.dumps(result))
    trial = generator_motion_trial(source, baseline)
    assert trial["lr"] == .01
    assert [trial["lr"] * trial[k] for k in ("d_lr_mult", "prior_lr_mult")] == pytest.approx([.0049, .0406])
    result["checked"] = check(.01, 1.6)
    source.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="decision disagrees"):
        generator_motion_trial(source, baseline)
    result["qualifies"] = False
    source.write_text(json.dumps(result))
    assert generator_motion_trial(source, baseline) is None


@pytest.mark.parametrize("wins", (True, False))
def test_individual_role_rate_shortlist_preserves_other_rates(tmp_path, monkeypatch, wins):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    def evaluate(args, arms, seeds):
        assert seeds == [22, 23, 24]
        rates = []
        for _, recipe, trial in arms:
            assert recipe == "unified_v12"
            gen = trial["lr"]
            rates.append((gen, gen * trial["d_lr_mult"], gen * trial["prior_lr_mult"]))
            assert (trial["prior_sigma"], trial["prior_reg"]) == (.15, .1)
        assert rates == [(.014, .021, .0406), (.010, .021, .0406), (.020, .021, .0406),
                         (.014, .021, .0287), (.014, .021, .0574)]
        values = [1.0, .9 if wins else 1.0, 1.1, 1.05, 1.02]
        return {study.key(arm): [value] * 9 for arm, value in zip(arms, values, strict=True)}

    monkeypatch.setattr(study, "evaluate", evaluate)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, stage="search")
    chosen = study.individual_rates(args)
    if wins:
        assert chosen["name"] == "noise150-prior100-g010"
        args.stage = "confirm"
        assert study.individual_rates(args) == chosen
    else:
        assert chosen is None
        args.stage = "confirm"
        with pytest.raises(ValueError, match="No individual role rate passed"):
            study.individual_rates(args)


def test_wide_noise_rivals_can_be_measured_without_promoting_failed_candidates(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    calls = []

    def evaluate(args, arms, seeds):
        calls.append((list(seeds), [a[2]["name"] for a in arms]))
        return {study.key(a): [1.] * 9 for a in arms}

    monkeypatch.setattr(study, "evaluate", evaluate)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, second_screen=None, stage="search")
    assert study.individual_rates(args) is None
    study.compare_noise_controls(args)
    first = json.loads((tmp_path / "wide_noise_first_rates.json").read_text())
    assert len(first["differences"]) == 4
    assert all(len(comparators) == 2 for comparators in first["differences"].values())
    assert (tmp_path / "wide_noise_second_controls.json").is_file()
    assert not (tmp_path / "wide_noise_second_rates.json").exists()
    assert calls[-1] == ([251, 252, 253], ["noise150-prior100", "noise250-prior100", "noise250-prior100"])


def test_shared_prior_reuses_only_identical_native_recipes(monkeypatch):
    from importlib.metadata import version

    pytest.importorskip("particlegan")
    if version("particlegan") != "0.12.0":
        pytest.skip("requires ParticleGAN 0.12.0")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from neuroko_matched_rate_replication import TRIAL, comparison, prior_equivalence

    matches, _ = comparison(TRIAL, False)
    checked = prior_equivalence("0.12.0", matches)
    for name in ("unified_v12", "unified_v11"):
        assert checked[name] == dict(native_weight=.02, shared_weight=.02, identical_full_recipe=True)
    assert checked["empf"] == dict(native_weight=.01, shared_weight=.02, identical_full_recipe=False)


def test_test_cells_use_training_coordinates():
    train = csr_matrix(np.array([[0., 0.], [3., 1.], [2., 1.], [1., 0.], [4., 3.]]))
    test = train[[2, 0, 3]]
    fit, projected = project_fold(train, test, n_features=2, n_components=2)
    np.testing.assert_allclose(projected, fit[[2, 0, 3]])
    _, shifted = project_fold(train, test * 10, n_features=2, n_components=2)
    assert not np.allclose(projected, shifted)


def test_normalization_uses_complete_library():
    matrix = csr_matrix([[1, 3], [4, 4]])
    result = normalize_counts(matrix)
    np.testing.assert_allclose(result.toarray(), np.log1p([[2500, 7500], [5000, 5000]]))
    np.testing.assert_array_equal(matrix.toarray(), [[1, 3], [4, 4]])
    with pytest.raises(ValueError, match="integer-valued"):
        normalize_counts(csr_matrix([[0.5, 1.0]]))


@pytest.fixture
def mock_release(tmp_path):
    import anndata as ad

    rng = np.random.default_rng(0)
    for name, cols in LAYOUT.items():
        rows, counts = [], []
        for group in ("g1", "g2"):
            for idx, target in enumerate(("INT", "A", "B")):
                for j in range(40):
                    rows.append((target, f"{target}_{j % 2}", group))
                    levels = np.full(6, 3.0)
                    levels[idx] = 8
                    counts.append(rng.poisson(levels) + 1)
        obs = pd.DataFrame(rows, columns=cols, index=[str(i) for i in range(len(rows))])
        var = pd.DataFrame(index=["A", "B", "C", "D", "E", "F"])
        ad.AnnData(csr_matrix(np.array(counts)), obs=obs, var=var).write_h5ad(tmp_path / f"{name}.h5ad")
    return tmp_path


def test_backed_group_holdouts_and_control_arms(mock_release):
    torch.set_num_threads(2)
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(mock_release, name, window_cells=116)) for name in LAYOUT]
        for screen in screens:
            assert screen.inventory()["pilot_intervals"] == [[0, 116], [120, 236]]
            assert screen.inventory()["control_cells_by_group"] == {"g1": 40, "g2": 40}
        assert select_targets(screens, 32, 6, ["B", "A"]) == ["B", "A"]
        with pytest.raises(ValueError, match="requested targets below 38"):
            select_targets(screens, 38, 6, ["A", "B"])
        report = evaluate(screens, min_cells=32, cap=40, max_targets=2, steps=2, seeds=(0,))
    assert report["targets"] == ["A", "B"]
    assert report["transfer"]["common_unique_gene_symbols"] == 6
    assert report["transfer"]["genes_after_excluding_panel_targets"] == 4
    for folds in report["folds"].values():
        assert [fold["held_out_group"] for fold in folds] == ["g1", "g2"]
        assert [row["test_cells"] for row in folds[0]["targets"]] == [40, 36]
        for fold in folds:
            for row in fold["targets"]:
                assert row["train_cells"] == row["test_cells"]
                assert row["expression_matched_target"] != row["target"]
                assert set(row["scores"]) == {
                    "control_only", "mean_shift", "empirical_target", "diagonal_gaussian",
                    "empf", "shuffled_empf", "expression_matched_target",
                }
                assert row["scores"]["control_only"]["shift_cosine"] is None
                assert np.isfinite(row["scores"]["empf"]["energy_distance"])


def test_fetch_promotes_only_verified_bytes(tmp_path, monkeypatch):
    from hashlib import md5

    from paper_watch.scripts import fetch_neuroko

    released = b"abcde"
    screen = "iNeurons"
    monkeypatch.setitem(fetch_neuroko.FILES, screen, (len(released), md5(released).hexdigest()))
    partial = tmp_path / f"{screen}.h5ad.part"
    partial.write_bytes(released[:2])
    downloaded = []

    def resume(command, *, check):
        assert check and command[command.index("--continue-at") + 1] == "-"
        assert command[command.index("--output") + 1] == str(partial)
        assert partial.read_bytes() == released[:2]
        partial.write_bytes(released)
        downloaded.append(command[-1])

    monkeypatch.setattr(fetch_neuroko.subprocess, "run", resume)
    path = fetch_neuroko.fetch(tmp_path, screen)
    assert path.read_bytes() == released
    assert not partial.exists()
    assert fetch_neuroko.fetch(tmp_path, screen) == path
    assert len(downloaded) == 1
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_neuroko.fetch(tmp_path, screen)


def test_rate_selection_keeps_outer_group_out_of_preprocessing_and_validation():
    from types import SimpleNamespace

    from alztest.neuroko_lr import validation_split

    panel = SimpleNamespace(groups=np.array(["a", "b", "c", "d"]))
    fit, validation, description = validation_split(panel, "b", ["INT", "A"])
    np.testing.assert_array_equal(fit, [True, False, False, True])
    np.testing.assert_array_equal(validation, [False, False, True, False])
    assert description == {"validation_group": "c"}


def test_rate_selection_in_one_batch_holds_out_whole_vectors():
    from types import SimpleNamespace

    from alztest.neuroko_lr import validation_split

    panel = SimpleNamespace(
        groups=np.repeat(["a", "b"], 192),
        labels=np.tile(np.repeat([0, 1], 96), 2),
        vectors=np.tile(np.repeat(["i1", "i2", "i3", "A1", "A2", "A3"], 32), 2),
    )
    fit, validation, description = validation_split(panel, "a", ["INT", "A"])
    assert not np.any((fit | validation) & (panel.groups == "a"))
    assert set(panel.vectors[fit]).isdisjoint(panel.vectors[validation])
    assert validation.sum() == 64
    assert fit.sum() == 128
    assert description["validation_group"] == "b"


def test_rate_refinement_brackets_an_upper_bound_winner():
    from alztest.neuroko_lr import RATES, finer_rates

    records = [{"base_rate": rate, "mean_energy_distance": 1 / rate} for rate in RATES]
    assert finer_rates(records) == (0.0792, 0.224)


def test_fixed_rate_fit_reuses_only_a_fully_verified_receipt(tmp_path):
    import json
    from paper_watch.scripts.compare_neuroko_v12 import reusable_fit

    path = tmp_path / "fit.json"
    assert not reusable_fit(path, True)
    cached = {"updates": 1200, "rate_trace": [{"update": 1200, "generator": 0.014}]}
    path.write_text(json.dumps(cached))
    assert reusable_fit(path, False)
    assert not reusable_fit(path, True)
    for count in (1199, 1200):
        path.write_text(json.dumps(dict(cached, constant_lr_verified_updates=count)))
        assert reusable_fit(path, True) == (count == 1200)
    cached.update(settings={"output_noise_std": 0.35}, constant_lr_verified_updates=1200)
    for count in (None, 1199, 1200):
        path.write_text(json.dumps(dict(cached, fixed_output_noise_verified_updates=count)))
        assert reusable_fit(path, True) == (count == 1200)
