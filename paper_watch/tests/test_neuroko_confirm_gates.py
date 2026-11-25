"""Holdout confirmation runs only the candidate that the frozen selection advanced."""
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def replication(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_matched_rate_replication as module

    calls = []
    monkeypatch.setattr(module, "confirm_candidate", lambda args, decision, trial, seeds, **_: calls.append(
        (trial["name"], seeds[0], seeds[-1])))
    return module, calls


def run(module, monkeypatch, tmp_path, rates):
    monkeypatch.setattr(sys, "argv", ["prog", "--data", str(tmp_path), "--wheels", str(tmp_path),
                                      "--output", str(tmp_path), "--stage", "confirm", "--rates", rates])
    module.main()


def freeze(tmp_path, module, rates, advances=True):
    trial, rule = (module.EMPF_RATES, module.EMPF_RULE) if rates == "empf" else (module.TRIAL, module.RULE)
    (tmp_path / "search.json").write_text(json.dumps({"rule": rule}))
    (tmp_path / "selection.json").write_text(json.dumps(
        {"trial": trial, "advances": advances, "winner": trial["name"] if advances else None}))


@pytest.mark.parametrize("rates, seeds", [("seed15", (180, 195)), ("empf", (196, 211))])
def test_confirm_runs_the_candidate_that_advanced(replication, monkeypatch, tmp_path, rates, seeds):
    module, calls = replication
    freeze(tmp_path, module, rates)
    run(module, monkeypatch, tmp_path, rates)
    trial = module.EMPF_RATES if rates == "empf" else module.TRIAL
    assert calls == [(trial["name"], *seeds)]


@pytest.mark.parametrize("searched, requested", [("seed15", "empf"), ("empf", "seed15")])
def test_confirm_refuses_a_selection_made_for_the_other_rates(replication, monkeypatch, tmp_path,
                                                              searched, requested):
    module, calls = replication
    freeze(tmp_path, module, searched)
    with pytest.raises(ValueError, match="was not made for --rates"):
        run(module, monkeypatch, tmp_path, requested)
    assert calls == []


def test_confirm_still_refuses_a_candidate_that_did_not_advance(replication, monkeypatch, tmp_path):
    module, calls = replication
    freeze(tmp_path, module, "seed15", advances=False)
    with pytest.raises(ValueError):
        run(module, monkeypatch, tmp_path, "seed15")
    assert calls == []
