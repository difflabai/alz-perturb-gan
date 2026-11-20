"""Tests for the condition-aware wide EMPF residual adapter (CPU, no training).

PYTHONPATH=<particlegan-0.10.0 site-packages> python -B -m pytest -q -p no:cacheprovider test_cond_residual.py
"""
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cond_residual as CR  # noqa: E402
import probe as PR  # noqa: E402
import synthetic as S  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope='module')
def rows():
    return {(r['width'], r['arm']): r for r in PR.table()}


@pytest.mark.parametrize('width', PR.WIDTHS)
def test_adapter_scores_swap_worse(rows, width):
    r = rows[(width, 'adapter')]
    assert r['correct'] == 0.0
    assert r['swapped'] > r['correct'] + 0.1
    assert r['halved'] > 0.1


@pytest.mark.parametrize('width', (65, 2150))
def test_native_blind_spot_documented(rows, width):
    r = rows[(width, 'native')]
    assert r['correct'] == 0.0 and r['swapped'] == 0.0 and r['halved'] > 0


@pytest.mark.parametrize('width', PR.WIDTHS)
@pytest.mark.parametrize('arm', ('native', 'adapter'))
def test_paired_row_reorder_control(rows, width, arm):
    r = rows[(width, arm)]
    assert r['correct_reordered'] == pytest.approx(r['correct'], abs=1e-6)
    assert r['swapped_reordered'] == pytest.approx(r['swapped'], abs=1e-6)


def test_width64_unchanged_by_adapter(rows):
    a, n = rows[(64, 'adapter')], rows[(64, 'native')]
    for k in ('correct', 'swapped', 'halved', 'penalty_correct', 'penalty_swapped'):
        assert a[k] == n[k]
    assert set(a['cond_calls'].values()) == {0}


@pytest.mark.parametrize('width', (65, 2150))
def test_adapter_routes_every_wide_call(rows, width):
    assert set(rows[(width, 'adapter')]['cond_calls'].values()) == {1}


def _run(width, adapter, condition, seed=3, conditioned=True):
    real = torch.randn(32, width, generator=torch.Generator().manual_seed(seed))
    fake = torch.randn(32, width, generator=torch.Generator().manual_seed(seed + 1)) * 1.3
    if conditioned:
        out = PR.residual(real, fake, condition, adapter)
        return out['ema_residual'], out['penalty'], out['grad']
    from particlegan import get_recipe
    recipe = get_recipe('empf', reg_every=1, batch_size=32)
    torch.manual_seed(0)
    critic = torch.nn.Sequential(torch.nn.Linear(width, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
    opt = recipe.make_critic_optimizer(critic)
    pen = recipe.make_critic_penalty(opt, input_domain='continuous', collect_stats=True)
    if adapter:
        CR.install(pen)
    v = pen(critic, real, fake); v.backward()
    return pen.last_stats['ema_residual'], float(v.detach()), critic[0].weight.grad.clone()


@pytest.mark.parametrize('width', (65, 2150))
def test_single_condition_is_bit_identical_to_native(width):
    c = torch.zeros(32, 7); c[:, 3] = 2.0
    n, a = _run(width, False, c), _run(width, True, c)
    assert n[0] == a[0] and n[1] == a[1] and torch.equal(n[2], a[2])


def test_unconditioned_call_is_untouched():
    n, a = _run(300, False, None, conditioned=False), _run(300, True, None, conditioned=False)
    assert n[0] == a[0] and n[1] == a[1] and torch.equal(n[2], a[2])


def test_residual_matches_masked_bruteforce():
    g = torch.Generator().manual_seed(11)
    r, f = torch.randn(32, 90, generator=g), torch.randn(32, 90, generator=g)
    c = torch.zeros(32, 7); c[:, 3] = torch.arange(32) % 4
    res, scale = CR.condition_group_chamfer_residual(r, f, [c])
    d = torch.cdist(r, f).pow(2); m = d.median()
    a = [min(d[i, j] for j in range(32) if c[i, 3] == c[j, 3]) for i in range(32)]
    b = [min(d[i, j] for i in range(32) if c[i, 3] == c[j, 3]) for j in range(32)]
    want = torch.sqrt(0.5 * (torch.stack(a).mean() + torch.stack(b).mean()) / m)
    assert float(res) == pytest.approx(float(want), rel=1e-5)
    assert float(scale) == pytest.approx(float(m.sqrt()), rel=1e-6)


def test_install_rejects_non_empf():
    with pytest.raises(TypeError):
        CR.install(object())


def test_synthetic_truth_and_controls():
    mean, sd = S.truth(128)
    # Latin square: every informative coordinate has the same pooled SD multiset.
    for j in range(S.INFO):
        assert sorted(sd[:, j]) == sorted(S.LEVELS)
    ctrl = S.controls(128)
    good, bad = ctrl['correct_sampler']['informative'], ctrl['swapped_spread_sampler']['informative']
    assert good['logsd_rmse'] < 0.05 and abs(good['coverage80'] - 0.8) < 0.02
    assert bad['logsd_rmse'] > 0.3 and bad['interval_score'] > good['interval_score']
