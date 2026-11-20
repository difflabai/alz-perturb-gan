"""Tests for the donor-latent study (CPU, no data needed except where marked).

Run in a fresh process:
    PYTHONPATH=/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages \
      /mnt/nvme/ai/conda/anaconda3/bin/python3 -B -m pytest -q analysis/donor_latent_20261120/test_donor_latent.py
"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip('torch')
import arm_dre  # noqa: E402


class Base(torch.nn.Module):
    """Minimal stand-in with the HookedCountGenerator interface (CountGenerator layer shapes)."""

    def __init__(self, mean, theta, zd=8, zc=8, h=128):
        super().__init__()
        self.zc = zc
        self.donor = torch.nn.Sequential(torch.nn.Linear(zd + 7, h), torch.nn.SiLU(), torch.nn.Linear(h, 2 * h))
        self.cell = torch.nn.Sequential(torch.nn.Linear(zc, h), torch.nn.SiLU())
        self.out = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, len(mean)))
        self.log_theta = torch.nn.Parameter(torch.zeros(len(mean)))

    def film(self, z, c):
        return self.donor(torch.cat([z, c], 1)).chunk(2, 1)

    def cell_hidden(self, noise, scale, shift, z, c):
        return self.cell(noise) * (1 + scale[:, None]) + shift[:, None]

    def logits(self, h, z, c):
        return self.out(h)

    def dispersion(self, z, c):
        return torch.nn.functional.softplus(self.log_theta) + .05

    def parameters_nb(self, z, c, k, library, gen=None):
        scale, shift = self.film(z, c)
        noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
        h = self.cell_hidden(noise, scale, shift, z, c)
        p = torch.softmax(self.logits(h, z, c), -1).clamp_min(1e-8)
        p = p / p.sum(-1, keepdim=True)
        return library.clamp_min(1)[..., None] * p, self.dispersion(z, c)


def make(cls, seed=0, genes=20):
    torch.manual_seed(seed)
    return cls(torch.ones(genes), torch.ones(genes))


def inputs(B=3, k=5):
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, 8, generator=g), torch.zeros(B, 7), torch.full((B, k), 100.)


def test_native_parameters_and_init_unchanged():
    base, dre = make(Base), make(arm_dre.dre_class(Base))
    for (n, p), (n2, p2) in zip(base.named_parameters(), dre.named_parameters()):
        assert n == n2 and torch.equal(p, p2)
    extra = {n for n, _ in dre.named_parameters()} - {n for n, _ in base.named_parameters()}
    assert extra == {'re_scale.weight', 're_loc.weight'}
    assert float(dre.re_scale.weight.abs().sum()) == 0 and float(dre.re_loc.weight.abs().sum()) == 0


def test_zero_init_forward_equals_native_for_same_noise():
    base, dre = make(Base), make(arm_dre.dre_class(Base))
    z, c, lib = inputs()
    mu0, th0 = base.parameters_nb(z, c, 5, lib, torch.Generator().manual_seed(7))
    mu1, th1 = dre.parameters_nb(z, c, 5, lib, torch.Generator().manual_seed(7))
    assert torch.equal(mu0, mu1) and torch.equal(th0, th1)


def test_random_effect_gradients_nonzero_and_effect_is_donor_level():
    dre = make(arm_dre.dre_class(Base))
    z, c, lib = inputs()
    mu, _ = dre.parameters_nb(z, c, 5, lib, torch.Generator().manual_seed(7))
    (mu.square().mean()).backward()
    assert float(dre.re_scale.weight.grad.abs().sum()) > 0 and float(dre.re_loc.weight.grad.abs().sum()) > 0
    # With nonzero maps, two bags with the same z, c and cell noise differ only through u (one draw per bag).
    with torch.no_grad():
        dre.re_loc.weight.normal_(); dre.re_scale.weight.normal_(std=.3)
    z1 = z[:1].repeat(2, 1); c1 = c[:1].repeat(2, 1); lib1 = lib[:1].repeat(2, 1)
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(2, 5, 8, generator=g); u = torch.randn(2, 4, generator=g)
    assert not torch.equal(u[0], u[1])
    mu2, _ = dre.parameters_nb(z1, c1, 5, lib1, torch.Generator().manual_seed(3))
    assert not torch.allclose(mu2[0], mu2[1])


def test_extra_draw_follows_native_noise():
    """u is drawn after the native cell noise, so the native noise stream prefix is unchanged."""
    dre = make(arm_dre.dre_class(Base))
    seen = []
    original = torch.randn

    def spy(*a, **kw):
        out = original(*a, **kw); seen.append(tuple(out.shape)); return out
    z, c, lib = inputs()
    torch.randn = spy
    try:
        dre.parameters_nb(z, c, 5, lib, torch.Generator().manual_seed(1))
    finally:
        torch.randn = original
    assert seen == [(3, 5, 8), (3, 4)]


def test_registration_and_plugin_hooks():
    class FakeArms:
        _ARMS = {}

        class Arm:
            def __init__(self, value, flag):
                self.value, self.flag, self.info = value, flag, {}

        @staticmethod
        def register(cls):
            FakeArms._ARMS[cls.name] = cls
            return cls
    cls = arm_dre.register(FakeArms)
    assert FakeArms._ARMS['g.dre'] is cls and arm_dre.register(FakeArms) is cls
    arm = cls(None, 'g.dre'); arm.setup(None)
    G = make(arm.generator_class(None, Base))
    assert [p.shape for p in arm.generator_grad_params(G)] == [(8, 4), (128, 4)]


def test_bootstrap_ci_paired_and_deterministic():
    import score
    rng = np.random.default_rng(0)
    d = [np.full(12, -.1), np.full(12, -.2)]
    r = score.boot_ci(d, rng)
    assert r['point'] == pytest.approx(-.15) and r['ci95'] == pytest.approx([-.15, -.15]) and r['excludes_zero']
    r2 = score.boot_ci([np.array([1., -1.] * 6)], np.random.default_rng(1))
    assert r2['ci95'][0] < 0 < r2['ci95'][1] and not r2['excludes_zero']


def test_per_donor_matches_frozen_summary():
    import score
    import interval_metrics as I
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(5, 32, 100)); observed = rng.normal(size=(3, 5, 100))
    mask = np.array([True, False, True, True, False])
    frozen = I.summarize(samples, observed, mask)
    mine = score.per_donor(samples, observed)
    assert np.mean(mine['sd_all50']['coverage'][mask]) == pytest.approx(frozen['sd']['coverage'])
    assert np.mean(mine['sd_all50']['interval_score'][mask]) == pytest.approx(frozen['sd']['interval_score'])
    assert np.mean(mine['combined_all100']['interval_score'][mask]) == pytest.approx(frozen['combined']['interval_score'])
    assert np.mean(mine['mean_all50']['width'][mask]) == pytest.approx(frozen['mean']['width'])


@pytest.mark.skipif(not Path('/ml2/alz/analysis/default_empf_20261117/runtime_source_manifest.json').exists(), reason='needs workspace')
def test_worktree_sources_match_pinned_application_sources():
    import dl_host
    assert dl_host.check_pinned()['n_pinned_sources_checked'] > 50
