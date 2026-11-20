"""Pin the ParticleGAN 0.10.0 EMPF conditional-spread probe as a regression test.

The probe in analysis/particlegan_review_20261117 records the reported
``ema_residual`` for a correct batch, a batch whose two groups exchange their
spreads, and a batch with both spreads halved. Passing cases pin the observed
v0.10.0 values. The strict xfail cases record, for this pinned artifact only, the
upstream requirement that the spread swap score worse than the correct pairing
at widths 65 and 2,150. The fixture refuses any version other than 0.10.0 loaded
from SITE, so a newer release gives setup errors here, not unexpected passes.
To check a candidate release, run probe_conditional_spread.py against it or copy
this test with a new pin; the xfails flip only if the pinned copy itself changes.

This characterizes one deterministic batch through the reported residual. It
says nothing about trained samples.

The frozen release is not a project dependency. The test runs the probe in a
CPU subprocess with the 0.10.0 site-packages on PYTHONPATH, so it cannot mix
with any other installed ParticleGAN. It skips when that directory is absent.
Set ALZ_PARTICLEGAN_0_10_SITE_PACKAGES to use a copy outside this checkout.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT/'analysis/particlegan_review_20261117/probe_conditional_spread.py'
SITE = Path(os.environ.get('ALZ_PARTICLEGAN_0_10_SITE_PACKAGES',
                           ROOT/'artifacts/native-releases/particlegan-0.10.0/site-packages'))
WIDTHS = (64, 65, 2150)

# Import the probe's own fixture and penalty call; only the loop is repeated here.
RUNNER = r'''
import importlib.metadata, importlib.util, json, sys
import particlegan, torch
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
out = {"version": importlib.metadata.version("particlegan"), "file": particlegan.__file__, "cases": {}}
for width in map(int, sys.argv[2:]):
    real, swapped, condition = probe.fixture(width)
    out["cases"][width] = {
        "correct": probe.observed_residual(real, real, condition),
        "swapped": probe.observed_residual(real, swapped, condition),
        "halved": probe.observed_residual(real, real * 0.5, condition)}
print(json.dumps(out))
'''

pytestmark = [
    pytest.mark.needs_particlegan,
    pytest.mark.skipif(not (SITE/'particlegan-0.10.0.dist-info').is_dir(),
                       reason=f'ParticleGAN 0.10.0 site-packages not found at {SITE}'),
]


@pytest.fixture(scope='module')
def residuals():
    env = dict(os.environ, PYTHONPATH=str(SITE), CUDA_VISIBLE_DEVICES='')
    run = subprocess.run([sys.executable, '-B', '-c', RUNNER, str(PROBE), *map(str, WIDTHS)],
                         env=env, capture_output=True, text=True, timeout=120)
    if run.returncode and "No module named 'torch'" in run.stderr:
        pytest.skip('torch is not importable in this interpreter')
    # pytest.fail, not assert: the strict xfail below must not absorb a broken setup.
    if run.returncode:
        pytest.fail(f'probe subprocess failed:\n{run.stderr}')
    out = json.loads(run.stdout.strip().splitlines()[-1])
    if out['version'] != '0.10.0' or not Path(out['file']).resolve().is_relative_to(SITE.resolve()):
        pytest.fail(f"loaded ParticleGAN {out['version']} from {out['file']}, not 0.10.0 from {SITE}")
    return {int(width): case for width, case in out['cases'].items()}


@pytest.mark.parametrize('width', WIDTHS)
def test_correct_pairing_reports_zero_residual(residuals, width):
    assert residuals[width]['correct'] == pytest.approx(0.0, abs=1e-9)


def test_width_64_conditional_path_separates_swapped_spreads(residuals):
    case = residuals[64]
    assert case['swapped'] == pytest.approx(0.931209, abs=1e-6)
    assert case['swapped'] > case['correct']


@pytest.mark.parametrize('width,expected', [(64, 0.617582), (65, 0.586302), (2150, 0.586302)])
def test_halved_spreads_positive_control(residuals, width, expected):
    case = residuals[width]
    assert case['halved'] == pytest.approx(expected, abs=1e-6)
    assert case['halved'] > case['correct']


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='v0.10.0 wide continuous EMPF path uses a pooled '
                   'nearest-neighbor residual that is invariant to cross-condition spread swaps')
@pytest.mark.parametrize('width', (65, 2150))
def test_wide_path_separates_swapped_spreads(residuals, width):
    assert residuals[width]['swapped'] > residuals[width]['correct']
