"""Current-state documentation stays consistent and its relative links resolve."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANNER = 'SUPERSEDED by STATUS.md (2026-11-20).'
SUPERSEDED = (
    'CONTINUATION.txt',
    'analysis/CELL_ENERGY_LIVE_CONTINUATION.txt',
    'analysis/EMA_READOUT_CONTINUATION.txt',
    'analysis/FROZEN_CONTROL_LIVE_CONTINUATION.txt',
    'analysis/LEARNED_CONTINUATION_REVIEW.txt',
    'analysis/LEARNED_CONTINUATION_V5_REVIEW.txt',
    'analysis/MOG_LIVE_CONTINUATION.txt',
    'analysis/ROUTED_E22_CPU_CONTINUATION.txt',
    'analysis/STAGE_PRIOR_LIVE_CONTINUATION.txt',
    'analysis/TURNOVER_CONTINUATION_REVIEW.txt',
    'analysis/empf_comparison_20261117/TAKEOVER.txt',
    'analysis/particlegan_upgrade_20261117/README.txt',
    'analysis/particlegan_upgrade_20261117/TAKEOVER.txt',
)
CHECKED_MARKDOWN = ('README.md', 'STATUS.md', 'docs/LEGACY_LAYOUT.md', 'analysis/INDEX.md')
LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\s]+)\)')

spec = importlib.util.spec_from_file_location('index_analysis', ROOT/'scripts/hygiene/index_analysis.py')
INDEX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INDEX)


def relative_links(path):
    for target in LINK.findall(path.read_text(encoding='utf-8')):
        if re.match(r'^[a-z][a-z0-9+.-]*:', target) or target.startswith('#'):
            continue
        yield target.split('#', 1)[0]


@pytest.mark.parametrize('name', CHECKED_MARKDOWN)
def test_relative_links_resolve(name):
    path = ROOT/name
    missing = [t for t in relative_links(path) if not (path.parent/t).exists()]
    assert not missing, f'{name}: {missing}'


@pytest.mark.parametrize('name', SUPERSEDED)
def test_historical_handoffs_point_to_status(name):
    first = (ROOT/name).read_text(encoding='utf-8').splitlines()[0]
    assert first.startswith(BANNER)


def test_status_names_the_configured_default():
    config = json.loads((ROOT/'configs/microglia_default.json').read_text())
    status = (ROOT/'STATUS.md').read_text(encoding='utf-8')
    assert config['name'] in status
    assert config['status'] in status
    assert config['generation']['mode'] in status
    assert f"ParticleGAN\n{config['training']['particlegan_version']}" in status or \
        f"ParticleGAN {config['training']['particlegan_version']}" in status
    assert 'coverage_empf_confirm_20261120' in status


def test_readme_map_lists_current_entry_points():
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    table = readme[readme.index('## Repository map'):readme.index('The original perturbation smoke')]
    for entry in ('`configs/`', '`scripts/microglia.py`', '`paper_watch/`', '`STATUS.md`',
                  '`runs/`', '`artifacts/`', 'LEGACY_LAYOUT.md'):
        assert entry in table
    assert '`src/alz_perturb_gan/`, `tests/`' not in table  # tests/ is not only legacy


def test_index_groups_studies_families_and_copies():
    rows = [
        ('analysis/INDEX.md', 'f'*40),
        ('analysis/a_20261101.json', '1'*40),
        ('analysis/a_20261103.json', '2'*40),
        ('analysis/a_x.py', '3'*40),
        ('analysis/lonely.txt', '4'*40),
        ('analysis/default_empf_20261117/README.md', '5'*40),
        ('analysis/old_study/model/x.py', '6'*40),
        ('analysis/other_study/model/x.py', '6'*40),
    ]
    text = INDEX.render(rows)
    assert text == INDEX.render(list(reversed(rows)))
    assert '| `a` | 3 | json 2, py 1 | 20261101-20261103 |' in text
    assert '| `(other)` | 1 |' in text
    assert '[README.md](default_empf_20261117/README.md)' in text
    assert '| 2 | `old_study/model/x.py`, `other_study/model/x.py` |' in text
    assert 'Tracked files: 7 (4 at the top level' in text  # INDEX.md itself is not data


def test_committed_index_names_every_current_study():
    text = (ROOT/'analysis/INDEX.md').read_text(encoding='utf-8')
    assert text.startswith('# Index of analysis/')
    for name, _ in INDEX.CURRENT:
        assert f'`{name}/`' in text
