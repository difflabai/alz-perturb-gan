"""Current-state documentation stays consistent and its relative links resolve."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANNER = 'SUPERSEDED by STATUS.md (2026-11-20).'
CHECKED_MARKDOWN = ('README.md', 'STATUS.md', 'docs/LEGACY_LAYOUT.md', 'analysis/INDEX.md')
LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\s]+)\)')

spec = importlib.util.spec_from_file_location('index_analysis', ROOT/'scripts/hygiene/index_analysis.py')
INDEX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INDEX)
spec = importlib.util.spec_from_file_location('build_record_registry',
                                              ROOT/'scripts/hygiene/build_record_registry.py')
REG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REG)
REGISTRY = json.loads((ROOT/REG.REGISTRY).read_text(encoding='utf-8'))
# Every handoff known when the glob replaced the closed allowlist; the glob must
# keep finding all of them (it may find more).
KNOWN_HANDOFFS = (
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
    'analysis/amplitude_challenge_20261114/TAKEOVER.txt',
    'analysis/branch_geometry_20261114/TAKEOVER.txt',
    'analysis/count_dependence_20261114/TAKEOVER.txt',
    'analysis/covariance_challenge_20261114/TAKEOVER.txt',
    'analysis/dispersion_challenge_20261114/TAKEOVER.txt',
    'analysis/empf_comparison_20261117/TAKEOVER.txt',
    'analysis/empirical_challenge_20261114/TAKEOVER.txt',
    'analysis/hybrid_challenge_20261114/TAKEOVER.txt',
    'analysis/improve_20261109/TAKEOVER_20261113.txt',
    'analysis/paired_challenge_20261114/TAKEOVER.txt',
    'analysis/particlegan_upgrade_20261117/TAKEOVER.txt',
)
HANDOFFS = REG.handoff_files(ROOT)


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


def test_glob_finds_every_known_handoff():
    assert set(KNOWN_HANDOFFS) <= set(HANDOFFS)
    assert all(p in REGISTRY['documents'] for p in HANDOFFS)


@pytest.mark.parametrize('name', sorted(set(HANDOFFS) | set(REG.EXTRA_SUPERSEDED)))
def test_historical_handoffs_point_to_status(name):
    """Each handoff is bannered or is the single CURRENT document in the registry."""
    entry = REGISTRY['documents'][name]
    if entry['state'] == 'CURRENT':
        return
    first = (ROOT/name).read_text(encoding='utf-8').splitlines()[0]
    assert first.startswith(BANNER)
    # The banner must say that CURRENT/AUTHORITATIVE/ACTIVE text below is history.
    assert REG.BANNER_HISTORICAL in first and 'out of date' in first
    assert entry == {'state': 'SUPERSEDED', 'by': 'STATUS.md', 'date': '2026-11-20'}


def test_exactly_one_document_is_current():
    current = [p for p, d in REGISTRY['documents'].items() if d['state'] == 'CURRENT']
    assert current == ['STATUS.md'] == [REGISTRY['current']]
    assert not REG.check_doc_registry(REGISTRY, ROOT)


def test_registry_is_fresh():
    assert (ROOT/REG.REGISTRY).read_text(encoding='utf-8') == REG.render(REG.build_doc_registry(ROOT))


def _fixture(tmp_path):
    (tmp_path/'analysis/study').mkdir(parents=True)
    (tmp_path/'STATUS.md').write_text('# Project status\n')
    (tmp_path/'CONTINUATION.txt').write_text(REG.BANNER_TEXT + '\n\nCURRENT AUTHORITATIVE: old\n')
    return tmp_path


def test_unbannered_takeover_fixture_fails(tmp_path):
    root = _fixture(tmp_path)
    registry = REG.build_doc_registry(root)
    assert not REG.check_doc_registry(registry, root)
    (root/'analysis/study/TAKEOVER.txt').write_text('CURRENT AUTHORITATIVE: do this next\n')
    with pytest.raises(ValueError, match='TAKEOVER.txt'):
        REG.build_doc_registry(root)
    assert any('TAKEOVER.txt' in e for e in REG.check_doc_registry(registry, root))
    # A banner that does not mark the text below as historical is not enough.
    (root/'analysis/study/TAKEOVER.txt').write_text('SUPERSEDED by STATUS.md (2026-11-20).\nx\n')
    with pytest.raises(ValueError):
        REG.build_doc_registry(root)
    (root/'analysis/study/TAKEOVER.txt').write_text(REG.BANNER_TEXT + '\nx\n')
    assert REG.build_doc_registry(root)['documents']['analysis/study/TAKEOVER.txt']['state'] == 'SUPERSEDED'


def test_handoff_name_variants_are_globbed(tmp_path):
    root = _fixture(tmp_path)
    for rel in ('analysis/X_HANDOFF.md', 'analysis/study/TAKEOVER_20261201.txt',
                'analysis/LIVE_CONTINUATION_V2.txt'):
        (root/rel).write_text('live\n')
    found = REG.handoff_files(root)
    assert {'analysis/X_HANDOFF.md', 'analysis/study/TAKEOVER_20261201.txt',
            'analysis/LIVE_CONTINUATION_V2.txt', 'CONTINUATION.txt'} == set(found)


def test_second_current_document_fails():
    registry = json.loads(json.dumps(REGISTRY))
    registry['documents']['CONTINUATION.txt'] = {'state': 'CURRENT'}
    assert any('exactly one CURRENT' in e for e in REG.check_doc_registry(registry, ROOT))


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
