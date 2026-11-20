"""The checkout matches the pinned runtime manifest, and the re-pin tool keeps every key."""
import datetime
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load('microglia_runtime_manifest', ROOT/'scripts/microglia_runtime.py')
F = load('refreeze_runtime_manifest', ROOT/'scripts/refreeze_runtime_manifest.py')


def test_checkout_matches_the_runtime_source_manifest():
    manifest = json.loads(R.SOURCE_MANIFEST.read_text())
    absent = [n for n in manifest['sources'] if not (Path(n) if Path(n).is_absolute() else ROOT/n).exists()]
    if absent:
        pytest.skip(f'{len(absent)} pinned sources are local artifacts absent here (e.g. {absent[0]})')
    R.check_sources()


def test_tracked_pinned_sources_match_the_manifest():
    """Runs in any checkout: every pinned file that git tracks must match its pin."""
    manifest = json.loads(R.SOURCE_MANIFEST.read_text())
    ok, missing, changed, borrowed, entry_ok = F.scan(manifest, ROOT)
    assert not changed, f'pinned sources changed without a re-pin: {sorted(changed)}'
    assert entry_ok
    assert all(n.startswith('artifacts/') for n in missing), missing


def make_tree(tmp_path):
    root = tmp_path/'repo'
    (root/'pkg').mkdir(parents=True)
    (root/'prev').mkdir()
    (root/'pkg/a.py').write_text('a = 1\n')
    (root/'pkg/b.py').write_text('b = 1\n')
    (root/'prev/old.py').write_text('old entrypoint\n')
    (root/'m').mkdir()
    manifest = root/'m/manifest.json'
    manifest.write_text(json.dumps({'schema': 1, 'ancestry_manifest_sha256': 'x'*64,
        'sources': {'pkg/a.py': F.sha(root/'pkg/a.py'), 'pkg/b.py': F.sha(root/'pkg/b.py')},
        'archived_previous_entrypoint': {'archive': 'prev/old.py', 'sha256': F.sha(root/'prev/old.py')}}, indent=2))
    return root, manifest


NOW = datetime.datetime(2026, 11, 20, 12, tzinfo=datetime.timezone.utc)


def test_refreeze_repins_changed_sources_and_archives_the_previous_manifest(tmp_path):
    root, manifest = make_tree(tmp_path)
    before = manifest.read_bytes()
    assert F.refreeze(manifest, root, 'no change', now=NOW) is None
    assert manifest.read_bytes() == before
    (root/'pkg/a.py').write_text('a = 2\n')
    result = F.refreeze(manifest, root, 'harden runtime', now=NOW)
    assert result['changed'] == ['pkg/a.py']
    new = json.loads(manifest.read_text())
    assert set(new['sources']) == {'pkg/a.py', 'pkg/b.py'}
    assert new['sources']['pkg/a.py'] == F.sha(root/'pkg/a.py')
    assert new['ancestry_manifest_sha256'] == 'x'*64
    prev = new['previous_runtime_manifest']
    assert prev['reason'] == 'harden runtime' and prev['changed'] == ['pkg/a.py']
    assert (root/prev['archive']).read_bytes() == before and prev['sha256'] == F.hashlib.sha256(before).hexdigest()


def test_refreeze_refuses_to_drop_a_missing_key(tmp_path):
    root, manifest = make_tree(tmp_path)
    before = manifest.read_bytes()
    (root/'pkg/b.py').unlink()
    with pytest.raises(ValueError, match='refusing to drop'):
        F.refreeze(manifest, root, 'drop b', now=NOW)
    assert manifest.read_bytes() == before
    assert not (root/'m/previous').exists()


def test_artifact_root_can_confirm_but_never_repin_artifacts(tmp_path):
    root, manifest = make_tree(tmp_path)
    other = tmp_path/'other'
    (other/'pkg').mkdir(parents=True)
    (other/'pkg/b.py').write_bytes((root/'pkg/b.py').read_bytes())
    (root/'pkg/b.py').unlink()
    (root/'pkg/a.py').write_text('a = 3\n')
    assert F.refreeze(manifest, root, 'with artifacts', artifact_root=other, now=NOW)['changed'] == ['pkg/a.py']
    (other/'pkg/b.py').write_text('tampered\n')
    with pytest.raises(ValueError, match='artifact-root'):
        F.refreeze(manifest, root, 'again', artifact_root=other, now=NOW + datetime.timedelta(hours=1))


def test_add_pins_a_new_path(tmp_path):
    root, manifest = make_tree(tmp_path)
    (root/'pkg/c.py').write_text('c\n')
    result = F.refreeze(manifest, root, 'add c', add=['pkg/c.py'], now=NOW)
    assert result['added'] == ['pkg/c.py']
    assert 'pkg/c.py' in json.loads(manifest.read_text())['sources']


def test_relocate_renames_a_key_only_for_a_byte_identical_file(tmp_path):
    root, manifest = make_tree(tmp_path)
    outside = tmp_path/'outside/ext.py'
    outside.parent.mkdir()
    outside.write_text('external = 1\n')
    pinned = json.loads(manifest.read_text())
    pinned['sources'][str(outside)] = F.sha(outside)
    manifest.write_text(json.dumps(pinned, indent=2))
    before = manifest.read_bytes()
    (root/'vendored').mkdir()
    (root/'vendored/ext.py').write_text('external = 2\n')  # not the pinned bytes
    with pytest.raises(ValueError, match='differs from the pinned digest'):
        F.refreeze(manifest, root, 'vendor', relocate=[(str(outside), 'vendored/ext.py')], now=NOW)
    with pytest.raises(ValueError, match='unpinned key'):
        F.refreeze(manifest, root, 'vendor', relocate=[('pkg/zzz.py', 'vendored/ext.py')], now=NOW)
    with pytest.raises(ValueError, match='repository-relative'):
        F.refreeze(manifest, root, 'vendor', relocate=[(str(outside), 'vendored/missing.py')], now=NOW)
    with pytest.raises(ValueError, match='already pinned'):
        F.refreeze(manifest, root, 'vendor', relocate=[(str(outside), 'pkg/a.py')], now=NOW)
    assert manifest.read_bytes() == before and not (root/'m/previous').exists()
    (root/'vendored/ext.py').write_bytes(outside.read_bytes())
    outside.unlink()  # the old absolute location may disappear once vendored
    result = F.refreeze(manifest, root, 'vendor', relocate=[(str(outside), 'vendored/ext.py')], now=NOW)
    assert result['relocated'] == {str(outside): 'vendored/ext.py'} and result['changed'] == []
    new = json.loads(manifest.read_text())
    assert list(new['sources']) == ['pkg/a.py', 'pkg/b.py', 'vendored/ext.py']
    assert new['sources']['vendored/ext.py'] == pinned['sources'][str(outside)]
    assert new['previous_runtime_manifest']['relocated'] == {str(outside): 'vendored/ext.py'}
    assert (root/new['previous_runtime_manifest']['archive']).read_bytes() == before
