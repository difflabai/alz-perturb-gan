"""Reject changed data and incompatible models before native inference."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('microglia_runtime', ROOT/'scripts/microglia_runtime.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_cache_checks_payloads_and_keeps_training_independent_of_heldout(tmp_path, monkeypatch):
    names = ('space.npz', 'train/counts.npz', 'train/metadata.npz',
             'heldout/counts.npz', 'heldout/metadata.npz', 'type_classifier.pt')
    (tmp_path/'manifest.json').write_text('{"inner_fold": 0}')
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    anchor = {'manifest_sha256': R.sha(tmp_path/'manifest.json'),
              'payload_sha256': {name: R.sha(tmp_path/name) for name in names}}
    pins = tmp_path/'anchors.json'
    pins.write_text(json.dumps({'caches': {'0': anchor}}))
    monkeypatch.setattr(R, 'CACHE_ANCHORS', pins)
    R.check_cache(tmp_path, 0, held=True)
    (tmp_path/'heldout/counts.npz').unlink()
    R.check_cache(tmp_path, 0)
    with pytest.raises((ValueError, FileNotFoundError)):
        R.check_cache(tmp_path, 0, held=True)
    (tmp_path/'train/counts.npz').write_bytes(b'changed under the same manifest')
    with pytest.raises(ValueError, match='cache payload'):
        R.check_cache(tmp_path, 0)


def checkpoint_header():
    saved = json.loads((ROOT/'analysis/empf_comparison_20261117/completion/fits/native/'
                       'pbulk/hvg2000/fold0/seed0/config.json').read_text())
    q = {key: saved[key] for key in ('arm', 'flags', 'panel', 'fold', 'seed',
         'cache_manifest_SHA256', 'recipe', 'particlegan_version', 'native_recipe')}
    q.update(status='completed', completed_updates=20000, fixed_endpoint=20000,
             released_hyperparameters=saved['hyperparameters'],
             penalty_options={'input_domain': 'continuous'},
             policy={'recipe': copy.deepcopy(saved['recipe']), 'completed_steps': 20000,
                     'optimizers': [{}, {'regularizer': {'record': {
                         'calls': 20000, 'observed_steps': 20000,
                         'empf': {'schema': 2, 'initialized': True, 'proto_initialized': False,
                                  'proto_sum': None, 'proto_count': None}}}}]})
    return q, saved


@pytest.mark.parametrize('damage', ['version', 'recipe', 'policy_recipe', 'regularizer', 'fold'])
def test_binary_header_cannot_disagree_with_valid_metadata(damage):
    q, saved = checkpoint_header()
    R.check_checkpoint(q, saved)
    q = copy.deepcopy(q)
    if damage == 'version': q['particlegan_version'] = '0.9.7'
    elif damage == 'recipe': q['recipe']['lr'] = .014
    elif damage == 'policy_recipe': q['policy']['recipe']['reg_every'] = 2
    elif damage == 'regularizer':
        q['policy']['optimizers'][1]['regularizer']['record']['calls'] = 10000
    else: q['fold'] = 1
    with pytest.raises(ValueError):
        R.check_checkpoint(q, saved)


def test_refreezing_sources_mid_run_does_not_relabel_loaded_code(tmp_path, monkeypatch):
    source = tmp_path/'runtime.py'
    source.write_text('old loaded code')
    manifest = tmp_path/'manifest.json'
    archive = tmp_path/'previous.py'
    archive.write_text('previous entrypoint')
    monkeypatch.setattr(R, 'ROOT', tmp_path)
    monkeypatch.setattr(R, 'SOURCE_MANIFEST', manifest)
    def freeze():
        manifest.write_text(json.dumps({'sources': {'runtime.py': R.sha(source)},
            'archived_previous_entrypoint': {'archive': 'previous.py', 'sha256': R.sha(archive)}}))
    freeze()
    started, digest = R.check_sources(), R.sha(manifest)
    R.check_unchanged_sources(started, digest)
    source.write_text('new code')
    freeze()
    R.check_sources()
    with pytest.raises(ValueError, match='during execution'):
        R.check_unchanged_sources(started, digest)
