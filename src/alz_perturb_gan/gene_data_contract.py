"""Metadata-only identity checks for a frozen real study bundle; never read X."""
from pathlib import Path
import numpy as np
import h5py
from .gene_features import file_sha256, json_hash


def _strings(group, name):
    node = group[name]
    if isinstance(node, h5py.Dataset):
        return node.asstr()[()].tolist()
    categories = node['categories'].asstr()[()].tolist()
    codes = node['codes'][()]
    if np.any(codes < 0): raise ValueError('source identity metadata is missing')
    return [categories[int(i)] for i in codes]


def validate_real_bundle_identity(protocol, control, cell_ids, provenance):
    """Bind declared tensor axes/identities to existing raw metadata and split.

    The immutable bundle and builder's full source SHA bind expression values.
    This check deliberately reads no raw expression and does not rehash 15 GB.
    """
    if provenance.get('protocol_sha256') != protocol.fingerprint:
        raise ValueError('bundle provenance protocol hash differs')
    gene_ids = provenance.get('expression_gene_ids', [])
    symbols = provenance.get('expression_gene_symbols', [])
    if len(gene_ids) != len(control) or len(symbols) != len(control):
        raise ValueError('expression axis metadata differs from tensor width')
    source = provenance.get('source', {})
    if len(source.get('sha256', '')) != 64 or not source.get('path'):
        raise ValueError('hash-bound raw source identity is missing')
    split = provenance.get('split_source', {})
    if not split.get('path') or not split.get('sha256') or file_sha256(split['path']) != split['sha256']:
        raise ValueError('original identity split hash differs')
    import json
    original = json.loads(Path(split['path']).read_text())
    train, exploratory = original['train_names'], original['eval_names']
    if tuple(train + exploratory) != protocol.library or tuple(exploratory) != protocol.exploratory or set(protocol.fit + protocol.calibration + protocol.inner) != set(train):
        raise ValueError('library order/identity groups differ from original split')
    source_path = Path(source['path'])
    with h5py.File(source_path, 'r') as raw:
        raw_symbols = _strings(raw['var'], raw['var'].attrs.get('_index', '_index'))
        raw_gene_ids = _strings(raw['var'], 'gene_id')
        if symbols != raw_symbols or gene_ids != raw_gene_ids:
            raise ValueError('expression gene order differs from raw source metadata')
        namespace = dict(zip(raw_symbols, raw_gene_ids))
        if any(namespace.get(n) != g for n,g in zip(protocol.library, protocol.gene_ids)):
            raise ValueError('perturbation stable IDs differ from raw source namespace')
        raw_ids = _strings(raw['obs'], raw['obs'].attrs.get('_index', '_index'))
        labels = _strings(raw['obs'], 'target_gene')
    if len(set(raw_ids)) != len(raw_ids): raise ValueError('raw source cell IDs are not unique')
    lookup = dict(zip(raw_ids, labels))
    for name in protocol.library:
        if any(lookup.get(cell) != name for cell in cell_ids[name]):
            raise ValueError('source cell ID belongs to a different perturbation')
    control_ids = provenance.get('control', {}).get('cell_ids', [])
    control_labels = {'non-targeting', 'non-targeting control', 'control'}
    if not control_ids or len(control_ids) != len(set(control_ids)) or any(str(lookup.get(c, '')).lower() not in control_labels for c in control_ids):
        raise ValueError('control source IDs do not identify non-targeting cells')
    if set(control_ids) & {c for ids in cell_ids.values() for c in ids}:
        raise ValueError('control and perturbation source cells overlap')
    return {'raw_metadata_path':str(source_path.resolve()), 'source_sha256_declared':source['sha256'],
            'source_expression_rehashed':False, 'expression_gene_ids_sha256':json_hash(gene_ids),
            'expression_gene_symbols_sha256':json_hash(symbols), 'original_split_sha256':split['sha256'],
            'protocol_sha256':protocol.fingerprint, 'selected_source_cell_count':sum(map(len,cell_ids.values())),
            'control_source_cell_count':len(control_ids), 'raw_source_metadata_verified':True}
