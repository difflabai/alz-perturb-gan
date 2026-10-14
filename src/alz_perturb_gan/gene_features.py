"""Frozen, hash-bound features; projections belong inside model forward."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _tau_supplements(packet_path, packet_hash, entry, symbols, stable, zeros,
                     mapping_path, mapping_hash, license_path, license_hash):
    expected = {'manifest_sha256':packet_hash, 'source_key':'string_spectral', 'matrix_sha256':entry['sha256']}
    checked = {}
    for label, asset, digest, kind in (
            ('mapping',mapping_path,mapping_hash,'tau_string_row_mapping_addon'),
            ('license',license_path,license_hash,'tau_string_license_receipt_addon')):
        if not asset or not digest: raise ValueError('hash-bound STRING mapping and license supplements required')
        asset = Path(asset).resolve()
        if not asset.is_relative_to(packet_path.parent) or file_sha256(asset) != digest:
            raise ValueError('STRING supplement path/hash mismatch')
        body = json.loads(asset.read_text())
        if body.get('schema_version') != 1 or body.get('status') != 'frozen' or body.get('kind') != kind:
            raise ValueError('STRING supplement is not the frozen supported contract')
        if any(body.get('applies_to',{}).get(k) != v for k,v in expected.items()):
            raise ValueError('STRING supplement belongs to another packet/matrix')
        checked[label] = {'path':str(asset), 'sha256':digest, 'content':body}
    mapping, license = checked['mapping']['content'], checked['license']['content']
    if mapping.get('alias_input_sha256') != entry['fit']['id_map_sha256'] or mapping.get('links_input_sha256') != entry['fit']['input_sha256'] or not mapping.get('rule'):
        raise ValueError('STRING mapping input provenance differs')
    rows = mapping.get('rows', [])
    if len(rows) != len(symbols) or [(r.get('symbol'),r.get('gene_id')) for r in rows] != list(zip(symbols,stable)):
        raise ValueError('STRING per-protein mapping identity/order differs')
    for row in rows:
        missing = row['symbol'] in zeros
        if row.get('missing') is not missing or row.get('feature_row_zero') is not missing:
            raise ValueError('STRING mapping missing status differs from matrix bytes')
        pid = row.get('string_protein_id'); edges = row.get('edges_ge_700')
        if not missing and (not pid or not str(pid).startswith('9606.') or not isinstance(edges,int) or edges < 1):
            raise ValueError('usable STRING row lacks a human protein/graph edge')
        if missing and (not row.get('missing_reason') or edges not in (0,None)):
            raise ValueError('missing STRING vector lacks an exclusion reason')
    required = ('asset_class','license_url','license_revision','license_text_sha256','source_license_statement_url',
                'source_license_statement_page_sha256','source_license_statement_fetched_utc','citation',
                'attribution_and_modification_notice','code_license')
    if license.get('license_id') != 'CC-BY-4.0' or license.get('local_research_reviewed') is not True or not all(license.get(k) for k in required):
        raise ValueError('STRING asset-specific reviewed license receipt is incomplete')
    source = license.get('source', {})
    if source.get('name') != 'STRING' or source.get('revision') != 'v12.0' or source.get('url') != entry.get('url'):
        raise ValueError('STRING license source revision differs')
    if not str(license['asset_class']).startswith('public annotation/network data'):
        raise ValueError('STRING license receipt describes a different asset class')
    for name, key in (('cc_by_4.0_legalcode.txt','license_text_sha256'),('string_access_page.html','source_license_statement_page_sha256')):
        asset = Path(checked['license']['path']).parent / name
        if file_sha256(asset) != license[key]: raise ValueError('saved STRING primary license bytes differ')
        checked['license'].setdefault('primary_assets',{})[name] = {'path':str(asset), 'sha256':license[key]}
    for label in checked:
        if file_sha256(checked[label]['path']) != checked[label]['sha256']:
            raise ValueError('STRING supplement changed during reading')
    return checked


@dataclass
class FeatureTable:
    names: tuple[str, ...]
    gene_ids: tuple[str, ...]
    values: torch.Tensor
    missing: torch.Tensor
    fit_names: tuple[str, ...]
    mean: torch.Tensor
    scale: torch.Tensor
    provenance: dict
    verified: bool = False

    @classmethod
    def from_array(cls, names, gene_ids, values, missing, fit_names, provenance, *, verified=False):
        names, ids, fit = tuple(names), tuple(gene_ids), tuple(fit_names)
        raw = torch.as_tensor(np.array(values, copy=True)).detach().float()
        mask = torch.as_tensor(missing, dtype=torch.bool)
        if len(set(names)) != len(names) or len(set(ids)) != len(ids) or len(ids) != len(names):
            raise ValueError('library names and stable gene IDs must be aligned and unique')
        if raw.ndim != 2 or raw.shape[0] != len(names) or not raw.shape[1] or mask.shape != (len(names),):
            raise ValueError('feature dimensions disagree with the common library')
        if not bool(torch.isfinite(raw).all()) or bool((raw[mask] != 0).any()):
            raise ValueError('features must be finite; missing rows must be explicit zeros')
        if not fit or len(set(fit)) != len(fit) or not set(fit) <= set(names):
            raise ValueError('feature standardization needs unique FIT identities')
        selected = torch.tensor([names.index(n) for n in fit])
        selected = selected[~mask[selected]]
        if not len(selected):
            raise ValueError('no observed FIT feature rows')
        mean, scale = raw[selected].mean(0), raw[selected].std(0, unbiased=False).clamp_min(1e-6)
        normalized = (raw - mean) / scale
        normalized[mask] = 0
        normalized = torch.cat([normalized, mask.float().unsqueeze(1)], 1)
        info = dict(provenance, library=list(names), gene_ids=list(ids), normalization_fit_names=list(fit),
                    missing_names=[n for n, m in zip(names, mask.tolist()) if m])
        return cls(names, ids, normalized, mask, fit, mean, scale, info, verified)

    @classmethod
    def load(cls, manifest_path, expected_manifest_sha256, *, names, gene_ids, fit_names):
        path = Path(manifest_path).resolve()
        if file_sha256(path) != expected_manifest_sha256:
            raise ValueError('feature manifest hash is not the cohort-frozen hash')
        manifest = json.loads(path.read_text())
        if manifest.get('schema_version') != 1 or manifest.get('status') != 'frozen':
            raise ValueError('draft or unsupported feature manifest')
        if manifest.get('coverage_kind') != 'usable_vectors' or manifest.get('species') != 9606:
            raise ValueError('identifier resolution is not usable human-vector coverage')
        source = manifest.get('source', {})
        if not all(source.get(k) for k in ('name', 'revision', 'url')):
            raise ValueError('source name, revision and primary URL are required')
        lic = manifest.get('license', {})
        required = ('asset_class', 'license_id', 'license_url', 'license_revision', 'license_sha256', 'citation', 'code_license')
        if not all(lic.get(k) for k in required) or lic.get('local_research_reviewed') is not True:
            raise ValueError('asset-class-specific local research license review is missing')
        if lic['asset_class'] not in ('supplied_table', 'model_weights', 'generated_output', 'annotation_data'):
            raise ValueError('unknown source/weight/output asset class')
        transform = manifest.get('transform', {})
        if transform.get('uses_expression_outcomes') is not False:
            raise ValueError('outcome-dependent feature preparation cannot enter this comparison')
        if transform.get('fit_scope') == 'fit_only':
            if set(transform.get('fit_names', [])) != set(fit_names):
                raise ValueError('feature transform was not fitted on this exact inner-FIT split')
        elif transform.get('fit_scope') != 'external_frozen':
            raise ValueError('local all-library transform is not FIT-only; declare a separate protocol')
        loaded = {}
        for kind in ('matrix', 'order'):
            item = manifest[kind]
            asset = (path.parent / item['path']).resolve()
            if not asset.is_relative_to(path.parent):
                raise ValueError('feature assets must belong to their manifest directory')
            if file_sha256(asset) != item['sha256']:
                raise ValueError(f'{kind} hash mismatch')
            loaded[kind] = asset
        order = json.loads(loaded['order'].read_text())
        if order['names'] != list(names) or order['gene_ids'] != list(gene_ids):
            raise ValueError('feature order or stable IDs differ from the common library')
        missing = order['missing']
        source_ids = order['source_ids']
        if len(missing) != len(names) or len(source_ids) != len(names):
            raise ValueError('coverage mappings are incomplete')
        if any((sid is None) != bool(miss) for sid, miss in zip(source_ids, missing)):
            raise ValueError('source-ID and missing-vector mappings disagree')
        matrix = np.load(loaded['matrix'], allow_pickle=False)
        if matrix.dtype not in (np.float32, np.float64) or matrix.ndim != 2 or matrix.shape[1] != manifest['dimension']:
            raise ValueError('matrix dtype/dimension differs from the verified manifest')
        provenance = {'manifest': manifest, 'manifest_path': str(path), 'manifest_sha256': expected_manifest_sha256,
                      'source_ids': source_ids, 'missing_policy': 'zero_plus_indicator'}
        table = cls.from_array(names, gene_ids, matrix, missing, fit_names, provenance, verified=True)
        # Detect a preparation process changing assets while they are read.
        for kind in ('matrix', 'order'):
            if file_sha256(loaded[kind]) != manifest[kind]['sha256']:
                raise ValueError('feature assets changed during loading')
        if file_sha256(path) != expected_manifest_sha256:
            raise ValueError('feature manifest changed during loading')
        return table

    @classmethod
    def load_tau_packet(cls, manifest_path, expected_manifest_sha256, *, source_key, names, gene_ids, fit_names,
                        mapping_path=None, expected_mapping_sha256=None, license_path=None, expected_license_sha256=None):
        """Read Tau's immutable aggregate packet; never rewrite/re-extract assets.

        Initial admissible source is STRING's outcome-free full-human graph.
        GO's unpinned cohort-only TF-IDF/SVD remains separate, not a STRING gate.
        This checks matrix/namespace provenance, not blanket redistribution rights.
        """
        path = Path(manifest_path).resolve()
        if file_sha256(path) != expected_manifest_sha256: raise ValueError('Tau packet hash changed')
        packet = json.loads(path.read_text())
        if source_key != 'string_spectral':
            raise ValueError('initial Tau adapter admits STRING only; other sources need their reviewed contract')
        order = packet['gene_order']; symbols, stable = order['symbols'], order['ensembl']
        if len(symbols) != len(set(symbols)) or len(stable) != len(set(stable)):
            raise ValueError('Tau namespace is not unique')
        lookup = dict(zip(symbols, stable))
        if set(symbols) != set(names) or any(lookup[n] != g for n,g in zip(names,gene_ids)):
            raise ValueError('Tau stable gene identities differ from the dataset')
        entry = packet['sources'][source_key]; fit = entry.get('fit', {})
        if entry.get('version') != 'STRING v12.0' or entry.get('license_model_or_data') != 'CC BY 4.0':
            raise ValueError('unreviewed STRING revision/license')
        if not str(fit.get('gene_universe', '')).startswith('all ') or fit.get('score_threshold') != 'combined_score >= 700':
            raise ValueError('STRING must use the declared outcome-free full-human graph')
        if not fit.get('input_sha256') or not fit.get('id_map_sha256'):
            raise ValueError('STRING graph and stable-ID alias inputs are not hash-bound')
        matrix_path = (path.parent / entry['file']).resolve()
        if not matrix_path.is_relative_to(path.parent) or file_sha256(matrix_path) != entry['sha256']:
            raise ValueError('STRING matrix hash/path mismatch')
        raw = np.load(matrix_path, allow_pickle=False)
        if raw.dtype != np.float32 or list(raw.shape) != entry['shape'] or raw.shape != (len(symbols),entry['dim']) or not np.isfinite(raw).all():
            raise ValueError('STRING matrix shape/dtype/finite contract failed')
        zeros = {symbols[i] for i in range(len(symbols)) if not np.any(raw[i])}
        if zeros != set(entry['all_zero_rows']) or zeros != set(entry['missing']):
            raise ValueError('STRING missing-vector evidence disagrees with bytes')
        supplements = _tau_supplements(path,expected_manifest_sha256,entry,symbols,stable,zeros,
            mapping_path,expected_mapping_sha256,license_path,expected_license_sha256)
        permutation = [symbols.index(n) for n in names]
        info = {'manifest_path':str(path), 'manifest_sha256':expected_manifest_sha256, 'source_key':source_key,
                'source':entry, 'source_order':symbols, 'source_row_permutation':permutation,
                'missing_policy':'zero_plus_indicator', 'asset_class':'annotation_data',
                'transform_scope':'external_graph_outcome_free',
                'license_scope':'CC-BY-4.0 source data; local research; preserve attribution; no blanket redistribution clearance',
                'supplements':supplements, 'gene_order_sha256':json_hash(order),
                'source_ids':[supplements['mapping']['content']['rows'][i]['string_protein_id'] for i in permutation],
                'mapping_checked':'per-row stable gene/protein identity, source order, graph membership and missing reasons verified; upstream graph bytes not rehashed'}
        table = cls.from_array(names, gene_ids, raw[permutation], [n in zeros for n in names], fit_names, info, verified=True)
        if file_sha256(matrix_path) != entry['sha256'] or file_sha256(path) != expected_manifest_sha256:
            raise ValueError('Tau artifacts changed while being read')
        return table

    def shuffled(self, seed):
        permutation = torch.arange(len(self.names))
        rng = torch.Generator().manual_seed(seed)
        for status in (False, True):
            rows = torch.nonzero(self.missing == status).flatten()
            permutation[rows] = rows[torch.randperm(len(rows), generator=rng)]
        info = dict(self.provenance, shuffle_seed=seed, row_permutation=permutation.tolist())
        # Standardization is kept fixed; missing-mask strata preserve missing behavior.
        return FeatureTable(self.names, self.gene_ids, self.values[permutation].clone(), self.missing.clone(),
                            self.fit_names, self.mean.clone(), self.scale.clone(), info, self.verified)


class FeatureProjector(nn.Module):
    def __init__(self, table, embedding_dim):
        super().__init__()
        self.names, self.gene_ids, self.fit_names = table.names, table.gene_ids, table.fit_names
        self.register_buffer('raw', table.values.clone())
        self.projection = nn.Linear(table.values.shape[1], embedding_dim)

    def forward(self, candidate_ids):
        if candidate_ids.dtype != torch.long:
            raise ValueError('candidate IDs must be integer library indices')
        return self.projection(self.raw[candidate_ids])
