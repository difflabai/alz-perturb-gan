"""Training-only selection and immutable common-library/cell-partition contracts."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os
import torch
from .gene_features import json_hash


@dataclass(frozen=True)
class StudyProtocol:
    library: tuple[str, ...]
    gene_ids: tuple[str, ...]
    fit: tuple[str, ...]
    inner: tuple[str, ...]
    exploratory: tuple[str, ...]
    final: tuple[str, ...] = ()
    calibration: tuple[str, ...] = ()  # held-out FIT identities; never trained on (plan v4 §1)
    split_seed: int = 1729
    evaluation_seed: int = 45
    negative_seed: int = 46
    draws: int = 64
    selection_split: str = 'inner'
    data_kind: str = 'real'

    def __post_init__(self):
        if len(set(self.library)) != len(self.library) or len(self.gene_ids) != len(self.library) or len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError('common library and stable IDs must be unique and aligned')
        groups = (self.fit, self.calibration, self.inner, self.exploratory, self.final)
        joined = [n for g in groups for n in g]
        if len(joined) != len(set(joined)) or set(joined) != set(self.library) or not self.fit or not self.inner:
            raise ValueError('FIT/calibration/inner/exploratory/final must partition the fixed library')
        if self.selection_split != 'inner' or self.draws < 1:
            raise ValueError('only training-only inner validation may select sources/arms')

    @property
    def fingerprint(self):
        d = asdict(self)
        if not d['calibration']: d.pop('calibration')  # protocols without a calibration group keep their historical hash
        return json_hash(d)

    def groups(self):
        return {k: getattr(self, k) for k in ('fit', 'inner', 'exploratory', 'final') if getattr(self, k)}

    @classmethod
    def from_data(cls, data, gene_ids, *, inner_fraction=.2, split_seed=1729, **kwargs):
        train = [p.name for p in data.named('train')]
        if not 0 < inner_fraction < 1 or len(train) < 3:
            raise ValueError('need at least three original training identities and an inner split')
        perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(split_seed))
        count = min(len(train) - 2, max(1, round(len(train) * inner_fraction)))
        inner_set = {train[i] for i in perm[:count].tolist()}
        return cls(tuple(p.name for p in data.named()), tuple(gene_ids), tuple(n for n in train if n not in inner_set),
                   tuple(n for n in train if n in inner_set), tuple(p.name for p in data.named('eval')),
                   split_seed=split_seed, **kwargs)


@dataclass
class StudyData:
    protocol: StudyProtocol
    control: torch.Tensor
    cells: dict[str, torch.Tensor]
    cell_ids: dict[str, tuple[str, ...]]
    source_provenance: dict
    candidate_cells: dict[str, torch.Tensor]
    candidate_ids: dict[str, tuple[str, ...]]
    query_profiles: dict[str, torch.Tensor]
    query_ids: dict[str, tuple[str, ...]]

    @classmethod
    def partition(cls, protocol, control, cells, cell_ids, *, seed=43, source_provenance=None):
        if set(cells) != set(protocol.library) or set(cell_ids) != set(protocol.library):
            raise ValueError('all library cells and stable cell IDs are required; no silent exclusion')
        candidates, candidate_ids, targets, query_ids = {}, {}, {}, {}
        all_ids = []
        generator = torch.Generator().manual_seed(seed)
        if control.ndim != 1 or not torch.isfinite(control).all():
            raise ValueError('control must be a finite expression vector')
        width = len(control)
        for name in protocol.library:
            x = cells[name].detach().float().cpu()
            ids = tuple(cell_ids[name])
            if x.ndim != 2 or x.shape[1] != width or len(x) != len(ids) or len(x) < 4 or not torch.isfinite(x).all():
                raise ValueError('each identity needs at least four finite cells aligned to unique IDs')
            if len(set(ids)) != len(ids):
                raise ValueError('duplicate cell IDs')
            all_ids.extend(ids)
            order = torch.randperm(len(x), generator=generator)
            cut = len(x) // 2
            left, right = order[:cut], order[cut:]
            candidates[name] = x[left].clone()
            candidate_ids[name] = tuple(ids[i] for i in left.tolist())
            targets[name] = x[right].mean(0)
            query_ids[name] = tuple(ids[i] for i in right.tolist())
        if len(all_ids) != len(set(all_ids)):
            raise ValueError('cell IDs must be globally unique across identities')
        return cls(protocol, control.detach().float().cpu(), cells, cell_ids, source_provenance or {},
                   candidates, candidate_ids, targets, query_ids)

    def training_tensors(self, exclude=()):
        # FIT only; protocol.calibration is disjoint from FIT by construction (plan v4 §1). `exclude` is a
        # legacy guard for 96-FIT protocols and must not be needed for v2 bundles.
        names = tuple(n for n in self.protocol.fit if n not in set(exclude))
        if set(names) & set(self.protocol.calibration): raise ValueError('calibration identity in training tensors')
        x = torch.cat([self.cells[n].detach().float().cpu() for n in names])
        ids = torch.cat([torch.full((len(self.cells[n]),), self.protocol.library.index(n), dtype=torch.long) for n in names])
        # Query target means and real candidate pools are disjoint within each FIT identity.
        real = torch.cat([self.candidate_cells[n] for n in names])
        counts = torch.tensor([len(self.candidate_cells[n]) for n in names])
        return {'outcomes': x, 'ids': ids, 'real_candidates': real, 'counts': counts,
                'starts': torch.cumsum(counts, 0) - counts,
                'fit_ids': torch.tensor([self.protocol.library.index(n) for n in names]),
                'targets': torch.stack([self.query_profiles[n] for n in names])}

    def selection_view(self, split):
        if split != 'inner':
            raise ValueError('exploratory/final outcomes are not source-selection inputs')
        return {n: self.query_profiles[n] for n in self.protocol.inner}


def validate_grant(grant, study, config, device, phase, max_wall_seconds):
    if max_wall_seconds <= 0: raise ValueError('positive wall budget required')
    if grant.get('approved') is not True or not grant.get('authorization_reference') or phase not in grant.get('phases',[]):
        raise PermissionError('explicit parent cohort/phase approval required')
    if grant.get('protocol_sha256')!=study.protocol.fingerprint or config.arm not in grant.get('arms',[]) or config.seed not in grant.get('seeds',[]):
        raise PermissionError('cohort protocol/arm/seed is outside the grant')
    if config.steps>grant.get('max_steps_per_arm',0) or max_wall_seconds>grant.get('max_wall_seconds_per_arm',0):
        raise PermissionError('fixed resource budget exceeded')
    if config.trainer != 'ka2' or 'particle_trainer' in grant:
        from .particle_integration import PARTICLEGAN_COMMIT
        if grant.get('particle_trainer') != config.trainer or grant.get('particlegan_commit') != PARTICLEGAN_COMMIT:
            raise PermissionError('ParticleGAN trainer/source is outside the grant')
    if device!='cpu':
        required=('gpu_uuid','capacity_receipt_sha256','ownership_reference','lease_receipt','capacity_valid_until')
        if not all(grant.get(k) for k in required):raise PermissionError('GPU capacity, ownership and shared lease evidence required')
        if device!='cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES')!=grant['gpu_uuid']:
            raise PermissionError('GPU UUID must be selected explicitly before initialization')
        expiry = datetime.fromisoformat(grant['capacity_valid_until'])
        if expiry.tzinfo is None: raise PermissionError('capacity expiry must include timezone')
        if datetime.now(timezone.utc)>=expiry:raise PermissionError('capacity confirmation expired')
    return grant
