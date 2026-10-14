"""Direct checkpoint construction and a true zero-optimizer-call evaluation guard."""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import hashlib
import torch
from .gene_features import FeatureTable, file_sha256, json_hash
from .gene_models import ModelConfig, build_models


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_hashes(models): return {name: state_hash(m) for name, m in models.modules().items()}


def _optimizer_classes():
    found, pending = set(), [torch.optim.Optimizer]
    while pending:
        cls = pending.pop()
        if cls in found: continue
        found.add(cls); pending.extend(cls.__subclasses__())
    return found


@contextmanager
def zero_update_guard():
    """Reject optimizer calls, even if their effects would later be overwritten."""
    calls = {'optimizer_step_calls': 0}
    def forbidden(*args, **kwargs):
        calls['optimizer_step_calls'] += 1
        raise RuntimeError('optimizer.step is forbidden in checkpoint evaluation')
    with ExitStack() as stack:
        for cls in _optimizer_classes():
            if 'step' in cls.__dict__: stack.enter_context(patch.object(cls, 'step', forbidden))
        yield calls
        if calls['optimizer_step_calls']:
            raise RuntimeError('an optimizer call was attempted during evaluation')


def save_checkpoint(path, models, protocol, table=None, *, training_receipt=None):
    if models.training_protocol_sha256 and models.training_protocol_sha256 != protocol.fingerprint:
        raise ValueError('cannot save fitted state under a different protocol')
    feature = None
    if table is not None:
        feature = {k: getattr(table, k) for k in ('names', 'gene_ids', 'values', 'missing', 'fit_names', 'mean', 'scale', 'provenance', 'verified')}
    payload = {'schema_version': 1, 'config': asdict(models.config), 'protocol': asdict(protocol), 'feature': feature,
               'state': {k: m.state_dict() for k, m in models.modules().items()},
               'feature_provenance': models.feature_provenance, 'training_receipt': training_receipt or {},
               'training_protocol_sha256':models.training_protocol_sha256}
    import io
    from .paths import create_new
    buf = io.BytesIO(); torch.save(payload, buf)
    path = create_new(Path(path).parent, Path(path).name, buf.getvalue())  # scoped dir, exclusive, no-follow
    return file_sha256(path)


def load_checkpoint(path, expected_sha256, *, device='cpu'):
    if file_sha256(path) != expected_sha256: raise ValueError('checkpoint hash mismatch')
    with zero_update_guard():
        blob = torch.load(path, map_location='cpu', weights_only=True)
        if blob.get('schema_version') == 1:
            cfg = ModelConfig(**blob['config'])
            feature = blob.get('feature')
            table = FeatureTable(**feature) if feature is not None else None
            kind = blob['protocol']['data_kind']
            models = build_models(cfg, table, data_kind=kind)
            for name, state in blob['state'].items(): getattr(models, name).load_state_dict(state, strict=True)
            models.feature_provenance = blob['feature_provenance']
            models.training_protocol_sha256 = blob.get('training_protocol_sha256')
            if models.training_protocol_sha256 and models.training_protocol_sha256 != json_hash(blob['protocol']):
                raise ValueError('checkpoint training protocol disagrees with stored split')
        else:
            # Existing run-2 checkpoint: plain modules, no trainer/optimizer construction.
            args = blob['args']; genes = blob['generator']['net.4.weight'].shape[0]
            particles, zdim = blob['prior']['z'].shape
            library = blob['generator']['net.0.weight'].shape[1] - genes - zdim
            cfg = ModelConfig(genes, library, hidden=args['hidden'], z_dim=zdim, particles=particles,
                              seed=args['seed'], steps=args['steps'], batch_size=args['batch_size'], lr=args['lr'])
            models = build_models(cfg)
            models.generator.backbone.load_state_dict(blob['generator'], strict=True)
            models.critic.load_state_dict(blob['critic'], strict=True)
            models.decision.backbone.load_state_dict(blob['decision'], strict=True)
            models.prior.load_state_dict(blob['prior'], strict=True)
            models.feature_provenance = {'source': 'legacy_onehot', 'checkpoint_args': args,
                                         'evaluation_only': True, 'training_receipt_missing': True}
        models.checkpoint_provenance = {'path':str(Path(path).resolve()), 'sha256':expected_sha256,
                                         'schema_version':blob.get('schema_version','legacy')}
        models.to(device)
    if file_sha256(path) != expected_sha256: raise ValueError('checkpoint changed during loading')
    return models
