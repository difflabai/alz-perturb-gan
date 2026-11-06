"""Exact native CFGen state restore, including unregistered covariate modules."""
from pathlib import Path
import fit_adapter as adapter
from state_fingerprint import fingerprint

def restore_checkpoint(checkpoint, device, expected_sha256):
    assert device in ('cpu', 'cuda:0')
    checkpoint = Path(checkpoint)
    assert adapter.sha(checkpoint) == expected_sha256
    saved = adapter.torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected = fingerprint(saved['state_dict'])
    feature_states = {k: v.state_dict() for k, v in saved['hyper_parameters']['feature_embeddings'].items()}
    expected_features = fingerprint(feature_states)
    model = adapter.restore(checkpoint, device)
    assert fingerprint(model.state_dict()) == expected
    assert fingerprint({k: v.state_dict() for k, v in model.feature_embeddings.items()}) == expected_features
    assert fingerprint(model.encoder_model.state_dict()) == fingerprint(saved['hyper_parameters']['encoder_model'].state_dict())
    assert not model.training and not any(m.training for m in model.encoder_model.modules())
    assert not any(p.requires_grad for p in model.encoder_model.parameters())
    for module in [model, *model.feature_embeddings.values()]:
        assert all(p.device.type == ('cpu' if device == 'cpu' else 'cuda') for p in module.parameters())
        assert all(p.device.type == ('cpu' if device == 'cpu' else 'cuda') for p in module.buffers())
        assert all(bool(adapter.torch.isfinite(t).all()) for t in module.state_dict().values())
    assert all(str(f.device) == device for f in model.feature_embeddings.values())
    return model, dict(native_state_fingerprint=expected, external_feature_fingerprint=expected_features,
                       checkpoint_SHA256=expected_sha256, native_global_step=saved['global_step'],
                       native_epoch=saved['epoch'], explicit_device=device)

def load_complete(cache, fit_dir, training_seed, device):
    """New generalized full-endpoint receipts only; never permit a warm start."""
    fit_dir = Path(fit_dir)
    receipt = adapter.read(fit_dir / 'fit_receipt.json')
    checkpoint = fit_dir / 'FM/final.ckpt'
    adapter.validate_complete(receipt, cache, checkpoint)
    assert receipt['training_seed'] == training_seed
    assert receipt['native_recipe'] == adapter.RECIPE and not receipt['network_attempts']
    for phase, epochs, updates in [('AE', 300, receipt['AE_updates']), ('FM', 1500, receipt['FM_updates'])]:
        path = fit_dir / phase / 'receipt.json'
        assert adapter.sha(path) == receipt[f'{phase}_receipt_SHA256']
        record = adapter.read(path)
        assert record['phase'] == phase
        assert record['completed_epochs'] == record['fixed_endpoint'] == epochs
        assert record['optimizer_updates'] == updates
        assert record['native_losses_finite'] and record['no_validation_or_early_selection']
        assert record['checkpoint_path'] == str(fit_dir / phase / 'final.ckpt')
        assert adapter.sha(record['checkpoint_path']) == record['checkpoint_SHA256']
    model, native = restore_checkpoint(checkpoint, device, receipt['FM_checkpoint_SHA256'])
    assert native['native_global_step'] == receipt['FM_updates']
    assert native['native_epoch'] == receipt['FM_epochs']
    assert fingerprint(model.encoder_model.state_dict()) == receipt['encoder_fingerprint']
    return model, receipt, native
