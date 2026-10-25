"""Caller-owned conditional loop uses complete develop optimizer bindings."""
from dataclasses import asdict
import pytest
import torch
from test_gene_harness import study, table, cfg
from alz_perturb_gan.gene_models import build_models, ModelConfig
from alz_perturb_gan.gene_training import train_arm
from alz_perturb_gan.gene_checkpoint import save_checkpoint, load_checkpoint, model_hashes
from alz_perturb_gan.particle_integration import PARTICLEGAN_COMMIT


def test_conditional_bcap_uses_protected_backward_and_sampled_rows(study, table, monkeypatch, tmp_path):
    import alz_perturb_gan.gene_training as training
    models=build_models(cfg(trainer='bcap'),data_kind='synthetic')
    before=model_hashes(models)
    calls=[]; original=training.constraint_geometry_backward
    def bound(loss,opt,protected):
        assert opt._sampled_rows is not None if hasattr(opt,'_sampled_rows') else True
        calls.append((opt,protected))
        return original(loss,opt,protected)
    monkeypatch.setattr(training,'constraint_geometry_backward',bound)
    out=train_arm(models,study,max_wall_seconds=30)
    assert len(calls)==2 and all(len(x[1])==1 for x in calls)
    assert out['particle_integration']['commit']==PARTICLEGAN_COMMIT
    assert out['particle_integration']['recipe']['critic_formulation']=='bcap'
    assert out['particle_integration']['recipe']['constraint_geometry_mode']=='direction_blend'
    assert out['steps']==2 and out['selection_scoring_during_fit'] is False
    assert before['generator']!=model_hashes(models)['generator']
    assert before['prior']!=model_hashes(models)['prior']
    path=tmp_path/'bcap.pt';digest=save_checkpoint(path,models,study.protocol,training_receipt=out)
    loaded=load_checkpoint(path,digest)
    assert loaded.config.trainer=='bcap' and model_hashes(loaded)==model_hashes(models)


def test_old_checkpoint_config_keeps_explicit_ka2():
    old=asdict(cfg());old.pop('trainer')
    assert build_models(ModelConfig(**old),data_kind='synthetic').recipe.critic_formulation=='ka2'


def test_conditional_bispf_requires_context_contract(study):
    models=build_models(cfg(trainer='bispf'),data_kind='synthetic')
    with pytest.raises(ValueError,match='conditional-context'): train_arm(models,study,max_wall_seconds=30)


@pytest.mark.parametrize('field',['resolved_recipe','serving_law','particle_integration'])
def test_checkpoint_refuses_incomplete_recipe_or_serving_identity(tmp_path,study,field):
    from alz_perturb_gan.gene_features import file_sha256
    models=build_models(cfg(trainer='bcap'),data_kind='synthetic')
    path=tmp_path/'original.pt';save_checkpoint(path,models,study.protocol)
    blob=torch.load(path,weights_only=True);blob.pop(field)
    bad=tmp_path/'missing.pt';torch.save(blob,bad)
    with pytest.raises((ValueError,KeyError)):
        load_checkpoint(bad,file_sha256(bad))


def test_new_grant_cannot_be_downgraded_to_a_different_trainer(study):
    from alz_perturb_gan.gene_protocol import validate_grant
    grant={'approved':True,'authorization_reference':'unit-test','phases':['train'],
           'protocol_sha256':study.protocol.fingerprint,'arms':['B0'],'seeds':[42],
           'max_steps_per_arm':2,'max_wall_seconds_per_arm':30,'particle_trainer':'bcap',
           'particlegan_commit':PARTICLEGAN_COMMIT}
    with pytest.raises(PermissionError,match='trainer/source'): validate_grant(grant,study,cfg(),'cpu','train',30)
    validate_grant(grant,study,cfg(trainer='bcap'),'cpu','train',30)
