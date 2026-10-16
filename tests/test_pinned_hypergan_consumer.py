"""Installed optional bridge must actually train the selected engine."""
import importlib.util
import json
import pytest


def test_pinned_bridge_preserves_bcap_through_public_training_and_checkpoint(tmp_path):
    if importlib.util.find_spec('hypergan') is None:
        pytest.skip('requires the pinned hypergan optional extra')
    from alz_perturb_gan.hypergan_bridge import describe_hypergan,train_condition_gan,INTEGRATION_COMMIT
    from alz_perturb_gan.data import make_synthetic
    from hypergan.checkpoints import read_checkpoint
    info=describe_hypergan()
    assert info['status']=='develop' and info['commit']==INTEGRATION_COMMIT,info
    data=make_synthetic(seed=9,cells_per=2,n_genes=8,n_single=4,n_combo=2,
                        holdout_singles=1,holdout_combos=1)
    run=tmp_path/'bridge'
    result=train_condition_gan(data,run,steps=2,batch_size=4)
    assert result['status']=='complete' and result['steps']==2
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['config']['particlegan_trainer']=='bcap'
    state=read_checkpoint(run)[2]
    assert state['particle_recipe']['critic_formulation']=='bcap'
    assert state['particle_recipe']['loss']=='non_saturating'
    assert 'direction_blend' in state['optimizers'][0]
