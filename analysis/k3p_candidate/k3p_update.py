"""Inactive K3P factory with the exact already-qualified count task update."""
from ka2_update import CONFIG,normalized_config,configure,bag_features,finite,gradnorm,advance
from particlegan import get_recipe

def recipe():
    return get_recipe('k3p',total_steps=20000,batch_size=32,num_particles=512,z_dim=CONFIG['z_dim'])
