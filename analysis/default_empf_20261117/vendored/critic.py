"""Nested invariant observation: the preserved100 moments plus50 fixed Fourier means.

No upstream edits or extra objective. Added input capacity/scale and dimension
normalization are model-choice confounds, not a pure causal bug-fix ablation.
"""
import math
import torch
from torch import nn

N_PC=50
N_FEATURES=150
EPSILON=1e-6
FREQUENCY_SEED=20261030
# Exactly the tested float64 tensor; converting to input dtype/device is fixed.
FREQUENCIES=(torch.randn(50,50,generator=torch.Generator().manual_seed(FREQUENCY_SEED),dtype=torch.float64)/math.sqrt(50.))[:25].clone()

def moment_features(pcs):
    ordered=pcs.sort(dim=1).values
    mean=ordered.mean(1)
    sd=(ordered.var(1,unbiased=False)+EPSILON).sqrt()
    return torch.cat([mean,sd],dim=1)

def bag_features(pcs):
    if pcs.ndim!=3 or pcs.shape[-1]!=N_PC or pcs.shape[1]<1:
        raise ValueError('Expected [donor,cell,50] frozen-PC coordinates')
    moments=moment_features(pcs)
    phase=pcs@FREQUENCIES.to(pcs).T
    fourier=(torch.cat([phase.cos(),phase.sin()],dim=-1)/5.).sort(dim=1).values.mean(1)
    return torch.cat([moments,fourier],dim=1)

class VectorCritic(nn.Module):
    def __init__(self,n_features=N_FEATURES,h=128):
        super().__init__()
        if n_features!=N_FEATURES:raise ValueError('Frozen nested representation has150 features')
        self.phi=nn.Sequential(nn.Linear(n_features+7,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
        self.rho=nn.Sequential(nn.Linear(h+7,h),nn.SiLU(),nn.Linear(h,1))
    def forward(self,x,c):
        if x.ndim!=2 or x.shape[1]!=N_FEATURES or c.shape!=(len(x),7):
            raise ValueError('Expected continuous donor vectors [B,150] and conditions [B,7]')
        f=self.phi(torch.cat([x,c],1))
        return self.rho(torch.cat([f,c],1)).squeeze(1)
