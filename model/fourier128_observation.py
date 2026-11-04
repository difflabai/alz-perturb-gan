"""Fixed richer characteristic-function observations; generated counts unchanged."""
import math
import torch
from torch import nn
from critic import moment_features

OPTIONS={'PC_dimensions':50,'moment_dimensions':100,'frequencies':128,'frequency_seed':20261030,'features':356,'hidden_width':128}
N_FEATURES=356
FREQUENCIES=torch.randn(128,50,generator=torch.Generator().manual_seed(20261030),dtype=torch.float64)/math.sqrt(50.)

def normalized_options(value):
    if not isinstance(value,dict) or value!=OPTIONS or any(type(value[k]) is not type(v) for k,v in OPTIONS.items()):raise ValueError('Fixed registered128frequency observation required')
    return dict(value)

def bag_features(pcs):
    if pcs.ndim!=3 or pcs.shape[-1]!=50 or pcs.shape[1]<1:raise ValueError('Expected bag/cell/frozenPC50')
    moments=moment_features(pcs)
    phase=pcs@FREQUENCIES.to(pcs).T
    fourier=(torch.cat([phase.cos(),phase.sin()],dim=-1)/math.sqrt(128.)).sort(dim=1).values.mean(1)
    return torch.cat([moments,fourier],dim=1)

class VectorCritic(nn.Module):
    def __init__(self,n_features=N_FEATURES,h=128):
        super().__init__()
        if n_features!=356 or h!=128:raise ValueError('Registered356feature128hidden critic required')
        self.phi=nn.Sequential(nn.Linear(n_features+7,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
        self.rho=nn.Sequential(nn.Linear(h+7,h),nn.SiLU(),nn.Linear(h,1))
    def forward(self,x,c):
        if x.ndim!=2 or x.shape[1]!=356 or c.shape!=(len(x),7):raise ValueError('Expected356continuous observations and7condition slots')
        f=self.phi(torch.cat([x,c],1))
        return self.rho(torch.cat([f,c],1)).squeeze(1)
