"""Bounded learned cell observations; the released penalty still sees150 vectors."""
import torch
from torch import nn
from critic import VectorCritic,moment_features,bag_features

OPTIONS={'enabled':False,'cell_hidden':64,'cell_features':25,'feature_scale':.2,'pool_epsilon':1e-6,'layernorm_epsilon':1e-5}

def normalized_options(value):
 if not isinstance(value,dict) or set(value)!=set(OPTIONS) or type(value['enabled']) is not bool or any(value[k]!=OPTIONS[k] for k in OPTIONS if k!='enabled'):
  raise ValueError('Registered learned cell observation configuration required')
 return dict(value)

class LearnedCellCritic(nn.Module):
 def __init__(self,head,options):
  super().__init__();self.options=normalized_options(options);self.head=head
  self.cell_encoder=nn.Sequential(nn.Linear(50,64),nn.SiLU(),nn.Linear(64,25),nn.LayerNorm(25,eps=1e-5,elementwise_affine=False))
 def features(self,pcs):
  if pcs.ndim!=3 or pcs.shape[-1]!=50 or pcs.shape[1]<1:raise ValueError('Expected bag/cell/frozenPC50')
  b,n,_=pcs.shape;cell=self.cell_encoder(pcs.reshape(-1,50)).reshape(b,n,25)
  ordered=cell.sort(dim=1).values
  learned=torch.cat([ordered.mean(1),(ordered.var(1,unbiased=False)+1e-6).sqrt()],dim=1)*.2
  return torch.cat([moment_features(pcs),learned],dim=1)
 def forward(self,features,condition):return self.head(features,condition)

def make_critic(options):
 options=normalized_options(options);head=VectorCritic(150,h=128)
 return LearnedCellCritic(head,options) if options['enabled'] else head

def observations(critic,pcs,options):
 return critic.features(pcs) if options['enabled'] else bag_features(pcs)
