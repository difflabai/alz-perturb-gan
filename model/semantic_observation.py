"""Frozen subtype bag observations, with explicit hard-forward/soft-backward law."""
import hashlib
import json
import torch
from torch import nn
from critic import VectorCritic as BaseCritic,bag_features as base_features

class SubtypeObservation:
 def __init__(self,classifier,*,enabled,temperature=1.,epsilon=1e-4):
  if type(enabled) is not bool or temperature!=1. or epsilon!=1e-4:raise ValueError('Registered semantic observation configuration required')
  self.enabled=enabled;self.temperature=temperature;self.epsilon=epsilon
  self.classifier=classifier.eval().requires_grad_(False)
  self.digest=self.state_digest()
 def state_digest(self):
  h=hashlib.sha256()
  for name,tensor in sorted(self.classifier.state_dict().items()):
   h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def metadata(self):return {'enabled':self.enabled,'temperature':self.temperature,'epsilon':self.epsilon,'classifier_state_SHA256':self.digest}
 def subtype_features(self,pcs):
  shape=pcs.shape
  if len(shape)!=3 or shape[-1]!=50 or shape[1]<1:raise ValueError('Expected bag/cell/frozenPC50')
  logits=self.classifier(pcs.reshape(-1,50)).reshape(shape[0],shape[1],6)
  hard=torch.nn.functional.one_hot(logits.argmax(-1),num_classes=6).to(pcs.dtype).sum(1)/shape[1]
  hard_root=hard.sqrt()
  soft=(logits/self.temperature).softmax(-1).sort(dim=1).values.mean(1)
  soft_root=(soft+self.epsilon).sqrt()
  return hard_root.detach()+(soft_root-soft_root.detach())
 def __call__(self,pcs):
  original=base_features(pcs)
  return torch.cat([original,self.subtype_features(pcs)],dim=-1) if self.enabled else original

def make_critic(n_features,h=128):
 if n_features not in (150,156):raise ValueError('Registered semantic feature dimension required')
 base=BaseCritic(150,h=h)
 if n_features==150:return base
 # Preserve all existing parameter values and forward behavior initially.
 old=base.phi[0]
 with torch.random.fork_rng(devices=[]):expanded=nn.Linear(156+7,h)
 with torch.no_grad():
  expanded.weight.zero_();expanded.weight[:,:150].copy_(old.weight[:,:150]);expanded.weight[:,156:].copy_(old.weight[:,150:]);expanded.bias.copy_(old.bias)
 base.phi[0]=expanded
 # The released vector critic validates150 inputs; use its exact architecture
 # and condition concatenation in a dimension-parameterized forward below.
 return SemanticCritic(base,n_features)

class SemanticCritic(nn.Module):
 def __init__(self,base,n_features):
  super().__init__();self.phi=base.phi;self.rho=base.rho;self.n_features=n_features
 def forward(self,x,c):
  if x.ndim!=2 or x.shape[1]!=self.n_features or c.shape!=(len(x),7):raise ValueError('Semantic critic feature/condition dimension mismatch')
  f=self.phi(torch.cat([x,c],1));return self.rho(torch.cat([f,c],1)).squeeze(1)
