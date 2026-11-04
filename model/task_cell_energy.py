"""Inactive fitting-only math prototype; no models, donor files or admission."""
import math
import torch

def pooled_cell_energy(real,fake,bag_weights):
    """Unbiased weighted energy over independent bags, normalized by sqrt(D).

    Real/fake bag i can share conditioning; cells within a bag are dependent.
    Exclude i==j blocks from every term, including XY. Weights must be fixed
    population importance weights, not normalized by their minibatch sum.
    """
    if (not isinstance(real,torch.Tensor) or not isinstance(fake,torch.Tensor)
            or real.ndim!=3 or real.shape!=fake.shape or real.shape[0]<2
            or min(real.shape[1:])<1 or real.device!=fake.device
            or real.dtype!=fake.dtype or not real.is_floating_point()):
        raise ValueError('matched floating N>=2 by M>=1 by D>=1 bag tensors required')
    n,_,d=real.shape
    if (not isinstance(bag_weights,torch.Tensor) or bag_weights.shape!=(n,)
            or bag_weights.device!=real.device or bag_weights.dtype!=real.dtype
            or bag_weights.requires_grad or not torch.isfinite(bag_weights).all()
            or not (bag_weights>0).all()):
        raise ValueError('fixed positive finite population importance weights required')
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        raise ValueError('finite cell coordinates required')
    def block_distances(x,y):
        difference=x[:,None,:,None,:]-y[None,:,None,:,:]
        return torch.linalg.vector_norm(difference,dim=-1).mean(dim=(-2,-1))
    xx=block_distances(fake,fake);yy=block_distances(real,real)
    xy=block_distances(fake,real)
    weights=bag_weights[:,None]*bag_weights[None,:]
    off_diagonal=~torch.eye(n,dtype=torch.bool,device=real.device)
    return ((2*xy-xx-yy)*weights)[off_diagonal].sum()/(n*(n-1)*math.sqrt(d))

def stage_importance_weights(stage_ids,donor_counts,*,dtype):
    """Population weights for equal stages under uniform fitting-donor draws."""
    if (not isinstance(stage_ids,torch.Tensor) or stage_ids.ndim!=1
            or stage_ids.dtype!=torch.long or not isinstance(donor_counts,torch.Tensor)
            or donor_counts.shape!=(4,) or donor_counts.dtype!=torch.long
            or donor_counts.device!=stage_ids.device or not (donor_counts>0).all()
            or not ((stage_ids>=0)&(stage_ids<4)).all()
            or dtype not in (torch.float32,torch.float64)):
        raise ValueError('four positive fitting-stage counts and integer stage IDs required')
    return donor_counts.sum().to(dtype)/(4*donor_counts[stage_ids].to(dtype))
