"""Pure count readout from one coupled native served E22 snapshot."""
import numpy as np
import torch
from generation import validate_counts
from routed_counts import make_context


@torch.no_grad()
def iter_routed_counts(ctx,served,forward,schedule,c,s):
    if forward.genes!=2000 or np.asarray(schedule.libraries).shape!=(200,400):
        raise ValueError('Frozen200bag/400cell/2000gene generation contract required')
    device=ctx.device;draw_seed=int(s)+500000000
    torch.manual_seed(draw_seed)
    query=torch.Generator(device=device).manual_seed(draw_seed+1)
    condition=torch.as_tensor(c,dtype=torch.float32,device=device)[None]
    for start in range(0,200,5):
        n=min(5,200-start)
        library=torch.as_tensor(schedule.libraries[start:start+n],dtype=torch.float32,device=device)
        context=make_context(condition.expand(n,-1),library,generator=query)
        packed=served.routed_forward(context,perturb=False,output_noise=False)
        raw=forward.raw_counts(packed)
        validate_counts(raw.reshape(n*400,-1),n*400,2000)
        yield (start,ctx.project(raw).cpu().numpy(),raw[:1].cpu().numpy(),
               raw.sum(-1).reshape(-1).cpu().numpy())
