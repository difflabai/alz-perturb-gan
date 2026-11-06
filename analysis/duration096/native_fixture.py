"""Small synthetic fitting context; no real ALZ donors or target scores."""
import torch
import released_adapter as adapter
from test_variant import fixture
def context(cache,fullwidth=False):
    ctx=fixture(cache);ctx.__class__=adapter.Context
    if fullwidth:
        ctx.X=torch.randint(1,8,(24,2000),generator=torch.Generator().manual_seed(774)).float()
        ctx.mean,ctx.sd=torch.zeros(2000),torch.ones(2000)
        ctx.W=torch.randn(2000,50,generator=torch.Generator().manual_seed(775))/2000
        ctx.trainpcs,ctx.mu=ctx.project(ctx.X),ctx.X.mean(0)
        ctx.theta=(ctx.mu.square()/(ctx.X.var(0,unbiased=False)-ctx.mu).clamp_min(1e-4)).clamp(.1,10000)
    return ctx
