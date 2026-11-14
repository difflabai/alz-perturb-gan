"""GPU forward adapter with the exact frozen CPU TRAIN reference geometry."""
import copy
import cov_common as C
import covmom as M

_cpu_setup=M.CovarianceArm.setup


@C.torch.no_grad()
def setup(self,ctx):
    if ctx.device.type=='cpu':
        return _cpu_setup(self,ctx)
    if ctx.device.type!='cuda':raise ValueError('only declared CPU/GPU devices')
    reference=C.T.Context(ctx.cache,'cpu',ctx.flags)
    source=next(a for a in reference.plugins if a.name=='d.covmom')
    self.projection=source.projection.to(ctx.device)
    self.center=source.center.to(ctx.device)
    self.basis=source.basis.copy();self.random_scale=source.random_scale.copy()
    self.scale=source.scale;self.reference_indices_sha256=source.reference_indices_sha256
    self.info=copy.deepcopy(source.info)
    C.B.require(C.T._same(self.state(),source.state()),'CPU/GPU reference state differs')


M.CovarianceArm.setup=setup
