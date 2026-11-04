"""Fixed ADNC-to-particle support; no added state, RNG stream, loss or output repair."""
import torch
CONFIG={'conditioning':'known_ADNC_stage','stages':4,'particles_per_stage':128,'num_particles':512,'partition':'contiguous_fixed_equal_groups','sampling':'uniform_with_replacement_within_stage'}
def stages(c):
    c=torch.as_tensor(c)
    if c.ndim==1:c=c[None]
    if c.ndim!=2 or c.shape[1]!=7 or not len(c):raise ValueError('ADNC condition rows require seven columns')
    value=c[:,3]*3;index=value.round()
    if not bool(torch.isfinite(value).all()) or not bool(((value-index).abs()<=1e-5).all()) or not bool(((index>=0)&(index<4)).all()):raise ValueError('Only the four known ADNC stages are supported')
    return index.long()
def sample_stage(prior,c,*,generator):
    if tuple(prior.z.shape)!=(512,8):raise ValueError('Matched512x8 native particle table required')
    stage=stages(c).to(prior.z.device)
    idx=stage*128+torch.randint(128,(len(stage),),device=prior.z.device,generator=generator)
    return prior(idx),idx
class StageView:
    def __init__(self,prior,stage):
        if type(stage) is not int or not 0<=stage<4:raise ValueError('Fixed known stage required')
        self.prior=prior;self.stage=stage
    @property
    def training(self):return self.prior.training
    def train(self,mode=True):self.prior.train(mode);return self
    def eval(self):return self.train(False)
    def sample(self,n,generator=None):
        if type(n) is not int or n<=0:raise ValueError('Positive integer batch size required')
        c=self.prior.z.new_zeros((n,7));c[:,3]=self.stage/3
        return sample_stage(self.prior,c,generator=generator)
def view_for_condition(prior,c):
    s=stages(c)
    if not bool((s==s[0]).all()):raise ValueError('Generation view requires one fixed stage')
    return StageView(prior,int(s[0]))
