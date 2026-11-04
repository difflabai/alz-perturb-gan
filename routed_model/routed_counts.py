"""Complete conditional Gamma-Poisson callback for public native RoutedRows."""
import math

import torch
from pg_model import critic_features, standardize_counts
from critic import bag_features

CONTEXT_CONFIG = {'query_dimension': 8, 'condition_dimension': 7, 'seed_streams': 3,
                  'words_per_seed': 4, 'word_bits': 16, 'effective_seed_bits': 64,
                  'output': 'raw_counts_and_existing_transformed_rate_observation'}


def validate_context(context):
    if (not isinstance(context, torch.Tensor) or context.ndim != 2 or not len(context)
            or context.shape[1] <= 27 or context.dtype != torch.float32
            or not bool(torch.isfinite(context).all())):
        raise ValueError('Finite float32 query/condition/libraries/seed-word context required')
    c, libraries, words = context[:,8:15], context[:,15:-12], context[:,-12:]
    s = c[:,3]*3
    if (not bool((c[:,[0,1,2,4,5,6]]==0).all())
            or not bool(((s-s.round()).abs()<=1e-5).all())
            or not bool(((s>=0)&(s<=3)).all())):
        raise ValueError('Only masked known ADNC stage conditions are supported')
    if not bool((libraries>=0).all()):
        raise ValueError('Nonnegative cell library exposures required')
    if not bool(((words>=0)&(words<65536)&(words==words.round())).all()):
        raise ValueError('Exact unsigned16bit seed words required')


def make_context(c, libraries, *, generator):
    if c.ndim!=2 or c.shape[1]!=7 or libraries.ndim!=2 or len(c)!=len(libraries):
        raise ValueError('Aligned batch-first condition and library arrays required')
    query=torch.randn(len(c),8,device=c.device,generator=generator)
    words=torch.randint(65536,(len(c),12),device=c.device,generator=generator).to(torch.float32)
    value=torch.cat([query,c.to(torch.float32),libraries.to(torch.float32),words],-1)
    validate_context(value)
    return value


def make_guard_context(c, libraries, *, rng):
    # Protected observations use their independent NumPy stream, never the
    # fitting policy's latent/noise generators.
    query=torch.as_tensor(rng.standard_normal((len(c),8)).astype('float32'),device=c.device)
    words=torch.as_tensor(rng.integers(65536,size=(len(c),12)).astype('float32'),device=c.device)
    value=torch.cat([query,c,libraries,words],-1)
    validate_context(value)
    return value


class CountForward:
    def __init__(self, mean, sd, W, *, temperature):
        if (mean.ndim!=1 or sd.shape!=mean.shape or W.shape!=(len(mean),50)
                or any(x.dtype!=torch.float32 for x in (mean,sd,W))
                or any(not bool(torch.isfinite(x).all()) for x in (mean,sd,W))
                or not bool((sd>0).all()) or not math.isfinite(temperature) or temperature<=0):
            raise ValueError('Frozen finite float32 standardized50PC space and native temperature required')
        self.genes=len(mean);self.mean=mean;self.sd=sd;self.W=W;self.temperature=temperature

    def route(self, models, context, candidate):
        validate_context(context)
        if tuple(candidate.table.shape)!=(512,8) or tuple(candidate.log_mass.shape)!=(512,):
            raise ValueError('Qualified512x8 bank and per-row represented mass required')
        return (context[:,:8]@candidate.table.T/math.sqrt(8)/self.temperature+candidate.log_mass).softmax(-1)

    @staticmethod
    def unpack_seeds(context):
        validate_context(context)
        words=context[:,-12:].detach().to(device='cpu',dtype=torch.int64)
        # Python integer arithmetic retains all64bits; no float64 or signed
        # int64 conversion of the assembled unsigned seed occurs.
        return [[sum(int(row[4*k+j])<<(16*j) for j in range(4)) for k in range(3)] for row in words]

    def generate(self, models, context, candidate, weights):
        seeds=self.unpack_seeds(context)
        if (candidate.codes is None or tuple(candidate.codes.shape)!=(len(context),8)
                or not bool(torch.isfinite(candidate.codes).all())):
            raise ValueError('Native candidate mixed codes, including optional DV12 perturbation, required')
        result=[]
        for i,(gamma_seed,cell_seed,count_seed) in enumerate(seeds):
            code=candidate.codes[i:i+1]
            c,lib=context[i:i+1,8:15],context[i:i+1,15:-12]
            with torch.random.fork_rng(devices=[code.device.index] if code.device.type=='cuda' else []):
                torch.manual_seed(gamma_seed)
                cell=torch.Generator(device=code.device).manual_seed(cell_seed)
                count=torch.Generator(device=code.device).manual_seed(count_seed)
                rate=models['generator'].rates(code,c,lib.shape[1],lib,cell)
                raw=torch.poisson(rate.detach(),generator=count)
                aux=critic_features(raw,rate,self.mean.to(code),self.sd.to(code))
                result.append(torch.cat([raw,aux],-1))
        return torch.cat(result,0)

    def targets(self, counts):
        if (counts.ndim!=3 or counts.shape[-1]!=self.genes
                or not bool(torch.isfinite(counts).all()) or not bool((counts>=0).all())
                or not bool((counts==counts.floor()).all())):
            raise ValueError('Batch/cell/gene target shape differs')
        return torch.cat([counts,standardize_counts(counts,self.mean,self.sd)],-1)

    def observation(self, packed):
        if packed.ndim!=3 or packed.shape[-1]!=2*self.genes:
            raise ValueError('Complete packed count/observation shape differs')
        return bag_features(packed[...,self.genes:]@self.W.to(packed))

    def features(self, models, context, samples, targets):
        return models['critic'].phi(torch.cat([self.observation(samples),context[:,8:15]],-1))

    def raw_counts(self, packed):
        raw=packed[...,:self.genes]
        if (packed.ndim!=3 or packed.shape[-1]!=2*self.genes
                or not bool(torch.isfinite(packed).all()) or not bool((raw>=0).all())
                or not bool((raw==raw.floor()).all())):
            raise ValueError('Direct generated integer count contract failed; no repair/retry')
        return raw
