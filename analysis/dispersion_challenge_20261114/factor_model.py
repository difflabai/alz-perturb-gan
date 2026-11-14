"""Fixed TRAIN scalar fit, inverse-precision normalization and routed generation."""
from contextlib import contextmanager
import math
import common as B

np=B.np;torch=B.torch;G=B.G;ZC=B.ZC
ORIGINAL_ITERATE=ZC.iterate_chunks
ORIGINAL_GENERATE=G.generate_arrays


def fit_scalar(function,target):
    B.require(math.isfinite(target) and 0<=target<=1,'invalid observed nonzero fraction')
    low,high=-1.,1.
    f_low,f_high=function(low),function(high)
    B.require(math.isfinite(f_low) and math.isfinite(f_high) and f_low<=f_high+1e-12,
              'nonfinite/nonmonotone expected nonzero bracket')
    clipped=None;iterations=0
    if target<=f_low:answer=low;clipped='low'
    elif target>=f_high:answer=high;clipped='high'
    else:
        for _ in range(12):
            midpoint=(low+high)/2
            if function(midpoint)<target:low=midpoint
            else:high=midpoint
        iterations=12;answer=(low+high)/2
    return answer,{'observed_nonzero':target,'F_low':f_low,'F_high':f_high,
                   'u':answer,'clipped':clipped,'bisection_steps':iterations,
                   'expected_nonzero_residual':function(answer)-target}


def normalized_factors(values):
    values=np.asarray(values,dtype=np.float64)
    B.require(values.ndim==1 and len(values)>0 and np.isfinite(values).all()
              and (np.abs(values)<=1).all(),'invalid bounded donor offsets')
    if np.ptp(values)==0:return np.zeros_like(values)
    return values+np.log(np.exp(-values).mean())


@torch.no_grad()
def fit_law(ctx,models,plans):
    targets=ZC.train_targets(ctx);fits=[];offsets=[]
    for index,(donor,held,_,_) in enumerate(targets):
        B.require(not held,'held-out target entered factor fit')
        cached=[];nvalues=0
        iterator=ORIGINAL_ITERATE(ctx,models,plans,targets,only={index})
        try:
            for ti,start,n,model,z,cond,libs,cell,count in iterator:
                # Include replayed current gene calibration, not base_nb's
                # intentional bypass of its outer pooled theta multiplier.
                mu,theta=model.parameters_nb(z,cond,G.K,libs,cell)
                B.require(bool(torch.isfinite(mu).all()) and bool((mu>=0).all()) and
                          bool(torch.isfinite(theta).all()) and bool((theta>0).all()),'invalid cached NB parameters')
                cached.append((mu.detach().double(),theta.detach().double()))
                nvalues+=mu.numel()
        finally:iterator.close()
        def expected(u):
            total=0.
            for mu,theta in cached:
                precision=theta*math.exp(u)
                nz=1-torch.exp(-precision*torch.log1p(mu/precision))
                total+=float(nz.sum())
            return total/nvalues
        observed=float((ctx.X[torch.as_tensor(ctx.by[index],device=ctx.X.device)]>0).double().mean())
        u,info=fit_scalar(expected,observed)
        info.update(donor=donor,generated_bags=G.M,generated_cells=G.M*G.K,generated_gene_values=nvalues)
        fits.append(info);offsets.append(u)
        del cached
    values=normalized_factors(offsets)
    return values,{'fit_donors':[t[0] for t in targets],'n_train_targets':len(targets),
                   'fitting':'12-step bounded bisection; u in[-1,1]',
                   'normalization':'h=u+log(mean(exp(-u)))',
                   'fit_precision':'float64 evaluation of fixed float32 model means/dispersion',
                   'donor_fits':fits,'h':values.tolist(),
                   'mean_inverse_precision_factor':float(np.exp(-values).mean()),
                   'interpretation':'independent empirical simulation offsets, not identified biological axes'}


class FactorRoute:
    def __init__(self,values,seed_for_donor):
        self.values=np.asarray(values,dtype=np.float32)
        B.require(self.values.ndim==1 and len(self.values)>0 and np.isfinite(self.values).all()
                  and (np.abs(self.values)<=2.000001).all(),'invalid factor law')
        self.seed_for_donor=seed_for_donor;self.assignments={};self.current=None
        self.draws=G.M

    def indices(self,donor):
        B.require(G.M==self.draws,'M changed during fixed factor routing')
        if donor not in self.assignments:
            self.assignments[donor]=np.random.default_rng(self.seed_for_donor(donor)).integers(
                len(self.values),size=self.draws).tolist()
        return self.assignments[donor]

    def patch(self,model):
        B.require(not getattr(model,'_donor_factor',False),'factor applied twice')
        original=model.parameters_nb
        def parameters_nb(z,c,k,library,gen=None):
            mu,theta=original(z,c,k,library,gen)
            B.require(self.current is not None and len(self.current)==len(z),'factor used outside routed chunk')
            # Preserve theta's original broadcast shape and RNG execution for
            # literal zero factors; multiplying by broadcasted ones can alter it.
            if not np.any(self.current):return mu,theta
            h=torch.as_tensor(self.current,dtype=theta.dtype,device=theta.device)
            return mu,theta*torch.exp(h[:,None,None])
        model.parameters_nb=parameters_nb;model._donor_factor=True

    def iterate(self,ctx,models,plans,targets,only=None):
        iterator=ORIGINAL_ITERATE(ctx,models,plans,targets,only=only)
        try:
            for chunk in iterator:
                ti,start,n=chunk[:3];donor=targets[ti][0]
                indices=self.indices(donor)[start:start+n]
                self.current=self.values[indices]
                try:yield chunk
                finally:self.current=None
        finally:iterator.close()

    @torch.no_grad()
    def generate(self,ctx,models,plans=None,*,T=None):
        targets=G.targets(ctx) if T is None else T
        if plans is None:plans,_=G.default_plans(ctx,targets)
        n=len(targets);gn=ctx.n_genes
        arrays={'pb_counts':np.zeros((n,G.M,gn),np.float32),'mean_log':np.zeros((n,G.M,gn),np.float32),
                'nonzero':np.zeros((n,G.M,gn),np.float32),'pcs':np.zeros((n,G.M,G.PC_CELLS,50),np.float16),
                'lib':np.zeros((n,G.M),np.float32),'sq_log':np.zeros((n,G.M,gn),np.float16)}
        zero=0
        iterator=self.iterate(ctx,models,plans,targets)
        try:
            for ti,start,num,model,z,cond,libs,cell,count in iterator:
                rate=model.rates(z,cond,G.K,libs,cell)
                counts=torch.poisson(rate,generator=count)
                B.require(bool(torch.isfinite(counts).all()) and bool((counts>=0).all()),'invalid generated counts')
                total=counts.sum(-1);zero+=int((total==0).sum())
                logcp=torch.log1p(counts*(1e4/total.clamp_min(1))[...,None])
                sl=slice(start,start+num)
                arrays['pb_counts'][ti,sl]=counts.sum(1).cpu().numpy()
                arrays['mean_log'][ti,sl]=logcp.mean(1).cpu().numpy()
                arrays['sq_log'][ti,sl]=logcp.square().mean(1).cpu().numpy().astype(np.float16)
                arrays['nonzero'][ti,sl]=(counts>0).float().mean(1).cpu().numpy()
                pcs=G.standardize_counts(counts[:,:G.PC_CELLS],ctx.mean,ctx.sd)@ctx.W
                B.require(bool(torch.isfinite(pcs).all()),'nonfinite generated PC coordinates')
                arrays['pcs'][ti,sl]=pcs.cpu().numpy().astype(np.float16)
                arrays['lib'][ti,sl]=total.mean(1).cpu().numpy()
        finally:iterator.close()
        raw=np.stack([t[2] for t in targets])
        arrays.update(targets=np.array([t[0] for t in targets]),is_heldout=np.array([t[1] for t in targets],dtype=bool),
                      stage=G.stage_of(raw).astype(np.int64),genes=np.asarray(ctx.genes).astype(str))
        stats={'targets':n,'heldout_targets':int(arrays['is_heldout'].sum()),'M':G.M,'K':G.K,
               'generated_cells':n*G.M*G.K,'zero_count_cells':zero,'zero_count_cell_rate':zero/(n*G.M*G.K),
               'pcs_float16_max_abs':float(np.abs(arrays['pcs'].astype(np.float32)).max()),
               'sq_log_float16_max':float(arrays['sq_log'].astype(np.float32).max()),
               'heldout_dir_present':bool(arrays['is_heldout'].any())}
        return arrays,stats

    @contextmanager
    def installed(self):
        B.require(ZC.iterate_chunks is ORIGINAL_ITERATE and G.generate_arrays is ORIGINAL_GENERATE,
                  'nested or incompatible generation routing')
        ZC.iterate_chunks=self.iterate;G.generate_arrays=self.generate
        try:yield self
        finally:
            ZC.iterate_chunks=ORIGINAL_ITERATE;G.generate_arrays=ORIGINAL_GENERATE;self.current=None
