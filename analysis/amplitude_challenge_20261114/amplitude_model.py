"""Concave TRAIN response fitting and independently routed composition amplitudes."""
import math
import amp_common as B
import factor_model as F
np=B.np;torch=B.torch;G=B.G;ZC=B.ZC


def fit_scalar(function,derivative,target):
    B.require(math.isfinite(target) and 0<=target<=1,'invalid observed nonzero')
    f0,f1=function(0.),function(1.);d0,d1=derivative(0.),derivative(1.)
    B.require(all(math.isfinite(x) for x in (f0,f1,d0,d1)) and
              0<=f0<=1 and 0<=f1<=1 and d0+1e-12>=d1,'invalid concave response endpoints')
    lo,hi=0.,1.;peak_steps=0
    if d0<=0:peak=lo=hi=0.
    elif d1>=0:peak=lo=hi=1.
    else:
        for _ in range(12):
            mid=(lo+hi)/2
            if derivative(mid)>0:lo=mid
            else:hi=mid
        peak=(lo+hi)/2;peak_steps=12
    fp=function(peak);roots=[]
    B.require(math.isfinite(fp) and 0<=fp<=1,'invalid peak response')
    degenerate=max(f0,fp,f1)-min(f0,fp,f1)<=1e-12
    if degenerate:answer=.5;clipped=False
    else:
        for left,right,fl,fr in ((0.,peak,f0,fp),(peak,1.,fp,f1)):
            if left==right or not min(fl,fr)<=target<=max(fl,fr):continue
            if target==fl:root=left;a=b=left;steps=0
            elif target==fr:root=right;a=b=right;steps=0
            else:
                a,b=left,right;increasing=fr>fl
                for _ in range(12):
                    mid=(a+b)/2
                    if (function(mid)<target)==increasing:a=mid
                    else:b=mid
                root=(a+b)/2;steps=12
            roots.append({'t':root,'segment':[left,right],'bracket':[a,b],
                          'resolution':b-a,'bisection_steps':steps})
        clipped=not roots
        if roots:answer=min((r['t'] for r in roots),key=lambda t:(abs(t-.5),t))
        else:answer=min((0.,peak,1.),key=lambda t:(abs(function(t)-target),abs(t-.5),t))
    return answer,{'observed_nonzero':target,'F_zero':f0,'F_one':f1,'dF_zero':d0,'dF_one':d1,
                   'peak':peak,'F_peak':fp,'peak_bracket':[lo,hi],
                   'peak_resolution':hi-lo,'peak_bisection_steps':peak_steps,
                   'roots':roots,'clipped':clipped,'degenerate':degenerate,'raw_t':answer,
                   'expected_nonzero_residual':function(answer)-target}


def normalize(values):
    raw=np.asarray(values,np.float64)
    B.require(raw.ndim==1 and len(raw)>0 and np.isfinite(raw).all()
              and ((0<=raw)&(raw<=1)).all(),'invalid raw amplitude law')
    positive=int(np.count_nonzero(raw>0));n=len(raw)
    info={'raw_t':raw.tolist(),'positive_count':positive,'positive_fraction':positive/n,
          'normalization':"t'=min(1,c*t); mean(t'^2)=.25"}
    if 4*positive<n:
        return None,dict(info,feasible=False,reason='fewer than25% strictly positive raw amplitudes')
    if 4*positive==n:
        c=1/float(raw[raw>0].min());normalized=np.where(raw>0,1.,0.);k=positive
    else:
        ordered=np.sort(raw)[::-1];c=None
        for k in range(positive):
            numerator=n*.25-k
            if numerator<=0:break
            denom=float(np.square(ordered[k:]).sum())
            trial=math.sqrt(numerator/denom)
            if (k==0 or trial*ordered[k-1]>=1) and trial*ordered[k]<=1:
                c=trial;break
        B.require(c is not None,'capped quadratic active set not found')
        normalized=np.minimum(1.,c*raw)
    moment=float(np.square(normalized).mean())
    moment32=float(np.square(normalized.astype(np.float32).astype(np.float64)).mean())
    B.require(abs(moment-.25)<1e-12 and abs(moment32-.25)<1e-6,'amplitude second moment changed')
    return normalized,dict(info,feasible=True,c=c,active_set_k=k,
                           saturated_count=int(np.count_nonzero(normalized==1)),normalized_t=normalized.tolist(),
                           second_moment=moment,generation_t_float32=normalized.astype(np.float32).tolist(),
                           second_moment_float32=moment32)


def cached_response(cached):
    n=sum(mu.numel() for mu,theta,base in cached)
    B.require(n>0,'empty cached response')
    def evaluate(t,gradient=False):
        total=0.
        for raw,theta,base in cached:
            mean=(1-t)*base+t*raw
            if gradient:values=(raw-base)*torch.exp((-theta-1)*torch.log1p(mean/theta))
            else:values=-torch.expm1(-theta*torch.log1p(mean/theta))
            total+=float(values.sum())
        return total/n
    return lambda t:evaluate(t),lambda t:evaluate(t,True)


@torch.no_grad()
def fit_law(ctx,models,plans):
    targets=ZC.train_targets(ctx);fits=[];values=[]
    for index,(donor,held,_,_) in enumerate(targets):
        B.require(not held,'held-out donor entered amplitude fit')
        cached=[];bags=cells=nvalues=0
        iterator=F.ORIGINAL_ITERATE(ctx,models,plans,targets,only={index})
        try:
            for ti,start,n,model,z,cond,libs,cell,count in iterator:
                # Use full replayed outer theta multiplier, never ZC.base_nb.
                mu,theta=model.parameters_nb(z,cond,G.K,libs,cell)
                B.require(bool(torch.isfinite(mu).all()) and bool((mu>=0).all())
                          and bool(torch.isfinite(theta).all()) and bool((theta>0).all())
                          and bool(torch.isfinite(libs).all()) and bool((libs>0).all()),'invalid cached NB parameters')
                raw=mu.detach().double();precision=theta.detach().double();library=libs.detach().double()[...,None]
                base=library*(raw.sum(1,keepdim=True)/library.sum(1,keepdim=True))
                cached.append((raw,precision,base));bags+=n;cells+=n*G.K;nvalues+=mu.numel()
        finally:iterator.close()
        function,derivative=cached_response(cached)
        observed=float((ctx.X[torch.as_tensor(ctx.by[index],device=ctx.X.device)]>0).double().mean())
        value,info=fit_scalar(function,derivative,observed)
        info.update(donor=donor,generated_bags=bags,generated_cells=cells,generated_gene_values=nvalues)
        fits.append(info);values.append(value)
        del cached,function,derivative
    law,normalization=normalize(values)
    return law,dict(normalization,fit_donors=[t[0] for t in targets],n_train_targets=len(targets),donor_fits=fits,
                    fitting='concave peak/segment roots;12 derivative and12 response bisections; exact endpoint roots',
                    fit_precision='float64 evaluation of fixed float32 fully calibrated NB parameters',
                    interpretation='independent scalar simulation random effect; degenerate/clipped fits do not establish donor variation')


class AmplitudeRoute(F.FactorRoute):
    def __init__(self,values,seed_for_donor):
        super().__init__(values,seed_for_donor)
        B.require(((0<=self.values)&(self.values<=1)).all(),'amplitude outside[0,1]')

    def patch(self,model):
        B.require(not getattr(model,'_donor_amplitude',False),'amplitude applied twice')
        original=model.parameters_nb
        def parameters_nb(z,c,k,library,gen=None):
            raw,theta=original(z,c,k,library,gen)
            B.require(self.current is not None and len(self.current)==len(z),'amplitude used outside routed chunk')
            if bool(np.all(self.current==.5)):
                return B.CS.shrink_mu(raw,.5),theta
            B.require(bool(torch.isfinite(library).all()) and bool((library>0).all()),'invalid supplied library')
            libs=library[...,None];base=libs*(raw.sum(1,keepdim=True)/libs.sum(1,keepdim=True))
            t=torch.as_tensor(self.current,dtype=raw.dtype,device=raw.device)[:,None,None]
            return (1-t)*base+t*raw,theta
        model.parameters_nb=parameters_nb;model._donor_amplitude=True
