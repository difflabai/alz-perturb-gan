"""One fixed marginal-preserving Gamma process coupling; explicit numerical limits."""
import math
import joint_common as J
from scipy.special import gammainc
B=J.B;np=J.np;torch=J.torch
TINY=torch.finfo(torch.float64).tiny


def partition(groups):
    raw=np.asarray(groups)
    B.require(raw.ndim==1 and len(raw)>0 and np.issubdtype(raw.dtype,np.number)
              and np.isrealobj(raw),'invalid group array')
    B.require(np.isfinite(raw).all() and ((0<=raw)&(raw<100)).all()
              and np.equal(raw,np.floor(raw)).all(),'groups must be integers in[0,99]')
    return raw.astype(np.int64,copy=True)


class GammaHead:
    def __init__(self,groups,coupling):
        self.groups=partition(groups)
        B.require(coupling in (0.,.5),'only literal0 or fixed.5 allowed')
        self.a=coupling;self.cache=None
        self.stats={'shared_increment_draws':0,'independent_draws':0,'shared_tiny_hits':0,'independent_tiny_hits':0,
                    'expected_shared_tiny_hits':0.,'expected_independent_tiny_hits':0.,
                    'shared_tiny_hit_variance':0.,'independent_tiny_hit_variance':0.,
                    'final_cast_underflows':0,'rate_error_bound_max':0.,'rate_error_bound_sum':0.,
                    'final_cast_lost_rate_mass':0.,'final_cast_lost_rate_max':0.,
                    'minimum_positive_time':None,'minimum_independent_shape':None,
                    'minimum_positive_increment':None,'minimum_gamma_sample':None,'exact_tied_time_increments':0}

    def configure(self,theta):
        # The frozen pbulk fits have a gene-only precision vector. Do not
        # silently reinterpret donor/cell-dependent precision as that vector.
        B.require(theta.numel()==len(self.groups),'diagnostic requires gene-only precision')
        precision=theta.detach().reshape(-1).double().clone()
        B.require(bool(torch.isfinite(precision).all()) and bool((precision>0).all()),'invalid Gamma precision')
        if self.cache is not None:
            B.require(torch.equal(self.cache['theta'],precision),'frozen theta changed during diagnostic')
            return self.cache
        order=[];increments=[];slices=[];sizes=np.zeros(len(self.groups),np.int64)
        for group in range(100):
            indices=np.flatnonzero(self.groups==group)
            indices=indices[np.argsort(precision.cpu().numpy()[indices],kind='stable')]
            start=len(order);order.extend(indices.tolist());slices.append((start,len(order)))
            times=self.a*precision[torch.as_tensor(indices,device=precision.device)]
            delta=torch.diff(times,prepend=torch.zeros(1,dtype=torch.float64,device=precision.device)) if len(indices) else times
            B.require(bool((delta>=0).all()),'negative Gamma time increment')
            increments.extend(delta.cpu().tolist());sizes[indices]=len(indices)
        order=torch.as_tensor(order,device=precision.device)
        inc=torch.as_tensor(increments,dtype=torch.float64,device=precision.device);positive=inc>0
        B.require(int(positive.sum())>0,'no positive Gamma increments')
        self.stats['minimum_positive_increment']=float(inc[positive].min())
        self.stats['minimum_positive_time']=float((self.a*precision).min())
        self.stats['minimum_independent_shape']=float(((1-self.a)*precision).min())
        self.stats['exact_tied_time_increments']=int((~positive).sum())
        self.cache={'theta':precision,'order':order,'increments':inc,'positive':positive,'slices':slices,
                    'independent_shape':(1-self.a)*precision,'sizes':torch.as_tensor(sizes,device=precision.device),
                    'shared_cdf':gammainc(inc[positive].cpu().numpy(),TINY),
                    'independent_cdf':gammainc(((1-self.a)*precision).cpu().numpy(),TINY)}
        return self.cache

    @torch.no_grad()
    def rates(self,mu,theta):
        B.require(self.a==.5,'literal0 must delegate original bound rates')
        B.require(mu.ndim==3 and mu.shape[-1]==len(self.groups)
                  and bool(torch.isfinite(mu).all()) and bool((mu>=0).all()),'invalid Gamma means')
        c=self.configure(theta);shape=mu.shape[:-1];draws=math.prod(shape)
        positive=c['positive'];inc=c['increments']
        shared_draw=torch.distributions.Gamma(inc[positive],torch.ones_like(inc[positive])).sample(shape)
        independent=torch.distributions.Gamma(c['independent_shape'],torch.ones_like(c['independent_shape'])).sample(shape)
        B.require(bool(torch.isfinite(shared_draw).all()) and bool((shared_draw>=0).all())
                  and bool(torch.isfinite(independent).all()) and bool((independent>=0).all()),'invalid Gamma samples')
        work=torch.zeros((*shape,len(self.groups)),dtype=torch.float64,device=mu.device)
        work[...,positive]=shared_draw
        for start,end in c['slices']:
            if start<end:work[...,start:end]=work[...,start:end].cumsum(-1)
        shared=torch.empty_like(work);shared[...,c['order']]=work
        scale=mu.double()/c['theta']
        rate=(shared+independent)*scale
        result=rate.to(mu.dtype)
        B.require(bool(torch.isfinite(result).all()) and bool((result>=0).all()),'nonfinite/negative correlated rate')
        bounds=(c['sizes'].double()+1)*TINY*scale
        s=self.stats
        s['rate_error_bound_max']=max(s['rate_error_bound_max'],float(bounds.max()))
        s['rate_error_bound_sum']+=float(bounds.sum())
        B.require(s['rate_error_bound_max']<1e-12 and s['rate_error_bound_sum']<1e-8,'Gamma underflow error bound exceeded')
        s['shared_increment_draws']+=draws*int(positive.sum());s['independent_draws']+=draws*len(self.groups)
        s['shared_tiny_hits']+=int((shared_draw==TINY).sum());s['independent_tiny_hits']+=int((independent==TINY).sum())
        for role in ('shared','independent'):
            probabilities=c[role+'_cdf']
            s['expected_'+role+'_tiny_hits']+=draws*float(probabilities.sum())
            s[role+'_tiny_hit_variance']+=draws*float((probabilities*(1-probabilities)).sum())
        lost=rate[(rate>0)&(result==0)]
        s['final_cast_underflows']+=lost.numel()
        s['final_cast_lost_rate_mass']+=float(lost.sum())
        if lost.numel():s['final_cast_lost_rate_max']=max(s['final_cast_lost_rate_max'],float(lost.max()))
        minimum=min(float(shared_draw.min()),float(independent.min()))
        s['minimum_gamma_sample']=minimum if s['minimum_gamma_sample'] is None else min(s['minimum_gamma_sample'],minimum)
        return result

    def patch(self,model):
        B.require(not getattr(model,'_joint_gamma_head',False),'count head patched twice')
        original=model.rates
        def rates(z,c,k,library,gen=None):
            if self.a==0:return original(z,c,k,library,gen)
            mu,theta=model.parameters_nb(z,c,k,library,gen)
            return self.rates(mu,theta)*(library>0)[...,None]
        model.rates=rates;model._joint_gamma_head=True


def projected_kernel(mu,theta,groups,projection,coupling=.5):
    """J.T @ deltaSigma @ J without forming the2000-square count kernel."""
    mu=np.asarray(mu,np.float64);theta=np.asarray(theta,np.float64);groups=partition(groups)
    projection=np.asarray(projection,np.float64)
    B.require(mu.ndim==theta.ndim==1 and mu.shape==theta.shape==groups.shape
              and projection.ndim==2 and projection.shape[0]==len(mu) and np.isfinite(mu).all() and (mu>=0).all()
              and np.isfinite(theta).all() and (theta>0).all() and np.isfinite(projection).all(),'invalid projected kernel input')
    B.require(math.isfinite(coupling) and 0<=coupling<=1,'invalid analytic coupling')
    answer=np.zeros((projection.shape[1],projection.shape[1]),np.float64)
    for group in range(100):
        indices=np.flatnonzero(groups==group);indices=indices[np.argsort(theta[indices],kind='stable')]
        if not len(indices):continue
        if len(indices)==1:continue
        weights=projection[indices]*(mu[indices]/theta[indices])[:,None]
        tail=np.zeros_like(weights);tail[:-1]=np.cumsum(weights[:0:-1],axis=0)[::-1]
        cross=(projection[indices]*mu[indices,None]).T@tail
        answer+=coupling*(cross+cross.T)
    return answer


def logcp_jacobian(mu,projection,gene_sd):
    mu=np.asarray(mu,np.float64);projection=np.asarray(projection,np.float64);sd=np.asarray(gene_sd,np.float64)
    B.require(mu.ndim==sd.ndim==1 and mu.shape==sd.shape and projection.ndim==2
              and projection.shape[0]==len(mu) and np.isfinite(projection).all(),'invalid Jacobian dimensions/projection')
    total=mu.sum()
    B.require(total>0 and np.isfinite(total) and (mu>=0).all() and np.isfinite(mu).all()
              and np.isfinite(sd).all() and (sd>0).all(),'invalid Jacobian means/scales')
    B.require(total!=1.,'logCP total=1 denominator kink is nondifferentiable')
    if total<1:return projection/sd[:,None]*(1e4/(1+1e4*mu))[:,None]
    weighted=projection/sd[:,None]*(1e4/(total+1e4*mu))[:,None]
    return weighted-(weighted*mu[:,None]).sum(0,keepdims=True)/total
