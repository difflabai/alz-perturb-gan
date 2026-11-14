"""Explicit signed covariance observations; native ParticleGAN updates unchanged.

The original summary already contains joint Fourier features. This block
adds direct quadratic coordinates, not a new donor model or extra loss.
Only imported explicitly by the isolated study wrapper; no r5 source edit.
"""
import cov_common as C
from arms import Arm,FeatureBlock,register
np=C.np;torch=C.torch


def covariance(x):
    if x.ndim!=3 or x.shape[-1]!=50 or x.shape[-2]<2:
        raise ValueError('covmom requires [bag,cells>=2,50]')
    y=x-x.mean(1,keepdim=True)
    return y.transpose(1,2)@y/(x.shape[1]-1)


def coordinates(pca,random):
    if pca.shape!=random.shape:raise ValueError('PCA/random cell layouts differ')
    p=covariance(pca);r=covariance(random)
    i,j=torch.triu_indices(50,50,offset=1,device=pca.device)
    u,v=torch.triu_indices(50,50,offset=0,device=pca.device)
    weights=torch.where(u==v,r.new_tensor(1.),r.new_tensor(2**.5))
    return torch.cat((p[:,i,j]*(2**.5),r[:,u,v]*weights),1)


def fit_center_scale(values,released):
    values=np.asarray(values,np.float32);released=np.asarray(released,np.float64)
    if values.ndim!=2 or values.shape[1]!=2500 or released.shape!=(len(values),150):
        raise ValueError('wrong reference geometry')
    if not np.isfinite(values).all() or not np.isfinite(released).all():raise ValueError('nonfinite references')
    center=values.astype(np.float64).mean(0).astype(np.float32)
    centered=(values-center).astype(np.float64)
    before=np.square(centered).sum(1).mean();target=np.square(released).sum(1).mean()
    if not before>0 or not target>0:raise ValueError('degenerate reference norm')
    scale=float(format(np.sqrt(target/before),'.6g'))
    after=((values-center)*np.float32(scale)).astype(np.float64)
    energies={'released150_reference_energy':float(target),'covmom_centered_reference_energy':float(before),
              'covmom_scaled_reference_energy':float(np.square(after).sum(1).mean()),
              'PCA_centered_reference_energy':float(np.square(centered[:,:1225]).sum(1).mean()),
              'random_centered_reference_energy':float(np.square(centered[:,1225:]).sum(1).mean()),
              'PCA_scaled_reference_energy':float(np.square(after[:,:1225]).sum(1).mean()),
              'random_scaled_reference_energy':float(np.square(after[:,1225:]).sum(1).mean())}
    return center,scale,energies


class CovarianceBlock(FeatureBlock):
    name='signed_covariance';width=2500

    def __init__(self,arm):
        super().__init__();self.arm=arm;self.info=dict(arm.info)

    def value(self,pca,random):
        return (coordinates(pca,random)-self.arm.center)*self.arm.scale

    def real(self,ctx,idx,ds):
        cf=C.T.standardize_counts(ctx.X[idx],ctx.mean,ctx.sd)
        return self.fake(ctx,cf,None,None)

    def fake(self,ctx,cf,rate,counts):
        return self.value(cf@ctx.W,cf@self.arm.projection)

    def state(self):return {'width':self.width,'sample_covariance_divisor':'cells-1'}


@register
class CovarianceArm(Arm):
    name='d.covmom';needs_Z=False

    @torch.no_grad()
    def setup(self,ctx):
        if ctx.device.type!='cpu':raise ValueError('fixed study is CPU-only')
        if set(ctx.flags)!=set(C.ARM.split(',')) or ctx.bag!=128 or ctx.batch_size!=32:
            raise ValueError('only fixed covmom+pbulk+derivedcap recipe is permitted')
        projection,_,basis,scale=C.DIAG.projections(ctx)
        self.projection=torch.tensor(projection[:,50:],dtype=torch.float32,device=ctx.device)
        self.basis=np.array(basis,copy=True);self.random_scale=np.array(scale,copy=True)
        values=[];released=[];indices=[]
        from released_update import bag_features
        for donor,rows in zip(ctx.ids,ctx.by):
            bags=np.stack([np.random.default_rng(C.B.AS.E.MX.stable_seed(
                'covariance-critic-reference',20261114,ctx.fold,str(donor),b)).choice(rows,128,replace=True)
                for b in range(16)])
            idx=torch.as_tensor(bags,device=ctx.device)
            cf=C.T.standardize_counts(ctx.X[idx],ctx.mean,ctx.sd)
            values.append(coordinates(cf@ctx.W,cf@self.projection).numpy())
            released.append(bag_features(ctx.trainpcs[idx]).numpy());indices.append(bags)
        center,self.scale,energies=fit_center_scale(np.concatenate(values),np.concatenate(released))
        self.center=torch.tensor(center,device=ctx.device)
        import hashlib
        self.reference_indices_sha256=hashlib.sha256(np.stack(indices).tobytes()).hexdigest()
        self.info={'features':2500,'PCA_offdiagonal':1225,'random_upper_including_diagonal':1275,
                   'cross_space_terms':False,'covariance_divisor':'cells-1','reference_bags_per_donor':16,
                   'reference_cells_per_bag':128,'reference_donors':len(ctx.ids),
                   'reference_indices_sha256':self.reference_indices_sha256,
                   'reference_weighting':'equal donors','scale':self.scale,'energies':energies,
                   'random_basis_seed':20261113,'random_sign_canonicalization':False,
                   'total_feature_width':4650,'native_kappa':C.KAPPA,
                   'change':'explicit signed quadratic observations; native PG update unchanged',
                   'gradient':'unchanged straight-through standardized counts via Gamma rate'}

    def feature_blocks(self,ctx):return [CovarianceBlock(self)]

    def state(self):
        return {'projection':self.projection.cpu().clone(),'random_basis':self.basis.copy(),
                'random_scale':self.random_scale.copy(),'center':self.center.cpu().clone(),
                'scale':self.scale,'reference_indices_sha256':self.reference_indices_sha256}
