"""Depth-qualified same-donor pair pooling with half-binomial thinning."""
import pair_common as B
np=B.np


def draw(cells,source_libraries,libraries,rng,parts):
    """parts=1 is a mandatory identity control; the fixed candidate uses2."""
    B.require(parts in (1,2),'only fixed kernel or identity control allowed')
    order=np.argsort(source_libraries,kind='stable');sorted_lib=source_libraries[order]
    B.require(len(order)>0 and sorted_lib[-1]>0,'empty/zero-depth source donor')
    first=np.searchsorted(sorted_lib,libraries,side='left');n_ok=len(order)-first
    output=np.zeros((len(libraries),cells.shape[1]),np.int64)
    for _ in range(parts):
        pick=np.where(n_ok>0,first+(rng.random(len(libraries))*np.maximum(n_ok,1)).astype(int),len(order)-1)
        rows=order[np.minimum(pick,len(order)-1)]
        X=cells[rows].toarray().astype(np.float64);sl=X.sum(1)
        p=np.minimum(1.,libraries/np.maximum(sl,1))/parts
        nzr,nzc=np.nonzero(X)
        X[nzr,nzc]=rng.binomial(X[nzr,nzc].astype(np.int64),p[nzr])
        output+=X.astype(np.int64)
    return output,int((n_ok==0).sum())


def generate(real,replicate,draws,role,parts=2):
    n,gn=len(real.donors),len(real.genes);K=256;kpc=64
    arrays=B.BL._alloc(n,draws,gn,kpc,real.W.shape[1])
    train=np.flatnonzero(~real.is_heldout);unmatched=0;total=0;assignments={}
    for t in range(n):
        rng=B.empirical_rng(real,t,replicate,role)
        libraries=rng.choice(real.libs[t],(draws,K),replace=True)
        pool=[i for i in train if i!=t]
        B.require(bool(pool),'no eligible TRAIN source donor')
        sources=B.BL._donor_cycle(rng,pool,draws)
        assignments[str(real.donors[t])]=[str(real.donors[s]) for s in sources]
        for m,source in enumerate(sources):
            counts,missing=draw(real.cells[source],real.libs[source],libraries[m],rng,parts)
            unmatched+=missing;total+=K
            pb,ml,nz,pcs,lib,sq=B.BL._summarise(counts,real,kpc)
            for key,value in zip(B.H.ARRAYS,(pb,ml,nz,pcs,lib,sq)):arrays[key][t,m]=value
    arrays.update(targets=real.donors.copy(),is_heldout=real.is_heldout.copy(),
                  stage=real.stage.copy(),genes=real.genes.copy())
    info={'M':draws,'K':K,'kpc':kpc,'targets':n,'generation_device':'cpu',
          'baseline':'paired_resample_blind','parts':parts,'empirical_role':role,'empirical_stream':replicate,
          'source_selection':'training donors only; target excluded when in training',
          'target_bag_source_donors':assignments,'frac_cells_depth_unmatched':unmatched/max(total,1),
          'depth_rule':'same qualified pool/deepest fallback as frozen baseline; two independent picks per bag cell',
          'count_rule':'sum_j Binomial(X_j,min(1,L/L_j)/parts)',
          'scientific_role':'artificial pooled cells from one TRAIN donor per whole bag'}
    return arrays,info
